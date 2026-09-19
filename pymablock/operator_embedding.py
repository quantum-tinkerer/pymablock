"""Operator embeddings generated from a source reference state."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache, cached_property

import sympy
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOperator,
    NumberOrderedForm,
    _NOFTransition,
    _number_symbols,
    _occupation_dimension,
    _one_term,
    find_operators,
    generator_types,
)

__all__ = ["Embedding"]


def _operator_sort_key(operator) -> tuple[int, str]:
    return generator_types.index(type(operator)), str(operator.name)


class Embedding(sympy.Expr):
    r"""Define an effective operator algebra or a finite retained basis.

    Pass this object as ``subspace_eigenvectors`` to ``block_diagonalize``.
    Source states outside the embedding remain available for virtual transitions.

    Parameters
    ----------
    generators : collections.abc.Mapping
        Target lowering operators mapped to source expressions. Supported targets
        are ``SigmaMinus``, ``FermionOp``, ``BosonOp``, and ``LadderOp``. For each
        ``LadderOp``, also give its independent ``NumberOperator`` image. Omit
        this argument to obtain finite matrices from an ordered reference list.
    reference : collections.abc.Mapping or collections.abc.Sequence
        With generators, map every source mode to its integer occupation in the
        target vacuum (index zero for a bilateral ladder). Applying the generator
        adjoints defines the other target states, including their phases.

        Without generators, list distinct occupation dictionaries in matrix-basis
        order. For a matrix source, use ``(matrix_index, occupations)`` pairs;
        a dictionary alone means component zero. All states must declare the same
        modes. Empty dictionaries support ordinary finite matrices.

    Notes
    -----
    Boson occupations are nonnegative, fermion and Pauli occupations are zero or
    one, and bilateral ladder indices may be any integer. Generator images must
    have one independent occupation shift per target and obey its ladder amplitudes.
    Orthonormal linear mode mixing is supported for initially empty modes.

    The perturbative solver requires Hermitian input and H0 diagonal in source
    occupations, including after mode rotation. Matrix sources also require H0
    diagonal in matrix indices and equal source shapes at every order. Reference
    lists produce finite SymPy matrices; generator mappings produce
    ``NumberOrderedForm`` objects, including for infinite targets. These types
    describe the retained block. Off-diagonal NOFs carry the embedding on their
    left or right; the complement block uses source operators. The solver rejects
    bosonic selections requiring occupation inequalities.

    Examples
    --------
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.pauli import SigmaMinus
    >>> a, s = BosonOp("a"), SigmaMinus("s")
    >>> embedding = Embedding({s: a}, reference={a: 0})
    >>> embedding.restrict(a).as_expr() == s
    True
    >>> embedding = Embedding(reference=[{a: 0}, {a: 1}, {a: 2}])
    >>> embedding.restrict(NumberOperator(a)) == sympy.diag(0, 1, 2)
    True

    """

    is_commutative = False

    def __new__(cls, generators=None, reference=None):
        """Compile a structurally reconstructible generator or reference map."""
        if generators is None or generators is sympy.S.NaN:
            reference = tuple(
                (0, state) if isinstance(state, (Mapping, sympy.Dict)) else state
                for state in reference
            )
            basis = _ReferenceBasis([(i, dict(state)) for i, state in reference])
            generators = sympy.S.NaN
            reference = sympy.Tuple(
                *(sympy.Tuple(i, sympy.Dict(state)) for i, state in reference)
            )
        else:
            basis = _GeneratorBasis(dict(generators), reference=dict(reference))
            generators, reference = sympy.Dict(generators), sympy.Dict(reference)
        result = sympy.Expr.__new__(cls, generators, reference)
        result._basis = basis
        return result

    @cached_property
    def _projector(self):
        from pymablock._operator_embedding import _projector

        return _projector(self._basis)

    @cached_property
    def _on_support(self):
        from pymablock._operator_embedding import _support_reducer

        return _support_reducer(self._basis)

    def _clean(self, value):
        return value.applyfunc(self._on_support)

    def _contract(self, value):
        result = self.restrict(self._clean(value))
        return result[0, 0] if isinstance(result, sympy.MatrixBase) else result

    @cache
    def _lift(self, value):
        """Substitute the source generator images, restricted to their representation."""
        basis = self._basis
        if isinstance(basis, _ReferenceBasis):
            return basis._source_scalar(value)
        coordinates = basis._occupation_left_inverse * (
            sympy.Matrix([NumberOperator(op) for op in basis.operators])
            - sympy.Matrix(basis.reference)
        )
        images = dict(zip(map(NumberOperator, basis._target_operators), coordinates))
        for op, generator in zip(basis._target_operators, basis._generators):
            images[op] = generator.form.as_expr()
            images[op.adjoint()] = generator.form.adjoint().as_expr()
        result = NumberOrderedForm.from_expr(
            value.as_expr().xreplace(images), basis.operators
        )
        return self._projector * result * self._projector

    def _attach(self, value, side):
        """Represent X W or W† X without normalizing the source operator."""
        if isinstance(self._basis, _ReferenceBasis) and (
            len(self._basis._references) != 1 or self._basis._references[0][0] != 0
        ):
            raise ValueError(
                "Reference lists use matrices of single-reference attachments"
            )
        value = self._basis._source_scalar(value)
        return NumberOrderedForm(
            value.operators, value.args[1], self, side, validate=False
        )

    def restrict(self, expression):
        """Return ``W† expression W`` in the retained representation.

        Source products are evaluated before compression, including intermediate
        states outside the retained space. Generator mappings return
        ``NumberOrderedForm``; reference lists return a SymPy matrix in list order.
        Generator coefficients retain symbolic spectator occupations. Call
        ``simplify()`` on the result when explicit binary reduction is needed.
        """
        return self._basis._pullback(self._basis._source_form(expression))


class _SourceBasis:
    """Shared normalization of source expressions and occupation symbols."""

    @cached_property
    def _source_placeholders(self):
        return _number_symbols(self.operators)

    def _at_occupations(self, expression, occupations):
        """Evaluate a source number coefficient at the supplied occupations."""
        return expression.xreplace(
            dict(zip(self._source_placeholders, occupations, strict=True))
        )

    def _source_scalar(self, expression) -> NumberOrderedForm:
        """Convert an expression into the source algebra of the embedding."""
        if isinstance(expression, NumberOrderedForm):
            if expression.operators == self.operators:
                return expression
            expression = expression.as_expr()
        expression = sympy.sympify(expression)
        if self._rotation:
            expression = expression.doit().xreplace(self._rotation)
        if set(find_operators(expression)) - set(self.operators):
            raise ValueError("Every source mode must be declared in the reference")
        result = NumberOrderedForm.from_expr(expression, operators=self.operators)
        return result.applyfunc(sympy.simplify) if self._rotation else result


class _GeneratorBasis(_SourceBasis):
    """Symbolic target algebra generated from one reference state."""

    def __init__(self, generators: Mapping, *, reference):
        """Compile the representation generated from the reference."""
        if not isinstance(generators, Mapping) or not isinstance(reference, Mapping):
            raise TypeError("Generators and reference must be mappings")
        if not reference:
            raise ValueError("Specify the source reference occupations")
        self._rotation, generators, reference = _rotate_linear_modes(
            generators, reference
        )
        self.operators, self.reference = _reference_state(reference)
        numbers = {
            op: value
            for op, value in generators.items()
            if isinstance(op, NumberOperator)
        }
        operators = tuple(op for op in generators if op not in numbers)
        if not all(isinstance(op, generator_types) for op in operators):
            raise TypeError("Keys must be target lowering generators or ladder numbers")
        if any(not op.is_annihilation for op in operators):
            raise ValueError("Target generators must be lowering operators")
        operators = tuple(sorted(operators, key=_operator_sort_key))
        self._target_operators = operators
        self._target_dimensions = tuple(map(_occupation_dimension, operators))
        self.coordinate_symbols = tuple(
            sympy.Dummy(
                f"target_{i}",
                integer=True,
                **({} if isinstance(op, LadderOp) else {"nonnegative": True}),
            )
            for i, op in enumerate(operators)
        )
        transitions = []
        for op in operators:
            form = self._source_form(generators[op])
            terms = tuple(_NOFTransition.from_form(form))
            if len(terms) != 1 or not any(terms[0].powers):
                raise NotImplementedError(
                    f"Image of {op} must have one nonzero source occupation shift "
                    "after resolving linear mode mixing"
                )
            transition = terms[0]
            parity = (
                sum(
                    power
                    for source, power in zip(
                        self.operators, transition.powers, strict=True
                    )
                    if isinstance(source, FermionOp)
                )
                % 2
            )
            if parity != isinstance(op, FermionOp):
                raise ValueError("Generator images must preserve fermionic parity")
            transitions.append(transition)
        self._generators = tuple(transitions)
        self._occupation_matrix = sympy.Matrix(
            len(self.operators), len(operators), lambda i, j: transitions[j].powers[i]
        )
        if self._occupation_matrix.rank() != len(operators):
            raise ValueError("Generator shifts must be independent")
        matrix = self._occupation_matrix
        self._occupation_left_inverse = (matrix.T * matrix).inv() * matrix.T
        self.source_occupations = tuple(
            origin + sum(matrix[i, j] * q for j, q in enumerate(self.coordinate_symbols))
            for i, origin in enumerate(self.reference)
        )
        self._validate_domains()
        self.phase = self._reference_phase()
        self._validate_generator_actions()
        expected_numbers = {
            NumberOperator(op) for op in operators if isinstance(op, LadderOp)
        }
        if set(numbers) != expected_numbers:
            raise ValueError(
                "Supply the independent number image for each target LadderOp"
            )
        for op, number in numbers.items():
            index = next(
                i for i, target in enumerate(operators) if NumberOperator(target) == op
            )
            form = self._source_form(number)
            if any(any(powers) for powers in form.terms):
                raise ValueError("A ladder number image must be occupation diagonal")
            expression = form.terms.get((0,) * len(self.operators), sympy.S.Zero)
            expression = self._at_occupations(expression, self.source_occupations)
            self._validate_identity(
                expression - self.coordinate_symbols[index],
                f"Ladder number image {op} must count from the target reference index zero",
            )

    def _source_form(self, expression):
        if isinstance(expression, sympy.MatrixBase):
            raise TypeError("Matrix sources require a list of reference states")
        return self._source_scalar(expression)

    def _validate_domains(self):
        """Check all generated occupations without enumerating target states."""
        for i, op in enumerate(self.operators):
            lower = upper = self.reference[i]
            for coefficient, target, size in zip(
                self._occupation_matrix.row(i),
                self._target_operators,
                self._target_dimensions,
                strict=True,
            ):
                if not coefficient:
                    continue
                if size is None:
                    if isinstance(target, LadderOp):
                        lower, upper = -sympy.oo, sympy.oo
                    elif coefficient > 0:
                        upper = sympy.oo
                    else:
                        lower = -sympy.oo
                else:
                    lower += min(0, coefficient) * (size - 1)
                    upper += max(0, coefficient) * (size - 1)
            if not isinstance(op, LadderOp) and lower < 0:
                raise ValueError("Generators leave the physical source occupation domain")
            if _occupation_dimension(op) is not None and upper >= _occupation_dimension(
                op
            ):
                raise ValueError("Generators overfill a source spin or fermion")

    def _lowering_weight(self, index):
        """Use the same generator action as compression and operator arithmetic."""
        powers = tuple(int(i == index) for i in range(len(self._target_operators)))
        transition = _NOFTransition(
            _one_term(self._target_operators, powers, sympy.S.One), powers
        )
        return transition.symbolic_action(self.coordinate_symbols).weight

    def _reference_phase(self):
        """Generate phases by applying creation operators to the reference."""
        phase = sympy.S.One
        for i, (transition, q, size) in enumerate(
            zip(
                self._generators,
                self.coordinate_symbols,
                self._target_dimensions,
                strict=True,
            )
        ):
            ratio = (
                self._lowering_weight(i)
                / transition.symbolic_action(self.source_occupations).weight
            )
            ratio = ratio.xreplace(
                dict.fromkeys(self.coordinate_symbols[:i], sympy.S.Zero)
            )
            if size is None:
                ratio = sympy.simplify(ratio)
                if ratio.free_symbols.intersection(self.coordinate_symbols):
                    raise NotImplementedError(
                        "Infinite target generators require a constant phase relative to their ladder weights"
                    )
                _require_identity(
                    ratio * sympy.conjugate(ratio) - 1,
                    f"Image of {self._target_operators[i]} must produce normalized target states",
                )
                phase *= ratio**q
            else:
                phase *= 1 - q + q * sympy.simplify(ratio.xreplace({q: sympy.S.One}))
        return sympy.factor(phase)

    def _validate_identity(self, expression, context, substitutions=None):
        """Check only finite coordinates actually occurring in a condition."""
        expression = sympy.simplify(expression.xreplace(substitutions or {}))
        if expression == 0:
            return
        for q, size in zip(self.coordinate_symbols, self._target_dimensions, strict=True):
            if size is not None and q in expression.free_symbols:
                for k in range(size):
                    self._validate_identity(
                        expression, f"{context}, {q}={k}", {q: sympy.Integer(k)}
                    )
                return
        _require_identity(expression, context)

    def _validate_generator_actions(self):
        """Verify normalization and all lowering actions, including cross signs."""
        self._validate_identity(
            self.phase * sympy.conjugate(self.phase) - 1,
            "Generator images must produce normalized target states",
        )
        for i, (transition, q, size) in enumerate(
            zip(
                self._generators,
                self.coordinate_symbols,
                self._target_dimensions,
                strict=True,
            )
        ):
            action = transition.symbolic_action(self.source_occupations).weight
            shifted_phase = self.phase.xreplace({q: q - 1})
            difference = action * self.phase - self._lowering_weight(i) * shifted_phase
            context = f"Image of {self._target_operators[i]} must obey the target algebra"
            if size is None:
                substitutions = (
                    {q: q + 1} if isinstance(self._target_operators[i], BosonOp) else {}
                )
                self._validate_identity(difference, context, substitutions)
            else:
                for k in range(1, size):
                    self._validate_identity(
                        difference, f"{context} at occupation {k}", {q: sympy.Integer(k)}
                    )

    @cached_property
    def _target_placeholders(self):
        return _number_symbols(self._target_operators)

    @cached_property
    def _target_zero(self):
        return NumberOrderedForm(self._target_operators, {}, validate=False)

    @cache
    def _target_shift(self, source_shift: tuple[int, ...]) -> tuple[int, ...] | None:
        """Find the target transition induced by a source occupation shift."""
        source = sympy.Matrix(source_shift)
        result = self._occupation_left_inverse * source
        if self._occupation_matrix * result != source:
            return None
        if any(
            not value.is_Integer or (size is not None and abs(value) >= size)
            for value, size in zip(result, self._target_dimensions, strict=True)
        ):
            return None
        return tuple(map(int, result))

    @cache
    def _project_transition(self, transition: _NOFTransition) -> NumberOrderedForm:
        """Translate one source transition without expanding spectator occupations."""
        powers = self._target_shift(transition.powers)
        if powers is None:
            return self._target_zero
        target_form = _one_term(self._target_operators, powers, sympy.S.One)
        (target_transition,) = _NOFTransition.from_form(target_form)
        source_weight = transition.symbolic_action(self.source_occupations).weight
        target_weight = target_transition.symbolic_action(self.coordinate_symbols).weight
        shifted = {q: q - p for q, p in zip(self.coordinate_symbols, powers)}
        # The phase has unit modulus on the retained domain. Its ratio cancels
        # unchanged factors without expanding binary occupation identities.
        amplitude = (
            source_weight / target_weight * self.phase / self.phase.xreplace(shifted)
        )
        # NOF coefficients sit between creation and annihilation operators.
        # A binary transition fixes its input occupation; spectators stay symbolic.
        initial = {
            q: sympy.Integer(p > 0) if p and size == 2 else n + max(p, 0)
            for q, n, p, size in zip(
                self.coordinate_symbols,
                self._target_placeholders,
                powers,
                self._target_dimensions,
            )
        }
        coefficient = sympy.factor_terms(amplitude.xreplace(initial))
        return _one_term(self._target_operators, powers, coefficient)

    @cache
    def _pullback(self, source):
        """Return ``W† source W`` without enumerating the discarded space."""
        result = self._target_zero
        for transition in _NOFTransition.from_form(source):
            result += self._project_transition(transition)
        return result


class _ReferenceBasis(_SourceBasis):
    """Finite target matrix in an ordered source occupation basis."""

    def __init__(self, reference):
        """Use an ordered orthonormal product basis for a finite matrix target."""
        if isinstance(reference, Mapping):
            raise TypeError("A matrix target requires a list of reference states")
        references = list(reference)
        if not references:
            raise ValueError("Specify at least one reference state")
        states = []
        for reference in references:
            component, occupations = (
                (0, reference) if isinstance(reference, Mapping) else reference
            )
            component = sympy.sympify(component)
            if not component.is_Integer or component < 0:
                raise ValueError("Matrix basis indices must be nonnegative integers")
            operators, state = _reference_state(occupations)
            if states and operators != self.operators:
                raise ValueError("Every reference must declare the same source modes")
            self.operators = operators
            states.append((int(component), state))
        if len(set(states)) != len(states):
            raise ValueError("Reference states must be distinct")
        self._references = tuple(states)
        self._reference_indices = {state: i for i, state in enumerate(states)}
        self._rotation = {}

    def _source_form(self, expression):
        """Use one matrix representation, including 1x1 scalar sources."""
        if not isinstance(expression, sympy.MatrixBase):
            if any(c for c, _ in self._references):
                raise ValueError(
                    "Nonzero reference matrix indices require a matrix source"
                )
            expression = sympy.ImmutableSparseMatrix([[expression]])
        if expression.rows != expression.cols:
            raise ValueError("Source matrices must be square")
        if any(component >= expression.rows for component, _ in self._references):
            raise ValueError("Reference matrix index lies outside the source matrix")
        return sympy.ImmutableSparseMatrix(expression.applyfunc(self._source_scalar))

    @cache
    def _pullback(self, source):
        entries = {}
        for (row, col), entry in source.todok().items():
            for transition in _NOFTransition.from_form(entry):
                for j, (component, state) in enumerate(self._references):
                    if component != col or (action := transition.apply(state)) is None:
                        continue
                    if (
                        i := self._reference_indices.get((row, action.output_state))
                    ) is not None:
                        entries[i, j] = entries.get((i, j), 0) + action.weight
        return sympy.ImmutableSparseMatrix(
            len(self._references), len(self._references), entries
        )


def _reference_state(reference):
    """Validate and order one product occupation state."""
    if not isinstance(reference, Mapping) or not all(
        isinstance(op, generator_types) and op.is_annihilation for op in reference
    ):
        raise TypeError("Reference keys must be source lowering generators")
    operators = tuple(sorted(reference, key=_operator_sort_key))
    state = tuple(sympy.sympify(reference[op]) for op in operators)
    for op, n in zip(operators, state, strict=True):
        size = _occupation_dimension(op)
        if (
            not n.is_Integer
            or (not isinstance(op, LadderOp) and n < 0)
            or (size is not None and n >= size)
        ):
            raise ValueError("Reference occupations lie outside the source algebra")
    return operators, state


def _rotate_linear_modes(generators, reference):
    """Complete orthonormal linear images to a passive source basis rotation.

    Each connected set of mixed modes is rotated independently. Its reference
    must be the empty Fock state, which the rotation preserves. Nonlinear images
    continue through the occupation-shift compiler after the same substitution.
    """
    linear = {}
    for target, image in generators.items():
        expression = (
            image.as_expr()
            if isinstance(image, NumberOrderedForm)
            else sympy.sympify(image)
        )
        modes = tuple(find_operators(expression))
        if len(modes) < 2 or not all(type(op) is type(modes[0]) for op in modes):
            continue
        if not isinstance(modes[0], (BosonOp, FermionOp)):
            continue
        coefficients = tuple(sympy.expand(expression).coeff(op) for op in modes)
        if any(not c.is_commutative for c in coefficients):
            continue
        if (
            sympy.expand(
                expression
                - sum(c * op for c, op in zip(coefficients, modes, strict=True))
            )
            != 0
        ):
            continue
        if not set(modes) <= reference.keys():
            raise ValueError("Every source mode must be declared in the reference")
        linear[target] = (modes, coefficients)
    if not linear:
        return {}, generators, reference

    groups = []
    for modes, _ in linear.values():
        group = set(modes)
        for other in groups[:]:
            if group & other:
                group |= other
                groups.remove(other)
        groups.append(group)

    rotation, compiled, reference = {}, dict(generators), dict(reference)
    for group in groups:
        modes = tuple(sorted(group, key=_operator_sort_key))
        if any(reference[op] != 0 for op in modes):
            raise NotImplementedError(
                "Linear mode mixing requires an empty reference in the mixed modes"
            )
        rows, targets = [], []
        for target, (support, coefficients) in linear.items():
            if not set(support) <= group:
                continue
            weights = dict(zip(support, coefficients, strict=True))
            row = sympy.Matrix([weights.get(op, 0) for op in modes])
            residuals = [
                (previous.conjugate().dot(row), "orthogonal") for previous in rows
            ]
            residuals.append((row.conjugate().dot(row) - 1, "normalized"))
            for residual, requirement in residuals:
                _require_identity(
                    residual, f"Linear image of {target} must be {requirement}"
                )
            rows.append(row)
            targets.append(target)
        # A two-mode completion is nonsingular even for symbolic rotation angles.
        if len(modes) == 2 and len(rows) == 1:
            a, b = rows[0]
            rows.append(sympy.Matrix([-sympy.conjugate(b), sympy.conjugate(a)]))
        # Complete only the discarded modes; declared images are never renormalized.
        for candidate in sympy.eye(len(modes)).columnspace():
            if len(rows) == len(modes):
                break
            row = candidate
            for previous in rows:
                row = row - previous * previous.conjugate().dot(row)
            row = row.applyfunc(sympy.simplify)
            norm = sympy.simplify(row.conjugate().dot(row))
            if norm == 0:
                continue
            if norm.is_zero is None and norm.is_positive is not True:
                raise NotImplementedError(
                    "Cannot prove a nonzero norm while completing the source rotation"
                )
            rows.append((row / sympy.sqrt(norm)).applyfunc(sympy.simplify))
            if len(rows) == len(modes):
                break
        rotated = tuple(
            type(modes[0])(f"__embedding_{sympy.Dummy().dummy_index}") for _ in modes
        )
        for i, op in enumerate(modes):
            image = sum(
                sympy.conjugate(row[i]) * new
                for row, new in zip(rows, rotated, strict=True)
            )
            rotation[op], rotation[op.adjoint()] = image, image.adjoint()
            del reference[op]
        reference.update(dict.fromkeys(rotated, 0))
        compiled.update(zip(targets, rotated, strict=False))
    for target, image in compiled.items():
        if target not in linear:
            expression = (
                image.as_expr()
                if isinstance(image, NumberOrderedForm)
                else sympy.sympify(image)
            )
            compiled[target] = expression.doit().xreplace(rotation)
    return rotation, compiled, reference


def _require_identity(expression, context):
    """Separate a contradicted identity from one not established symbolically."""
    residual = sympy.simplify(expression)
    if residual == 0:
        return
    message = f"{context}; residual: {residual}"
    if residual.is_zero is False:
        raise ValueError(message)
    raise NotImplementedError(f"Cannot establish {message}")
