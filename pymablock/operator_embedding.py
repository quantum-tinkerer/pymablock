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
    _divide_coefficients,
    _NOFTransition,
    _number_symbols,
    _occupation_dimension,
    _spectral_projector,
    find_operators,
    generator_types,
)
from pymablock.series import BlockSeries, zero

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
        """Select the joint spectrum of retained number operators and constraints."""
        basis = self._basis
        numbers = basis._source_placeholders
        if isinstance(basis, _ReferenceBasis):
            spectra = list(zip(numbers, ((n,) for n in basis._references[0][1])))
        else:
            offsets = sympy.Matrix(numbers) - sympy.Matrix(basis.reference)
            spectra = [
                (sympy.expand(normal.dot(offsets)), (0,))
                for normal in basis._occupation_matrix.T.nullspace()
            ]
            target = basis._occupation_left_inverse * offsets
            physical = {
                n: sympy.Dummy(integer=True, nonnegative=True)
                for source, n in zip(basis.operators, numbers)
                if not isinstance(source, LadderOp)
            }
            for q, op, size in zip(
                target, basis._target_operators, basis._target_dimensions
            ):
                if (
                    isinstance(op, BosonOp)
                    and q.xreplace(physical).is_nonnegative is not True
                ):
                    raise NotImplementedError(
                        "This bosonic embedding requires an occupation inequality"
                    )
                spectra.append((q, range(size) if size is not None else sympy.S.Integers))
        indicator = sympy.prod(
            _spectral_projector(q, spectrum) for q, spectrum in spectra
        )
        return NumberOrderedForm(basis.operators, {(0,) * len(numbers): indicator}) * 1

    def _contract(self, value):
        result = self.restrict(value)
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
            image = NumberOrderedForm(
                generator.operators,
                {generator.powers: generator.coefficient},
                validate=False,
            )
            images[op] = image.as_expr()
            images[op.adjoint()] = image.adjoint().as_expr()
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

    def _prepare(self, hamiltonian):
        """Return rectangular Hamiltonian blocks and their Sylvester solver."""
        if hamiltonian.shape:
            raise ValueError("Structured embeddings require an unseparated Hamiltonian.")
        basis = self._basis
        finite = isinstance(basis, _ReferenceBasis)
        origin = (0,) * hamiltonian.n_infinite
        source_h0 = basis._source_form(hamiltonian[origin])
        h0 = source_h0 if finite else sympy.ImmutableMatrix([[source_h0]])
        modes, numbers = basis.operators, basis._source_placeholders
        if any(
            i != j or any(any(p) for p in x.terms) for (i, j), x in h0.todok().items()
        ):
            raise ValueError("Structured embeddings currently require diagonal H0")
        vacuum = (0,) * len(modes)
        if finite:
            entry_embedding = Embedding(reference=[dict(zip(modes, vacuum))])
            w = sympy.zeros(h0.rows, len(basis._references))
            for col, (row, state) in enumerate(basis._references):
                monomial = NumberOrderedForm(
                    modes, {tuple(-n for n in state): sympy.S.One}
                )
                (transition,) = _NOFTransition.from_form(monomial)
                w[row, col] = entry_embedding._attach(
                    monomial / transition.apply(vacuum).weight, 1
                )
            w = sympy.ImmutableMatrix(w)
        else:
            entry_embedding = self
            w = sympy.ImmutableMatrix([[self._attach(sympy.S.One, 1)]])
        frames = (w, sympy.eye(h0.rows) - w * w.adjoint())
        retained = w.adjoint() * h0 * w
        energies = [
            x.terms.get(vacuum, sympy.S.Zero) if x != 0 else sympy.S.Zero
            for x in h0.diagonal()
        ]

        coordinates = () if finite else basis.coordinate_symbols
        occupations = vacuum if finite else basis.source_occupations
        sizes = () if finite else basis._target_dimensions
        binary = [q for q, size in zip(coordinates, sizes) if size == 2]

        def divide_scalar(value, row, col):
            if value == 0 or value.is_zero:
                return sympy.S.Zero
            value = basis._source_scalar(value.source)
            energy = (
                retained[col, col]
                if finite
                else basis._at_occupations(energies[col], occupations)
            )
            terms = {}
            for transition in _NOFTransition.from_form(value):
                powers = transition.powers
                action = transition.symbolic_action(occupations)
                if action.weight == 0:
                    continue
                middle = [n - max(p, 0) for n, p in zip(occupations, powers)]
                coefficient = basis._at_occupations(value.terms[powers], middle)
                denominator = sympy.expand(
                    basis._at_occupations(energies[row], action.output_state) - energy
                )
                try:
                    result = _divide_coefficients(
                        coefficient, denominator, binary, coordinates, action.weight
                    )
                except ValueError as error:
                    raise ZeroDivisionError(str(error)) from error
                if not finite:
                    incoming = sympy.Matrix(
                        [n + max(p, 0) for n, p in zip(numbers, powers)]
                    )
                    target = basis._occupation_left_inverse * (
                        incoming - sympy.Matrix(basis.reference)
                    )
                    result = result.xreplace(dict(zip(coordinates, target)))
                terms[powers] = result
            return NumberOrderedForm(
                basis.operators, terms, entry_embedding, 1, validate=False
            )

        def solve(value, index):
            reverse = index[:2] == (0, 1)
            source = value.adjoint() if reverse else value
            result = sympy.ImmutableMatrix(
                source.rows, source.cols, lambda i, j: divide_scalar(source[i, j], i, j)
            )
            return -result.adjoint() if reverse else result

        def evaluate(i, j, *order):
            source = hamiltonian[tuple(order)]
            if source is zero or (i != j and tuple(order) == origin):
                return zero
            source = basis._source_form(source)
            if not finite:
                source = sympy.ImmutableMatrix([[source]])
            if source.shape != h0.shape:
                raise ValueError(
                    "All Hamiltonian coefficients must have the same source matrix shape"
                )
            result = frames[i].adjoint() * source * frames[j]
            return zero if result.is_zero_matrix else result if finite else result[0, 0]

        options = dict(
            n_infinite=hamiltonian.n_infinite, dimension_names=hamiltonian.dimension_names
        )
        blocks = BlockSeries(eval=evaluate, shape=(2, 2), **options)
        return blocks, solve


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
        self._validate_generator_algebra()
        self.phase = self._reference_phase()
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
        transition = _NOFTransition(self._target_operators, powers, sympy.S.One)
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
                phase *= ratio**q
            else:
                phase *= 1 - q + q * sympy.simplify(ratio.xreplace({q: sympy.S.One}))
        return sympy.factor(phase)

    def _validate_identity(self, expression, context):
        """Reduce polynomial binary identities without visiting occupation states."""
        expression = sympy.expand(expression)
        for q, size in zip(self.coordinate_symbols, self._target_dimensions):
            if size == 2 and q in expression.free_symbols and expression.is_polynomial(q):
                expression = sympy.rem(expression, q**2 - q, q)
        _require_identity(expression, context)

    def _validate_generator_algebra(self):
        """Check local norms and graded commutation on the retained lattice.

        The occupation boundaries fix the vacuum and binary truncation. Norms
        then fix the same-mode algebra; pairwise lowering relations also fix
        the mixed adjoint relations by reversing an edge of each lattice square.
        """
        weights = [
            g.symbolic_action(self.source_occupations).weight for g in self._generators
        ]
        coordinates = self.coordinate_symbols
        active = {
            q: sympy.S.One if size == 2 else q + 1
            for q, size in zip(coordinates, self._target_dimensions)
        }
        for i, q in enumerate(coordinates):
            norm = (
                weights[i] * sympy.conjugate(weights[i]) - self._lowering_weight(i) ** 2
            )
            self._validate_identity(
                norm.xreplace({q: active[q]}),
                "Generator images must produce normalized target states",
            )
            for j, other in enumerate(coordinates[:i]):
                sign = (
                    -1
                    if all(
                        isinstance(self._target_operators[k], FermionOp) for k in (i, j)
                    )
                    else 1
                )
                relation = weights[j] * weights[i].xreplace(
                    {other: other - 1}
                ) - sign * weights[i] * weights[j].xreplace({q: q - 1})
                self._validate_identity(
                    relation.xreplace({q: active[q], other: active[other]}),
                    "Generator images must obey the target algebra",
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
        target_transition = _NOFTransition(self._target_operators, powers, sympy.S.One)
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
        return NumberOrderedForm(
            self._target_operators, {powers: coefficient}, validate=False
        )

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


def _complete_orthonormal_rows(rows):
    """Keep the declared rows fixed and complete their orthonormal basis."""
    gram = rows * rows.adjoint() - sympy.eye(rows.rows)
    for i in range(rows.rows):
        for j in range(i + 1):
            requirement = "normalized" if i == j else "orthogonal"
            _require_identity(gram[i, j], f"Linear images must be {requirement}")
    # This completion stays nonsingular for all symbolic two-mode angles.
    if rows.shape == (1, 2):
        a, b = rows
        return rows.col_join(sympy.Matrix([[-sympy.conjugate(b), sympy.conjugate(a)]]))
    for candidate in sympy.eye(rows.cols).columnspace():
        if rows.rows == rows.cols:
            break
        row = candidate.T
        row = (row - row * rows.adjoint() * rows).applyfunc(sympy.simplify)
        norm = sympy.simplify((row * row.adjoint())[0])
        if norm == 0:
            continue
        if norm.is_zero is None and norm.is_positive is not True:
            raise NotImplementedError(
                "Cannot prove a nonzero norm while completing the source rotation"
            )
        rows = rows.col_join((row / sympy.sqrt(norm)).applyfunc(sympy.simplify))
    return rows


def _rotate_linear_modes(generators, reference):
    """Recognize linear images, complete each mixed group, and change source basis.

    Groups must start in their vacuum. Keeping disconnected groups separate
    preserves unmixed modes and allows independent bosonic and fermionic rotations.
    """
    expressions = {
        target: image.as_expr()
        if isinstance(image, NumberOrderedForm)
        else sympy.sympify(image)
        for target, image in generators.items()
    }
    linear = {}
    for target, expression in expressions.items():
        modes = tuple(find_operators(expression))
        if len(modes) < 2 or not isinstance(modes[0], (BosonOp, FermionOp)):
            continue
        if not all(type(op) is type(modes[0]) for op in modes):
            continue
        expression = sympy.expand(expression)
        coefficients = {op: expression.coeff(op) for op in modes}
        if any(not c.is_commutative for c in coefficients.values()):
            continue
        if sympy.expand(expression - sum(c * op for op, c in coefficients.items())) != 0:
            continue
        if not coefficients.keys() <= reference.keys():
            raise ValueError("Every source mode must be declared in the reference")
        linear[target] = coefficients
    if not linear:
        return {}, generators, reference

    groups = []
    for coefficients in linear.values():
        group = set(coefficients)
        for other in groups[:]:
            if group & other:
                group |= other
                groups.remove(other)
        groups.append(group)

    rotation, compiled, reference = {}, {}, dict(reference)
    for group in groups:
        modes = tuple(sorted(group, key=_operator_sort_key))
        if any(reference[op] != 0 for op in modes):
            raise NotImplementedError(
                "Linear mode mixing requires an empty reference in the mixed modes"
            )
        targets = [
            target
            for target, coefficients in linear.items()
            if coefficients.keys() <= group
        ]
        rows = _complete_orthonormal_rows(
            sympy.Matrix(
                [[linear[target].get(op, 0) for op in modes] for target in targets]
            )
        )
        rotated = tuple(
            type(modes[0])(f"__embedding_{sympy.Dummy().dummy_index}") for _ in modes
        )
        for op, image in zip(modes, rows.adjoint() * sympy.Matrix(rotated), strict=True):
            rotation[op], rotation[op.adjoint()] = image, image.adjoint()
            del reference[op]
        reference.update(dict.fromkeys(rotated, 0))
        compiled.update(zip(targets, rotated, strict=False))
    compiled.update(
        {
            target: expression.doit().xreplace(rotation)
            for target, expression in expressions.items()
            if target not in linear
        }
    )
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
