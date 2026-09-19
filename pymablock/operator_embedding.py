"""Operator embeddings generated from a source reference state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache, cached_property
from itertools import product
from math import prod
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Embedding"]


def _operator_sort_key(operator) -> tuple[int, str]:
    return generator_types.index(type(operator)), str(operator.name)


@dataclass(frozen=True)
class _TargetSpace:
    """Target generators and their occupation domains."""

    operators: tuple
    dimensions: tuple[int | None, ...]

    @cached_property
    def states(self) -> tuple[tuple[int, ...], ...]:
        if None in self.dimensions:
            raise ValueError("An infinite target has no enumerated basis")
        return tuple(product(*(range(size) for size in self.dimensions)))

    @property
    def dimension(self) -> int:
        if None in self.dimensions:
            return sympy.oo
        return prod(self.dimensions)


class Embedding:
    r"""Embed target generators or a finite reference basis into a source algebra.

    ``generators`` maps target lowering operators to source expressions. The
    reference represents the target vacuum (index zero for a bilateral ladder).
    Applying the adjoints, in canonical target order and with the target ladder
    normalization, defines an isometry ``W``. The actual generator image is
    ``W g W† = P G P``: source actions outside the target's occupation range are
    allowed, and remain available during perturbation theory.

    Parameters
    ----------
    generators : collections.abc.Mapping
        Target ``SigmaMinus``, ``FermionOp``, ``BosonOp``, or ``LadderOp`` lowering
        generators mapped to source expressions. Adjoints and number operators
        are derived. For ``LadderOp``, also supply
        ``NumberOperator(target): source_number_expression``; its number is an
        independent generator. Omit this mapping for a finite matrix target.
    reference : collections.abc.Mapping or collections.abc.Sequence
        With generators, map every source mode to its occupation in the target
        vacuum. Without generators, supply an ordered list of such dictionaries:
        each is one retained basis state. For a matrix source, use
        ``(matrix_index, occupations)`` pairs; a dictionary alone means index zero.
        All references must declare the same source modes. Boson occupations are
        nonnegative integers, fermion and Pauli occupations are zero or one, and
        bilateral ladder indices may be any integer. Empty dictionaries support
        ordinary finite matrix sources. Reference states must be distinct.

    Notes
    -----
    The symbolic compiler supports source expressions with one occupation shift
    per target generator and a product occupation reference. It derives the
    occupation changes, checks ladder amplitudes and parity, and fixes relative
    phases from the generators. Orthonormal linear combinations of boson or
    fermion annihilators are rotated automatically when those source modes start
    empty. Other multiple-shift superpositions are not supported. Infinite target
    modes support constant unit phases. Generator mappings return NumberOrderedForm;
    use its ``to_matrix()`` method for an explicit finite representation.

    Reference lists return SymPy matrices in the supplied order. The source may
    be a scalar operator expression or a square SymPy matrix with operator
    entries. All Hamiltonian coefficients must have the same matrix shape.
    Perturbation theory requires H0 diagonal in both source matrix indices and
    occupations. The reference list defines one retained subspace; all other
    source states remain available for virtual transitions.

    The reference fixes overall phase to one. Target fermions use the canonical
    NOF ordering; no source-to-target fermion phase convention is an extra input.
    ``restrict(A)`` evaluates the full source expression before compression.

    Examples
    --------
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.pauli import SigmaMinus
    >>> a, s = BosonOp("a"), SigmaMinus("s")
    >>> embedding = Embedding({s: a}, reference={a: 0})
    >>> embedding.restrict(a).as_expr() == s
    True

    A finite target needs only a list of references:

    >>> matrix_embedding = Embedding(reference=[{a: 0}, {a: 1}, {a: 2}])
    >>> matrix_embedding.restrict(NumberOperator(a)) == sympy.diag(0, 1, 2)
    True

    """

    def __init__(self, generators: Mapping | None = None, *, reference):
        """Compile the representation generated from the reference."""
        self._references = None
        if generators is None:
            self._init_references(reference)
            return
        if not isinstance(generators, Mapping) or not isinstance(reference, Mapping):
            raise TypeError("Generators and reference must be mappings")
        if not reference:
            raise ValueError("Specify the source reference occupations")
        if not all(
            isinstance(op, generator_types) and op.is_annihilation for op in reference
        ):
            raise TypeError("Reference keys must be source lowering generators")
        self._rotation, generators, reference = _rotate_linear_modes(
            generators, reference
        )
        self.operators = tuple(sorted(reference, key=_operator_sort_key))
        self.reference = tuple(sympy.sympify(reference[op]) for op in self.operators)
        for op, value in zip(self.operators, self.reference, strict=True):
            if (
                not value.is_Integer
                or (not isinstance(op, LadderOp) and value < 0)
                or (
                    _occupation_dimension(op) is not None
                    and value >= _occupation_dimension(op)
                )
            ):
                raise ValueError("Reference occupations lie outside the source algebra")
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
        self.target = _TargetSpace(
            operators, tuple(map(_occupation_dimension, operators))
        )
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
            expression = expression.xreplace(
                dict(zip(self._source_placeholders, self.source_occupations, strict=True))
            )
            self._validate_identity(
                expression - self.coordinate_symbols[index],
                f"Ladder number image {op} must count from the target reference index zero",
            )

    def _init_references(self, references):
        """Use an ordered orthonormal product basis for a finite matrix target."""
        if isinstance(references, Mapping):
            raise TypeError("A matrix target requires a list of reference states")
        references = list(references)
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
            if not isinstance(occupations, Mapping) or not all(
                isinstance(op, generator_types) and op.is_annihilation
                for op in occupations
            ):
                raise TypeError(
                    "Reference occupations must map source lowering generators to integers"
                )
            operators = tuple(sorted(occupations, key=_operator_sort_key))
            if states and operators != self.operators:
                raise ValueError("Every reference must declare the same source modes")
            self.operators = operators
            state = tuple(sympy.sympify(occupations[op]) for op in operators)
            if any(
                not n.is_Integer
                or (not isinstance(op, LadderOp) and n < 0)
                or (
                    _occupation_dimension(op) is not None
                    and n >= _occupation_dimension(op)
                )
                for op, n in zip(operators, state, strict=True)
            ):
                raise ValueError("Reference occupations lie outside the source algebra")
            states.append((int(component), state))
        if len(set(states)) != len(states):
            raise ValueError("Reference states must be distinct")
        self._references = tuple(states)
        self._reference_indices = {state: i for i, state in enumerate(states)}
        self._rotation = {}

    def _validate_domains(self):
        """Check all generated occupations without enumerating target states."""
        for i, op in enumerate(self.operators):
            lower = upper = self.reference[i]
            for coefficient, target, size in zip(
                self._occupation_matrix.row(i),
                self.target.operators,
                self.target.dimensions,
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
        powers = tuple(int(i == index) for i in range(len(self.target.operators)))
        transition = _NOFTransition(
            _one_term(self.target.operators, powers, sympy.S.One), powers
        )
        return transition.symbolic_action(self.coordinate_symbols).weight

    def _reference_phase(self):
        """Generate phases by applying creation operators to the reference."""
        phase = sympy.S.One
        for i, (transition, q, size) in enumerate(
            zip(
                self._generators,
                self.coordinate_symbols,
                self.target.dimensions,
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
                    f"Image of {self.target.operators[i]} must produce normalized target states",
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
        for q, size in zip(self.coordinate_symbols, self.target.dimensions, strict=True):
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
                self.target.dimensions,
                strict=True,
            )
        ):
            action = transition.symbolic_action(self.source_occupations).weight
            shifted_phase = self.phase.xreplace({q: q - 1})
            difference = action * self.phase - self._lowering_weight(i) * shifted_phase
            context = f"Image of {self.target.operators[i]} must obey the target algebra"
            if size is None:
                substitutions = (
                    {q: q + 1} if isinstance(self.target.operators[i], BosonOp) else {}
                )
                self._validate_identity(difference, context, substitutions)
            else:
                for k in range(1, size):
                    self._validate_identity(
                        difference, f"{context} at occupation {k}", {q: sympy.Integer(k)}
                    )

    def restrict(self, expression):
        """Return ``W† expression W`` after evaluating the full source product."""
        return self._pullback(self._source_form(expression))

    def encode(self, state: tuple[int, ...]) -> tuple[int, ...]:
        """Return occupations in the compiled source basis for a target tuple.

        With linear mode mixing, these are occupations of the rotated modes,
        not of the original source operators.
        """
        if self._references is not None:
            raise TypeError(
                "Reference-list embeddings already specify their source basis"
            )
        if len(state) != len(self.target.operators):
            raise ValueError("State has the wrong number of target occupations")
        for value, op, size in zip(
            state, self.target.operators, self.target.dimensions, strict=True
        ):
            if (
                not sympy.sympify(value).is_Integer
                or (not isinstance(op, LadderOp) and value < 0)
                or (size is not None and value >= size)
            ):
                raise ValueError("State lies outside the target occupation basis")
        substitutions = dict(
            zip(self.coordinate_symbols, map(sympy.Integer, state), strict=True)
        )
        return tuple(
            int(value.xreplace(substitutions)) for value in self.source_occupations
        )

    @cached_property
    def _source_placeholders(self):
        return _number_symbols(self.operators)

    @cached_property
    def _target_placeholders(self):
        return _number_symbols(self.target.operators)

    @cached_property
    def _target_identity(self):
        if self._references is not None:
            return sympy.ImmutableSparseMatrix.eye(len(self._references))
        return _one_term(
            self.target.operators, (0,) * len(self.target.operators), sympy.S.One
        )

    @cached_property
    def _target_zero(self):
        if self._references is not None:
            return sympy.ImmutableSparseMatrix.zeros(len(self._references))
        return NumberOrderedForm(self.target.operators, {}, validate=False)

    @cache
    def _target_shift(self, source_shift: tuple[int, ...]) -> tuple[int, ...] | None:
        """Find the target transition induced by a source occupation shift."""
        source = sympy.Matrix(source_shift)
        result = self._occupation_left_inverse * source
        if self._occupation_matrix * result != source:
            return None
        if any(
            not value.is_Integer or (size is not None and abs(value) >= size)
            for value, size in zip(result, self.target.dimensions, strict=True)
        ):
            return None
        return tuple(map(int, result))

    def _pullback_weight(
        self,
        transition: _NOFTransition,
        _target_shift: tuple[int, ...],
    ) -> sympy.Expr:
        """Return the target diagonal weight of a pulled-back transition."""
        source_action = transition.symbolic_action(self.source_occupations)
        shifted = {
            symbol: symbol - power
            for symbol, power in zip(self.coordinate_symbols, _target_shift, strict=True)
        }
        phase_ratio = self.phase * sympy.conjugate(self.phase.xreplace(shifted))
        amplitude = source_action.weight * phase_ratio
        amplitude = amplitude.xreplace(self._transition_support(_target_shift))
        return sympy.expand(amplitude.xreplace(self._initial_to_middle(_target_shift)))

    def _transition_support(
        self, _target_shift: tuple[int, ...]
    ) -> dict[sympy.Symbol, sympy.Expr]:
        """Return support values forced by the retained transition."""
        return {
            symbol: sympy.S.One if power > 0 else sympy.S.Zero
            for symbol, power, size in zip(
                self.coordinate_symbols,
                _target_shift,
                self.target.dimensions,
                strict=True,
            )
            if power and size == 2
        }

    def _initial_to_middle(
        self, _target_shift: tuple[int, ...]
    ) -> dict[sympy.Symbol, sympy.Expr]:
        """Translate input coordinates to NOF middle coordinates."""
        support = self._transition_support(_target_shift)
        return {
            symbol: placeholder + max(power, 0)
            for symbol, placeholder, power in zip(
                self.coordinate_symbols,
                self._target_placeholders,
                _target_shift,
                strict=True,
            )
            if symbol not in support
        }

    @cache
    def _support_substitutions(
        self,
        transition: _NOFTransition,
        _target_shift: tuple[int, ...],
    ) -> dict[sympy.Symbol, sympy.Expr]:
        """Infer Boolean coordinates fixed by a composed transition."""
        after_target = {
            symbol: symbol - power
            for symbol, power in zip(self.coordinate_symbols, _target_shift, strict=True)
        }
        before_source = tuple(
            occupation.xreplace(after_target) for occupation in self.source_occupations
        )
        equations = list(transition.support_equations(before_source))
        equations.extend(
            symbol - (1 if power > 0 else 0)
            for symbol, power, size in zip(
                self.coordinate_symbols,
                _target_shift,
                self.target.dimensions,
                strict=True,
            )
            if power and size == 2
        )
        return _boolean_solutions(equations, self.coordinate_symbols)

    def _source_form(self, expression):
        """Normalize scalar or matrix source expressions without truncation."""
        if isinstance(expression, sympy.MatrixBase):
            if self._references is None:
                raise TypeError("Matrix sources require a list of reference states")
            if expression.rows != expression.cols:
                raise ValueError("Source matrices must be square")
            if any(component >= expression.rows for component, _ in self._references):
                raise ValueError("Reference matrix index lies outside the source matrix")
            return sympy.ImmutableSparseMatrix(expression.applyfunc(self._source_scalar))
        if self._references is not None and any(c for c, _ in self._references):
            raise ValueError("Nonzero reference matrix indices require a matrix source")
        return self._source_scalar(expression)

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

    @cache
    def _project_transition(
        self,
        transition: _NOFTransition,
    ) -> NumberOrderedForm:
        """Pull back one transition and encode it as a target NOF."""
        target_powers = self._target_shift(transition.powers)
        if target_powers is None:
            return self._target_zero
        amplitude = self._pullback_weight(transition, target_powers)
        if amplitude == 0:
            return self._target_zero
        # Divide out the target monomial's ladder weight and Fock sign: the
        # coefficient supplies only the remaining source matrix element.
        target_form = _one_term(self.target.operators, target_powers, sympy.S.One)
        (target_transition,) = _NOFTransition.from_form(target_form)
        target_weight = target_transition.symbolic_action(self.coordinate_symbols).weight
        target_weight = target_weight.xreplace(
            self._transition_support(target_powers)
        ).xreplace(self._initial_to_middle(target_powers))
        amplitude = sympy.cancel(amplitude / target_weight)
        # Coefficients are already in the NOF middle coordinates. Multiplying
        # by a diagonal operator on the right would shift boson coefficients.
        return (
            _one_term(self.target.operators, target_powers, amplitude)
            * self._target_identity
        )

    @cache
    def _pullback(self, source):
        """Return ``W† source W`` without enumerating the discarded space."""
        source = self._source_form(source)
        result = self._target_zero
        if self._references is not None:
            entries = {}
            for row, column, entry in _source_entries(source):
                for transition in _NOFTransition.from_form(entry):
                    for j, (component, state) in enumerate(self._references):
                        if component != column:
                            continue
                        action = transition.apply(state)
                        if action is None:
                            continue
                        i = self._reference_indices.get((row, action.output_state))
                        if i is not None:
                            entries[i, j] = entries.get((i, j), 0) + action.weight
            return sympy.ImmutableSparseMatrix(result.rows, result.cols, entries)
        for transition in _NOFTransition.from_form(source):
            result += self._project_transition(transition)
        return NumberOrderedForm(
            result.operators,
            {
                powers: coefficient
                for powers, coefficient in result.terms.items()
                if coefficient != 0
            },
            validate=False,
        )


def _source_entries(source):
    """Iterate matrix entries, treating scalar sources as one component."""
    if isinstance(source, sympy.MatrixBase):
        for (row, column), entry in source.todok().items():
            yield row, column, entry
    else:
        yield 0, 0, source


def _boolean_solutions(
    equations: Sequence[sympy.Expr],
    variables: tuple[sympy.Symbol, ...],
) -> dict[sympy.Symbol, sympy.Expr]:
    """Return Boolean variables uniquely fixed by a small equation system."""
    if not equations:
        return {}
    matrix, right_hand_side = sympy.linear_eq_to_matrix(equations, variables)
    reduced, pivots = matrix.row_join(right_hand_side).rref()
    substitutions = {}
    for row, pivot in enumerate(pivots):
        if pivot >= len(variables):
            return {}
        if any(
            reduced[row, column] != 0
            for column in range(len(variables))
            if column != pivot
        ):
            continue
        value = reduced[row, -1]
        if value in (sympy.S.Zero, sympy.S.One):
            substitutions[variables[pivot]] = value
    return substitutions


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
