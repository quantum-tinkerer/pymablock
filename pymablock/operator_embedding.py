"""Source occupation-state selections and their target operators."""

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
from sympy.physics.quantum.pauli import SigmaMinus
from sympy.physics.quantum.spin import JminusOp, JzOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOperator,
    NumberOrderedForm,
    generator_types,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ["Embedding"]


def _operator_sort_key(operator) -> tuple[int, str]:
    if isinstance(operator, JminusOp):
        return len(generator_types), str(operator.name)
    return generator_types.index(type(operator)), str(operator.name)


@dataclass(frozen=True)
class _TargetSpace:
    """Declared generators and their finite occupation bases."""

    operators: tuple
    dimensions: tuple[int, ...]

    @cached_property
    def states(self) -> tuple[tuple[int, ...], ...]:
        return tuple(product(*(range(size) for size in self.dimensions)))

    @property
    def dimension(self) -> int:
        return prod(self.dimensions)


class Embedding:
    """Select source occupation states and define their target operators.

    Parameters
    ----------
    target : collections.abc.Sequence or collections.abc.Mapping
        Target ``SigmaMinus`` or ``FermionOp`` annihilation generators. Their
        dimension is two. For higher spins, pass e.g. ``{JminusOp("S"): 3}``;
        occupations may then depend on ``JzOp("S")`` in units of hbar=1.
        Generators are ordered canonically by type and name. A target containing
        a higher spin uses finite matrices ordered by increasing occupations
        (increasing magnetic quantum number for higher spins).
    occupations : collections.abc.Mapping
        Source annihilation generators mapped to integer affine expressions in
        target ``NumberOperator`` objects, or target ``JzOp`` objects. These are
        occupation constraints, not substitutions for source operators. Source
        modes must be listed even when their occupation is fixed.

    Notes
    -----
    Spin targets use the source occupation-basis phase convention. Fermionic
    targets currently require a direct one-to-one assignment of each retained
    fermion to a source fermion. Other source occupations may depend on target
    spins or be fixed. The fermionic phase accounts for spectator occupations
    and mode permutations so that retained fermionic generators map to their
    declared source generators. Spin and fermion targets may be combined.

    The map must be injective and have physical integer source occupations.
    Only full-rank affine occupation rules are supported. This restricted
    contract permits validation without enumerating a source or target basis.
    Arbitrary superpositions require a prior source-basis rotation or explicit
    subspace eigenvector matrices; this constructor does not perform rotations.

    Examples
    --------
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.pauli import SigmaMinus
    >>> from pymablock.number_ordered_form import NumberOperator as N
    >>> a, s = BosonOp("a"), SigmaMinus("s")
    >>> embedding = Embedding(target=(s,), occupations={a: N(s)})
    >>> embedding.encode((1,))
    (1,)

    """

    def __init__(
        self,
        *,
        target: Sequence | Mapping,
        occupations: Mapping,
    ):
        """Compile and validate the target-to-source occupation map."""
        if not occupations:
            raise ValueError("An embedding must specify source occupations")
        dimensions = dict(target) if isinstance(target, Mapping) else None
        operators = tuple(target)
        if len(operators) != len(set(operators)):
            raise ValueError("Target generators must be distinct")
        if not all(isinstance(op, (SigmaMinus, FermionOp, JminusOp)) for op in operators):
            raise TypeError("Targets must be spin or fermion lowering operators")
        if any(isinstance(op, FermionOp) and not op.is_annihilation for op in operators):
            raise ValueError("Target fermions must be annihilation operators")
        operators = tuple(sorted(operators, key=_operator_sort_key))
        sizes = []
        for op in operators:
            size = dimensions[op] if dimensions is not None else 2
            if isinstance(op, JminusOp) and dimensions is None:
                raise ValueError("A higher-spin target requires an explicit dimension")
            size = sympy.sympify(size)
            if not size.is_Integer or size < 2:
                raise ValueError("Target dimensions must be integers of at least two")
            if not isinstance(op, JminusOp) and size != 2:
                raise ValueError("SigmaMinus and fermion targets have dimension two")
            sizes.append(int(size))
        self.target = _TargetSpace(operators, tuple(sizes))
        fermions = tuple(op for op in operators if isinstance(op, FermionOp))
        self.target_is_nof = not any(isinstance(op, JminusOp) for op in operators)
        if not all(isinstance(op, generator_types) for op in occupations):
            raise TypeError("Source keys must be supported annihilation generators")
        if not all(op.is_annihilation for op in occupations):
            raise ValueError("Source keys must be annihilation generators")
        self.operators = tuple(sorted(occupations, key=_operator_sort_key))
        self.coordinate_symbols = tuple(
            sympy.Dummy(f"target_{i}", integer=True, nonnegative=True)
            for i in range(len(operators))
        )
        substitutions = {
            (JzOp(op.name) if isinstance(op, JminusOp) else NumberOperator(op)): (
                symbol - sympy.Rational(size - 1, 2)
                if isinstance(op, JminusOp)
                else symbol
            )
            for op, symbol, size in zip(
                operators, self.coordinate_symbols, sizes, strict=True
            )
        }
        self.source_occupations = tuple(
            sympy.expand(sympy.sympify(occupations[op]).xreplace(substitutions))
            for op in self.operators
        )
        self._validate_occupations()
        self.phase = sympy.S.One
        if fermions:
            self.phase = self._fermion_phase()

    def _validate_occupations(self) -> None:
        """Prove the supported affine map is physical and injective."""
        rows = []
        origin = dict.fromkeys(self.coordinate_symbols, sympy.S.Zero)
        for operator, expression in zip(
            self.operators, self.source_occupations, strict=True
        ):
            constant = expression.xreplace(origin)
            coefficients = tuple(
                sympy.diff(expression, x) for x in self.coordinate_symbols
            )
            if not constant.is_Integer or any(not c.is_Integer for c in coefficients):
                raise ValueError(
                    "Occupations must be integer affine expressions in declared target numbers"
                )
            if (
                sympy.expand(
                    expression
                    - constant
                    - sum(
                        c * x
                        for c, x in zip(
                            coefficients, self.coordinate_symbols, strict=True
                        )
                    )
                )
                != 0
            ):
                raise ValueError("Only affine occupation rules are supported")
            minimum = constant + sum(
                min(0, c) * (size - 1)
                for c, size in zip(coefficients, self.target.dimensions, strict=True)
            )
            maximum = constant + sum(
                max(0, c) * (size - 1)
                for c, size in zip(coefficients, self.target.dimensions, strict=True)
            )
            if not isinstance(operator, LadderOp) and minimum < 0:
                raise ValueError("Source occupations must be nonnegative")
            if isinstance(operator, (FermionOp, SigmaMinus)) and maximum > 1:
                raise ValueError(
                    "Source spin and fermion occupations must be zero or one"
                )
            rows.append(coefficients)
        matrix = sympy.Matrix(rows)
        if matrix.rank() != len(self.coordinate_symbols):
            raise ValueError("Occupation rules must have full column rank (be injective)")
        self._occupation_matrix = matrix
        self._occupation_left_inverse = (matrix.T * matrix).inv() * matrix.T

    def _fermion_phase(self) -> sympy.Expr:
        """Fix relative signs for direct retained fermionic generators."""
        fermion_coordinates = {
            i: coordinate
            for i, (operator, coordinate) in enumerate(
                zip(self.target.operators, self.coordinate_symbols, strict=True)
            )
            if isinstance(operator, FermionOp)
        }
        mapped = []
        spectator_parity = sympy.S.One
        phase = sympy.S.One
        for operator, occupation in zip(
            self.operators, self.source_occupations, strict=True
        ):
            matches = [
                i
                for i, coordinate in fermion_coordinates.items()
                if occupation == coordinate
            ]
            if isinstance(operator, FermionOp) and matches:
                index = matches[0]
                phase *= 1 - occupation + occupation * spectator_parity
                for earlier in mapped:
                    if earlier > index:
                        phase *= 1 - 2 * fermion_coordinates[earlier] * occupation
                mapped.append(index)
            else:
                if occupation.free_symbols.intersection(fermion_coordinates.values()):
                    raise ValueError(
                        "Fermion targets require direct source fermion assignments"
                    )
                if isinstance(operator, FermionOp):
                    spectator_parity *= 1 - 2 * occupation
        if sorted(mapped) != sorted(fermion_coordinates):
            raise ValueError("Each target fermion must map to exactly one source mode")
        return sympy.expand(phase)

    def encode(self, state: tuple[int, ...]) -> tuple[int, ...]:
        """Return source occupations for one target basis state (without its phase)."""
        if len(state) != len(self.target.dimensions) or any(
            sympy.sympify(value).is_Integer is not True or not 0 <= value < size
            for value, size in zip(state, self.target.dimensions)
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
        return _number_symbols(self.target.operators) if self.target_is_nof else ()

    @cached_property
    def _target_identity(self):
        if self.target_is_nof:
            return _one_term(
                self.target.operators, (0,) * len(self.target.operators), sympy.S.One
            )
        return sympy.ImmutableMatrix.eye(self.target.dimension)

    @cached_property
    def _target_zero(self):
        if self.target_is_nof:
            return NumberOrderedForm(self.target.operators, {}, validate=False)
        return sympy.ImmutableMatrix.zeros(self.target.dimension)

    @cache
    def _target_shift(self, source_shift: tuple[int, ...]) -> tuple[int, ...] | None:
        """Find the binary target transition for a source occupation shift."""
        source = sympy.Matrix(source_shift)
        result = self._occupation_left_inverse * source
        if self._occupation_matrix * result != source:
            return None
        if any(not value.is_Integer or abs(value) > 1 for value in result):
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
            for symbol, power in zip(self.coordinate_symbols, _target_shift, strict=True)
            if power
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
            for symbol, power in zip(self.coordinate_symbols, _target_shift, strict=True)
            if power
        )
        return _boolean_solutions(equations, self.coordinate_symbols)

    def _source_form(self, expression) -> NumberOrderedForm:
        """Convert an expression into the source algebra of the embedding."""
        if isinstance(expression, NumberOrderedForm):
            if expression.operators == self.operators:
                return expression
            expression = expression.as_expr()
        return NumberOrderedForm.from_expr(
            sympy.sympify(expression),
            operators=self.operators,
        )

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
        # The source action includes its Fock sign. A target fermion monomial
        # supplies a Fock sign of its own; remove it from the coefficient so
        # that evaluating the target operator does not count it twice.
        target_form = _one_term(self.target.operators, target_powers, sympy.S.One)
        (target_transition,) = _NOFTransition.from_form(target_form)
        target_weight = target_transition.symbolic_action(self.coordinate_symbols).weight
        target_weight = target_weight.xreplace(
            self._transition_support(target_powers)
        ).xreplace(self._initial_to_middle(target_powers))
        amplitude *= target_weight  # Inverse of a sign.
        # Multiplication applies the public NOF binary-number normalization.
        return target_form * _one_term(
            self.target.operators, (0,) * len(target_powers), amplitude
        )

    @cached_property
    def _finite_source_index(self):
        """Source occupations belonging to a finite target, without coefficients."""
        return {
            self.encode(state): index for index, state in enumerate(self.target.states)
        }

    @cache
    def _finite_transition_action(
        self,
        transition: _NOFTransition,
        retained_row: int,
    ):
        """Apply a transition to the source image of one retained state."""
        target_state = self.target.states[retained_row]
        return transition.apply(self.encode(target_state))

    @cache
    def _pullback(
        self, source: NumberOrderedForm
    ) -> NumberOrderedForm | sympy.MatrixBase:
        """Return ``W† source W`` without enumerating the source Hilbert space."""
        source = self._source_form(source)
        transitions = tuple(_NOFTransition.from_form(source))
        if self.target_is_nof:
            result = self._target_zero
            for transition in transitions:
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

        target = self.target
        source_to_target = self._finite_source_index
        matrix = sympy.MutableSparseMatrix(target.dimension, target.dimension, {})
        for column, state in enumerate(target.states):
            source_state = self.encode(state)
            for transition in transitions:
                action = transition.apply(source_state)
                if action is None:
                    continue
                if (row := source_to_target.get(action.output_state)) is not None:
                    initial_phase = self.phase.xreplace(
                        dict(zip(self.coordinate_symbols, state, strict=True))
                    )
                    final_phase = self.phase.xreplace(
                        dict(
                            zip(self.coordinate_symbols, target.states[row], strict=True)
                        )
                    )
                    matrix[row, column] += (
                        sympy.conjugate(final_phase) * initial_phase * action.weight
                    )
        return sympy.ImmutableMatrix(matrix)


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


def _one_term(
    operators,
    powers: tuple[int, ...],
    coefficient: sympy.Expr,
) -> NumberOrderedForm:
    return NumberOrderedForm(
        tuple(operators),
        {powers: coefficient},
        validate=False,
    )


@cache
def _number_symbols(operators: tuple) -> tuple[sympy.Symbol, ...]:
    """Obtain coefficient coordinates using public NOF term inspection.

    A number operator has one diagonal term whose coefficient is its occupation
    symbol. Query that term rather than depending on private placeholder names
    or metadata. This works with both plain and packed NOF storage.
    """
    powers = (0,) * len(operators)
    return tuple(
        NumberOrderedForm.from_expr(NumberOperator(op), operators=operators).terms[powers]
        for op in operators
    )


@dataclass(frozen=True, slots=True)
class _WeightedTransition:
    """One partial transition between occupation states."""

    output_state: tuple[sympy.Expr, ...]
    weight: sympy.Expr


@dataclass(frozen=True)
class _NOFTransition:
    """The occupation shift and amplitude of one NOF term."""

    form: NumberOrderedForm
    powers: tuple[int, ...]

    @classmethod
    def from_form(cls, form: NumberOrderedForm) -> Iterable[_NOFTransition]:
        """Read the number-ordered term interface, independently of storage."""
        for powers, coefficient in form.terms.items():
            powers = tuple(map(int, powers))
            yield cls(
                NumberOrderedForm(
                    form.operators,
                    {powers: coefficient},
                    validate=False,
                ),
                powers,
            )

    @cached_property
    def operators(self) -> tuple:
        """Return the ordered annihilation generators."""
        return tuple(self.form.operators)

    @cached_property
    def placeholders(self) -> tuple[sympy.Symbol, ...]:
        """Return the number-operator placeholders of the source algebra."""
        return _number_symbols(self.operators)

    @cached_property
    def fermion_indices(self) -> tuple[int, ...]:
        """Return source indices that contribute fermionic parity."""
        return tuple(
            index
            for index, operator in enumerate(self.operators)
            if isinstance(operator, FermionOp)
        )

    def apply(self, state: Sequence[int]) -> _WeightedTransition | None:
        """Apply this term to a concrete occupation state."""
        action = self.symbolic_action(state)
        return None if action.weight == 0 else action

    def symbolic_action(self, occupations: Sequence[sympy.Expr]) -> _WeightedTransition:
        """Apply this term to symbolic occupations."""
        ((_, coefficient),) = self.form.terms.items()
        current = list(map(sympy.sympify, occupations))
        amplitude = sympy.S.One

        for index, power in enumerate(self.powers):
            for _ in range(max(power, 0)):
                factor = self._symbolic_generator(current, index, annihilate=True)
                if factor == 0:
                    return _WeightedTransition(tuple(current), sympy.S.Zero)
                amplitude *= factor

        amplitude *= coefficient.xreplace(
            dict(zip(self.placeholders, current, strict=True))
        )

        for index in reversed(range(len(self.powers))):
            for _ in range(max(-self.powers[index], 0)):
                factor = self._symbolic_generator(current, index, annihilate=False)
                if factor == 0:
                    return _WeightedTransition(tuple(current), sympy.S.Zero)
                amplitude *= factor

        return _WeightedTransition(
            tuple(current),
            sympy.expand(amplitude),
        )

    def support_equations(
        self, input_occupations: Sequence[sympy.Expr]
    ) -> tuple[sympy.Expr, ...]:
        """Return exact occupation constraints implied by this transition."""
        equations = []
        for occupation, operator, power in zip(
            input_occupations, self.operators, self.powers, strict=True
        ):
            if isinstance(operator, (FermionOp, SigmaMinus)) and power:
                equations.append(occupation - (1 if power > 0 else 0))
            # Bosonic annihilation requires occupation >= power, not equality.
            # Its vanishing channels are handled by their transition weights.
        return tuple(equations)

    def _symbolic_generator(
        self,
        state: list[sympy.Expr],
        index: int,
        *,
        annihilate: bool,
    ) -> sympy.Expr:
        operator = self.operators[index]
        occupation = state[index]
        if isinstance(operator, BosonOp):
            factor = sympy.sqrt(occupation if annihilate else occupation + 1)
        elif isinstance(operator, LadderOp):
            factor = sympy.S.One
        elif isinstance(operator, SigmaMinus):
            factor = occupation if annihilate else 1 - occupation
        elif isinstance(operator, FermionOp):
            factor = (occupation if annihilate else 1 - occupation) * sympy.prod(
                1 - 2 * state[earlier]
                for earlier in self.fermion_indices
                if earlier < index
            )
        else:  # pragma: no cover - guarded by NumberOrderedForm
            raise TypeError(f"Unsupported source operator: {operator!r}")
        state[index] += -1 if annihilate else 1
        return sympy.sympify(factor)
