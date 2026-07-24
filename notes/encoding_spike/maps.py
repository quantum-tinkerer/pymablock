"""Sparse weighted maps for projection-aware second quantization.

A term ``MapKey(input_state, output_state): coefficient`` represents

``coefficient * |output_state><input_state|``.

The representation is deliberately close to ``NumberOrderedForm``: structural map
keys are stored separately from held scalar coefficients, equal keys combine directly,
and multiplication is composition of structural keys. ``MapRule`` extends this storage
to rectangular target-to-complement maps by recording only source occupations that
actually occur as outputs.
"""

# The protocol-like arithmetic classes are implementation details of the spike.
# ruff: noqa: D101, D102, D103, D105, D107

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any

import sympy
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock._packed_binary import masks_from_monomial
from pymablock.number_ordered_form import LadderOp, NumberOrderedForm

from .encoding import (
    FermionEmbedding,
    FermionSpace,
    OccupationEncoding,
    TargetSpace,
)
from .packed import PackedForm, nof_to_packed_fermions, packed_powers
from .tensor import TensorOperator

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    Target = TargetSpace | FermionSpace
    Encoding = OccupationEncoding | FermionEmbedding


@dataclass(frozen=True, order=True, slots=True)
class MapKey:
    """One partial map between occupation states."""

    input_state: tuple[int, ...]
    output_state: tuple[int, ...]

    def adjoint(self) -> MapKey:
        return type(self)(self.output_state, self.input_state)


@dataclass(frozen=True, slots=True)
class SourceSpace:
    """An occupation basis with an ordered set of source generators."""

    operators: tuple[Any, ...]

    @property
    def fermion_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, operator in enumerate(self.operators)
            if isinstance(operator, FermionOp)
        )

    @property
    def packed_basis(self) -> tuple[str, ...]:
        return tuple(str(operator.name) for operator in self.operators)

    @staticmethod
    def _apply_packed_monomial(
        monomial: int,
        state: tuple[int, ...],
    ) -> tuple[tuple[int, ...], sympy.Expr] | None:
        creators, numbers, annihilators = masks_from_monomial(
            monomial,
            num_modes=len(state),
        )
        output = list(state)
        amplitude = sympy.S.One
        for mode in reversed(range(len(state))):
            active = 1 << mode
            if numbers & active:
                if not output[mode]:
                    return None
                continue
            if not (creators | annihilators) & active:
                continue
            parity = sum(output[:mode]) % 2
            if annihilators & active:
                if not output[mode]:
                    return None
                output[mode] = 0
            else:
                if output[mode]:
                    return None
                output[mode] = 1
            if parity:
                amplitude = -amplitude
        return tuple(output), amplitude

    def _apply_nof_term(
        self,
        powers: Sequence[int],
        coefficient: sympy.Expr,
        placeholders: Sequence[sympy.Symbol],
        source_state: tuple[int, ...],
    ) -> tuple[tuple[int, ...], sympy.Expr] | None:
        state = list(source_state)
        amplitude = sympy.S.One

        for index, power in enumerate(powers):
            for _ in range(max(int(power), 0)):
                operator = self.operators[index]
                occupation = state[index]
                if isinstance(operator, BosonOp):
                    if occupation == 0:
                        return None
                    amplitude *= sympy.sqrt(occupation)
                    state[index] = occupation - 1
                elif isinstance(operator, LadderOp):
                    state[index] = occupation - 1
                elif isinstance(operator, SigmaMinus):
                    if occupation == 0:
                        return None
                    state[index] = 0
                elif isinstance(operator, FermionOp):
                    if occupation == 0:
                        return None
                    if (
                        sum(
                            state[earlier]
                            for earlier in self.fermion_indices
                            if earlier < index
                        )
                        % 2
                    ):
                        amplitude = -amplitude
                    state[index] = 0
                else:  # pragma: no cover - guarded by NumberOrderedForm
                    raise TypeError(f"Unsupported source operator: {operator!r}")

        amplitude *= coefficient.xreplace(
            dict(zip(placeholders, map(sympy.Integer, state), strict=True))
        )
        if amplitude == 0:
            return None

        for index in reversed(range(len(powers))):
            for _ in range(max(-int(powers[index]), 0)):
                operator = self.operators[index]
                occupation = state[index]
                if isinstance(operator, BosonOp):
                    amplitude *= sympy.sqrt(occupation + 1)
                    state[index] = occupation + 1
                elif isinstance(operator, LadderOp):
                    state[index] = occupation + 1
                elif isinstance(operator, SigmaMinus):
                    if occupation == 1:
                        return None
                    state[index] = 1
                elif isinstance(operator, FermionOp):
                    if occupation == 1:
                        return None
                    if (
                        sum(
                            state[earlier]
                            for earlier in self.fermion_indices
                            if earlier < index
                        )
                        % 2
                    ):
                        amplitude = -amplitude
                    state[index] = 1
                else:  # pragma: no cover - guarded by NumberOrderedForm
                    raise TypeError(f"Unsupported source operator: {operator!r}")

        return tuple(state), amplitude

    @cache
    def apply(
        self,
        source: TensorOperator | NumberOrderedForm | PackedForm,
        state: tuple[int, ...],
    ) -> tuple[tuple[tuple[int, ...], sympy.Expr], ...]:
        """Apply a sparse source operator to one occupation state."""
        if isinstance(source, TensorOperator):
            source = source.native
        if isinstance(source, PackedForm):
            if source.basis != self.packed_basis:
                raise ValueError("Packed operator basis does not match the source space")
        elif source.operators != self.operators:
            raise ValueError("Number-ordered operators do not match the source space")
        combined: dict[tuple[int, ...], sympy.Expr] = {}
        if isinstance(source, PackedForm):
            for monomial, coefficient in source.items:
                action = self._apply_packed_monomial(monomial, state)
                if action is None:
                    continue
                output, amplitude = action
                combined[output] = (
                    combined.get(output, sympy.S.Zero) + coefficient * amplitude
                )
        else:
            placeholders = source._number_operator_placeholders
            for powers, coefficient in source.terms.items():
                action = self._apply_nof_term(
                    tuple(map(int, powers)),
                    coefficient,
                    placeholders,
                    state,
                )
                if action is None:
                    continue
                output, amplitude = action
                combined[output] = combined.get(output, sympy.S.Zero) + amplitude
        return tuple(
            (output, coefficient)
            for output, coefficient in sorted(combined.items())
            if coefficient != 0
        )


@dataclass(frozen=True, slots=True)
class ComplementSpace:
    """The occupation-basis complement of a retained realization."""

    ambient: SourceSpace
    retained_states: frozenset[tuple[int, ...]]

    def contains(self, state: tuple[int, ...]) -> bool:
        return state not in self.retained_states


@dataclass(frozen=True, order=True, slots=True)
class MapRule:
    """One actual map arrow from a right-space state to a left-space state."""

    input_state: tuple[int, ...]
    output_state: tuple[int, ...]


class ExplicitOperatorMap:
    """Sparse state-arrow map retained as an enumerable reference backend."""

    __slots__ = ("items", "left_space", "right_space")

    def __init__(
        self,
        left_space: Any,
        right_space: Any,
        items: Iterable[tuple[tuple[tuple[int, ...], tuple[int, ...]], object]],
    ):
        combined = {}
        for key, coefficient in items:
            input_state, output_state = map(tuple, key)
            if not left_space.contains(output_state):
                continue
            coefficient = sympy.sympify(coefficient)
            if coefficient == 0:
                continue
            key = (input_state, output_state)
            combined[key] = combined.get(key, sympy.S.Zero) + coefficient
            if combined[key] == 0:
                del combined[key]
        self._initialize(left_space, right_space, combined.items())

    def _initialize(self, left_space, right_space, items):
        self.left_space = left_space
        self.right_space = right_space
        self.items = tuple(sorted(items))

    @classmethod
    def _from_items(cls, left_space, right_space, items):
        combined = {}
        for key, coefficient in items:
            if coefficient == 0:
                continue
            combined[key] = combined.get(key, sympy.S.Zero) + coefficient
            if combined[key] == 0:
                del combined[key]
        result = cls.__new__(cls)
        result._initialize(left_space, right_space, combined.items())
        return result

    def _require_same_spaces(self, other):
        if self.left_space != other.left_space or self.right_space != other.right_space:
            raise ValueError("Maps must have equal left and right spaces")

    def __add__(self, other):
        self._require_same_spaces(other)
        return type(self)._from_items(
            self.left_space,
            self.right_space,
            (*self.items, *other.items),
        )

    def __neg__(self):
        return type(self)._from_items(
            self.left_space,
            self.right_space,
            ((key, -coefficient) for key, coefficient in self.items),
        )

    def scale(self, factor):
        return type(self)._from_items(
            self.left_space,
            self.right_space,
            ((key, coefficient * factor) for key, coefficient in self.items),
        )

    def right_multiply(self, matrix):
        states = tuple(self.right_space.states)
        inputs_by_output = {}
        for row, column in matrix.todok():
            inputs_by_output.setdefault(states[row], []).append(
                (states[column], matrix[row, column])
            )
        return type(self)._from_items(
            self.left_space,
            self.right_space,
            (
                (
                    (new_input, output_state),
                    map_coefficient * matrix_coefficient,
                )
                for (input_state, output_state), map_coefficient in self.items
                for new_input, matrix_coefficient in inputs_by_output.get(input_state, ())
            ),
        )

    def left_apply(self, action):
        return type(self)._from_items(
            self.left_space,
            self.right_space,
            (
                (
                    (input_state, tuple(new_output)),
                    map_coefficient * action_coefficient,
                )
                for (input_state, output_state), map_coefficient in self.items
                for new_output, action_coefficient in action(output_state)
                if self.left_space.contains(tuple(new_output))
            ),
        )

    def inner(self, other):
        self._require_same_spaces(other)
        states = tuple(self.right_space.states)
        index = {state: position for position, state in enumerate(states)}
        left_by_output = {}
        for (input_state, output_state), coefficient in self.items:
            left_by_output.setdefault(output_state, []).append((input_state, coefficient))

        matrix = sympy.MutableSparseMatrix(len(states), len(states), {})
        for (right_input, output_state), right_coefficient in other.items:
            for left_input, left_coefficient in left_by_output.get(output_state, ()):
                matrix[index[left_input], index[right_input]] += (
                    sympy.conjugate(left_coefficient) * right_coefficient
                )
        return sympy.ImmutableSparseMatrix(matrix)


@dataclass(frozen=True, slots=True)
class OffDiagonalMapKey:
    """One operator-labelled map from the target into the source complement.

    ``source_monomial`` is either a number-ordered power tuple or a packed
    fermion monomial.  Together with ``target_transition = |out><in|`` it
    represents ``Q X W |out><in|``.
    """

    source_monomial: tuple[int, ...] | int
    target_transition: MapKey


@dataclass(frozen=True, slots=True)
class OccupationMapKey:
    """One algebraic occupation map between differently generated spaces."""

    input_symbols: tuple[sympy.Symbol, ...]
    output_expressions: tuple[sympy.Expr, ...]

    def evaluate(self, state: tuple[int, ...]) -> tuple[int, ...]:
        substitutions = dict(zip(self.input_symbols, state, strict=True))
        return tuple(
            int(expression.xreplace(substitutions))
            for expression in self.output_expressions
        )


@dataclass(frozen=True, slots=True)
class AffineBand:
    """A finite-support affine relation between two occupation spaces.

    One parameter point ``z`` denotes

    ``coefficient(z) * |left_coordinates(z)><right_coordinates(z)|``.

    Both sides are parameterized independently.  In particular, taking an adjoint
    only swaps the two sides and never attempts to invert an encoding such as
    ``n = 2*m``.  The finite support is the lowering boundary used by the current
    spike; the affine relation itself remains compact during map arithmetic.
    """

    left_space: Any
    right_space: Any
    parameters: tuple[sympy.Symbol, ...]
    left_coordinates: tuple[sympy.Expr, ...]
    right_coordinates: tuple[sympy.Expr, ...]
    coefficient: sympy.Expr
    support: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        left = tuple(sympy.expand(value) for value in self.left_coordinates)
        right = tuple(sympy.expand(value) for value in self.right_coordinates)
        coefficient = sympy.sympify(self.coefficient)
        support = tuple(sorted(set(map(tuple, self.support))))
        if len(set(parameters)) != len(parameters):
            raise ValueError("Affine-band parameters must be unique")
        if any(len(point) != len(parameters) for point in support):
            raise ValueError("An affine-band support point has the wrong dimension")
        for expression in (*left, *right):
            if parameters and sympy.Poly(expression, *parameters).total_degree() > 1:
                raise ValueError("Band coordinates must be affine in their parameters")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "left_coordinates", left)
        object.__setattr__(self, "right_coordinates", right)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "support", support)

    def evaluate(
        self,
        point: tuple[int, ...],
    ) -> tuple[MapRule, sympy.Expr]:
        """Evaluate one parameter point without enumerating either ambient space."""
        point = tuple(point)
        if point not in self.support:
            raise ValueError("The parameter point lies outside the band support")
        substitutions = dict(zip(self.parameters, map(sympy.Integer, point), strict=True))
        left = tuple(
            int(expression.xreplace(substitutions))
            for expression in self.left_coordinates
        )
        right = tuple(
            int(expression.xreplace(substitutions))
            for expression in self.right_coordinates
        )
        return MapRule(right, left), self.coefficient.xreplace(substitutions)

    @property
    def items(self) -> tuple[tuple[MapRule, sympy.Expr], ...]:
        """Lower the band to explicit arrows for validation or fallback execution."""
        return tuple(self.evaluate(point) for point in self.support)

    def adjoint(self) -> AffineBand:
        """Swap the two parameterized spaces without inverting the relation."""
        return type(self)(
            self.right_space,
            self.left_space,
            self.parameters,
            self.right_coordinates,
            self.left_coordinates,
            sympy.conjugate(self.coefficient),
            self.support,
        )

    def compose(self, other: AffineBand) -> AffineBand:
        """Compose bands that use the same parameterization of their join.

        This is the cheap closed case used by an embedding and its adjoint.  A
        general affine join may require new integer parameters and congruence
        constraints; the spike leaves that case explicit rather than silently
        producing a non-integral parameterization.
        """
        if self.right_space != other.left_space:
            raise ValueError("Affine bands have incompatible intermediate spaces")
        if self.parameters != other.parameters or self.support != other.support:
            raise NotImplementedError(
                "Composition currently requires a shared parameter domain"
            )
        if self.right_coordinates != other.left_coordinates:
            raise NotImplementedError(
                "Composition currently requires identical intermediate coordinates"
            )
        return type(self)(
            self.left_space,
            other.right_space,
            self.parameters,
            self.left_coordinates,
            other.right_coordinates,
            sympy.expand(self.coefficient * other.coefficient),
            self.support,
        )

    def __matmul__(self, other: AffineBand) -> AffineBand:
        return self.compose(other)

    @property
    def fixed_shift(self) -> tuple[int, ...] | None:
        """Return a constant same-coordinate shift, if the band has one."""
        if len(self.left_coordinates) != len(self.right_coordinates):
            return None
        shifts = tuple(
            sympy.expand(left - right)
            for left, right in zip(
                self.left_coordinates,
                self.right_coordinates,
                strict=True,
            )
        )
        if any(shift.free_symbols & set(self.parameters) for shift in shifts):
            return None
        if any(not shift.is_Integer for shift in shifts):
            return None
        return tuple(map(int, shifts))

    @property
    def is_identity(self) -> bool:
        """Whether this band covers a finite space with unit diagonal arrows."""
        if (
            self.left_space != self.right_space
            or self.left_coordinates != self.right_coordinates
        ):
            return False
        states = getattr(self.right_space, "states", None)
        if states is None or len(states) != len(self.items):
            return False
        return all(
            key.input_state == key.output_state and coefficient == 1
            for key, coefficient in self.items
        ) and {key.input_state for key, _coefficient in self.items} == set(states)


def state_matrix(
    target: Target,
    terms: Iterable[tuple[MapKey, object]],
) -> sympy.ImmutableMatrix:
    """Build a native SymPy matrix from occupation-state arrows."""
    matrix = sympy.MutableSparseMatrix(target.dimension, target.dimension, {})
    for key, coefficient in terms:
        coefficient = sympy.sympify(coefficient)
        if coefficient == 0:
            continue
        matrix[target.index[key.output_state], target.index[key.input_state]] += (
            coefficient
        )
    return sympy.ImmutableMatrix(matrix)


def matrix_unit(
    target: Target,
    row: int,
    column: int,
    coefficient: object,
) -> sympy.ImmutableMatrix:
    """Return one native finite-space matrix unit."""
    matrix = sympy.MutableSparseMatrix(target.dimension, target.dimension, {})
    matrix[row, column] = coefficient
    return sympy.ImmutableMatrix(matrix)


def matrix_entries(
    matrix: sympy.MatrixBase,
) -> tuple[tuple[int, int, sympy.Expr], ...]:
    """Return nonzero ``(row, column, coefficient)`` matrix entries."""
    return tuple(
        (row, column, matrix[row, column])
        for row, column in matrix.todok()
        if matrix[row, column] != 0
    )


def matrix_items(
    target: Target,
    matrix: sympy.MatrixBase,
) -> tuple[tuple[MapKey, sympy.Expr], ...]:
    """Return a matrix as occupation-state arrows."""
    return tuple(
        (
            MapKey(target.states[column], target.states[row]),
            coefficient,
        )
        for row, column, coefficient in matrix_entries(matrix)
    )


@dataclass(frozen=True, slots=True)
class EmbeddingMapForm:
    """Weighted algebraic map from target to source occupations."""

    target: Target
    source_operators: tuple[Any, ...]
    items: tuple[tuple[OccupationMapKey, sympy.Expr], ...]


class MapEmbedding:
    """Compile an encoding into sparse maps and source-term actions."""

    target_is_packed = False

    def __init__(self, encoding: Encoding):
        if not isinstance(encoding, (OccupationEncoding, FermionEmbedding)):
            raise TypeError("Map execution needs an occupation or fermion embedding")
        self.encoding = encoding
        self.target = encoding.target
        self.source_operators = tuple(encoding.operators)
        self.source_space = SourceSpace(self.source_operators)
        self.coordinates = tuple(getattr(self.target, "coordinates", ()))
        self.target_operators = ()
        self.target_basis = tuple(
            getattr(coordinate, "name", str(index))
            for index, coordinate in enumerate(self.coordinates)
        )
        self.source_is_packed = all(
            isinstance(operator, FermionOp) for operator in self.source_operators
        )
        self._fermion_indices = tuple(
            index
            for index, operator in enumerate(self.source_operators)
            if isinstance(operator, FermionOp)
        )
        phases = tuple(
            getattr(
                encoding,
                "phases",
                (sympy.S.One,) * self.target.dimension,
            )
        )
        encoded = tuple(encoding.encode(state) for state in self.target.states)
        if len(set(encoded)) != len(encoded):
            raise ValueError("The embedding map must be injective")
        self.complement_space = ComplementSpace(
            self.source_space,
            frozenset(encoded),
        )
        self._source_to_target = dict(zip(encoded, self.target.states, strict=True))
        self._target_phases = dict(zip(self.target.states, phases, strict=True))
        occupation_key, phase_expression = self._algebraic_map()
        self.map = EmbeddingMapForm(
            self.target,
            self.source_operators,
            ((occupation_key, phase_expression),),
        )
        self.bridge = AffineBand(
            self.source_space,
            self.target,
            occupation_key.input_symbols,
            occupation_key.output_expressions,
            occupation_key.input_symbols,
            phase_expression,
            self.target.states,
        )

    def _algebraic_map(self) -> tuple[OccupationMapKey, sympy.Expr]:
        if isinstance(self.encoding, OccupationEncoding):
            symbols = tuple(
                sympy.Symbol(coordinate.name, integer=True)
                for coordinate in self.target.coordinates
            )
            values = dict(zip(self.target.coordinates, symbols, strict=True))
            outputs = tuple(
                sympy.sympify(
                    (
                        values[expression]
                        if expression in values
                        else expression.function(values)
                    )
                )
                for operator in self.source_operators
                for expression in (self.encoding._expressions[operator],)
            )
            phase = sympy.S.One
        else:
            symbols = tuple(
                sympy.Symbol(str(mode.name), integer=True) for mode in self.target.modes
            )
            target_symbols = dict(zip(self.target.modes, symbols, strict=True))
            outputs = tuple(
                (
                    target_symbols[value]
                    if isinstance(value, FermionOp)
                    else sympy.Integer(value)
                )
                for source in self.source_operators
                for value in (self.encoding._modes[source],)
            )
            fixed_occupied = 0
            phase_factors = []
            for source in self.source_operators:
                value = self.encoding._modes[source]
                if isinstance(value, FermionOp):
                    if fixed_occupied % 2:
                        phase_factors.append(1 - 2 * target_symbols[value])
                elif value == 1:
                    fixed_occupied += 1
            phase = sympy.prod(phase_factors)

        if symbols and any(
            sympy.Poly(output, *symbols).total_degree() > 1 for output in outputs
        ):
            raise ValueError("Map execution currently requires affine occupations")
        return OccupationMapKey(symbols, outputs), phase

    @cached_property
    def target_identity(self) -> sympy.ImmutableMatrix:
        return sympy.ImmutableMatrix(sympy.eye(self.target.dimension))

    @cached_property
    def target_zero(self) -> sympy.ImmutableMatrix:
        return sympy.ImmutableMatrix(sympy.zeros(self.target.dimension))

    def source_nof(
        self,
        expression: sympy.Expr | NumberOrderedForm,
    ) -> NumberOrderedForm:
        if isinstance(expression, NumberOrderedForm):
            if expression.operators == self.source_operators:
                return expression
            return NumberOrderedForm.from_expr(
                expression.as_expr(),
                operators=self.source_operators,
            )
        return NumberOrderedForm.from_expr(
            sympy.sympify(expression),
            operators=self.source_operators,
        )

    def source_form(
        self,
        expression: sympy.Expr | NumberOrderedForm | PackedForm | TensorOperator,
    ) -> TensorOperator:
        if isinstance(expression, TensorOperator):
            native = expression.native
            if (
                isinstance(native, PackedForm)
                and native.basis != self.source_space.packed_basis
            ):
                raise ValueError("Packed operator basis does not match the source space")
            if (
                isinstance(native, NumberOrderedForm)
                and native.operators != self.source_operators
            ):
                raise ValueError("NOF operators do not match the source space")
            return expression
        if isinstance(expression, PackedForm):
            if not self.source_is_packed:
                raise TypeError("Packed source terms require a purely fermionic source")
            return TensorOperator.from_factor(expression)
        form = self.source_nof(expression)
        native = nof_to_packed_fermions(form) if self.source_is_packed else form
        return TensorOperator.from_factor(native)

    def source_terms(self, source: TensorOperator):
        source = self.source_form(source).native
        if isinstance(source, NumberOrderedForm):
            for powers, coefficient in source.terms.items():
                powers = tuple(map(int, powers))
                yield (
                    powers,
                    coefficient,
                    TensorOperator.from_factor(
                        NumberOrderedForm(
                            self.source_operators,
                            {powers: coefficient},
                            validate=False,
                        )
                    ),
                )
            return
        for monomial, coefficient in source.items:
            yield (
                packed_powers(monomial, num_modes=len(self.source_operators)),
                coefficient,
                TensorOperator.from_factor(
                    PackedForm.monomial(source.basis, monomial, coefficient)
                ),
            )

    def apply_source(
        self,
        source: TensorOperator,
        state: tuple[int, ...],
    ) -> tuple[tuple[tuple[int, ...], sympy.Expr], ...]:
        """Apply a source form without constructing a source basis."""
        return self.source_space.apply(source, state)

    @staticmethod
    def _apply_packed_monomial(
        monomial: int,
        state: tuple[int, ...],
    ) -> tuple[tuple[int, ...], sympy.Expr] | None:
        creators, numbers, annihilators = masks_from_monomial(
            monomial,
            num_modes=len(state),
        )
        output = list(state)
        amplitude = sympy.S.One
        for mode in reversed(range(len(state))):
            active = 1 << mode
            if numbers & active:
                if not output[mode]:
                    return None
                continue
            if not (creators | annihilators) & active:
                continue
            parity = sum(output[:mode]) % 2
            if annihilators & active:
                if not output[mode]:
                    return None
                output[mode] = 0
            else:
                if output[mode]:
                    return None
                output[mode] = 1
            if parity:
                amplitude = -amplitude
        return tuple(output), amplitude

    def _apply_nof_term(
        self,
        powers: Sequence[int],
        coefficient: sympy.Expr,
        placeholders: Sequence[sympy.Symbol],
        source_state: tuple[int, ...],
    ) -> tuple[tuple[int, ...], sympy.Expr] | None:
        state = list(source_state)
        amplitude = sympy.S.One

        for index, power in enumerate(powers):
            for _ in range(max(int(power), 0)):
                operator = self.source_operators[index]
                occupation = state[index]
                if isinstance(operator, BosonOp):
                    if occupation == 0:
                        return None
                    amplitude *= sympy.sqrt(occupation)
                    state[index] = occupation - 1
                elif isinstance(operator, LadderOp):
                    state[index] = occupation - 1
                elif isinstance(operator, SigmaMinus):
                    if occupation == 0:
                        return None
                    state[index] = 0
                elif isinstance(operator, FermionOp):
                    if occupation == 0:
                        return None
                    if (
                        sum(
                            state[earlier]
                            for earlier in self._fermion_indices
                            if earlier < index
                        )
                        % 2
                    ):
                        amplitude = -amplitude
                    state[index] = 0
                else:  # pragma: no cover - guarded by NumberOrderedForm
                    raise TypeError(f"Unsupported source operator: {operator!r}")

        amplitude *= coefficient.xreplace(
            dict(zip(placeholders, map(sympy.Integer, state), strict=True))
        )
        if amplitude == 0:
            return None

        for index in reversed(range(len(powers))):
            for _ in range(max(-int(powers[index]), 0)):
                operator = self.source_operators[index]
                occupation = state[index]
                if isinstance(operator, BosonOp):
                    amplitude *= sympy.sqrt(occupation + 1)
                    state[index] = occupation + 1
                elif isinstance(operator, LadderOp):
                    state[index] = occupation + 1
                elif isinstance(operator, SigmaMinus):
                    if occupation == 1:
                        return None
                    state[index] = 1
                elif isinstance(operator, FermionOp):
                    if occupation == 1:
                        return None
                    if (
                        sum(
                            state[earlier]
                            for earlier in self._fermion_indices
                            if earlier < index
                        )
                        % 2
                    ):
                        amplitude = -amplitude
                    state[index] = 1
                else:  # pragma: no cover - guarded by NumberOrderedForm
                    raise TypeError(f"Unsupported source operator: {operator!r}")

        return tuple(state), amplitude

    @cache
    def apply_source_term(
        self,
        source: TensorOperator,
        target_row: int,
    ) -> tuple[tuple[tuple[int, ...], sympy.Expr], ...]:
        target_state = self.target.states[target_row]
        source_state = self.encoding.encode(target_state)
        source = self.source_form(source).native
        if isinstance(source, PackedForm):
            if len(source.items) != 1:
                raise ValueError("A source term must contain one packed monomial")
            monomial, coefficient = source.items[0]
            action = self._apply_packed_monomial(monomial, source_state)
            if action is None:
                return ()
            output, amplitude = action
            return ((output, coefficient * amplitude),)
        if len(source.terms) != 1:
            raise ValueError("A source term must contain one number-ordered monomial")
        powers, coefficient = next(iter(source.terms.items()))
        action = self._apply_nof_term(
            tuple(map(int, powers)),
            coefficient,
            source._number_operator_placeholders,
            source_state,
        )
        return () if action is None else (action,)

    @cache
    def to_target(
        self,
        source: TensorOperator,
    ) -> sympy.ImmutableMatrix:
        source = self.source_form(source).native
        terms = []
        if isinstance(source, PackedForm):
            for monomial, coefficient in source.items:
                one_term = PackedForm.monomial(
                    source.basis,
                    monomial,
                    coefficient,
                )
                terms.extend(self._pullback_term(one_term))
        else:
            placeholders = source._number_operator_placeholders
            for powers, coefficient in source.terms.items():
                powers = tuple(map(int, powers))
                for target_state in self.target.states:
                    source_state = self.encoding.encode(target_state)
                    action = self._apply_nof_term(
                        powers,
                        coefficient,
                        placeholders,
                        source_state,
                    )
                    if action is None:
                        continue
                    output, amplitude = action
                    output_target = self._source_to_target.get(output)
                    if output_target is None:
                        continue
                    terms.append(
                        (
                            MapKey(target_state, output_target),
                            sympy.conjugate(self._target_phases[output_target])
                            * amplitude
                            * self._target_phases[target_state],
                        )
                    )
            return state_matrix(self.target, terms)
        return state_matrix(self.target, terms)

    def _pullback_term(
        self,
        source: PackedForm,
    ) -> Iterable[tuple[MapKey, sympy.Expr]]:
        for column, target_state in enumerate(self.target.states):
            for output, amplitude in self.apply_source_term(source, column):
                output_target = self._source_to_target.get(output)
                if output_target is None:
                    continue
                yield (
                    MapKey(target_state, output_target),
                    sympy.conjugate(self._target_phases[output_target])
                    * amplitude
                    * self._target_phases[target_state],
                )
