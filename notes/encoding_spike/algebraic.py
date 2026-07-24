"""Algebraic implicit backend for occupation embeddings.

Unlike :mod:`notes.encoding_spike.encoding`, this module does not construct a complete
source basis. Its main map backend composes source action into sparse rules from target
states to the virtual source occupations that actually occur. The previous factored
``Q X W |out><in|`` representation remains available as a reference. Native NOF and
packed factors share one tensor-product operator interface, while finite retained
indices use Pymablock's existing outer-matrix convention.
"""

# The small protocol classes below are implementation details of the spike.
# ruff: noqa: D101, D102, D105, D107

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any

import sympy
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize as _pymablock_block_diagonalize
from pymablock._packed_binary import masks_from_monomial
from pymablock.number_ordered_form import LadderOp, NumberOrderedForm
from pymablock.series import BlockSeries, zero

from .encoding import (
    CompiledOperator,
    Coordinate,
    FermionEmbedding,
    FermionSpace,
    OccupationEncoding,
    PauliPolynomial,
    TargetOperator,
    TargetSpace,
)
from .maps import (
    AffineBand,
    ComplementSpace,
    ExplicitOperatorMap,
    MapEmbedding,
    MapKey,
    OffDiagonalMapKey,
    matrix_entries,
    matrix_items,
    matrix_unit,
    state_matrix,
)
from .packed import (
    BooleanPolynomial,
    PackedForm,
    nof_to_packed_fermions,
    packed_monomial_term,
    packed_powers,
    packed_to_fermion_nof,
    packed_to_spin_nof,
    spin_term_from_boolean,
)
from .tensor import TensorOperator

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def _is_zero_form(form: NumberOrderedForm) -> bool:
    return not form


SourceForm = TensorOperator


def _is_zero_operator(form: SourceForm) -> bool:
    return not bool(form)


def _adjoint(form: SourceForm):
    return form.adjoint()


def _one_form(operators: Sequence[Any]) -> NumberOrderedForm:
    return NumberOrderedForm(tuple(operators), {(0,) * len(operators): sympy.S.One})


def _one_term_form(
    operators: Sequence[Any], powers: tuple, coefficient: sympy.Expr
) -> NumberOrderedForm:
    return NumberOrderedForm(tuple(operators), {powers: coefficient}, validate=False)


TargetForm = TensorOperator | sympy.ImmutableMatrix


def _immutable_matrix(matrix: sympy.MatrixBase) -> sympy.ImmutableMatrix:
    return (
        matrix
        if isinstance(matrix, sympy.ImmutableMatrix)
        else sympy.ImmutableMatrix(matrix)
    )


def _target_is_zero(form: TargetForm) -> bool:
    if isinstance(form, sympy.MatrixBase):
        return not form.todok()
    return not form


def _target_simplify(form: TargetForm) -> TargetForm:
    if isinstance(form, sympy.MatrixBase):
        return _immutable_matrix(
            form.applyfunc(lambda value: sympy.factor(sympy.cancel(value)))
        )
    return form.simplify()


def _target_entries(form: TargetForm):
    if isinstance(form, sympy.MatrixBase):
        return matrix_entries(form)
    raise TypeError("The target operator does not have finite matrix entries")


def _target_matrix(form: TargetForm) -> sympy.ImmutableMatrix:
    if isinstance(form, sympy.MatrixBase):
        return _immutable_matrix(form)
    raise TypeError("The target operator requires algebraic readout")


class OperatorMap:
    """Original factored-map implementation retained by the reference spike."""

    def __init__(
        self,
        embedding,
        terms: Iterable[tuple[SourceForm, TargetForm]],
    ):
        combined: dict[SourceForm, TargetForm] = {}
        for source, target in terms:
            if _is_zero_operator(source) or _target_is_zero(target):
                continue
            combined[source] = combined.get(source, embedding.target_zero) + target
        self.embedding = embedding
        self.terms = tuple(
            (source, target)
            for source, target in combined.items()
            if not _target_is_zero(target)
        )

    @property
    def left_space(self):
        if hasattr(self, "_left_space"):
            return self._left_space
        return self.embedding.complement_space

    @property
    def right_space(self):
        if hasattr(self, "_right_space"):
            return self._right_space
        return self.embedding.target_space

    @classmethod
    def from_source(cls, embedding, source):
        return cls(embedding, ((source, embedding.target_identity),)).or_zero()

    @classmethod
    def zero(cls, embedding):
        return cls(embedding, ())

    def or_zero(self):
        return self if self.terms else zero

    def __bool__(self):
        return bool(self.terms)

    def _require_same_embedding(self, other):
        if self.embedding is not other.embedding:
            raise ValueError("Operator maps must use the same embedding")

    def __add__(self, other):
        if other is zero:
            return self
        if not isinstance(other, OperatorMap):
            return NotImplemented
        self._require_same_embedding(other)
        return type(self)(self.embedding, (*self.terms, *other.terms)).or_zero()

    def __neg__(self):
        return type(self)(
            self.embedding,
            ((source, -target) for source, target in self.terms),
        ).or_zero()

    def __sub__(self, other):
        return self + (-other)

    def scale(self, factor):
        return type(self)(
            self.embedding,
            ((source, target * factor) for source, target in self.terms),
        ).or_zero()

    def __truediv__(self, divisor):
        return self.scale(sympy.S.One / divisor)

    def right(self, target):
        target = getattr(target, "form", target)
        return type(self)(
            self.embedding,
            ((source, coefficient * target) for source, coefficient in self.terms),
        ).or_zero()

    def left(self, source):
        terms = []
        for inner_source, target in self.terms:
            terms.append((source * inner_source, target))
            terms.append((source, -self.embedding.to_target(inner_source) * target))
        return type(self)(self.embedding, terms).or_zero()

    def inner(self, other):
        self._require_same_embedding(other)
        result = self.embedding.target_zero
        for left_source, left_target in self.terms:
            left_adjoint = _adjoint(left_source)
            projected_left = self.embedding.to_target(left_adjoint)
            for right_source, right_target in other.terms:
                kernel = self.embedding.to_target(left_adjoint * right_source)
                kernel -= projected_left * self.embedding.to_target(right_source)
                result += left_target.adjoint() * kernel * right_target
        return TargetBlock.build(self.embedding, result)

    def __mul__(self, other):
        if isinstance(other, TargetBlock):
            return self.right(other)
        if isinstance(other, RowBlock):
            self._require_same_embedding(other.column)
            return RankOneQBlock(self, other.column)
        return NotImplemented

    def adjoint(self):
        return self.embedding.adjoint_map(self)


def _target_matrix_unit(
    embedding: MapEmbedding | AlgebraicEmbedding,
    exemplar: TargetForm,  # noqa: ARG001
    row: int,
    column: int,
    coefficient: object,
) -> TargetForm:
    return matrix_unit(embedding.encoding.target, row, column, coefficient)


class AlgebraicEmbedding:
    """Operator-level form of an occupation encoding.

    Binary product targets use packed operators. A non-binary target is retained as
    one explicit finite block while the source Hilbert space remains algebraic.
    """

    def __init__(self, encoding: OccupationEncoding | FermionEmbedding):
        if not isinstance(encoding, (OccupationEncoding, FermionEmbedding)):
            raise TypeError("Expected an occupation map or fermion embedding")
        self.encoding = encoding
        self.source_operators = tuple(encoding.operators)
        self.target_is_fermionic = isinstance(encoding, FermionEmbedding)
        if self.target_is_fermionic:
            self.coordinates = tuple(encoding.target.modes)
            self.target_is_packed = True
            self.target_operators = tuple(encoding.target.modes)
        else:
            self.coordinates = tuple(encoding.target.coordinates)
            self.target_is_packed = all(
                coordinate.values == (0, 1) for coordinate in self.coordinates
            )
            self.target_operators = (
                tuple(
                    SigmaMinus(sympy.Symbol(coordinate.name))
                    for coordinate in self.coordinates
                )
                if self.target_is_packed
                else ()
            )
        self._coordinate_symbols = tuple(
            sympy.Symbol(f"_target_{index}", integer=True, nonnegative=True)
            for index in range(len(self.coordinates))
        )
        if self.target_is_fermionic:
            target_occupations = dict(
                zip(self.target_operators, self._coordinate_symbols, strict=True)
            )
            self._source_occupations = tuple(
                sympy.sympify(
                    (target_occupations[value] if isinstance(value, FermionOp) else value)
                )
                for value in (
                    encoding._modes[operator] for operator in self.source_operators
                )
            )
        else:
            values = dict(zip(self.coordinates, self._coordinate_symbols, strict=True))
            self._source_occupations = tuple(
                sympy.sympify(
                    (
                        values[encoding._expressions[operator]]
                        if isinstance(encoding._expressions[operator], Coordinate)
                        else encoding._expressions[operator].function(values)
                    )
                )
                for operator in self.source_operators
            )
        self._target_placeholders = (
            tuple(_one_form(self.target_operators)._number_operator_placeholders)
            if self.target_is_packed
            else ()
        )
        source_identity = _one_form(self.source_operators)
        self._source_placeholders = tuple(source_identity._number_operator_placeholders)
        self._fermion_indices = tuple(
            index
            for index, operator in enumerate(self.source_operators)
            if isinstance(operator, FermionOp)
        )
        self.source_is_packed = self.target_is_packed and all(
            isinstance(operator, FermionOp) for operator in self.source_operators
        )
        self.target_basis = tuple(coordinate.name for coordinate in self.coordinates)
        self._source_occupation_boolean = (
            tuple(
                BooleanPolynomial.from_expr(occupation, self._coordinate_symbols)
                for occupation in self._source_occupations
            )
            if self.target_is_packed
            else ()
        )
        if self.target_is_packed:
            occupation_matrix = sympy.Matrix(
                [
                    [
                        sympy.diff(occupation, symbol)
                        for symbol in self._coordinate_symbols
                    ]
                    for occupation in self._source_occupations
                ]
            )
            affine = all(
                not entry.free_symbols.intersection(self._coordinate_symbols)
                for entry in occupation_matrix
            )
            if affine and occupation_matrix.rank() == len(self.coordinates):
                self._occupation_matrix = occupation_matrix
                self._occupation_left_inverse = (
                    occupation_matrix.T * occupation_matrix
                ).inv() * occupation_matrix.T
            else:
                self._occupation_matrix = None
                self._occupation_left_inverse = None
        else:
            self._occupation_matrix = None
            self._occupation_left_inverse = None

    @property
    def complement_space(self):
        """The formal source complement defined by this embedding.

        The embedding itself is the descriptor: its ``W`` fixes both the
        retained image and the complementary projector, without a basis.
        """
        return self

    @property
    def target_space(self):
        return self.encoding.target

    @staticmethod
    def adjoint_map(operator_map: OperatorMap):
        return RowBlock(operator_map)

    def source_form(
        self,
        expression: sympy.Expr | NumberOrderedForm | PackedForm | TensorOperator,
    ) -> TensorOperator:
        """Convert a source expression to the canonical source algebra."""
        if isinstance(expression, TensorOperator):
            return expression
        if isinstance(expression, PackedForm):
            if not self.source_is_packed:
                raise TypeError("Packed source terms require a purely binary source")
            return TensorOperator.from_factor(expression)
        if isinstance(expression, NumberOrderedForm):
            if expression.operators == self.source_operators:
                form = expression
            else:
                form = NumberOrderedForm.from_expr(
                    expression.as_expr(), operators=self.source_operators
                )
        else:
            form = NumberOrderedForm.from_expr(
                sympy.sympify(expression), operators=self.source_operators
            )
        native = nof_to_packed_fermions(form) if self.source_is_packed else form
        return TensorOperator.from_factor(native)

    def source_nof(self, expression: sympy.Expr | NumberOrderedForm) -> NumberOrderedForm:
        """Return the source NumberOrderedForm used for diagonal energy formulas."""
        if isinstance(expression, NumberOrderedForm):
            if expression.operators == self.source_operators:
                return expression
            return NumberOrderedForm.from_expr(
                expression.as_expr(), operators=self.source_operators
            )
        return NumberOrderedForm.from_expr(
            sympy.sympify(expression), operators=self.source_operators
        )

    @cached_property
    def target_identity(self) -> TargetForm:
        if self.target_is_packed:
            return TensorOperator.from_factor(PackedForm.identity(self.target_basis))
        return _immutable_matrix(sympy.eye(self.encoding.target.dimension))

    @cached_property
    def target_zero(self) -> TargetForm:
        if self.target_is_packed:
            return TensorOperator.from_factor(PackedForm.zero(self.target_basis))
        return _immutable_matrix(sympy.zeros(self.encoding.target.dimension))

    @cache
    def _target_shift(self, powers: tuple) -> tuple[int, ...] | None:
        if self._occupation_left_inverse is not None:
            source_shift = sympy.Matrix(powers)
            result = self._occupation_left_inverse * source_shift
            if self._occupation_matrix * result != source_shift:
                return None
            if any(not value.is_Integer for value in result):
                return None
            integer_result = tuple(map(int, result))
            return (
                integer_result
                if all(abs(value) <= 1 for value in integer_result)
                else None
            )

        shifts = sympy.symbols(f"_shift_0:{len(self.coordinates)}", integer=True)
        shifted = {
            symbol: symbol - shift
            for symbol, shift in zip(self._coordinate_symbols, shifts, strict=True)
        }
        equations = [
            sympy.expand(occupation.xreplace(shifted) - occupation + power)
            for occupation, power in zip(self._source_occupations, powers, strict=True)
        ]
        solution = sympy.solve(equations, shifts, dict=True)
        if len(solution) != 1 or any(shift not in solution[0] for shift in shifts):
            return None
        result = tuple(sympy.simplify(solution[0][shift]) for shift in shifts)
        if any(not value.is_Integer for value in result):
            return None
        integer_result = tuple(int(value) for value in result)
        if any(abs(value) > 1 for value in integer_result):
            return None
        return integer_result

    def _source_matrix_element(
        self,
        powers: tuple,
        coefficient: sympy.Expr,
    ) -> sympy.Expr:
        state = list(self._source_occupations)
        amplitude = sympy.S.One

        # NumberOrderedForm acts with annihilators in ascending source-mode order.
        for index, power in enumerate(powers):
            for _ in range(max(int(power), 0)):
                operator = self.source_operators[index]
                occupation = state[index]
                if isinstance(operator, BosonOp):
                    amplitude *= sympy.sqrt(occupation)
                elif isinstance(operator, LadderOp):
                    pass
                elif isinstance(operator, SigmaMinus):
                    amplitude *= occupation
                elif isinstance(operator, FermionOp):
                    parity = sympy.prod(
                        1 - 2 * state[earlier]
                        for earlier in self._fermion_indices
                        if earlier < index
                    )
                    amplitude *= occupation * parity
                else:  # pragma: no cover - guarded by NumberOrderedForm
                    raise TypeError(f"Unsupported source operator: {operator!r}")
                state[index] = occupation - 1

        amplitude *= coefficient.xreplace(
            dict(zip(self._source_placeholders, state, strict=True))
        )

        # Creators act in descending source-mode order.
        for index in reversed(range(len(powers))):
            for _ in range(max(-int(powers[index]), 0)):
                operator = self.source_operators[index]
                occupation = state[index]
                if isinstance(operator, BosonOp):
                    amplitude *= sympy.sqrt(occupation + 1)
                elif isinstance(operator, LadderOp):
                    pass
                elif isinstance(operator, SigmaMinus):
                    amplitude *= 1 - occupation
                elif isinstance(operator, FermionOp):
                    parity = sympy.prod(
                        1 - 2 * state[earlier]
                        for earlier in self._fermion_indices
                        if earlier < index
                    )
                    amplitude *= (1 - occupation) * parity
                else:  # pragma: no cover - guarded by NumberOrderedForm
                    raise TypeError(f"Unsupported source operator: {operator!r}")
                state[index] = occupation + 1

        return sympy.expand(amplitude)

    def source_terms(self, source: TensorOperator):
        """Yield ``(powers, middle coefficient, one-term source operator)``."""
        source = self.source_form(source).native
        if isinstance(source, NumberOrderedForm):
            for powers, coefficient in source.terms.items():
                powers = tuple(map(int, powers))
                yield (
                    powers,
                    coefficient,
                    TensorOperator.from_factor(
                        _one_term_form(self.source_operators, powers, coefficient)
                    ),
                )
            return

        for monomial, coefficient in source.items:
            powers = packed_powers(monomial, num_modes=len(self.source_operators))
            yield (
                powers,
                coefficient,
                TensorOperator.from_factor(
                    PackedForm.monomial(source.basis, monomial, coefficient)
                ),
            )

    @cache
    def _project_source_term(
        self, powers: tuple[int, ...], coefficient: sympy.Expr
    ) -> PackedForm:
        target_powers = self._target_shift(powers)
        if target_powers is None:
            return PackedForm.zero(self.target_basis)

        amplitude = self._source_matrix_element(powers, coefficient)
        support = {
            symbol: sympy.S.One if power > 0 else sympy.S.Zero
            for symbol, power in zip(self._coordinate_symbols, target_powers, strict=True)
            if power
        }
        amplitude = amplitude.xreplace(support)
        initial_to_middle = {
            symbol: placeholder + max(power, 0)
            for symbol, placeholder, power in zip(
                self._coordinate_symbols,
                self._target_placeholders,
                target_powers,
                strict=True,
            )
            if symbol not in support
        }
        amplitude = sympy.expand(amplitude.xreplace(initial_to_middle))
        if amplitude == 0:
            return PackedForm.zero(self.target_basis)
        target_coefficient = BooleanPolynomial.from_expr(
            amplitude, self._target_placeholders
        )
        return spin_term_from_boolean(
            self.target_basis, target_powers, target_coefficient
        )

    @cache
    def _project_packed_monomial(self, monomial: int) -> PackedForm:
        """Project one coefficient-free structural source monomial."""
        powers = packed_powers(monomial, num_modes=len(self.source_operators))
        target_powers = self._target_shift(powers)
        if target_powers is None:
            return PackedForm.zero(self.target_basis)

        creators, numbers, annihilators = masks_from_monomial(
            monomial, num_modes=len(self.source_operators)
        )
        state = list(self._source_occupation_boolean)
        amplitude = BooleanPolynomial.scalar(self._coordinate_symbols, sympy.S.One)

        # Packed monomials are products of local factors in ascending source-mode
        # order, so their action on a ket proceeds in descending mode order.
        for mode in reversed(range(len(self.source_operators))):
            active = 1 << mode
            if numbers & active:
                amplitude *= state[mode]
                continue
            if not (creators | annihilators) & active:
                continue
            parity = BooleanPolynomial.scalar(self._coordinate_symbols, sympy.S.One)
            for earlier in range(mode):
                parity *= 1 - 2 * state[earlier]
            if annihilators & active:
                amplitude *= state[mode] * parity
                state[mode] = BooleanPolynomial.scalar(
                    self._coordinate_symbols, sympy.S.Zero
                )
            else:
                amplitude *= (1 - state[mode]) * parity
                state[mode] = BooleanPolynomial.scalar(
                    self._coordinate_symbols, sympy.S.One
                )

        support = {
            symbol: sympy.S.One if power > 0 else sympy.S.Zero
            for symbol, power in zip(self._coordinate_symbols, target_powers, strict=True)
            if power
        }
        initial_to_middle = {
            symbol: placeholder + max(power, 0)
            for symbol, placeholder, power in zip(
                self._coordinate_symbols,
                self._target_placeholders,
                target_powers,
                strict=True,
            )
            if symbol not in support
        }
        expression = amplitude.as_expr().xreplace(support).xreplace(initial_to_middle)
        target_coefficient = BooleanPolynomial.from_expr(
            expression, self._target_placeholders
        )
        return spin_term_from_boolean(
            self.target_basis, target_powers, target_coefficient
        )

    @cache
    def to_target(self, source: TensorOperator) -> TargetForm:
        """Return ``W† source W`` without constructing either basis."""
        source = self.source_form(source).native
        if not self.target_is_packed:
            if not isinstance(source, NumberOrderedForm):  # pragma: no cover
                raise TypeError("Finite targets currently require number-ordered source")
            compiled = CompiledOperator(source.as_expr(), self.source_operators)
            target = self.encoding.target
            source_to_target = {
                self.encoding.encode(state): index
                for index, state in enumerate(target.states)
            }
            matrix = sympy.MutableSparseMatrix(target.dimension, target.dimension, {})
            for column, state in enumerate(target.states):
                for output, coefficient in compiled.apply(
                    self.encoding.encode(state)
                ).items():
                    if (row := source_to_target.get(output)) is not None:
                        matrix[row, column] += coefficient
            return _immutable_matrix(matrix)

        result = PackedForm.zero(self.target_basis)
        if isinstance(source, PackedForm):
            for monomial, coefficient in source.items:
                result += self._project_packed_monomial(monomial) * coefficient
            return TensorOperator.from_factor(result)

        for powers, coefficient, _source_term in self.source_terms(source):
            result += self._project_source_term(powers, coefficient)
        return TensorOperator.from_factor(result)


@dataclass(frozen=True)
class TargetBlock:
    embedding: AlgebraicEmbedding | MapEmbedding
    form: TargetForm

    @classmethod
    def build(cls, embedding: AlgebraicEmbedding | MapEmbedding, form: TargetForm):
        return zero if _target_is_zero(form) else cls(embedding, form)

    def __add__(self, other):
        if other is zero:
            return self
        if not isinstance(other, TargetBlock):
            return NotImplemented
        return type(self).build(self.embedding, self.form + other.form)

    def __neg__(self):
        return type(self).build(self.embedding, -self.form)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, divisor: int):
        return type(self).build(self.embedding, self.form * sympy.Rational(1, divisor))

    def __mul__(self, other):
        if isinstance(other, TargetBlock):
            return type(self).build(self.embedding, self.form * other.form)
        if isinstance(other, RowBlock):
            return type(other)(other.column.right(self.adjoint()))
        return NotImplemented

    def adjoint(self):
        return type(self).build(self.embedding, self.form.adjoint())


@dataclass(frozen=True)
class MapComponent:
    """One ``left_action ◦ bridge ◦ right_action`` map component."""

    left_action: TensorOperator
    bridge: AffineBand
    right_action: TargetForm


class BandMap(OperatorMap):
    """A compressed sum of operators acting around one affine bridge.

    Every component denotes ``Q left_action bridge right_action``.  The bridge
    owns the source and retained spaces, while the existing NOF, packed, and
    sparse-target representations remain intact.  Construction performs cheap
    exact compression along both factor axes; it does not search for a globally
    minimal tensor factorization.
    """

    def __init__(
        self,
        embedding: MapEmbedding,
        terms: Iterable[tuple[TensorOperator, TargetForm]],
    ):
        if not isinstance(embedding, MapEmbedding):
            raise TypeError("BandMap requires an affine map embedding")
        raw_terms = tuple(terms)
        normalized = self._compress(embedding, raw_terms)
        self.embedding = embedding
        self.bridge = embedding.bridge
        self.components = tuple(
            MapComponent(source, self.bridge, target) for source, target in normalized
        )
        # Retain the OperatorMap protocol used by the perturbative recurrence.
        self.terms = tuple(
            (component.left_action, component.right_action)
            for component in self.components
        )
        self.uncompressed_count = len(raw_terms)

    @staticmethod
    def _compress(
        embedding: MapEmbedding,
        terms: Iterable[tuple[TensorOperator, TargetForm]],
    ) -> tuple[tuple[TensorOperator, TargetForm], ...]:
        pairs = []
        for source, target in terms:
            if _is_zero_operator(source) or _target_is_zero(target):
                continue
            pairs.append((source, target))

        while pairs:
            previous_count = len(pairs)

            by_left: dict[TensorOperator, TargetForm] = {}
            for source, target in pairs:
                by_left[source] = by_left.get(source, embedding.target_zero) + target
            pairs = [
                (source, target)
                for source, target in by_left.items()
                if not _target_is_zero(target)
            ]

            by_right: dict[TargetForm, TensorOperator] = {}
            for source, target in pairs:
                if target in by_right:
                    previous = by_right[target]
                    if type(previous) is not type(source):
                        raise TypeError("Cannot combine unlike source-operator domains")
                    by_right[target] = previous + source
                else:
                    by_right[target] = source
            pairs = [
                (source, target)
                for target, source in by_right.items()
                if not _is_zero_operator(source)
            ]

            if len(pairs) == previous_count:
                break

        return tuple(pairs)

    @property
    def compression(self) -> tuple[int, int]:
        """Return ``(input components, retained components)`` for the last build."""
        return self.uncompressed_count, len(self.components)

    def lower(self):
        """Lower to explicit arrows, preserving :class:`RuleMap` as a reference."""
        result = zero
        for component in self.components:
            lowered = RuleMap.from_source(self.embedding, component.left_action)
            if lowered is zero:
                continue
            lowered = lowered.right(component.right_action)
            result = lowered if result is zero else result + lowered
        return result

    @cached_property
    def factored(self):
        """Compile a contraction index without replacing the stored components."""
        return OffDiagonalMap(self.embedding, self.terms)

    def inner(self, other: OperatorMap):
        if not isinstance(other, BandMap):
            return super().inner(other)
        if self.bridge != other.bridge:
            raise ValueError("Band maps must use the same affine bridge")
        return self.factored.inner(other.factored)

    def adjoint(self):
        return AdjointBandMap(self)


class OffDiagonalMap(OperatorMap):
    """Canonical sparse map from the retained space into the complement.

    An item ``(X_key, |out><in|): coefficient`` denotes
    ``coefficient * Q X W |out><in|``.  The virtual output of ``X`` is not
    enumerated.  Number-operator coefficients are evaluated after ``out`` fixes
    the source occupation on which the number-ordered monomial acts.
    """

    def __init__(
        self,
        embedding: MapEmbedding,
        terms: Iterable[tuple[TensorOperator, TargetForm]],
    ):
        if not isinstance(embedding, MapEmbedding):
            raise TypeError("OffDiagonalMap requires a map embedding")
        combined: dict[OffDiagonalMapKey, sympy.Expr] = {}
        for source, target in terms:
            if _is_zero_operator(source) or _target_is_zero(target):
                continue
            target_matrix = _target_matrix(target)
            for source_key, source_coefficient, placeholders in self._source_items(
                source
            ):
                for target_key, target_coefficient in matrix_items(
                    embedding.target,
                    target_matrix,
                ):
                    source_coefficient_at_target = self._coefficient_at_target(
                        embedding,
                        source_key,
                        source_coefficient,
                        placeholders,
                        target_key.output_state,
                    )
                    coefficient = source_coefficient_at_target * target_coefficient
                    if coefficient == 0:
                        continue
                    key = OffDiagonalMapKey(source_key, target_key)
                    combined[key] = combined.get(key, sympy.S.Zero) + coefficient
                    if combined[key] == 0:
                        del combined[key]
        self.embedding = embedding
        self.items = tuple(
            sorted(
                combined.items(),
                key=lambda item: (
                    (
                        0,
                        item[0].source_monomial,
                    )
                    if isinstance(item[0].source_monomial, int)
                    else (
                        1,
                        item[0].source_monomial,
                    ),
                    item[0].target_transition,
                ),
            )
        )

    @staticmethod
    def _source_items(source: TensorOperator):
        source = source.native
        if isinstance(source, PackedForm):
            for monomial, coefficient in source.items:
                yield monomial, coefficient, ()
            return
        placeholders = tuple(source._number_operator_placeholders)
        for powers, coefficient in source.terms.items():
            yield tuple(map(int, powers)), coefficient, placeholders

    @staticmethod
    def _coefficient_at_target(
        embedding: MapEmbedding,
        source_key: tuple[int, ...] | int,
        coefficient: sympy.Expr,
        placeholders: Sequence[sympy.Symbol],
        target_output: tuple[int, ...],
    ) -> sympy.Expr:
        if isinstance(source_key, int):
            return coefficient

        state = list(embedding.encoding.encode(target_output))
        for index, power in enumerate(source_key):
            for _ in range(max(power, 0)):
                operator = embedding.source_operators[index]
                occupation = state[index]
                if isinstance(operator, BosonOp):
                    if occupation == 0:
                        return sympy.S.Zero
                    state[index] = occupation - 1
                elif isinstance(operator, LadderOp):
                    state[index] = occupation - 1
                elif isinstance(operator, (SigmaMinus, FermionOp)):
                    if occupation == 0:
                        return sympy.S.Zero
                    state[index] = 0
                else:  # pragma: no cover - guarded by NumberOrderedForm
                    raise TypeError(f"Unsupported source operator: {operator!r}")
        return coefficient.xreplace(
            dict(zip(placeholders, map(sympy.Integer, state), strict=True))
        )

    @cached_property
    def terms(self) -> tuple[tuple[TensorOperator, TargetForm], ...]:
        grouped: dict[
            tuple[int, ...] | int,
            list[tuple[MapKey, sympy.Expr]],
        ] = {}
        for key, coefficient in self.items:
            grouped.setdefault(key.source_monomial, []).append(
                (key.target_transition, coefficient)
            )

        source_basis = tuple(
            str(operator.name) for operator in self.embedding.source_operators
        )
        return tuple(
            (
                TensorOperator.from_factor(
                    (
                        PackedForm.monomial(source_basis, source_key)
                        if isinstance(source_key, int)
                        else NumberOrderedForm(
                            self.embedding.source_operators,
                            {source_key: sympy.S.One},
                            validate=False,
                        )
                    )
                ),
                state_matrix(self.embedding.target, target_terms),
            )
            for source_key, target_terms in grouped.items()
        )

    def __bool__(self) -> bool:
        return bool(self.items)

    def or_zero(self):
        return self if self.items else zero

    def adjoint(self):
        return AdjointOffDiagonalMap(self)

    def inner(self, other: OperatorMap):
        if not isinstance(other, OffDiagonalMap):
            return super().inner(other)

        result_terms = []
        to_target = self.embedding.to_target
        for left_source, left_target in self.terms:
            left_adjoint = _adjoint(left_source)
            projected_left = to_target(left_adjoint)
            left_by_output: dict[
                tuple[int, ...],
                list[tuple[tuple[int, ...], sympy.Expr]],
            ] = {}
            for key, coefficient in matrix_items(
                self.embedding.target,
                _target_matrix(left_target),
            ):
                left_by_output.setdefault(key.output_state, []).append(
                    (key.input_state, coefficient)
                )
            for right_source, right_target in other.terms:
                kernel = to_target(left_adjoint * right_source)
                kernel -= projected_left * to_target(right_source)
                right_by_output: dict[
                    tuple[int, ...],
                    list[tuple[tuple[int, ...], sympy.Expr]],
                ] = {}
                for key, coefficient in matrix_items(
                    self.embedding.target,
                    _target_matrix(right_target),
                ):
                    right_by_output.setdefault(key.output_state, []).append(
                        (key.input_state, coefficient)
                    )

                for kernel_key, kernel_coefficient in matrix_items(
                    self.embedding.target,
                    _target_matrix(kernel),
                ):
                    for left_input, left_coefficient in left_by_output.get(
                        kernel_key.output_state, ()
                    ):
                        for right_input, right_coefficient in right_by_output.get(
                            kernel_key.input_state, ()
                        ):
                            result_terms.append(
                                (
                                    MapKey(right_input, left_input),
                                    sympy.conjugate(left_coefficient)
                                    * kernel_coefficient
                                    * right_coefficient,
                                )
                            )
        return TargetBlock.build(
            self.embedding,
            state_matrix(self.embedding.target, result_terms),
        )


class RuleMap(OperatorMap):
    """Recurrence adapter for an explicit enumerable state map."""

    def __init__(
        self,
        left_space: ComplementSpace,
        right_space: TargetSpace | FermionSpace,
        items: Iterable[
            tuple[
                tuple[tuple[int, ...], tuple[int, ...]],
                object,
            ]
        ],
    ):
        self._set_map(
            ExplicitOperatorMap(
                left_space,
                right_space,
                (
                    (key, coefficient)
                    for key, coefficient in items
                    if left_space.contains(key[1])
                ),
            )
        )

    @classmethod
    def _from_map(cls, operator_map: ExplicitOperatorMap):
        result = cls.__new__(cls)
        result._set_map(operator_map)
        return result

    def _set_map(self, operator_map: ExplicitOperatorMap) -> None:
        self.map = operator_map
        self._left_space = operator_map.left_space
        self._right_space = operator_map.right_space
        self.items = operator_map.items

    @classmethod
    def from_source(
        cls,
        embedding: MapEmbedding,
        source: TensorOperator,
    ):
        items = []
        for target_state in embedding.target.states:
            source_state = embedding.encoding.encode(target_state)
            phase = embedding._target_phases[target_state]
            for output_state, coefficient in embedding.apply_source(source, source_state):
                items.append(
                    (
                        (target_state, output_state),
                        coefficient * phase,
                    )
                )
        return cls(
            embedding.complement_space,
            embedding.target,
            items,
        ).or_zero()

    def __bool__(self) -> bool:
        return bool(self.items)

    def _require_same_spaces(self, other: RuleMap) -> None:
        self.map._require_same_spaces(other.map)

    def __add__(self, other):
        if other is zero:
            return self
        if not isinstance(other, RuleMap):
            return NotImplemented
        return type(self)._from_map(self.map + other.map).or_zero()

    def __neg__(self):
        return type(self)._from_map(-self.map).or_zero()

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, divisor: int):
        return self.scale(sympy.Rational(1, divisor))

    def scale(self, factor: object):
        return type(self)._from_map(self.map.scale(factor)).or_zero()

    def or_zero(self):
        return self if self.items else zero

    def right(self, target: TargetBlock):
        target = getattr(target, "form", target)
        return (
            type(self)
            ._from_map(self.map.right_multiply(_target_matrix(target)))
            .or_zero()
        )

    def left(self, source: TensorOperator):
        return (
            type(self)
            ._from_map(
                self.map.left_apply(
                    lambda state: self.left_space.ambient.apply(source, state)
                )
            )
            .or_zero()
        )

    def inner(self, other: OperatorMap):
        if not isinstance(other, RuleMap):
            return NotImplemented
        result = self.map.inner(other.map)
        return TargetBlock.build(self.right_space, result)

    def __mul__(self, other):
        if isinstance(other, TargetBlock):
            return self.right(other)
        if isinstance(other, RowBlock):
            if not isinstance(other.column, RuleMap):
                return NotImplemented
            self._require_same_spaces(other.column)
            return RankOneQBlock(self, other.column)
        return NotImplemented

    def adjoint(self):
        return AdjointRuleMap(self)


@dataclass(frozen=True)
class RowBlock:
    """Adjoint of a formal complement column."""

    column: OperatorMap

    def __add__(self, other):
        if other is zero:
            return self
        if not isinstance(other, RowBlock):
            return NotImplemented
        result = self.column + other.column
        return zero if result is zero else type(self)(result)

    def __neg__(self):
        result = -self.column
        return zero if result is zero else type(self)(result)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, divisor: int):
        result = self.column / divisor
        return zero if result is zero else type(self)(result)

    def __mul__(self, other):
        if isinstance(other, OperatorMap):
            result = self.column.inner(other)
            if isinstance(result, TargetBlock):
                return result
            return TargetBlock.build(self.column.embedding, result)
        if isinstance(other, QBlock):
            result = other.adjoint().apply(self.column)
            return zero if result is zero else type(self)(result)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, OperatorMap):
            return RankOneQBlock(other, self.column)
        return NotImplemented

    def adjoint(self):
        return self.column


class AdjointOffDiagonalMap(RowBlock):
    """Adjoint view of a target-to-complement sparse map."""

    column: OffDiagonalMap


class AdjointBandMap(RowBlock):
    """Adjoint view of a compressed operator-band map."""

    column: BandMap

    @property
    def left_space(self):
        return self.column.right_space

    @property
    def right_space(self):
        return self.column.left_space


class AdjointRuleMap(RowBlock):
    """Adjoint view of an explicitly composed rule map."""

    column: RuleMap

    @property
    def left_space(self):
        return self.column.right_space

    @property
    def right_space(self):
        return self.column.left_space


class QBlock:
    """An endomorphism of the formal complement module."""

    embedding: AlgebraicEmbedding | MapEmbedding

    def apply(self, column: OperatorMap):  # pragma: no cover - abstract protocol
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, OperatorMap):
            return self.apply(other)
        if isinstance(other, QBlock):
            return ProductQBlock(self, other)
        return NotImplemented

    def __add__(self, other):
        if other is zero:
            return self
        if not isinstance(other, QBlock):
            return NotImplemented
        return SumQBlock((self, other))

    def __neg__(self):
        return ScaledQBlock(-sympy.S.One, self)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, divisor: int):
        return ScaledQBlock(sympy.Rational(1, divisor), self)


@dataclass(frozen=True)
class SourceQBlock(QBlock):
    embedding: AlgebraicEmbedding | MapEmbedding
    source: TensorOperator

    def apply(self, column: OperatorMap):
        if isinstance(column, RuleMap):
            return column.left(self.source)
        return column.left(self.source)

    def adjoint(self):
        return type(self)(self.embedding, _adjoint(self.source))


@dataclass(frozen=True)
class RankOneQBlock(QBlock):
    left: OperatorMap
    right: OperatorMap

    @property
    def embedding(self):
        return self.left.embedding

    def apply(self, column: OperatorMap):
        coefficient = self.right.inner(column)
        if isinstance(coefficient, TargetBlock):
            coefficient = coefficient.form
        return zero if _target_is_zero(coefficient) else self.left.right(coefficient)

    def adjoint(self):
        return type(self)(self.right, self.left)


@dataclass(frozen=True)
class SumQBlock(QBlock):
    terms: tuple[QBlock, ...]

    @property
    def embedding(self):
        return self.terms[0].embedding

    def apply(self, column: OperatorMap):
        result = zero
        for term in self.terms:
            result = result + term.apply(column)
        return result

    def adjoint(self):
        return type(self)(tuple(term.adjoint() for term in self.terms))


@dataclass(frozen=True)
class ProductQBlock(QBlock):
    left: QBlock
    right: QBlock

    @property
    def embedding(self):
        return self.left.embedding

    def apply(self, column: OperatorMap):
        right_result = self.right.apply(column)
        return zero if right_result is zero else self.left.apply(right_result)

    def adjoint(self):
        return type(self)(self.right.adjoint(), self.left.adjoint())


@dataclass(frozen=True)
class ScaledQBlock(QBlock):
    factor: sympy.Expr
    block: QBlock

    @property
    def embedding(self):
        return self.block.embedding

    def apply(self, column: OperatorMap):
        result = self.block.apply(column)
        if result is zero:
            return zero
        if isinstance(result, RuleMap):
            return result.scale(self.factor)
        return result.scale(self.factor)

    def adjoint(self):
        return type(self)(sympy.conjugate(self.factor), self.block.adjoint())


class AlgebraicProblem:
    """Two-block algebraic lowering of ``[H0, V]``."""

    def __init__(
        self,
        hamiltonian: Sequence[sympy.Expr],
        encoding: OccupationEncoding | FermionEmbedding,
        *,
        map_storage: bool = False,
        composed_maps: bool = True,
        band_maps: bool = False,
    ):
        if len(hamiltonian) != 2:
            raise NotImplementedError(
                "The algebraic prototype currently supports [H0, V]"
            )
        self.embedding = (
            MapEmbedding(encoding) if map_storage else AlgebraicEmbedding(encoding)
        )
        self.composed_maps = map_storage and composed_maps
        self.band_maps = map_storage and band_maps
        h0_nof = self.embedding.source_nof(hamiltonian[0])
        self.H0 = self.embedding.source_form(hamiltonian[0])
        self.V = self.embedding.source_form(hamiltonian[1])
        zero_powers = (0,) * len(self.embedding.source_operators)
        if any(tuple(powers) != zero_powers for powers in h0_nof.terms):
            raise ValueError("The algebraic prototype requires diagonal H0")
        self.source_energy = h0_nof.terms.get(zero_powers, sympy.S.Zero)
        self._source_energy_placeholders = tuple(h0_nof._number_operator_placeholders)
        target_h0 = self.embedding.to_target(self.H0)
        target_h0_native = (
            target_h0.native if isinstance(target_h0, TensorOperator) else None
        )
        if isinstance(target_h0_native, PackedForm):
            if any(
                any(packed_powers(monomial, num_modes=target_h0_native.num_modes))
                for monomial, _coefficient in target_h0_native.items
            ):
                raise ValueError("The embedding image must be invariant under H0")
            self.target_energy = self.source_energy.xreplace(
                dict(
                    zip(
                        self.embedding._source_placeholders,
                        self.embedding._source_occupations,
                        strict=True,
                    )
                )
            )
            self._finite_target_energy = None
        else:
            if any(row != column for row, column, _ in _target_entries(target_h0)):
                raise ValueError("The retained finite block must diagonalize H0")
            target_h0_matrix = _target_matrix(target_h0)
            self._finite_target_energy = tuple(
                target_h0_matrix[index, index]
                for index in range(self.embedding.encoding.target.dimension)
            )
            self.target_energy = None
        self._h0_compiled = (
            None
            if map_storage
            else CompiledOperator(
                h0_nof.as_expr(),
                self.embedding.source_operators,
            )
        )
        if not map_storage:
            self.source_energy = sympy.simplify(self.source_energy)
        if self.target_energy is not None:
            self.target_energy = sympy.simplify(self.target_energy)

    @cache
    def _source_energy_at(self, state: tuple[int, ...]) -> sympy.Expr:
        if self._h0_compiled is not None:
            return self._h0_compiled.diagonal(state)
        return self.source_energy.xreplace(
            dict(
                zip(
                    self._source_energy_placeholders,
                    map(sympy.Integer, state),
                    strict=True,
                )
            )
        )

    @cache
    def _finite_denominator(
        self,
        target_column: int,
        source_state: tuple[int, ...],
    ) -> sympy.Expr:
        return sympy.factor(
            self._finite_target_energy[target_column]
            - self._source_energy_at(source_state)
        )

    @cache
    def _support_substitutions(
        self,
        source_powers: tuple,
        target_powers: tuple,
    ) -> dict[sympy.Symbol, sympy.Expr]:
        initial = self.embedding._coordinate_symbols
        after_target = {
            symbol: symbol - power
            for symbol, power in zip(initial, target_powers, strict=True)
        }
        equations = []
        for occupation, operator, power in zip(
            self.embedding._source_occupations,
            self.embedding.source_operators,
            source_powers,
            strict=True,
        ):
            before_source = occupation.xreplace(after_target)
            if isinstance(operator, (FermionOp, SigmaMinus)) and power:
                equations.append(before_source - (1 if power > 0 else 0))
            elif isinstance(operator, BosonOp) and power > 0:
                # Equality is sufficient for the binary target coordinates supported
                # by this prototype. General bosonic targets need inequality support.
                equations.append(before_source - power)

        equations.extend(
            symbol - (1 if power > 0 else 0)
            for symbol, power in zip(initial, target_powers, strict=True)
            if power
        )
        if not equations:
            return {}
        try:
            matrix, right_hand_side = sympy.linear_eq_to_matrix(equations, initial)
        except sympy.NonlinearError:
            solutions = sympy.solve(equations, initial, dict=True)
            if len(solutions) != 1:
                return {}
            return {
                symbol: value
                for symbol, value in solutions[0].items()
                if value in (sympy.S.Zero, sympy.S.One)
            }

        reduced, pivots = matrix.row_join(right_hand_side).rref()
        substitutions = {}
        for row, pivot in enumerate(pivots):
            if pivot >= len(initial):
                return {}
            if any(
                reduced[row, column] != 0
                for column in range(len(initial))
                if column != pivot
            ):
                continue
            value = reduced[row, -1]
            if value in (sympy.S.Zero, sympy.S.One):
                substitutions[initial[pivot]] = value
        return substitutions

    @cache
    def _energy_denominator(
        self,
        source_powers: tuple,
        target_powers: tuple,
    ) -> BooleanPolynomial:
        target_initial = self.embedding._coordinate_symbols
        after_target = tuple(
            symbol - power
            for symbol, power in zip(target_initial, target_powers, strict=True)
        )
        encoded_after_target = tuple(
            occupation.xreplace(
                dict(
                    zip(
                        self.embedding._coordinate_symbols,
                        after_target,
                        strict=True,
                    )
                )
            )
            for occupation in self.embedding._source_occupations
        )
        source_output = tuple(
            occupation - power
            for occupation, power in zip(encoded_after_target, source_powers, strict=True)
        )
        source_energy = self.source_energy.xreplace(
            dict(
                zip(
                    self.embedding._source_placeholders,
                    source_output,
                    strict=True,
                )
            )
        )
        target_energy = self.target_energy
        denominator = sympy.expand(target_energy - source_energy)
        denominator = denominator.xreplace(
            self._support_substitutions(source_powers, target_powers)
        )
        initial_to_middle = {
            symbol: placeholder + max(power, 0)
            for symbol, placeholder, power in zip(
                target_initial,
                self.embedding._target_placeholders,
                target_powers,
                strict=True,
            )
        }
        return BooleanPolynomial.from_expr(
            denominator.xreplace(initial_to_middle),
            self.embedding._target_placeholders,
        )

    def solve_sylvester(self, value, index):
        if value is zero:
            return zero
        if index[:2] != (0, 1) or not isinstance(value, RowBlock):
            raise TypeError("The algebraic Sylvester solver expects the P-Q block")
        if isinstance(value.column, RuleMap):
            return self._solve_sylvester_rules(value)
        if not self.embedding.target_is_packed:
            return self._solve_sylvester_finite(value)

        solved_terms = []
        for source, target in value.column.terms:
            if not isinstance(target, TensorOperator) or not isinstance(
                target.native, PackedForm
            ):
                raise TypeError("Expected a packed tensor target")
            target_native = target.native
            for (
                source_powers,
                source_coefficient,
                source_term,
            ) in self.embedding.source_terms(source):
                for target_monomial, target_coefficient in target_native.items:
                    target_powers = packed_powers(
                        target_monomial, num_modes=target_native.num_modes
                    )
                    denominator = self._energy_denominator(
                        tuple(source_powers), target_powers
                    )
                    if not denominator:
                        leakage_kernel = self.embedding.to_target(
                            _adjoint(source_term) * source_term
                        ) - self.embedding.to_target(source_term).adjoint() * (
                            self.embedding.to_target(source_term)
                        )
                        target_term = TensorOperator.from_factor(
                            packed_monomial_term(
                                target_native,
                                target_monomial,
                                target_coefficient,
                            )
                        )
                        leakage_norm = _target_simplify(
                            target_term.adjoint() * leakage_kernel * target_term
                        )
                        if _target_is_zero(leakage_norm):
                            continue
                        raise ZeroDivisionError(
                            "A virtual algebraic channel is degenerate with the target: "
                            f"source powers {tuple(source_powers)}, "
                            f"target powers {tuple(target_powers)}, "
                            f"source coefficient {source_coefficient}, "
                            f"target coefficient {target_coefficient}"
                        )
                    target_term = TensorOperator.from_factor(
                        packed_monomial_term(
                            target_native,
                            target_monomial,
                            target_coefficient,
                        )
                    )
                    solved_target = target_term * TensorOperator.from_factor(
                        denominator.reciprocal().diagonal(self.embedding.target_basis)
                    )
                    solved_terms.append((source_term, solved_target))

        column = type(value.column)(self.embedding, solved_terms).or_zero()
        return zero if column is zero else type(value)(column)

    def _solve_sylvester_rules(self, value: RowBlock):
        column = value.column
        solved_items = []
        for key, coefficient in column.items:
            input_state, output_state = key
            target_column = column.right_space.index[input_state]
            denominator = self._finite_denominator(
                target_column,
                output_state,
            )
            if denominator == 0:
                raise ZeroDivisionError(
                    "A virtual map channel is degenerate with the retained block: "
                    f"source state {output_state}, "
                    f"target state {input_state}"
                )
            solved_items.append((key, coefficient / denominator))
        result = type(column)(
            column.left_space,
            column.right_space,
            solved_items,
        ).or_zero()
        return zero if result is zero else type(value)(result)

    @cache
    def _finite_source_action(
        self,
        source_term: SourceForm,
        row: int,
    ):
        if isinstance(self.embedding, MapEmbedding):
            return self.embedding.apply_source_term(source_term, row)
        native = source_term.native
        if not isinstance(native, NumberOrderedForm):  # pragma: no cover
            raise TypeError("The matrix target requires number-ordered source terms")
        target_state = self.embedding.encoding.target.states[row]
        source_state = self.embedding.encoding.encode(target_state)
        compiled = CompiledOperator(native.as_expr(), self.embedding.source_operators)
        return tuple(compiled.apply(source_state).items())

    def _solve_sylvester_finite(self, value: RowBlock):
        solved_terms = []
        for source, target in value.column.terms:
            if not isinstance(target, sympy.MatrixBase):
                raise TypeError("Expected a finite target block")
            for (
                source_powers,
                source_coefficient,
                source_term,
            ) in self.embedding.source_terms(source):
                for row, column, target_coefficient in _target_entries(target):
                    action = self._finite_source_action(source_term, row)
                    if not action:
                        continue
                    if (
                        len(action) != 1
                    ):  # pragma: no cover - one NOF term is deterministic
                        raise ValueError(
                            "A source term produced multiple occupation states"
                        )
                    output_state, _amplitude = action[0]
                    denominator = self._finite_denominator(column, output_state)
                    target_term = _target_matrix_unit(
                        self.embedding,
                        target,
                        row,
                        column,
                        target_coefficient,
                    )
                    if denominator == 0:
                        projected = self.embedding.to_target(source_term)
                        leakage_kernel = (
                            self.embedding.to_target(_adjoint(source_term) * source_term)
                            - projected.adjoint() * projected
                        )
                        leakage_norm = _target_simplify(
                            target_term.adjoint() * leakage_kernel * target_term
                        )
                        if _target_is_zero(leakage_norm):
                            continue
                        raise ZeroDivisionError(
                            "A virtual finite-target channel is degenerate with the "
                            f"retained block: source powers {tuple(source_powers)}, "
                            f"target matrix element {(row, column)}, "
                            f"source coefficient {source_coefficient}"
                        )
                    solved_terms.append((source_term, target_term / denominator))

        result = type(value.column)(self.embedding, solved_terms).or_zero()
        return zero if result is zero else type(value)(result)

    def block_series(self) -> BlockSeries:
        embedding = self.embedding
        target_h0 = TargetBlock.build(embedding, embedding.to_target(self.H0))
        target_v = TargetBlock.build(embedding, embedding.to_target(self.V))
        if self.band_maps:
            column_type = BandMap
        elif self.composed_maps:
            column_type = RuleMap
        elif isinstance(embedding, MapEmbedding):
            column_type = OffDiagonalMap
        else:
            column_type = OperatorMap
        column_v = column_type.from_source(embedding, self.V)
        row_v = zero if column_v is zero else column_v.adjoint()
        q_h0 = SourceQBlock(embedding, self.H0)
        q_v = SourceQBlock(embedding, self.V)
        data = {
            (0, 0, 0): target_h0,
            (1, 1, 0): q_h0,
            (0, 0, 1): target_v,
            (0, 1, 1): row_v,
            (1, 0, 1): column_v,
            (1, 1, 1): q_v,
        }
        return BlockSeries(data=data, shape=(2, 2), n_infinite=1, name="H")


@dataclass(frozen=True)
class AlgebraicTargetOperator:
    """A retained operator in packed, finite-matrix, or sparse-map form."""

    embedding: AlgebraicEmbedding | MapEmbedding
    operator: TargetForm

    @property
    def packed(self) -> PackedForm:
        if isinstance(self.operator, TensorOperator) and isinstance(
            self.operator.native, PackedForm
        ):
            return self.operator.native
        raise TypeError("This target is not represented by a packed factor")

    @cached_property
    def form(self) -> NumberOrderedForm | sympy.MatrixBase:
        """Return the packed-spin readout or explicit retained finite block."""
        if isinstance(self.operator, sympy.MatrixBase):
            return self.operator
        if isinstance(self.operator, TensorOperator):
            native = self.operator.native
            if isinstance(native, PackedForm):
                if self.embedding.target_is_fermionic:
                    return packed_to_fermion_nof(
                        native,
                        self.embedding.target_operators,
                    )
                return packed_to_spin_nof(native, self.embedding.target_operators)
            return native
        raise TypeError(f"Unexpected target operator: {type(self.operator)}")

    def as_expr(self) -> sympy.Expr:
        if isinstance(self.form, sympy.MatrixBase):
            raise TypeError("A finite target block has no single generator expression")
        return self.form.as_expr()

    def matrix(self) -> sympy.ImmutableMatrix:
        """Materialize a matrix only for validation or final inspection."""
        if isinstance(self.operator, sympy.MatrixBase):
            return _immutable_matrix(self.operator)
        target = self.embedding.encoding.target
        compiled = self.as_expr()
        # The reference state-action compiler is intentionally used only at readout.
        from .encoding import CompiledOperator

        action = CompiledOperator(compiled, self.embedding.target_operators)
        matrix = sympy.MutableSparseMatrix(target.dimension, target.dimension, {})
        for column, state in enumerate(target.states):
            for output, coefficient in action.apply(state).items():
                row = target.index.get(output)
                if row is not None:
                    matrix[row, column] += coefficient
        return sympy.ImmutableMatrix(matrix)

    def pauli(self) -> PauliPolynomial:
        if isinstance(self.embedding.encoding.target, FermionSpace):
            raise ValueError("Pauli expansion requires a product-coordinate target")
        target_operator = TargetOperator(
            self.embedding.encoding.target,
            self.matrix(),
        )
        return target_operator.pauli()

    @property
    def generators(self) -> tuple[FermionOp, ...]:
        target = self.embedding.encoding.target
        if not isinstance(target, FermionSpace):
            raise TypeError("This target is not a fermionic Fock space")
        return target.modes

    def matrix_element(self, bra: tuple[int, ...], ket: tuple[int, ...]) -> sympy.Expr:
        target = self.embedding.encoding.target
        return self.matrix()[target.index[tuple(bra)], target.index[tuple(ket)]]


class AlgebraicEffectiveSeries:
    def __init__(self, series, problem: AlgebraicProblem, to_order: int):
        self._series = series
        self.problem = problem
        self.to_order = to_order

    def __getitem__(self, order: int) -> AlgebraicTargetOperator:
        if not 0 <= order <= self.to_order:
            raise IndexError(f"Order {order} lies outside 0..{self.to_order}")
        value = self._series[0, 0, order]
        if value is zero:
            form = self.problem.embedding.target_zero
        elif isinstance(value, TargetBlock):
            form = value.form
        else:  # pragma: no cover - typed block multiplication should prevent this
            raise TypeError(f"Unexpected retained block: {type(value)}")
        return AlgebraicTargetOperator(self.problem.embedding, form)


def block_diagonalize_algebraic(
    hamiltonian: Sequence[sympy.Expr],
    *,
    encoding: OccupationEncoding | FermionEmbedding,
    to_order: int,
):
    """Run Pymablock's recurrence over the formal embedding module."""
    problem = AlgebraicProblem(hamiltonian, encoding)
    series = _pymablock_block_diagonalize(
        problem.block_series(),
        solve_sylvester=problem.solve_sylvester,
    )
    return AlgebraicEffectiveSeries(series[0], problem, to_order), *series[1:]


def block_diagonalize_maps(
    hamiltonian: Sequence[sympy.Expr],
    *,
    encoding: OccupationEncoding | FermionEmbedding,
    to_order: int,
):
    """Run the recurrence with sparse maps on the retained Hilbert space."""
    problem = AlgebraicProblem(hamiltonian, encoding, map_storage=True)
    series = _pymablock_block_diagonalize(
        problem.block_series(),
        solve_sylvester=problem.solve_sylvester,
    )
    return AlgebraicEffectiveSeries(series[0], problem, to_order), *series[1:]


def block_diagonalize_factored_maps(
    hamiltonian: Sequence[sympy.Expr],
    *,
    encoding: OccupationEncoding | FermionEmbedding,
    to_order: int,
):
    """Run the retained-map backend with factored ``Q X W A`` columns."""
    problem = AlgebraicProblem(
        hamiltonian,
        encoding,
        map_storage=True,
        composed_maps=False,
    )
    series = _pymablock_block_diagonalize(
        problem.block_series(),
        solve_sylvester=problem.solve_sylvester,
    )
    return AlgebraicEffectiveSeries(series[0], problem, to_order), *series[1:]


def block_diagonalize_band_maps(
    hamiltonian: Sequence[sympy.Expr],
    *,
    encoding: OccupationEncoding | FermionEmbedding,
    to_order: int,
):
    """Run the recurrence over compressed ``operator ◦ band ◦ operator`` maps."""
    problem = AlgebraicProblem(
        hamiltonian,
        encoding,
        map_storage=True,
        composed_maps=False,
        band_maps=True,
    )
    series = _pymablock_block_diagonalize(
        problem.block_series(),
        solve_sylvester=problem.solve_sylvester,
    )
    return AlgebraicEffectiveSeries(series[0], problem, to_order), *series[1:]
