"""Projected operator algebra and perturbation theory for generated embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import sympy

from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main
from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.operator_embedding import (
    Embedding,
    _NOFTransition,
    _ReferenceEmbedding,
)
from pymablock.series import BlockSeries, zero

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

Operator: TypeAlias = NumberOrderedForm | sympy.MatrixBase


def _is_zero(value: NumberOrderedForm | sympy.MatrixBase) -> bool:
    """Test exact structural zero without simplifying coefficients."""
    if isinstance(value, sympy.MatrixBase):
        return all(
            _is_zero(entry) if isinstance(entry, NumberOrderedForm) else entry == 0
            for entry in value
        )
    return not any(coefficient != 0 for coefficient in value.terms.values())


def _adjoint(value: Operator) -> Operator:
    """Return an algebra-native adjoint."""
    result = value.adjoint()
    if isinstance(value, NumberOrderedForm) and result.operators != value.operators:
        # SymPy may cache an equal expression with a different list of unused
        # generators. Restore the declared basis using the public conversion.
        return NumberOrderedForm.from_expr(result.as_expr(), operators=value.operators)
    return result


class OperatorMap:
    """A canonical sum of formal complement columns ``Q X W A``."""

    __slots__ = ("embedding", "terms")

    def __init__(
        self,
        embedding: Embedding,
        terms: Iterable[tuple[Operator, Operator]],
    ):
        """Combine equal source factors and discard exact structural zeros."""
        combined: dict[Operator, Operator] = {}
        for source, target in terms:
            if _is_zero(source) or _is_zero(target):
                continue
            combined[source] = combined.get(source, embedding._target_zero) + target

        self.embedding = embedding
        self.terms = tuple(
            (source, target)
            for source, target in combined.items()
            if not _is_zero(target)
        )

    @classmethod
    def from_source(
        cls,
        embedding: Embedding,
        source: Operator,
    ) -> OperatorMap | object:
        """Construct ``Q X W`` from one source-space operator ``X``."""
        return cls(
            embedding,
            ((source, embedding._target_identity),),
        ).or_zero()

    def or_zero(self) -> OperatorMap | object:
        """Use Pymablock's structural zero sentinel for an empty map."""
        return self if self.terms else zero

    def __bool__(self) -> bool:
        """Return whether the map has any stored terms."""
        return bool(self.terms)

    def _require_same_embedding(self, other: OperatorMap) -> None:
        if self.embedding is not other.embedding:
            raise ValueError("Operator maps must use the same embedding")

    def __add__(self, other: object) -> OperatorMap | object:
        """Add two maps defined by the same embedding."""
        if other is zero:
            return self
        if not isinstance(other, OperatorMap):
            return NotImplemented
        self._require_same_embedding(other)
        return type(self)(
            self.embedding,
            (*self.terms, *other.terms),
        ).or_zero()

    def __neg__(self) -> OperatorMap | object:
        """Negate every retained-space factor."""
        return self.right(-sympy.S.One)

    def __sub__(self, other: object) -> OperatorMap | object:
        """Subtract a map defined by the same embedding."""
        return self + (-other)

    def __mul__(self, factor: object) -> OperatorMap | object:
        """Multiply every term by a scalar."""
        try:
            factor = sympy.sympify(factor)
        except sympy.SympifyError:
            return NotImplemented
        if not factor.is_commutative:
            return NotImplemented
        return self.right(factor)

    __rmul__ = __mul__

    def __truediv__(self, divisor: object) -> OperatorMap | object:
        """Divide every term by a scalar."""
        return self.right(sympy.S.One / divisor)

    def right(self, target: Operator) -> OperatorMap | object:
        """Compose with a retained-space operator on the right."""
        return type(self)(
            self.embedding,
            ((source, coefficient * target) for source, coefficient in self.terms),
        ).or_zero()

    def left(self, source: Operator) -> OperatorMap | object:
        """Apply a source operator while preserving the outer projection.

        This uses ``Q S Q X W A = Q S X W A - Q S W (W† X W) A``.
        """
        terms = []
        for inner_source, target in self.terms:
            terms.append((source * inner_source, target))
            terms.append((source, -self.embedding._pullback(inner_source) * target))
        return type(self)(self.embedding, terms).or_zero()

    def inner(self, other: OperatorMap) -> Operator:
        """Return ``self† other`` in the retained operator algebra."""
        self._require_same_embedding(other)

        result = self.embedding._target_zero
        for left_source, left_target in self.terms:
            left_adjoint = left_source.adjoint()
            projected_left = self.embedding._pullback(left_adjoint)
            for right_source, right_target in other.terms:
                kernel = self.embedding._pullback(left_adjoint * right_source)
                kernel -= projected_left * self.embedding._pullback(right_source)
                result += _adjoint(left_target) * kernel * right_target
        return result

    def adjoint(self) -> AdjointOperatorMap:
        """Return a lightweight Q→P adjoint view."""
        return AdjointOperatorMap(self)


@dataclass(frozen=True, slots=True)
class AdjointOperatorMap:
    """Adjoint view of a P→Q :class:`OperatorMap`."""

    column: OperatorMap

    def __add__(self, other: object) -> AdjointOperatorMap | object:
        """Add adjoint maps with the same embedding."""
        if other is zero:
            return self
        if not isinstance(other, AdjointOperatorMap):
            return NotImplemented
        result = self.column + other.column
        return result.adjoint()

    def __neg__(self) -> AdjointOperatorMap | object:
        """Negate the underlying column."""
        result = -self.column
        return result.adjoint()

    def __sub__(self, other: object) -> AdjointOperatorMap | object:
        """Subtract an adjoint map."""
        return self + (-other)

    def __truediv__(self, divisor: object) -> AdjointOperatorMap | object:
        """Divide the underlying column by a scalar."""
        result = self.column / divisor
        return result.adjoint()

    def adjoint(self) -> OperatorMap:
        """Return the underlying P→Q column."""
        return self.column


@dataclass(frozen=True)
class ModuleEndomorphism:
    """A complement action and its adjoint, composed without expanding Q."""

    action: Callable
    adjoint_action: Callable

    @classmethod
    def source(cls, source):
        """Represent ``Q source Q``."""
        return cls(lambda c: c.left(source), lambda c: c.left(source.adjoint()))

    @classmethod
    def rank_one(cls, left, right):
        """Represent ``left right†``."""
        left._require_same_embedding(right)
        return cls(
            lambda c: left.right(right.inner(c)), lambda c: right.right(left.inner(c))
        )

    def apply(self, column):
        """Apply the action, propagating an empty column."""
        return zero if column is zero else self.action(column)

    def adjoint(self):
        """Exchange the forward and adjoint actions."""
        return type(self)(self.adjoint_action, self.action)

    def compose(self, other):
        """Compose actions; reverse the order for the adjoint."""
        return type(self)(
            lambda c: self.apply(other.apply(c)),
            lambda c: other.adjoint().apply(self.adjoint().apply(c)),
        )

    def __add__(self, other):
        """Add actions and their adjoints."""
        if other is zero:
            return self
        if not isinstance(other, ModuleEndomorphism):
            return NotImplemented
        return type(self)(
            lambda c: self.apply(c) + other.apply(c),
            lambda c: self.adjoint().apply(c) + other.adjoint().apply(c),
        )

    def __neg__(self):
        return self / -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, divisor):
        return type(self)(
            lambda c: self.apply(c) * (sympy.S.One / divisor),
            lambda c: self.adjoint().apply(c) * (sympy.S.One / sympy.conjugate(divisor)),
        )


# The four block types have shapes PP, QP, PQ, QQ respectively.
# This is the complete multiplication table for conformable blocks.
_BLOCK_PRODUCTS = {
    (Operator, Operator): lambda a, b: a * b,
    (Operator, AdjointOperatorMap): lambda a, b: b.column.right(a.adjoint()).adjoint(),
    (OperatorMap, Operator): OperatorMap.right,
    (OperatorMap, AdjointOperatorMap): ModuleEndomorphism.rank_one,
    (AdjointOperatorMap, OperatorMap): lambda a, b: a.column.inner(b),
    (AdjointOperatorMap, ModuleEndomorphism): lambda a, b: b.adjoint()
    .apply(a.column)
    .adjoint(),
    (ModuleEndomorphism, OperatorMap): ModuleEndomorphism.apply,
    (ModuleEndomorphism, ModuleEndomorphism): ModuleEndomorphism.compose,
}


def multiply_projected(left, right):
    """Multiply conformable blocks of the projected algebra."""
    kinds = tuple(
        Operator if isinstance(x, (NumberOrderedForm, sympy.MatrixBase)) else type(x)
        for x in (left, right)
    )
    try:
        multiply = _BLOCK_PRODUCTS[kinds]
    except KeyError:
        raise TypeError(
            f"Cannot multiply projected blocks {type(left)} and {type(right)}"
        ) from None
    return multiply(left, right)


class _EmbeddingProblem:
    """Shared P/Q recurrence, independent of the retained representation."""

    def __init__(self, hamiltonian: BlockSeries, embedding: Embedding):
        if hamiltonian.shape:
            raise ValueError("Structured embeddings require an unseparated Hamiltonian.")
        self.embedding = embedding
        self.hamiltonian = hamiltonian
        zero_order = (0,) * hamiltonian.n_infinite
        h0 = self.embedding._source_form(hamiltonian[zero_order])
        self.source_shape = h0.shape if isinstance(h0, sympy.MatrixBase) else None
        self._prepare(h0)

    def _diagonal_energy(self, entry):
        zero_powers = (0,) * len(self.embedding.operators)
        if any(
            powers != zero_powers and coefficient != 0
            for powers, coefficient in entry.terms.items()
        ):
            raise ValueError("Structured embeddings currently require diagonal H0")
        return sympy.expand(entry.terms.get(zero_powers, sympy.S.Zero))

    def solve_sylvester(self, value, index):
        """Solve a P-Q equation using the compiled representation's channels."""
        if value is zero:
            return zero
        if index[:2] != (0, 1) or not isinstance(value, AdjointOperatorMap):
            raise TypeError("The embedding solver expects the P-Q block")
        return (
            OperatorMap(self.embedding, self._solve_terms(value.column.terms))
            .or_zero()
            .adjoint()
        )

    def block_series(self) -> BlockSeries:
        """Represent the retained and discarded Hamiltonian blocks."""
        embedding = self.embedding
        zero_order = (0,) * self.hamiltonian.n_infinite

        def evaluate(*index):
            row, column, *order = index
            source = self.hamiltonian[tuple(order)]
            if source is zero:
                return zero
            source = embedding._source_form(source)
            shape = source.shape if isinstance(source, sympy.MatrixBase) else None
            if shape != self.source_shape:
                raise ValueError(
                    "All Hamiltonian coefficients must have the same source matrix shape"
                )
            if _is_zero(source):
                return zero
            if row == column == 0:
                result = embedding._pullback(source)
                return zero if _is_zero(result) else result
            if tuple(order) == zero_order and row != column:
                return zero
            if (row, column) == (1, 0):
                return OperatorMap.from_source(embedding, source)
            if (row, column) == (0, 1):
                column_map = OperatorMap.from_source(embedding, source)
                return column_map.adjoint()
            return ModuleEndomorphism.source(source)

        return BlockSeries(
            eval=evaluate,
            shape=(2, 2),
            n_infinite=self.hamiltonian.n_infinite,
            dimension_names=self.hamiltonian.dimension_names,
            name="H",
        )


class _SymbolicProblem(_EmbeddingProblem):
    """Occupation-dependent denominators in the symbolic target algebra."""

    def _prepare(self, h0):
        self.source_energy = self._diagonal_energy(h0)

        # Selecting occupation states of diagonal H0 is automatically invariant.
        self.target_energy = self.source_energy.xreplace(
            dict(
                zip(
                    self.embedding._source_placeholders,
                    self.embedding.source_occupations,
                    strict=True,
                )
            )
        )

    def _solve_terms(self, terms):
        for source_form, target_form in terms:
            for source in _NOFTransition.from_form(source_form):
                for target in _NOFTransition.from_form(target_form):
                    denominator = self._channel_denominator(source, target)
                    yield (
                        source.form,
                        self._divide_channel(source, target.form, denominator),
                    )

    def _channel_denominator(
        self,
        source: _NOFTransition,
        target: _NOFTransition,
    ) -> sympy.Expr:
        encoded_after_target = self.embedding._shifted_occupations(target.powers)
        source_output = tuple(
            occupation - power
            for occupation, power in zip(encoded_after_target, source.powers, strict=True)
        )
        source_value = self.source_energy.xreplace(
            dict(zip(self.embedding._source_placeholders, source_output, strict=True))
        )
        denominator = sympy.expand(self.target_energy - source_value)
        denominator = denominator.xreplace(
            self.embedding._support_substitutions(source, target.powers)
        )
        return denominator.xreplace(self.embedding._initial_to_middle(target.powers))

    def _zero_denominator_is_projected_out(
        self,
        source: _NOFTransition,
        target: NumberOrderedForm,
    ) -> bool:
        projected = self.embedding._pullback(source.form)
        leakage = (
            self.embedding._pullback(source.form.adjoint() * source.form)
            - projected.adjoint() * projected
        )
        norm = target.adjoint() * leakage * target
        return all(sympy.cancel(coefficient) == 0 for coefficient in norm.terms.values())

    def _divide_channel(
        self, source: _NOFTransition, target: NumberOrderedForm, denominator: sympy.Expr
    ) -> NumberOrderedForm:
        """Divide middle coefficients, resolving finite occupation sectors."""
        variables = tuple(
            (symbol, size - abs(int(power)))
            for symbol, size, power in zip(
                self.embedding._target_placeholders,
                self.embedding.target.dimensions,
                next(iter(target.terms)),
                strict=True,
            )
            if size is not None and symbol in denominator.free_symbols
        )

        sectors = []
        has_zero = False

        def visit(expression, mask, remaining):
            nonlocal has_zero
            expression = sympy.expand(expression)
            remaining = tuple(x for x in remaining if x[0] in expression.free_symbols)
            if remaining:
                (symbol, size), *rest = remaining
                for value in range(size):
                    visit(
                        expression.xreplace({symbol: sympy.Integer(value)}),
                        mask
                        * (sympy.S.One if size == 1 else symbol if value else 1 - symbol),
                        rest,
                    )
                return
            expression = sympy.simplify(expression)
            if expression == 0:
                has_zero = True
                sector = target.applyfunc(lambda coefficient: coefficient * mask)
                if not self._zero_denominator_is_projected_out(source, sector):
                    raise ZeroDivisionError(
                        "A virtual channel is degenerate with the retained space"
                    )
            else:
                sectors.append(mask / expression)

        visit(denominator, sympy.S.One, variables)
        inverse = sympy.Add(*sectors) if has_zero else 1 / denominator
        result = target.applyfunc(lambda coefficient: coefficient * inverse)
        return result * self.embedding._target_identity


class _MatrixProblem(_EmbeddingProblem):
    """Energy differences for a finite reference basis."""

    def _prepare(self, h0):
        self.source_energies = [sympy.S.Zero] * h0.rows
        for (row, column), entry in h0.todok().items():
            if row != column and not _is_zero(entry):
                raise ValueError("Structured embeddings currently require diagonal H0")
            if row == column:
                self.source_energies[row] = self._diagonal_energy(entry)
        self.reference_energies = [
            self._energy(c, state) for c, state in self.embedding._references
        ]

    def _energy(self, component, state):
        return self.source_energies[component].xreplace(
            dict(zip(self.embedding._source_placeholders, state, strict=True))
        )

    def _solve_terms(self, terms):
        embedding = self.embedding
        for source, target in terms:
            for factor, j, state, _ in embedding._actions(source):
                if state in embedding._reference_indices:
                    continue  # Q removes the entire retained subspace.
                energy = self._energy(*state)
                quotient = {}
                for (_, k), coefficient in target[j, :].todok().items():
                    denominator = sympy.simplify(self.reference_energies[k] - energy)
                    if denominator == 0:
                        raise ZeroDivisionError(
                            "A virtual channel is degenerate with the retained space"
                        )
                    quotient[j, k] = coefficient / denominator
                yield factor, sympy.ImmutableSparseMatrix(*target.shape, quotient)


def block_diagonalize(
    hamiltonian: BlockSeries,
    embedding: Embedding,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """Run the standard recurrence over a structured embedding."""
    problem_type = (
        _MatrixProblem if isinstance(embedding, _ReferenceEmbedding) else _SymbolicProblem
    )
    problem = problem_type(hamiltonian, embedding)
    outputs, _ = series_computation(
        {"H": problem.block_series()},
        algorithm=main,
        scope={
            "solve_sylvester": problem.solve_sylvester,
            "use_linear_operator": np.zeros((2, 2), dtype=bool),
            "two_block_optimized": True,
            "commuting_blocks": [True, True],
        },
        operator=multiply_projected,
    )
    return outputs["H_tilde"], outputs["U"], outputs["U†"]
