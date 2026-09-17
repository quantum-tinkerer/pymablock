"""Projected operator algebra and perturbation theory for occupation selections."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import sympy

from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main
from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.operator_embedding import Embedding, _NOFTransition, _one_term
from pymablock.series import BlockSeries, zero

if TYPE_CHECKING:
    from collections.abc import Iterable

TargetOperator: TypeAlias = NumberOrderedForm | sympy.MatrixBase


def _is_zero(value: NumberOrderedForm | sympy.MatrixBase) -> bool:
    """Test exact structural zero without simplifying coefficients."""
    if isinstance(value, sympy.MatrixBase):
        return not value.todok()
    return not any(coefficient != 0 for coefficient in value.terms.values())


def _adjoint(value: TargetOperator) -> TargetOperator:
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
        terms: Iterable[tuple[NumberOrderedForm, TargetOperator]],
    ):
        """Combine equal source factors and discard exact structural zeros."""
        combined: dict[NumberOrderedForm, TargetOperator] = {}
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
        source: NumberOrderedForm,
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
        return type(self)(
            self.embedding,
            ((source, -target) for source, target in self.terms),
        ).or_zero()

    def __sub__(self, other: object) -> OperatorMap | object:
        """Subtract a map defined by the same embedding."""
        if other is zero:
            return self
        if not isinstance(other, OperatorMap):
            return NotImplemented
        return self + (-other)

    def scale(self, factor: object) -> OperatorMap | object:
        """Multiply every term by a scalar."""
        factor = sympy.sympify(factor)
        return type(self)(
            self.embedding,
            ((source, target * factor) for source, target in self.terms),
        ).or_zero()

    def __mul__(self, factor: object) -> OperatorMap | object:
        """Multiply every term by a scalar."""
        try:
            factor = sympy.sympify(factor)
        except sympy.SympifyError:
            return NotImplemented
        if not factor.is_commutative:
            return NotImplemented
        return self.scale(factor)

    def __rmul__(self, factor: object) -> OperatorMap | object:
        """Multiply every term by a scalar."""
        return self * factor

    def __truediv__(self, divisor: object) -> OperatorMap | object:
        """Divide every term by a scalar."""
        return self.scale(sympy.S.One / divisor)

    def right(self, target: TargetOperator) -> OperatorMap | object:
        """Compose with a retained-space operator on the right."""
        return type(self)(
            self.embedding,
            ((source, coefficient * target) for source, coefficient in self.terms),
        ).or_zero()

    def left(self, source: NumberOrderedForm) -> OperatorMap | object:
        """Apply a source operator while preserving the outer projection.

        This uses ``Q S Q X W A = Q S X W A - Q S W (W† X W) A``.
        """
        terms = []
        for inner_source, target in self.terms:
            terms.append((source * inner_source, target))
            terms.append((source, -self.embedding._pullback(inner_source) * target))
        return type(self)(self.embedding, terms).or_zero()

    def inner(self, other: OperatorMap) -> TargetOperator:
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
        return zero if result is zero else type(self)(result)

    def __neg__(self) -> AdjointOperatorMap | object:
        """Negate the underlying column."""
        result = -self.column
        return zero if result is zero else type(self)(result)

    def __sub__(self, other: object) -> AdjointOperatorMap | object:
        """Subtract an adjoint map."""
        if not isinstance(other, AdjointOperatorMap):
            return NotImplemented
        return self + (-other)

    def __truediv__(self, divisor: object) -> AdjointOperatorMap | object:
        """Divide the underlying column by a scalar."""
        result = self.column / divisor
        return zero if result is zero else type(self)(result)

    def adjoint(self) -> OperatorMap:
        """Return the underlying P→Q column."""
        return self.column


@dataclass(frozen=True)
class ModuleEndomorphism:
    """A lazy exact endomorphism of the formal complement module.

    This is the algebraic analogue of a linear operator: it stores source
    actions, rank-one maps, sums, products, and scalar multiples without
    choosing a basis for the complement.
    """

    operation: str
    operands: tuple

    @classmethod
    def source(
        cls,
        embedding: Embedding,
        source: NumberOrderedForm,
    ) -> ModuleEndomorphism:
        """Represent ``Q source Q``."""
        return cls("source", (embedding, source))

    @classmethod
    def rank_one(
        cls,
        left: OperatorMap,
        right: OperatorMap,
    ) -> ModuleEndomorphism:
        """Represent ``left right†``."""
        left._require_same_embedding(right)
        return cls("rank_one", (left, right))

    def apply(self, column: OperatorMap):
        """Apply the lazy endomorphism to one complement column."""
        operation = self.operation
        if operation == "source":
            _embedding, source = self.operands
            return column.left(source)
        if operation == "rank_one":
            left, right = self.operands
            coefficient = right.inner(column)
            return zero if _is_zero(coefficient) else left.right(coefficient)
        if operation == "sum":
            result = zero
            for term in self.operands:
                result = result + term.apply(column)
            return result
        if operation == "product":
            left, right = self.operands
            result = right.apply(column)
            return zero if result is zero else left.apply(result)
        if operation == "scale":
            factor, block = self.operands
            result = block.apply(column)
            return zero if result is zero else result.scale(factor)
        raise AssertionError(f"Unknown module operation: {operation}")

    def adjoint(self) -> ModuleEndomorphism:
        """Return the lazy adjoint endomorphism."""
        operation = self.operation
        if operation == "source":
            embedding, source = self.operands
            return type(self).source(embedding, source.adjoint())
        if operation == "rank_one":
            left, right = self.operands
            return type(self).rank_one(right, left)
        if operation == "sum":
            return type(self)("sum", tuple(term.adjoint() for term in self.operands))
        if operation == "product":
            left, right = self.operands
            return type(self)("product", (right.adjoint(), left.adjoint()))
        if operation == "scale":
            factor, block = self.operands
            return type(self)("scale", (sympy.conjugate(factor), block.adjoint()))
        raise AssertionError(f"Unknown module operation: {operation}")

    def __add__(self, other):
        """Add exact module endomorphisms."""
        if other is zero:
            return self
        if not isinstance(other, ModuleEndomorphism):
            return NotImplemented
        return type(self)("sum", (self, other))

    def __neg__(self):
        """Negate this endomorphism lazily."""
        return type(self)("scale", (-sympy.S.One, self))

    def __sub__(self, other):
        """Subtract exact module endomorphisms."""
        return self + (-other)

    def __truediv__(self, divisor):
        """Divide this endomorphism by a scalar."""
        return type(self)("scale", (sympy.S.One / divisor, self))


def multiply_projected(left, right):
    """Multiply blocks in the algebra induced by one compression ``W† X W``."""
    if isinstance(left, OperatorMap):
        if isinstance(right, AdjointOperatorMap):
            return ModuleEndomorphism.rank_one(left, right.column)
        if isinstance(right, (NumberOrderedForm, sympy.MatrixBase)):
            return left.right(right)
    if isinstance(left, AdjointOperatorMap):
        if isinstance(right, OperatorMap):
            return left.column.inner(right)
        if isinstance(right, ModuleEndomorphism):
            result = right.adjoint().apply(left.column)
            return zero if result is zero else result.adjoint()
    if isinstance(left, ModuleEndomorphism):
        if isinstance(right, OperatorMap):
            return left.apply(right)
        if isinstance(right, ModuleEndomorphism):
            return ModuleEndomorphism("product", (left, right))
    if isinstance(left, (NumberOrderedForm, sympy.MatrixBase)):
        if isinstance(right, AdjointOperatorMap):
            result = right.column.right(left.adjoint())
            return zero if result is zero else result.adjoint()
        if isinstance(right, (NumberOrderedForm, sympy.MatrixBase)):
            return left * right
    raise TypeError(f"Cannot multiply projected blocks {type(left)} and {type(right)}")


class _EmbeddingProblem:
    """Lower one scalar Hamiltonian series to the formal P/Q block algebra."""

    def __init__(self, hamiltonian: BlockSeries, embedding: Embedding):
        if hamiltonian.shape:
            raise ValueError("Structured embeddings require an unseparated Hamiltonian.")
        self.embedding = embedding
        self.hamiltonian = hamiltonian
        zero_order = (0,) * hamiltonian.n_infinite
        h0 = self.embedding._source_form(hamiltonian[zero_order])
        zero_powers = (0,) * len(self.embedding.operators)
        if any(tuple(map(int, powers)) != zero_powers for powers in h0.terms):
            raise ValueError("Structured embeddings currently require diagonal H0")
        self.source_energy = sympy.simplify(h0.terms.get(zero_powers, sympy.S.Zero))

        # Selecting occupation states of diagonal H0 is automatically invariant.
        self.target_energy = self.source_energy.xreplace(
            dict(
                zip(
                    self.embedding._source_placeholders,
                    embedding.source_occupations,
                    strict=True,
                )
            )
        )
        self.finite_target_energy = (
            None
            if embedding.target_is_nof
            else tuple(
                self.target_energy.xreplace(
                    dict(zip(embedding.coordinate_symbols, state, strict=True))
                )
                for state in embedding.target.states
            )
        )

    @cache
    def _source_energy_at(self, state: tuple[int, ...]) -> sympy.Expr:
        return self.source_energy.xreplace(
            dict(
                zip(
                    self.embedding._source_placeholders,
                    map(sympy.Integer, state),
                    strict=True,
                )
            )
        )

    @cache
    def _target_reciprocal(self, denominator: sympy.Expr) -> NumberOrderedForm:
        return _one_term(
            self.embedding.target.operators,
            (0,) * len(self.embedding.target.operators),
            sympy.S.One / denominator,
        )

    def _channel_denominator(
        self,
        source: _NOFTransition,
        target: _NOFTransition,
    ) -> sympy.Expr:
        after_target = tuple(
            symbol - power
            for symbol, power in zip(
                self.embedding.coordinate_symbols, target.powers, strict=True
            )
        )
        encoded_after_target = tuple(
            occupation.xreplace(
                dict(zip(self.embedding.coordinate_symbols, after_target, strict=True))
            )
            for occupation in self.embedding.source_occupations
        )
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
        target: NumberOrderedForm | sympy.MatrixBase,
    ) -> bool:
        projected = self.embedding._pullback(source.form)
        leakage = (
            self.embedding._pullback(source.form.adjoint() * source.form)
            - projected.adjoint() * projected
        )
        norm = target.adjoint() * leakage * target
        return _is_zero(norm)

    def solve_sylvester(self, value, index):
        """Solve the P-Q Sylvester equation transition by transition."""
        if value is zero:
            return zero
        if index[:2] != (0, 1) or not isinstance(value, AdjointOperatorMap):
            raise TypeError("The embedding solver expects the P-Q block")
        if not self.embedding.target_is_nof:
            return self._solve_finite(value)

        solved_terms = []
        for _source_form, target_form in value.column.terms:
            for source in _NOFTransition.from_form(_source_form):
                for target in _NOFTransition.from_form(target_form):
                    denominator = self._channel_denominator(source, target)
                    target_term = target.form
                    quotient = self._divide_channel(source, target_term, denominator)
                    if not _is_zero(quotient):
                        solved_terms.append((source.form, quotient))
        column = OperatorMap(self.embedding, solved_terms).or_zero()
        return zero if column is zero else column.adjoint()

    def _divide_channel(
        self, source: _NOFTransition, target: NumberOrderedForm, denominator: sympy.Expr
    ) -> NumberOrderedForm:
        """Divide on binary sectors, discarding unsupported resonances first."""
        variables = tuple(
            symbol
            for symbol in self.embedding._target_placeholders
            if symbol in denominator.free_symbols
        )

        sectors = []
        has_zero = False

        def visit(expression, mask, remaining):
            nonlocal has_zero
            expression = sympy.expand(expression)
            remaining = tuple(x for x in remaining if x in expression.free_symbols)
            if remaining:
                symbol, *rest = remaining
                for value in (0, 1):
                    visit(
                        expression.xreplace({symbol: sympy.Integer(value)}),
                        mask * (symbol if value else 1 - symbol),
                        rest,
                    )
                return
            expression = sympy.simplify(expression)
            if expression == 0:
                has_zero = True
                sector = _one_term(
                    self.embedding.target.operators,
                    (0,) * len(self.embedding.target.operators),
                    mask,
                )
                if not self._zero_denominator_is_projected_out(source, target * sector):
                    raise ZeroDivisionError(
                        "A virtual channel is degenerate with the retained space"
                    )
            else:
                sectors.append(mask / expression)

        visit(denominator, sympy.S.One, variables)
        if not has_zero:
            return target * self._target_reciprocal(denominator)
        inverse = _one_term(
            self.embedding.target.operators,
            (0,) * len(self.embedding.target.operators),
            sympy.Add(*sectors),
        )
        return target * inverse

    def _solve_finite(self, value: AdjointOperatorMap):
        solved_terms = []
        for _source_form, target in value.column.terms:
            for source in _NOFTransition.from_form(_source_form):
                for (row, column), target_coefficient in target.todok().items():
                    action = self.embedding._finite_transition_action(source, row)
                    # A weighted transition has one output occupation. If it
                    # returns to P, the outer Q removes it exactly, regardless
                    # of how complicated its symbolic coefficient is.
                    if (
                        action is None
                        or action.output_state in self.embedding._finite_source_index
                    ):
                        continue
                    denominator = sympy.factor(
                        self.finite_target_energy[column]
                        - self._source_energy_at(action.output_state)
                    )
                    target_term = sympy.ImmutableSparseMatrix(
                        self.embedding.target.dimension,
                        self.embedding.target.dimension,
                        {(row, column): target_coefficient},
                    )
                    if denominator == 0:
                        if self._zero_denominator_is_projected_out(source, target_term):
                            continue
                        raise ZeroDivisionError(
                            "A virtual channel is degenerate with the retained space"
                        )
                    solved_terms.append(
                        (
                            source.form,
                            target_term / denominator,
                        )
                    )
        column = OperatorMap(self.embedding, solved_terms).or_zero()
        return zero if column is zero else column.adjoint()

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
                return zero if column_map is zero else column_map.adjoint()
            return ModuleEndomorphism.source(embedding, source)

        return BlockSeries(
            eval=evaluate,
            shape=(2, 2),
            n_infinite=self.hamiltonian.n_infinite,
            dimension_names=self.hamiltonian.dimension_names,
            name="H",
        )


def block_diagonalize(
    hamiltonian: BlockSeries,
    embedding: Embedding,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """Run the standard recurrence over a structured embedding."""
    problem = _EmbeddingProblem(hamiltonian, embedding)
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
