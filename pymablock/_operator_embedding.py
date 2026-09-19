"""Projected operator algebra and perturbation theory for generated embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import sympy

from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main
from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.operator_embedding import Embedding, _NOFTransition, _source_entries
from pymablock.series import BlockSeries, zero

if TYPE_CHECKING:
    from collections.abc import Iterable

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
        source: Operator,
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
    """Lower a Hamiltonian series to the formal P/Q block algebra."""

    def __init__(self, hamiltonian: BlockSeries, embedding: Embedding):
        if hamiltonian.shape:
            raise ValueError("Structured embeddings require an unseparated Hamiltonian.")
        self.embedding = embedding
        self.hamiltonian = hamiltonian
        zero_order = (0,) * hamiltonian.n_infinite
        h0 = self.embedding._source_form(hamiltonian[zero_order])
        self.source_shape = h0.shape if isinstance(h0, sympy.MatrixBase) else None
        if embedding._references is not None:
            self.source_energies = [sympy.S.Zero] * (h0.rows if self.source_shape else 1)
            for row, column, entry in _source_entries(h0):
                if row != column and not _is_zero(entry):
                    raise ValueError(
                        "Structured embeddings currently require diagonal H0"
                    )
                if row == column:
                    self.source_energies[row] = self._diagonal_energy(entry)
            self.reference_energies = [
                self._energy(c, state) for c, state in embedding._references
            ]
            return
        self.source_energy = self._diagonal_energy(h0)

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

    def _diagonal_energy(self, entry):
        zero_powers = (0,) * len(self.embedding.operators)
        if any(
            powers != zero_powers and coefficient != 0
            for powers, coefficient in entry.terms.items()
        ):
            raise ValueError("Structured embeddings currently require diagonal H0")
        return sympy.expand(entry.terms.get(zero_powers, sympy.S.Zero))

    def _energy(self, component, state):
        return self.source_energies[component].xreplace(
            dict(zip(self.embedding._source_placeholders, state, strict=True))
        )

    def _solve_matrix(self, terms):
        """Resolve virtual transitions from the whole retained reference basis."""
        embedding = self.embedding
        solved = []
        for source_form, target in terms:
            for row, column, entry in _source_entries(source_form):
                for transition in _NOFTransition.from_form(entry):
                    quotient = {}
                    for j, (component, state) in enumerate(embedding._references):
                        if component != column:
                            continue
                        action = transition.apply(state)
                        if (
                            action is None
                            or (row, action.output_state) in embedding._reference_indices
                        ):
                            continue
                        energy = self._energy(row, action.output_state)
                        for k in range(target.cols):
                            if target[j, k] == 0:
                                continue
                            denominator = sympy.simplify(
                                self.reference_energies[k] - energy
                            )
                            if denominator == 0:
                                raise ZeroDivisionError(
                                    "A virtual channel is degenerate with the retained space"
                                )
                            quotient[j, k] = target[j, k] / denominator
                    if quotient:
                        source = transition.form
                        if self.source_shape:
                            source = sympy.ImmutableSparseMatrix(
                                *self.source_shape, {(row, column): source}
                            )
                        solved.append(
                            (
                                source,
                                sympy.ImmutableSparseMatrix(
                                    target.rows, target.cols, quotient
                                ),
                            )
                        )
        return solved

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
        if _is_zero(norm):
            return True
        coefficients = (
            norm.terms.values() if isinstance(norm, NumberOrderedForm) else norm
        )
        return all(sympy.cancel(coefficient) == 0 for coefficient in coefficients)

    def solve_sylvester(self, value, index):
        """Solve the P-Q Sylvester equation transition by transition."""
        if value is zero:
            return zero
        if index[:2] != (0, 1) or not isinstance(value, AdjointOperatorMap):
            raise TypeError("The embedding solver expects the P-Q block")
        if self.embedding._references is not None:
            column = OperatorMap(
                self.embedding, self._solve_matrix(value.column.terms)
            ).or_zero()
            return zero if column is zero else column.adjoint()
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
                sector = NumberOrderedForm(
                    target.operators,
                    {
                        powers: coefficient * mask
                        for powers, coefficient in target.terms.items()
                    },
                    validate=False,
                )
                if not self._zero_denominator_is_projected_out(source, sector):
                    raise ZeroDivisionError(
                        "A virtual channel is degenerate with the retained space"
                    )
            else:
                sectors.append(mask / expression)

        visit(denominator, sympy.S.One, variables)
        inverse = sympy.Add(*sectors) if has_zero else 1 / denominator
        result = NumberOrderedForm(
            target.operators,
            {
                powers: coefficient * inverse
                for powers, coefficient in target.terms.items()
            },
            validate=False,
        )
        return result * self.embedding._target_identity

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
