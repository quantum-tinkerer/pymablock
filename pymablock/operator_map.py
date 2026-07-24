"""Projection-aware maps between a retained space and its complement.

For an embedding ``W`` and ``Q = 1 - W W†``, one stored term ``(X, A)``
represents ``Q X W A``.  ``X`` is a source-space
:class:`~pymablock.number_ordered_form.NumberOrderedForm`; ``A`` is an
operator on the retained space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

import sympy

from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.series import zero

if TYPE_CHECKING:
    from collections.abc import Iterable

TargetOperator: TypeAlias = NumberOrderedForm | sympy.MatrixBase

__all__ = ["OperatorMap"]


def _is_zero(value: NumberOrderedForm | sympy.MatrixBase) -> bool:
    """Test exact structural zero without simplifying coefficients."""
    if isinstance(value, sympy.MatrixBase):
        return not value.todok()
    return not bool(value)


def _adjoint(value: TargetOperator) -> TargetOperator:
    """Return an algebra-native adjoint."""
    return value.adjoint()


class OperatorEmbeddingBackend(Protocol):
    """Internal contract required by :class:`OperatorMap`."""

    target_zero: TargetOperator
    target_identity: TargetOperator
    target_space: object
    complement_space: object

    def pullback(self, source: NumberOrderedForm) -> TargetOperator:
        """Return ``W† source W``."""


class OperatorMap:
    """A canonical sum of formal complement columns ``Q X W A``."""

    __slots__ = ("embedding", "terms")

    def __init__(
        self,
        embedding: OperatorEmbeddingBackend,
        terms: Iterable[tuple[NumberOrderedForm, TargetOperator]],
    ):
        """Combine equal source factors and discard exact structural zeros."""
        combined: dict[NumberOrderedForm, TargetOperator] = {}
        for source, target in terms:
            if not source or _is_zero(target):
                continue
            combined[source] = combined.get(source, embedding.target_zero) + target

        self.embedding = embedding
        self.terms = tuple(
            (source, target)
            for source, target in combined.items()
            if not _is_zero(target)
        )

    @property
    def left_space(self) -> object:
        """The source-space complement in which the map takes values."""
        return self.embedding.complement_space

    @property
    def right_space(self) -> object:
        """The retained space on which the map acts."""
        return self.embedding.target_space

    @classmethod
    def from_source(
        cls,
        embedding: OperatorEmbeddingBackend,
        source: NumberOrderedForm,
    ) -> OperatorMap | object:
        """Construct ``Q X W`` from one source-space operator ``X``."""
        return cls(
            embedding,
            ((source, embedding.target_identity),),
        ).or_zero()

    @classmethod
    def zero(cls, embedding: OperatorEmbeddingBackend) -> OperatorMap:
        """Construct the zero map without returning the series zero sentinel."""
        return cls(embedding, ())

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
            terms.append((source, -self.embedding.pullback(inner_source) * target))
        return type(self)(self.embedding, terms).or_zero()

    def inner(self, other: OperatorMap) -> TargetOperator:
        """Return ``self† other`` in the retained operator algebra."""
        self._require_same_embedding(other)

        result = self.embedding.target_zero
        for left_source, left_target in self.terms:
            left_adjoint = left_source.adjoint()
            projected_left = self.embedding.pullback(left_adjoint)
            for right_source, right_target in other.terms:
                kernel = self.embedding.pullback(left_adjoint * right_source)
                kernel -= projected_left * self.embedding.pullback(right_source)
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
        embedding: OperatorEmbeddingBackend,
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
