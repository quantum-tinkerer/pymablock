"""Tests for projection-aware algebraic operator maps."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import sympy

from pymablock.operator_map import ModuleEndomorphism, OperatorMap
from pymablock.series import zero


@dataclass(frozen=True)
class MatrixEmbedding:
    """Small exact embedding used only to lower formal maps in tests."""

    name: str = "W"

    @property
    def bridge(self):
        return sympy.ImmutableMatrix([[1, 0], [0, 1], [0, 0]])

    @property
    def projector(self):
        bridge = self.bridge
        return sympy.eye(3) - bridge * bridge.adjoint()

    @property
    def target_zero(self):
        return sympy.ImmutableMatrix(sympy.zeros(2))

    @property
    def target_identity(self):
        return sympy.ImmutableMatrix(sympy.eye(2))

    @property
    def complement_space(self):
        return f"Q({self.name})"

    @property
    def target_space(self):
        return f"P({self.name})"

    def pullback(self, source):
        return self.bridge.adjoint() * source * self.bridge


def _lower(operator_map):
    embedding = operator_map.embedding
    result = sympy.zeros(3, 2)
    for source, target in operator_map.terms:
        result += embedding.projector * source * embedding.bridge * target
    return sympy.ImmutableMatrix(result)


def test_storage_is_canonical_and_defines_spaces() -> None:
    embedding = MatrixEmbedding()
    source = sympy.ImmutableMatrix([[0, 0, 0], [0, 0, 0], [1, 2, 0]])
    first = sympy.ImmutableMatrix([[1, 2], [0, 0]])
    second = sympy.ImmutableMatrix([[0, -2], [3, 0]])

    operator_map = OperatorMap(
        embedding,
        (
            (source, first),
            (source, second),
            (source, sympy.zeros(2)),
        ),
    )

    assert operator_map.terms == ((source, first + second),)
    assert operator_map.left_space == "Q(W)"
    assert operator_map.right_space == "P(W)"
    assert not OperatorMap.zero(embedding)
    assert operator_map + (-operator_map) is zero


def test_left_and_right_actions_match_explicit_projection() -> None:
    embedding = MatrixEmbedding()
    source = sympy.ImmutableMatrix([[0, 0, 0], [0, 0, 0], [1, 2, 0]])
    left = sympy.ImmutableMatrix([[1, 0, 1], [0, 2, 0], [3, 0, 4]])
    right = sympy.ImmutableMatrix([[1, 2], [3, 4]])
    operator_map = OperatorMap.from_source(embedding, source)

    assert _lower(operator_map.left(left)) == (
        embedding.projector * left * _lower(operator_map)
    )
    assert _lower(operator_map.right(right)) == _lower(operator_map) * right


def test_inner_product_matches_explicit_maps() -> None:
    embedding = MatrixEmbedding()
    x = sympy.ImmutableMatrix([[0, 0, 1], [0, 0, 2], [1, 3, 0]])
    y = sympy.ImmutableMatrix([[1, 0, 0], [0, 1, 0], [4, 5, 0]])
    a = sympy.ImmutableMatrix([[1, 2], [0, 1]])
    b = sympy.ImmutableMatrix([[2, 0], [3, 1]])
    left = OperatorMap(embedding, ((x, a),))
    right = OperatorMap(embedding, ((y, b),))

    assert left.inner(right) == _lower(left).adjoint() * _lower(right)


def test_arithmetic_requires_one_embedding() -> None:
    first_embedding = MatrixEmbedding("first")
    second_embedding = MatrixEmbedding("second")
    source = sympy.ImmutableMatrix([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    first = OperatorMap.from_source(first_embedding, source)
    second = OperatorMap.from_source(second_embedding, source)

    assert _lower(3 * first / 2) == sympy.Rational(3, 2) * _lower(first)
    with pytest.raises(ValueError, match="same embedding"):
        _ = first + second
    with pytest.raises(ValueError, match="same embedding"):
        first.inner(second)


def test_lazy_complement_endomorphisms_match_explicit_projection() -> None:
    embedding = MatrixEmbedding()
    source = sympy.ImmutableMatrix([[0, 0, 1], [0, 0, 2], [1, 3, 0]])
    left = sympy.ImmutableMatrix([[1, 0, 1], [0, 2, 0], [3, 0, 4]])
    column = OperatorMap.from_source(embedding, source)

    source_action = ModuleEndomorphism.source(embedding, left)
    assert _lower(source_action.apply(column)) == (
        embedding.projector * left * embedding.projector * _lower(column)
    )

    rank_one = ModuleEndomorphism.rank_one(column, column)
    assert _lower(rank_one.apply(column)) == (
        _lower(column) * _lower(column).adjoint() * _lower(column)
    )
