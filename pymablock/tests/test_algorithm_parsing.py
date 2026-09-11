"""Tests for compiling custom series algorithms."""

import numpy as np
import pytest

from pymablock.algorithm_parsing import series_computation
from pymablock.series import BlockSeries, zero


@pytest.mark.parametrize("evaluated", [False, True])
def test_nested_algorithm_callbacks_receive_one_index(evaluated):
    h = BlockSeries(
        eval=lambda i, j, k: np.array([[1.0 + i + 2 * j + 3 * k]]),
        shape=(2, 2),
        n_infinite=1,
    )
    calls = []

    def inner(value, index):
        calls.append(("inner", index))
        assert isinstance(value, np.ndarray if evaluated else BlockSeries)
        return value if evaluated else value[index]

    def outer(value, index):
        calls.append(("outer", index))
        assert isinstance(value, np.ndarray)
        return 2 * value

    def offdiag(value, index):
        assert index[0] == index[1]
        return value

    series, _ = series_computation(
        {"H": h},
        _nested_expression_algorithm if evaluated else _nested_whole_series_algorithm,
        scope={"inner": inner, "outer": outer, "offdiag": offdiag},
    )
    for i, j in np.ndindex(2, 2):
        index = (i, j, 1)
        expected = (-2 if evaluated else 2) * h[index]
        np.testing.assert_allclose(series["out"][index], expected)
        assert calls.count(("inner", index)) == 1
        assert calls.count(("outer", index)) == 1


def _nested_whole_series_algorithm():
    with "out":
        start = 0  # noqa: F841
        if offdiagonal:  # noqa: F821
            outer(inner("H"))  # noqa: F821
    return "out"


def _nested_expression_algorithm():
    with "out":
        start = 0  # noqa: F841
        if offdiagonal:  # noqa: F821
            outer(inner(-"H"))  # noqa: F821
    return "out"


@pytest.mark.parametrize("sentinel", [False, True])
@pytest.mark.parametrize("kind", ["chained", "parenthesized", "denominator"])
def test_nested_algorithm_division_preserves_zero(kind, sentinel):
    source = zero if sentinel else np.array([[1.0, 2.0], [3.0, 4.0]])
    h = BlockSeries(eval=lambda *_index: source, shape=(1, 1), n_infinite=1)
    algorithm, factor = {
        "chained": (_division_chained_algorithm, 1 / 6),
        "parenthesized": (_division_parenthesized_algorithm, -1 / 6),
        "denominator": (_division_denominator_algorithm, 1.5),
    }[kind]
    result, _ = series_computation({"H": h}, algorithm)
    actual = result["out"][0, 0, 1]
    if sentinel:
        assert actual is zero
    else:
        np.testing.assert_allclose(actual, source * factor)


def _division_chained_algorithm():
    with "out":
        start = 0  # noqa: F841
        "H" / 2 / 3
    return "out"


def _division_parenthesized_algorithm():
    with "out":
        start = 0  # noqa: F841
        -("H" / 2) / 3
    return "out"


def _division_denominator_algorithm():
    with "out":
        start = 0  # noqa: F841
        "H" / (2 / 3)
    return "out"
