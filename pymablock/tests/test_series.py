from operator import mul
from typing import Any

import numpy as np
import pytest
import sympy
from scipy import sparse

from pymablock.series import AlgebraElement, BlockSeries, cauchy_dot_product, zero


@pytest.fixture(
    scope="module",
    params=[
        [np.index_exp[1, 2, 3], AttributeError],
        [np.index_exp[:, 2, 3], (5,)],
        [np.index_exp[1, :, 3], IndexError],
        [np.index_exp[:3, [1, 2], 3], (3, 2)],
        [np.index_exp[1, 2, 3, 4], IndexError],
        [np.index_exp[:, [1, 2], [3, 4]], (5, 2)],
        [np.index_exp[:, [1, 2], :3], (5, 2, 3)],
        [np.index_exp[0, :5, :3], (5, 3)],
        [np.index_exp[6, 3, 3], IndexError],
    ],
)
def possible_keys_and_errors(request):
    return request.param


@pytest.fixture(autouse=True, scope="session")
def clear_log():
    yield
    AlgebraElement.log = []


def test_indexing(possible_keys_and_errors: tuple[tuple[tuple[int, ...]], Any]) -> None:
    """
    Test that indexing works like in numpy arrays.

    Parameters
    ----------
    possible_keys_and_errors: tuple of (key, shape)
    """
    key, shape = possible_keys_and_errors
    try:
        a = BlockSeries(lambda *x: x, data=None, shape=(5,), n_infinite=2)
        assert shape == a[key].shape
    except Exception as e:
        assert isinstance(e, shape)


def test_infinite_views():
    test = BlockSeries(lambda *x: x, data=None, shape=(3, 3, 3), n_infinite=2)
    np.testing.assert_equal(
        test[:, :, 0][:, :, [2, 1, 0], :3], test[:, :, 0, [2, 1, 0], :3]
    )


def test_slicing():
    """
    Regression test for #154.
    """
    H = BlockSeries(
        data={
            (0, 0, 0): sparse.eye(2, format="csr"),
        },
        shape=(1, 1),
        n_infinite=1,
    )
    H[0, 0, :3]
    H[:1, :1][0, 0, :3]
    H[0, 0][:3]
    assert H[0, 0, 3] is zero
    assert H[:1, :1][0, 0, 3] is zero


def test_fibonacci_series() -> None:
    """Test that we can implement the Fibonacci series."""
    F = BlockSeries(
        eval=lambda x: F[x - 2] + F[x - 1],
        data={(0,): 0, (1,): 1},
        shape=(),
        n_infinite=1,
    )

    assert F[6] == 8


def test_cleanup():
    """Test that BlockSeries data is not corrupted by an exception."""

    def raising_eval(i):
        if i:
            raise ValueError
        return i

    problematic = BlockSeries(raising_eval, shape=(), n_infinite=1)
    problematic[0]
    with pytest.raises(ValueError):
        problematic[1]
    # Check that a repeated call raises the same error
    with pytest.raises(ValueError):
        problematic[1]


def test_recursion_detection():
    """Test that BlockSeries detects recursion."""
    recursive = BlockSeries(lambda i: recursive[i], shape=(), n_infinite=1)
    with pytest.raises(RuntimeError) as excinfo:
        recursive[0]
    assert "Infinite recursion loop detected in" in str(excinfo.value.__cause__)
    assert recursive.name in str(excinfo.value.__cause__)


def test_cauchy_dot_product():
    """Test that cauchy dot product reduces to a dot"""
    n = 5
    test_value = np.random.randn(n, n)
    a = BlockSeries(
        data={(i, j, 2): x for (i, j), x in np.ndenumerate(test_value)},
        shape=(n, n),
        n_infinite=1,
    )
    b = BlockSeries(
        data={(i, j, 1): x for (i, j), x in np.ndenumerate(test_value)},
        shape=(n, n),
        n_infinite=1,
    )
    result = cauchy_dot_product(a, b, operator=mul)
    np.testing.assert_allclose(result[:, :, 3].astype(float), test_value @ test_value)


def test_printing():
    """Test that BlockSeries prints nicely."""
    a = BlockSeries(
        lambda *x: x,
        data=None,
        shape=(5, 5),
        n_infinite=2,
        dimension_names=("i", sympy.Symbol("j")),
        name="test",
    )
    assert str(a) == "test_(5 × 5 × ∞_(i) × ∞_(j))"  # noqa: RUF001


def test_algebra_element_algebra():
    a = AlgebraElement("a")
    b = AlgebraElement("b")
    c = AlgebraElement("c")

    AlgebraElement.log = []

    t1 = 2 * a * b / 2
    assert AlgebraElement.call_counts()["__mul__"] == 1
    assert AlgebraElement.call_counts()["__rmul__"] == 1
    assert AlgebraElement.call_counts()["__truediv__"] == 1
    AlgebraElement.log = []

    t2 = a * (-b) * c.adjoint()
    assert AlgebraElement.call_counts()["__mul__"] == 2
    AlgebraElement.log = []

    assert t2.to_sympy() == a.to_sympy() * (-b.to_sympy()) * c.to_sympy().adjoint()
    assert not any(symbol.is_commutative for symbol in t2.to_sympy().free_symbols)

    (t1 - c) * (t2 + a)
    assert AlgebraElement.call_counts()["__mul__"] == 1
    assert AlgebraElement.call_counts()["__sub__"] == 1
    # subtraction = addition of negative
    assert AlgebraElement.call_counts()["__add__"] == 2
