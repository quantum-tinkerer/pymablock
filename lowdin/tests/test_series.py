from typing import Any
from operator import mul

import numpy as np
import pytest

from lowdin.series import BlockSeries, cauchy_dot_product


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
        assert type(e) == shape


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
    with pytest.raises(RuntimeError, match="Infinite recursion loop detected"):
        recursive[0]


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
    np.testing.assert_allclose(
        result[:, :, 3].astype(float), test_value @ test_value
    )
