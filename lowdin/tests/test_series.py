from typing import Any

import numpy as np
import pytest

from lowdin.series import BlockSeries


@pytest.fixture(
    scope="module",
    params=[
        [np.index_exp[1, 2, 3], AttributeError],  # ()
        [np.index_exp[:, 2, 3], (5,)],  # (5,)
        [np.index_exp[1, :, 3], IndexError],  # Should raise an error
        [np.index_exp[:3, [1, 2], 3], (3, 2)],  # (3, 2)
        [np.index_exp[1, 2, 3, 4], IndexError],  # Should raise an error
        [np.index_exp[:, [1, 2], [3, 4]], (5, 2)],  # (5, 2)
        [np.index_exp[:, [1, 2], :3], (5, 2, 3)],  # (5, 2, 3)
        [np.index_exp[0, :5, :3], (5, 3)],  # (5, 3)
        [np.index_exp[6, 3, 3], IndexError],  # Should raise an error
    ],
)


def possible_keys_and_errors(request):
    return request.param


def test_indexing(possible_keys_and_errors: tuple[tuple[tuple[int, ...]], Any]) -> None:
    """
    Test that indexing works like in numpy arrays.

    possible_keys_and_errors: tuple of (key, shape)
    """
    key, shape = possible_keys_and_errors
    try:
        a = BlockSeries(lambda *x: x, data=None, shape=(5,), n_infinite=2)
        assert shape == a.evaluated[key].shape
    except Exception as e:
        assert type(e) == shape


def test_fibonacci_series() -> None:
    F = BlockSeries(
        eval=lambda x: F.evaluated[x-2] + F.evaluated[x-1],
        data={(0,): 0, (1,): 1},
        shape=(), n_infinite=1,
    )

    np.testing.assert_allclose(8, F.evaluated[6])
