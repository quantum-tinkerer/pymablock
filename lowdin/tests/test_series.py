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


def test_indices(possible_keys_and_errors):
    key, shape = possible_keys_and_errors
    try:
        a = BlockSeries(lambda x: x, data=None, shape=(5,), n_infinite=2)
        assert shape == a.evaluated[key].shape
    except Exception as e:
        assert type(e) == shape
