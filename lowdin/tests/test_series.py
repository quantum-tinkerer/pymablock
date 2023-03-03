# +
import numpy as np

from lowdin.series import BlockOperatorSeries

# +
# First dimension is finite and of size 5, the second two infinite
possible_keys = [
    np.index_exp[1, 2, 3], # ()
    np.index_exp[:, 2, 3], # (5,)
    np.index_exp[1, :, 3],  # Should raise an error
    np.index_exp[:3, [1, 2], 3], # (3, 2)
    np.index_exp[1, 2, 3, 4],  # Should raise an error
    np.index_exp[:, [1, 2], [3, 4]], # (5, 2)
    np.index_exp[:, [1, 2], :3], # (5, 2, 3)
    np.index_exp[0, :5, :3], # (5, 3)
    np.index_exp[6, 3, 3],
]

shapes = [
    AttributeError,
    (5,),
    IndexError,
    (3, 2),
    IndexError,
    (5, 2),
    (5, 2, 3),
    (5, 3),
]

for key, shape in zip(possible_keys, shapes):
    try:
        a = BlockOperatorSeries(lambda x: x, data=None, shape=(5,), n_infinite=2)
        assert shape == a.evaluated[key].shape
    except Exception as e:
        assert type(e) == shape
