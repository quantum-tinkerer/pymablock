import numpy as np

class _Evaluated:
    def __init__(self, original):
        self.original = original

    def __getitem__(self, item):
        self.assert_valid_infinite_dimension(item[-self.original.n_infinite:])
        indices = self.array_indices(item)
        data = self.original.data

        # If the indices are a tuple, then we are evaluating a single item
        if isinstance(indices, tuple):
            if indices not in data:
                data[indices] = self.original.eval(indices)
            return data[indices]
        
        # Gather the data into correct shape
        output = np.zeros_like(indices, dtype=object)
        for pos, idx in np.ndenumerate(indices):
            if idx not in data:
                data[idx] = self.original.eval(idx)
            output[pos] = data[idx]
        return output
    
    def assert_valid_infinite_dimension(self, item):
        """Assert that the index in the infinite dimensions are positive and finite"""
        for order in item:
            if isinstance(order, slice):
                if order.stop is None:
                    raise ValueError("Cannot evaluate infinite series")
                elif isinstance(order.start, int) and order.start < 0:
                    raise ValueError("Cannot evaluate negative order")

    def array_indices(self, item):
        """Return the indices of the array that would be used to store the data"""
        trial_shape = self.original.shape + tuple(
            [
                order.stop if isinstance(order, slice) else np.max(order) + 1
                for order in item[-self.original.n_infinite:]
            ]
        )
        trial = np.zeros(trial_shape, dtype=object)
        trial[item] = 1
        for entry in zip(*np.where(trial)):
            trial[entry] = entry  # Mimic the computation
        return trial[item]
    
class BlockOperatorSeries:
    def __init__(self, eval, shape=(), n_infinite=1):
        """An infinite series that caches its items.

        Parameters
        ----------
        eval : callable
            A function that takes an index and returns the corresponding item.
        shape : tuple of int
            The shape of the finite dimensions.
        n_infinite : int
            The number of infinite dimensions.
        """
        self.eval = eval
        self.evaluated = _Evaluated(self)
        self.data = {}
        self.shape = shape
        self.n_infinite = n_infinite


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
]

shapes = [
    (),
    (5,),
    None,
    (3, 2),
    None,
    (5, 2),
    (5, 2, 3),
    (5, 3),
]

for key, expected_shape in zip(possible_keys, shapes):
    try:
        a = BlockOperatorSeries(lambda x: x, shape=(5,), n_infinite=2)
        assert expected_shape==a.evaluated[key].shape
    except (ValueError, AttributeError, IndexError):
        continue