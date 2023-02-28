import numpy as np

class _Evaluated:
    def __init__(self, original):
        self.original = original

    def __getitem__(self, item):
        self.check_finite(item[-self.original.n_infinite:])
        self.check_bounds(item[:-self.original.n_infinite])

        data = self.original.data
        trial_shape = self.original.shape + tuple(
            [
                order.stop if isinstance(order, slice) else np.max(order) + 1
                for order in item[-self.original.n_infinite:]
            ]
        )
        trial = np.zeros(trial_shape, dtype=object)
        trial[item] = 1
        for entry in zip(*np.where(trial)):
            if entry not in data:
                data[entry] = self.original.eval(entry)
            trial[entry] = data[entry]
        return trial[item]

    def check_finite(self, item):
        for order in item:
            if isinstance(order, slice):
                if order.stop is None:
                    raise IndexError("Cannot evaluate infinite series")
                elif isinstance(order.start, int) and order.start < 0:
                    raise IndexError("Cannot evaluate negative order")

    def check_bounds(self, item):
        if not isinstance(item, tuple):
            if len(item) != len(self.original.shape) + self.original.n_infinite:
                raise IndexError("Wrong number of indices")
        for i, order in enumerate(item):
            if isinstance(order, int):
                if order > self.original.shape[i]:
                    raise IndexError("Index larger than shape")
            elif isinstance(order.stop, int) and order.stop > self.original.shape[i]:
                    raise IndexError("Index larger than shape")


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
    print(key)
    try:
        a = BlockOperatorSeries(lambda x: x, shape=(5,), n_infinite=2)
        assert shape == a.evaluated[key].shape
    except Exception as e:
        assert type(e) == shape