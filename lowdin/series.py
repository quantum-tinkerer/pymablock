# %%
from itertools import product
from functools import reduce
from operator import matmul

import numpy as np
from sympy.physics.quantum import Dagger
import tinyarray as ta

# %%
class _Evaluated:
    def __init__(self, original):
        self.original = original

    def __getitem__(self, item):
        """ Evaluate the series at the given index.

        Parameters
        ----------
        item : tuple of int or slice

        Returns
        -------
        The item at the given index.
        """
        self.check_finite(item[-self.original.n_infinite:])
        self.check_number_perturbations(item)

        trial_shape = self.original.shape + tuple(
            [
                order.stop if isinstance(order, slice) else np.max(order) + 1
                for order in item[-self.original.n_infinite:]
            ]
        )
        trial = np.zeros(trial_shape, dtype=object)
        trial[item] = 1

        data = self.original.data
        for entry in zip(*np.where(trial)):
            if entry not in data:
                data[entry] = self.original.eval(entry)
            trial[entry] = data[entry]
        return trial[item]

    def check_finite(self, item):
        """ Check that the indices of the infinite dimension are finite and positive."""
        for order in item:
            if isinstance(order, slice):
                if order.stop is None:
                    raise IndexError("Cannot evaluate infinite series")
                elif isinstance(order.start, int) and order.start < 0:
                    raise IndexError("Cannot evaluate negative order")

    def check_number_perturbations(self, item):
        """ Check that the number of indices is correct."""
        if not isinstance(item, tuple):
            if len(item) != len(self.original.shape) + self.original.n_infinite:
                raise IndexError("Wrong number of indices")


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
