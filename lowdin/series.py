# %%
from itertools import product
from functools import reduce
from operator import matmul

import numpy as np
import numpy.ma as ma
from sympy.physics.quantum import Dagger
import tinyarray as ta

# %%
class Zero:
    """
    A class that behaves like zero in all operations.
    This is used to avoid having to check for zero terms in the sum.
    """

    def __mul__(self, other=None):
        return self

    def __add__(self, other):
        return other
    
    def adjoint(self):
        return self

    __neg__ = __truediv__ = __rmul__ = __mul__


_zero = Zero()

def _zero_sum(terms):
    """
    Sum that returns a singleton _zero if empty and omits _zero terms

    terms : iterable of terms to sum.

    Returns:
    Sum of terms, or _zero if terms is empty.
    """
    return sum((term for term in terms if not isinstance(term, Zero)), start=_zero)
# %%
class _Evaluated:
    def __init__(self, original):
        self.original = original

    def __getitem__(self, item):
        """Evaluate the series at the given index.

        Parameters
        ----------
        item : tuple of int or slice

        Returns
        -------
        The item at the given index.
        """
        self.check_finite(item[-self.original.n_infinite :])
        self.check_number_perturbations(item)

        trial_shape = self.original.shape + tuple(
            [
                order.stop if isinstance(order, slice) else np.max(order) + 1
                for order in item[-self.original.n_infinite :]
            ]
        )
        trial = np.zeros(trial_shape, dtype=object)
        trial[item] = 1

        data = self.original.data
        for entry in zip(*np.where(trial)):
            if entry not in data:
                data[entry] = _zero  # avoid recursion
                data[entry] = self.original.eval(entry)
            trial[entry] = data[entry]
        return trial[item]

    def check_finite(self, item):
        """Check that the indices of the infinite dimension are finite and positive."""
        for order in item:
            if isinstance(order, slice):
                if order.stop is None:
                    raise IndexError("Cannot evaluate infinite series")
                elif isinstance(order.start, int) and order.start < 0:
                    raise IndexError("Cannot evaluate negative order")

    def check_number_perturbations(self, item):
        """Check that the number of indices is correct."""
        if len(item) != len(self.original.shape) + self.original.n_infinite:
            raise IndexError("Wrong number of indices")


class BlockOperatorSeries:
    def __init__(self, eval=None, data=None, shape=(), n_infinite=1):
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
        self.eval = (lambda _:_zero) if eval is None else eval
        self.evaluated = _Evaluated(self)
        self.data = data or {}
        self.shape = shape
        self.n_infinite = n_infinite


def cauchy_dot_product(*series, op=None, hermitian=False):
    """
    Product of series with no finite dimensions.

    series : (BlockOperatorSeries) series to be multiplied.
    op : (optional) callable for multiplying terms.
    hermitian : (optional) bool for whether to compute hermitian conjugate.

    Returns:
    (BlockOperatorSeries) Product of series.
    """
    if op is None:
        op = matmul

    starts, ends = zip(*(factor.shape for factor in series))
    start, *rest_starts = starts
    *rest_ends, end = ends
    if rest_starts != rest_ends:
        raise ValueError("Factors must have finite dimensions compatible with dot product.")

    if len(set(factor.n_infinite for factor in series)) > 1:
        raise ValueError("Factors must have equal number of infinite dimensions.")
    def eval(index):
        return product_by_order(index, *series, op=op, hermitian=hermitian)
    return BlockOperatorSeries(eval=eval, data=None, shape=(start, end), n_infinite=series[0].n_infinite)


# %%
def product_by_order(index, *series, op=None, hermitian=False):
    """
    Compute sum of all product of terms of wanted order.

    order : int or tinyarray containing the order of the product.
    series : (BlockOperatorSeries) series to be multiplied.
    op : (optional) callable for multiplying terms.
    hermitian : (optional) bool for whether to compute hermitian conjugate.

    Returns:
    Sum of all products of terms of wanted order.
    """
    if op is None:
        op = matmul
    hermitian = hermitian and index[0] == index[1]

    n_infinite = series[0].n_infinite
    order = index[-n_infinite:]
    lower_orders = tuple(slice(None, dim+1) for dim in order)
    all_blocks = (slice(None), slice(None))
    
    #TODO: start and end only need 1 index
    data = [factor.evaluated[all_blocks + lower_orders] for factor in series]
    def mask(x):
        mask = np.zeros_like(x)
        for i, val in np.ndenumerate(x):
            mask[i] = isinstance(val, Zero)
        return mask
    data = [ma.masked_array(factor, mask=mask(factor)) for factor in data]
    contributing_products = []
    for combination in product(*(ma.ndenumerate(factor, ) for factor in data)):
        combination = list(combination)
        matrix_indices = tuple(key[:-n_infinite] for key, _ in combination)
        starts, ends = zip(*(indices for indices in matrix_indices))
        start, *rest_starts = starts
        *rest_ends, end = ends
        if rest_starts != rest_ends:
            continue
        if (start, end) != index[:2]:
            continue
        key = tuple(ta.array(key[-n_infinite:]) for key, _ in combination)
        if sum(key) != order:
            continue
        values = [value for _, value in combination if value is not None]
        if any(isinstance(value, Zero) for value in values):
            continue
        if hermitian and key > tuple(reversed(key)):
            # exclude half of the reversed partners to prevent double counting
            continue
        temp = reduce(op, values)
        if hermitian and key == tuple(reversed(key)):
            temp /= 2
        contributing_products.append(temp)
    result = _zero_sum(contributing_products)
    if hermitian and not isinstance(result, Zero):
        result += Dagger(result)
    return result
