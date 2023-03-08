# %%
from itertools import product
from functools import reduce
from operator import matmul

import numpy as np
import numpy.ma as ma
from sympy.physics.quantum import Dagger
import tinyarray as ta

PENDING = object()

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

    adjoint = conjugate = __neg__ = __truediv__ = __rmul__ = __mul__

    def __eq__(self, other):
        return isinstance(other, Zero)


_zero = Zero()


@np.vectorize
def _mask(entry):
    return isinstance(entry, Zero)


def _zero_sum(terms):
    """
    Sum that returns a singleton _zero if empty and omits _zero terms

    terms : iterable of terms to sum.

    Returns:
    Sum of terms, or _zero if terms is empty.
    """
    return sum((term for term in terms if _zero != term), start=_zero)


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
        self.check_finite(item[-self.original.n_infinite:])
        self.check_number_perturbations(item)

        trial_shape = self.original.shape + tuple(
            [
                order.stop if isinstance(order, slice) else np.max(order, initial=0) + 1
                for order in item[-self.original.n_infinite:]
            ]
        )
        trial = np.zeros(trial_shape, dtype=object)
        one_entry = np.isscalar(trial[item])
        trial[item] = 1

        data = self.original.data
        for entry in zip(*np.where(trial)):
            if entry not in data:
                data[entry] = PENDING
                if (result := self.original.eval(entry)) is PENDING:
                    raise RuntimeError("Recursion detected")
                data[entry] = result
            trial[entry] = data[entry]

        result = trial[item]
        if not one_entry:
            return ma.masked_where(_mask(result), result)
        return result # return one item

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
            raise IndexError(
                "Wrong number of indices",
                (len(item), len(self.original.shape), self.original.n_infinite),
            )


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
        self.eval = (lambda _: _zero) if eval is None else eval
        self.evaluated = _Evaluated(self)
        self.data = data or {}
        self.shape = shape
        self.n_infinite = n_infinite


def cauchy_dot_product(*series, op=None, hermitian=False, recursive=False):
    """
    Product of series with no finite dimensions.

    series : (BlockOperatorSeries) series to be multiplied.
    op : (optional) callable for multiplying terms.
    hermitian : (optional) bool for whether to compute hermitian conjugate.

    Returns:
    (BlockOperatorSeries) Product of series.
    """
    if len(series) < 2:
        raise ValueError("Must have at least two series to multiply.")
    if op is None:
        op = matmul

    starts, ends = zip(*(factor.shape for factor in series))
    start, *rest_starts = starts
    *rest_ends, end = ends
    if rest_starts != rest_ends:
        raise ValueError(
            "Factors must have finite dimensions compatible with dot product."
        )

    if len(set(factor.n_infinite for factor in series)) > 1:
        raise ValueError("Factors must have equal number of infinite dimensions.")

    def eval(index):
        return product_by_order(index, *series, op=op, hermitian=hermitian, recursive=recursive)

    return BlockOperatorSeries(
        eval=eval, data=None, shape=(start, end), n_infinite=series[0].n_infinite
    )


# %%
def product_by_order(index, *series, op=None, hermitian=False, recursive=False):
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
    start, end, *order = index
    hermitian = hermitian and start == end

    n_infinite = series[0].n_infinite

    def generate_orders(order, start=None, end=None):
        mask = (slice(None), slice(None)) + (-1,) * len(order)
        trial = ma.ones((2, 2) + tuple([dim + 1 for dim in order]), dtype=object)
        if start is not None:
            if recursive:
                trial[mask] = ma.masked
            trial[int(not start)] = ma.masked
        elif end is not None:
            if recursive:
                trial[mask] = ma.masked
            trial[:, int(not end)] = ma.masked
        return trial

    data = (
        [generate_orders(order, start=start)]
        + [generate_orders(order)] * (len(series) - 2) # Actually wrong
        + [generate_orders(order, end=end)]
    )

    for item, factor in zip(data, series):
        item[ma.where(item)] = factor.evaluated[ma.where(item)]
  
    contributing_products = []
    for combination in product(*(ma.ndenumerate(factor) for factor in data)):
        combination = list(combination)
        starts, ends = zip(*(key[:-n_infinite] for key, _ in combination))
        if starts[1:] != ends[:-1]:
            continue
        key = tuple(ta.array(key[-n_infinite:]) for key, _ in combination)
        if sum(key) != order:
            continue
        values = [value for _, value in combination if value is not None]
        if hermitian and key > tuple(reversed(key)):
            continue
        temp = reduce(op, values)
        if hermitian and key == tuple(reversed(key)):
            temp /= 2
        contributing_products.append(temp)
    result = _zero_sum(contributing_products)
    if hermitian and _zero != result:
        result += Dagger(result)
    return result