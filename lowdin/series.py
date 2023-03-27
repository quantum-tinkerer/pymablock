# %%
from itertools import product
from functools import reduce
from operator import matmul

import numpy as np
import numpy.ma as ma
from sympy.physics.quantum import Dagger
import tinyarray as ta

PENDING = object()  # sentinel value for pending evaluation
one = object()   # singleton for identity operator

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

    def __eq__(self, other):
        return isinstance(other, Zero)

    def __hash__(self):
        return super().__hash__()

    adjoint = conjugate = all = __neg__ = __truediv__ = __rmul__ = __mul__

zero = Zero()
_mask = np.vectorize((lambda entry: isinstance(entry, Zero)), otypes=[np.bool])


def _zero_sum(terms):
    """
    Sum that returns a singleton zero if empty and omits zero terms

    terms : iterable of terms to sum.

    Returns:
    Sum of terms, or zero if terms is empty.
    """
    return sum((term for term in terms if zero != term), start=zero)


class _Evaluated:
    def __init__(self, original):
        self.original = original

    def __getitem__(self, item):
        """Evaluate the series at the given index, following numpy's indexing rules.

        Parameters
        ----------
        item : tuple of int or slice

        Returns
        -------
        The item or items at the given index.
        """
        self.check_finite(item[-self.original.n_infinite :])
        self.check_number_perturbations(item)

        # Create trial array to use for indexing
        trial_shape = self.original.shape + tuple(
            [
                order.stop if isinstance(order, slice) else np.max(order, initial=0) + 1
                for order in item[-self.original.n_infinite :]
            ]
        )
        trial = np.zeros(trial_shape, dtype=object)
        one_entry = np.isscalar(trial[item])
        trial[item] = 1

        data = self.original.data
        for index in zip(*np.where(trial)):
            if index not in data:
                # Calling eval gives control away; mark that this value is evaluated
                # To be able to catch recursion and data corruption.
                data[index] = PENDING
                data[index] = self.original.eval(index)
            if data[index] is PENDING:
                raise RuntimeError("Infinite recursion loop detected")
            trial[index] = data[index]


        result = trial[item]
        if not one_entry:
            return ma.masked_where(_mask(result), result)
        return result  # return one item

    def check_finite(self, orders):
        """Check that the indices of the infinite dimension are finite and positive."""
        for order in orders:
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


class BlockSeries:
    def __init__(self, eval=None, data=None, shape=(), n_infinite=1):
        """An infinite series that caches its items.
        The series has finite and infinite dimensions.

        Parameters
        ----------
        eval : callable
            A function that takes an index and returns the corresponding item.
        data : dict
            A dictionary of items so start with. The keys should be tuples of indices.
        shape : tuple of int
            The shape of the finite dimensions.
        n_infinite : int
            The number of infinite dimensions.
        """
        self.eval = (lambda _: zero) if eval is None else eval
        self.evaluated = _Evaluated(self)
        self.data = data or {}
        self.shape = shape
        self.n_infinite = n_infinite


def cauchy_dot_product(*series, op=None, hermitian=False, exclude_last=None):
    """
    Multivariate Cauchy product of block operator series

    Notes:
    This treats a singleton `one` as the identity operator.

    series : list of BlockSeries to be multiplied.
    op : (optional) callable for multiplying factors.
    hermitian : (optional) bool for whether to use hermiticity.
    exclude_last : (optional) None or list of bools on whether to exclude last order on each term.

    Returns:
    (BlockSeries) Product of series.
    """
    if len(series) < 2:
        return series[0] if series else one
    if op is None:
        op = matmul
    if exclude_last is None:
        exclude_last = [False] * len(series)

    starts, ends = zip(*(factor.shape for factor in series))
    start, *rest_starts = starts
    *rest_ends, end = ends
    if rest_starts != rest_ends:
        raise ValueError(
            "Factors must have finite dimensions compatible with dot product."
        )

    if len(set(factor.n_infinite for factor in series)) > 1:
        raise ValueError("Factors must have equal number of infinite dimensions.")

    return BlockSeries(
        eval=lambda index: product_by_order(
            index, *series, op=op, hermitian=hermitian, exclude_last=exclude_last
        ),
        data=None,
        shape=(start, end),
        n_infinite=series[0].n_infinite,
    )


def _generate_orders(orders, start=None, end=None, last=True):
    """
    Generate array of lower orders to be used in product_by_order.

    orders : (tuple) maximum orders of each infinite dimension.
    start : (optional) 0 or 1 to choose row index of block.
    end : (optional) 0 or 1 to choose column index of block.
    last : (optional) bool for whether to keep last order.
        This is useful to avoid recursion errors.

    Returns:
    (numpy.ma.MaskedArray) Array of orders.
    """
    mask = (slice(None), slice(None)) + (-1,) * len(orders)
    trial = ma.ones((2, 2) + tuple([dim + 1 for dim in orders]), dtype=object)
    if start is not None:
        trial[int(not start)] = ma.masked
    if end is not None:
        trial[:, int(not end)] = ma.masked
    if not last:
        trial[mask] = ma.masked
    return trial


# %%
def product_by_order(index, *series, op=None, hermitian=False, exclude_last=None):
    """
    Compute sum of all product of factors of wanted order.

    index : (tuple) index of wanted order.
    series : (BlockSeries) series to be multiplied.
    op : (optional) callable for multiplying factors.
    hermitian : (optional) bool for whether to use hermiticity.
    exclude_last : (optional) None or list of bools on whether to exclude last order on each series.

    Returns:
    Sum of all products that contribute to the wanted order.
    """
    if op is None:
        op = matmul
    start, end, *orders = index
    hermitian = hermitian and start == end

    n_infinite = series[0].n_infinite

    if exclude_last is None:
        exclude_last = [False] * len(series)
    starts = [start] + [None] * (len(series) - 1)
    ends = [None] * (len(series) - 1) + [end]
    data = [
        _generate_orders(orders, start=start, end=end, last=not(last))
        for start, end, last in zip(starts, ends, exclude_last)
    ]

    for indices, factor in zip(data, series):
        indices[ma.where(indices)] = factor.evaluated[ma.where(indices)]

    contributing_products = []
    for combination in product(*(ma.ndenumerate(factor) for factor in data)):
        combination = list(combination)
        starts, ends = zip(*(key[:-n_infinite] for key, _ in combination))
        if starts[1:] != ends[:-1]:
            continue
        key = tuple(ta.array(key[-n_infinite:]) for key, _ in combination)
        if sum(key) != orders:
            continue
        values = [value for _, value in combination if value is not one]
        if hermitian and key > tuple(reversed(key)):
            continue
        temp = reduce(op, values)
        if hermitian and key == tuple(reversed(key)):
            temp /= 2
        contributing_products.append(temp)
    result = _zero_sum(contributing_products)
    if hermitian and zero != result:
        result += Dagger(result)
    return result
