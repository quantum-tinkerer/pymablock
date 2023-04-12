# %%
from itertools import product
from functools import reduce
from operator import matmul
from typing import Any, Optional, Callable

import numpy as np
import numpy.ma as ma
from sympy.physics.quantum import Dagger
import tinyarray as ta

PENDING = object()  # sentinel value for pending evaluation
one = object()  # singleton for identity operator

# %%
class Zero:
    """
    A class that behaves like zero in all operations.
    This is used to avoid having to check for zero terms in the sum.
    """

    def __mul__(self, other: Any = None) -> "Zero":
        return self

    def __add__(self, other: Any) -> Any:
        return other

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Zero)

    adjoint = conjugate = all = __neg__ = __truediv__ = __rmul__ = __mul__


zero = Zero()
_mask = np.vectorize((lambda entry: isinstance(entry, Zero)), otypes=[bool])


def _zero_sum(terms: list[Any]) -> Any:
    """
    Sum that returns a singleton zero if empty and omits zero terms

    Parameters
    ----------
    terms : Terms to sum over with zero as default value.

    Returns
    -------
    Sum of terms, or zero if terms is empty.
    """
    return sum((term for term in terms if zero != term), start=zero)


class _Evaluated:
    def __init__(self, original: "BlockSeries") -> None:
        self.original = original

    def __getitem__(
        self, item: int | slice | tuple[int | slice, ...] 
    ) -> ma.MaskedArray[Any] | Any:
        """
        Evaluate the series at the given index, following numpy's indexing rules.

        Parameters
        ----------
        item : index at which to evaluate the series.

        Returns
        -------
        The item or items at the given index.
        """
        if not isinstance(item, tuple):  # Allow indexing with integer
            item = (item,)

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
                data[index] = self.original.eval(*index)
            if data[index] is PENDING:
                raise RuntimeError("Infinite recursion loop detected")
            trial[index] = data[index]

        result = trial[item]
        if not one_entry:
            return ma.masked_where(_mask(result), result)
        return result  # return one item

    def check_finite(self, orders: tuple[int | slice, ...]):
        """
        Check that the indices of the infinite dimension are finite and positive.

        Parameters
        ----------
        orders : indices of the infinite dimension.
        """
        for order in orders:
            if isinstance(order, slice):
                if order.stop is None:
                    raise IndexError("Cannot evaluate infinite series")
                elif isinstance(order.start, int) and order.start < 0:
                    raise IndexError("Cannot evaluate negative order")

    def check_number_perturbations(self, item: tuple[int | slice, ...]):
        """
        Check that the number of indices is correct.

        Parameters
        ----------
        item : indices to check.
        """
        if len(item) != len(self.original.shape) + self.original.n_infinite:
            raise IndexError(
                "Wrong number of indices",
                (len(item), len(self.original.shape), self.original.n_infinite),
            )


class BlockSeries:
    def __init__(
        self,
        eval: Optional[Callable[[tuple[int, ...]], Any]] = None,
        data: Optional[dict[tuple[int, ...], Any]] = None,
        shape: tuple[int, ...] = (),
        n_infinite: int = 1,
    ) -> None:
        """An infinite series that caches its items.
        The series has finite and infinite dimensions.

        Parameters
        ----------
        eval : Function that takes an index and returns the corresponding item.
        data : Dictionary {index: value} to start with.
        shape : Shape of the finite dimensions.
        n_infinite : Number of infinite dimensions.
        """
        self.eval = (lambda *_: zero) if eval is None else eval
        self.evaluated = _Evaluated(self)
        self.data = data or {}
        self.shape = shape
        self.n_infinite = n_infinite


def cauchy_dot_product(
    *series: BlockSeries,
    op: Optional[Callable] = None,
    hermitian: bool = False,
    exclude_last: Optional[list[bool]] = None
):
    """
    Multivariate Cauchy product of BlockSeries.

    Notes:
    This treats a singleton `one` as the identity operator.

    Parameters
    ----------
    series :
        Series to multiply using their block structure.
    op :
        (optional) Function for multiplying elements of the series.
        Default is matrix multiplication matmul.
    hermitian :
        (optional) if True, hermiticity is used to reduce computations to 1/2.
    exclude_last :
        (optional) whether to exclude last order on each term.
        This is useful to avoid infinite recursion on some algorithms.

    Returns
    -------
    `~lowdin.series.BlockSeries`
        A new series that is the Cauchy dot product of the given series.
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
        eval=lambda *index: product_by_order(
            index, *series, op=op, hermitian=hermitian, exclude_last=exclude_last
        ),
        data=None,
        shape=(start, end),
        n_infinite=series[0].n_infinite,
    )


def _generate_orders(
    orders: tuple[int, ...],
    start: Optional[int] = None,
    end: Optional[int] = None,
    last: bool = True
) -> ma.MaskedArray:
    """
    Generate array of lower orders to be used in product_by_order.

    Parameters
    ----------
    orders : maximum orders of each infinite dimension.
    start : (optional) 0 or 1 row index of block.
    end : (optional) 0 or 1 column index of block.
    last : Whether to keep last order, True by default.
        This is useful to avoid recursion errors.

    Returns
    -------
    Array of lower orders to be used in product_by_order.
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
def product_by_order(
    index: tuple[int, ...],
    *series: BlockSeries,
    op: Optional[Callable] = None,
    hermitian: bool = False,
    exclude_last: Optional[list[bool]] = None
) -> Any:
    """
    Compute sum of all product of factors of a wanted order.

    Parameters
    ----------
    index :
        Index of the wanted order.
    series :
        Series to multiply using their block structure.
    op :
        Function for multiplying elements of the series.
        Default is matrix multiplication matmul.
    hermitian :
        if True, hermiticity is used to reduce computations to 1/2.
    exclude_last :
        whether to exclude last order on each term.
        This is useful to avoid infinite recursion on some algorithms.

    Returns
    -------
    Any
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
        _generate_orders(orders, start=start, end=end, last=not (last))
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
            temp = 1/2 * temp
        contributing_products.append(temp)
    result = _zero_sum(contributing_products)
    if hermitian and zero != result:
        result += Dagger(result)
    return result
