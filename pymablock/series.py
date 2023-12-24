from __future__ import annotations
from itertools import product, compress, chain
from functools import reduce
from operator import matmul
from typing import Any, Optional, Callable, Union, Iterable
from secrets import token_hex

import numpy as np
import numpy.ma as ma
import sympy
from sympy.physics.quantum import Dagger

__all__ = ["BlockSeries", "cauchy_dot_product", "one", "zero"]

# Common types
OneItem = Union[int, slice, list[int]]
Item = Union[OneItem, tuple[OneItem, ...]]


class Pending:
    """Sentinel value representing a pending evaluation."""

    def __repr__(self) -> str:
        return "pending"


PENDING = Pending()
del Pending


class One:
    """Sentinel value representing the identity operator."""

    def __repr__(self) -> str:
        return "one"


one = One()
del One


class Zero:
    """
    A class that behaves like zero in all operations.
    This is used to avoid having to check for zero terms in the sum.
    """

    def __mul__(self, other: Any = None) -> Zero:
        return self

    def __add__(self, other: Any) -> Any:
        return other

    def __sub__(self, other: Any) -> Any:
        return -other

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Zero)

    adjoint = conjugate = all = __neg__ = __truediv__ = __rmul__ = __mul__

    __radd__ = __rsub__ = __add__


zero = Zero()
_mask = np.vectorize((lambda entry: isinstance(entry, Zero)), otypes=[bool])


def _zero_sum(terms: Iterable[Any]) -> Any:
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


def safe_divide(numerator, denominator):
    try:
        return numerator / denominator
    except TypeError:
        return (1 / denominator) * numerator


class BlockSeries:
    def __init__(
        self,
        eval: Optional[Callable] = None,
        data: Optional[dict[tuple[int, ...], Any]] = None,
        shape: tuple[int, ...] = (),
        n_infinite: int = 1,
        dimension_names: Optional[tuple[Union[str, sympy.Symbol], ...]] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        An infinite series that caches its items.

        The series has finite and infinite dimensions.

        Parameters
        ----------
        eval :
            Function that takes an index and returns the corresponding item.
        data :
            Dictionary {index: value} with initial data of the series.
        shape :
            Shape of the finite dimensions.
        n_infinite :
            Number of infinite dimensions.
        dimension_names :
            Names of the infinite dimensions.
            This is used for symbolic series.
        name :
            Name of the series. Used for printing.
        """
        self.eval = (lambda *_: zero) if eval is None else eval
        # Avoid accidentally mutating data
        self._data = {} if data is None else data.copy()
        self.shape = shape
        self.n_infinite = n_infinite
        self.dimension_names = dimension_names or tuple(
            f"n_{i}" for i in range(n_infinite)
        )
        self.name = name or f"Series_{token_hex(4)}"

    def __getitem__(self, item: Item) -> Any:
        """
        Evaluate the series at indices, following numpy's indexing rules.

        Parameters
        ----------
        item :
            Numpy-style index. If only the finite dimensions are specified, the
            result is a BlockSeries view with the infinite dimensions left
            intact.

        Returns
        -------
        The item or items at the given index.
        """
        if not isinstance(item, tuple):  # Allow indexing with integer
            item = (item,)

        if len(item) == len(self.shape):
            # Make an intermediate scalar BlockSeries that packs all finite
            # dimensions into a single item
            packed = BlockSeries(
                eval=lambda *index: self[item + index],
                shape=(),
                n_infinite=self.n_infinite,
            )
            return BlockSeries(
                eval=lambda *index: packed[index[-self.n_infinite :]][
                    index[: -self.n_infinite]
                ],
                shape=np.empty(self.shape)[item].shape,
                n_infinite=self.n_infinite,
                dimension_names=self.dimension_names,
            )

        self._check_finite(item[-self.n_infinite :])
        self._check_number_perturbations(item)

        # Create trial array to use for indexing
        trial_shape = self.shape + tuple(
            [
                order.stop if isinstance(order, slice) else np.max(order, initial=0) + 1
                for order in item[-self.n_infinite :]
            ]
        )
        trial = np.zeros(trial_shape, dtype=object)
        one_entry = np.isscalar(trial[item])
        trial[item] = 1

        data = self._data
        for index in zip(*np.where(trial)):
            if index not in data:
                # Calling eval gives control away; mark that this value is evaluated
                # To be able to catch recursion and data corruption.
                data[index] = PENDING
                try:
                    data[index] = self.eval(*index)
                except RuntimeError as error:
                    # Catch recursion errors with an informative message
                    data.pop(index, None)
                    raise RuntimeError(f"Failed to evaluate {self}[{index}]") from error
                except BaseException:
                    # Catching BaseException to clean up also after keyboard interrupt
                    data.pop(index, None)
                    raise
            if data[index] is PENDING:
                raise RuntimeError(
                    f"Infinite recursion loop detected in {self}[{index}]"
                )
            trial[index] = data[index]

        result = trial[item]
        if not one_entry:
            return ma.masked_where(_mask(result), result)
        return result  # return one item

    def __str__(self) -> str:
        dimensions = chain(
            map(str, self.shape),
            (f"{name}: ∞" for name in self.dimension_names),
        )

        return f"{self.name}_({' × '.join(dimensions)})"

    def _check_finite(self, orders: tuple[OneItem, ...]):
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

    def _check_number_perturbations(self, item: tuple[OneItem, ...]):
        """
        Check that the number of indices is correct.

        Parameters
        ----------
        item : indices to check.
        """
        if len(item) != len(self.shape) + self.n_infinite:
            raise IndexError(
                "Wrong number of indices",
                (len(item), len(self.shape), self.n_infinite),
            )


def cauchy_dot_product(
    *series: BlockSeries,
    operator: Optional[Callable] = None,
    hermitian: bool = False,
) -> BlockSeries:
    """
    Multivariate Cauchy product of `~pymablock.series.BlockSeries`.

    Notes:
    This treats a singleton ``one`` as the identity operator.

    Parameters
    ----------
    series :
        Series to multiply using their block structure.
    operator :
        (optional) Function for multiplying elements of the series.
        Default is matrix multiplication matmul.
    hermitian :
        (optional) if True, hermiticity is used to reduce computations to 1/2.

    Returns
    -------
    `~pymablock.series.BlockSeries`
        A new series that is the Cauchy dot product of the given series.
    """
    if not series:
        raise ValueError("Need at least one series to multiply")
    if len(series) < 2:
        return series[0]
    if operator is None:
        operator = matmul

    # For the last term to be included, all other terms should have a non-empty
    # 0th order. The 0th order access below may be slightly inefficient, but in
    # practice it doesn't matter because of caching.
    exclude_last = [False] * len(series)
    zero_0th_orders = [
        np.all(factor[(slice(None), slice(None)) + series[0].n_infinite * (0,)].mask)
        for factor in series
    ]
    for i, empty in enumerate(zero_0th_orders):
        if not empty:
            continue
        for j in range(len(exclude_last)):
            if i != j:
                exclude_last[j] = True

    starts, ends = zip(*(factor.shape for factor in series))
    start, *rest_starts = starts
    *rest_ends, end = ends
    if rest_starts != rest_ends:
        raise ValueError(
            "Factors must have finite dimensions compatible with dot product."
        )

    if len(set(factor.n_infinite for factor in series)) > 1:
        raise ValueError("Factors must have equal number of infinite dimensions.")

    product = BlockSeries(
        data=None,
        shape=(start, end),
        n_infinite=series[0].n_infinite,
        dimension_names=series[0].dimension_names,
        name=" @ ".join(factor.name for factor in series),
    )

    def eval(*index):
        if index[0] > index[1] and hermitian:
            return Dagger(product[(index[1], index[0], *index[2:])])
        return product_by_order(
            index,
            *series,
            operator=operator,
            hermitian=hermitian,
            exclude_last=exclude_last,
        )

    product.eval = eval
    return product


def product_by_order(
    index: tuple[int, ...],
    *series: BlockSeries,
    operator: Optional[Callable] = None,
    hermitian: bool = False,
    exclude_last: Optional[list[bool]] = None,
) -> Any:
    """
    Compute sum of all product of factors of a wanted order.

    Parameters
    ----------
    index :
        Index of the wanted order.
    series :
        Series to multiply using their block structure.
    operator :
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
    if operator is None:
        operator = matmul
    start, end, *orders = index
    hermitian = hermitian and start == end

    n_infinite = series[0].n_infinite
    if exclude_last is None:
        exclude_last = [False] * len(series)

    # Form masked arrays with correct shapes and required elements unmasked
    data = [
        ma.ones(
            factor.shape + tuple([dim + 1 for dim in orders]),
            dtype=object,
        )
        for factor in series
    ]
    data[0][np.arange(data[0].shape[0]) != start] = ma.masked
    data[-1][:, np.arange(data[-1].shape[1]) != end] = ma.masked
    for values in compress(data, exclude_last):
        values[(slice(None), slice(None)) + (-1,) * len(orders)] = ma.masked

    # Fill these with the evaluated data
    for values, factor in zip(data, series):  # type: ignore
        values[ma.where(values)] = factor[ma.where(values)]

    terms = []
    daggered_terms = []
    for combination in product(*(ma.ndenumerate(factor) for factor in data)):
        combination = list(combination)
        starts, ends = zip(*(key[:-n_infinite] for key, _ in combination))
        if starts[1:] != ends[:-1]:  # type: ignore
            continue
        key = tuple(key[-n_infinite:] for key, _ in combination)
        if list(map(sum, zip(*key))) != orders:  # Vector sum of key
            continue
        values = [value for _, value in combination if value is not one]
        if hermitian and key > tuple(reversed(key)):
            continue
        # Take care of the case when we have a product of all ones
        term = reduce(operator, values) if values else one
        if not hermitian or key == tuple(reversed(key)):
            terms.append(term)
        else:
            daggered_terms.append(term)
    result = _zero_sum(daggered_terms)
    if hermitian:
        result = _zero_sum((result, Dagger(result)))
    result = _zero_sum(terms + [result])
    return result
