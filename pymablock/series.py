import sys
from operator import matmul
from typing import Any, Optional, Callable, Union, Iterable
from secrets import token_hex

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any

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

    def __mul__(self, other: Any = None) -> Self:
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
        dimensions = (
            *map(str, self.shape),
            *(f"{name}: ∞" for name in self.dimension_names),
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

    Notes: This treats a singleton ``one`` as the identity operator.

    Parameters
    ----------
    series :
        Series to multiply using their block structure.
    operator :
        (optional) Function for multiplying elements of the series. Default is
        matrix multiplication matmul.
    hermitian :
        (optional) whether to assume that the result is Hermitian in order to
        reduce computational costs.

    Returns
    -------
    `~pymablock.series.BlockSeries`
        A new series that is the Cauchy dot product of the given series.
    """
    if not series:
        raise ValueError("Need at least one series to multiply")
    elif len(series) == 1:
        return series[0]
    elif len(series) > 2:
        # Use associativity to reuse intermediate results This might be possible
        # to speed up further if series had any promises about the orders they
        # contains.
        product = cauchy_dot_product(
            cauchy_dot_product(*series[:2], operator=operator),
            *series[2:],
            operator=operator,
        )
        if not hermitian:
            return product
        # Same, but use Hermiticity to obtain terms with index[0] > index[1]
        nonhermitian_eval = product.eval

        def eval(*index):
            if index[0] > index[1]:
                return Dagger(product[(index[1], index[0], *index[2:])])
            return nonhermitian_eval(*index)

        product.eval = eval
        return product

    if operator is None:
        operator = matmul

    first, second = series
    if first.n_infinite != second.n_infinite:
        raise ValueError("Factors must have equal number of infinite dimensions.")
    if first.dimension_names != second.dimension_names:
        raise ValueError("All series must have the same dimension names.")

    if first.shape[1] != second.shape[0]:
        raise ValueError(
            "Factors must have finite dimensions compatible with dot product."
        )

    # For the highest order of a factor to be included, the other factor must
    # have at least some 0th order terms. We check this explicitly to support
    # recurrent definitions.
    exclude_last = [
        np.all(factor[(slice(None), slice(None)) + series[0].n_infinite * (0,)].mask)
        for factor in (second, first)
    ]

    product = BlockSeries(
        data=None,
        shape=(first.shape[0], second.shape[1]),
        n_infinite=first.n_infinite,
        dimension_names=first.dimension_names,
        name=f"{first.name} @ {second.name}",
    )

    def eval(*index):
        if index[0] > index[1] and hermitian:
            return Dagger(product[(index[1], index[0], *index[2:])])
        return product_by_order(
            index,
            first,
            second,
            operator=operator,
            hermitian=hermitian,
            exclude_last=exclude_last,
        )

    product.eval = eval
    return product


def product_by_order(
    index: tuple[int, ...],
    first: BlockSeries,
    second: BlockSeries,
    operator: Optional[Callable] = None,
    hermitian: bool = False,
    exclude_last: tuple[bool, bool] = (False, False),
) -> Any:
    """
    Compute sum of all product of factors of a wanted order.

    Parameters
    ----------
    index :
        Index of the wanted order.
    first :
        First factor.
    second :
        Second factor.
    operator :
        Function for multiplying elements of the series. Default is matrix
        multiplication matmul.
    hermitian :
        if True, hermiticity is used to reduce computations to 1/2.
    exclude_last :
        whether to exclude last order on each term. This is useful to avoid
        infinite recursion on some algorithms.

    Returns
    -------
    Any
        Sum of all products that contribute to the wanted order.
    """
    if operator is None:
        operator = matmul
    start, end, *orders = index
    hermitian = hermitian and start == end
    series = (first, second)

    # Form masked arrays with correct shapes and required elements unmasked
    data = [
        ma.ones(
            factor.shape + tuple([dim + 1 for dim in orders]),
            dtype=object,
        )
        for factor in series
    ]
    data[0][np.arange(data[0].shape[0]) != start] = ma.masked
    data[1][:, np.arange(data[-1].shape[1]) != end] = ma.masked
    for values, exclude in zip(data, exclude_last):
        values.mask[(slice(None), slice(None)) + (-1,) * len(orders)] |= exclude

    # Fill these with the evaluated data
    for values, factor in zip(data, series):  # type: ignore
        values[ma.where(values)] = factor[ma.where(values)]

    result = zero
    for first_index, value in ma.ndenumerate(data[0]):
        start, middle, *orders_1st = first_index
        orders_1st = tuple(orders_1st)
        orders_2nd = tuple(i - j for i, j in zip(orders, orders_1st))
        if any(order < 0 for order in orders_2nd):
            continue
        if hermitian and orders_1st > orders_2nd:
            continue
        if (other := data[1][(middle, end) + orders_2nd]) is ma.masked:
            continue
        # Take care of the sentinel value one
        values = [i for i in (value, other) if i is not one]
        if not values:
            term = one
        elif len(values) == 1:
            term = values[0]
        else:
            term = operator(values[0], values[1])
        if not hermitian or orders_1st == orders_2nd:
            result = _zero_sum((result, term))
        else:
            result = _zero_sum((result, term, Dagger(term)))

    return result
