import sys
from operator import matmul, mul
from typing import Any, Optional, Callable, Union
from secrets import token_hex
from functools import wraps, reduce
from itertools import product
from collections import Counter

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

    def __contains__(self, item) -> bool:
        """
        Check if the given index has been evaluated.

        Parameters
        ----------
        item :
            Index to check.

        Returns
        -------
        bool
            Whether the given index has been evaluated.
        """
        return self._data.get(item) is not zero

    def __str__(self) -> str:
        dimensions = (
            *map(str, self.shape),
            *(f"∞_({name})" for name in self.dimension_names),
        )

        return f"{self.name}_({' × '.join(dimensions)})"

    def pop(self, item: Item, default: Any, /) -> Any:
        """
        Remove and return the value for a given index.

        Parameters
        ----------
        item :
            Index to remove.
        default :
            Value to return if the index is not present.

        Returns
        -------
        Any
            Value for the given index.
        """
        return self._data.pop(item, default)

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
        Series to multiply using their block structure. Must be at least two.
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
    if len(series) > 2:
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
        )

    product.eval = eval
    return product


def product_by_order(
    index: tuple[int, ...],
    first: BlockSeries,
    second: BlockSeries,
    operator: Optional[Callable] = None,
    hermitian: bool = False,
) -> Any:
    """
    Compute sum of all product of factors of a wanted order.

    Only queries the highest order of a series if the the other series has
    some 0th order terms. This is needed to support recurrent definitions.

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

    Returns
    -------
    Any
        Sum of all products that contribute to the wanted order.
    """
    if operator is None:
        operator = matmul
    start, end, *orders = index
    hermitian = hermitian and start == end

    result = zero
    for middle, *orders_1st in product(
        range(first.shape[1]), *(range(dim + 1) for dim in orders)
    ):
        orders_1st = tuple(orders_1st)
        orders_2nd = tuple(i - j for i, j in zip(orders, orders_1st))
        first_index = (start, middle) + orders_1st
        second_index = (middle, end) + orders_2nd

        if hermitian and orders_1st > orders_2nd:
            continue

        if (first_index not in first) or (second_index not in second):
            continue

        # Total complexity of computing an element is ~П(n_i + 1)²
        def cost(orders):
            return reduce(mul, ((i + 1) ** 2 for i in orders))

        if cost(orders_1st) <= cost(orders_2nd):
            if (first_value := first[first_index]) is zero:
                continue
            if (second_value := second[second_index]) is zero:
                continue
        else:
            if (second_value := second[second_index]) is zero:
                continue
            if (first_value := first[first_index]) is zero:
                continue

        # Take care of the sentinel value one
        values = [i for i in (first_value, second_value) if i is not one]
        if not values:
            term = one
        elif len(values) == 1:
            term = values[0]
        else:
            term = operator(values[0], values[1])
        if not hermitian or orders_1st == orders_2nd:
            result = result + term  # No += to avoid mutating data.
        else:
            result = result + term + Dagger(term)

    return result


def _log_call(func):
    """
    Log a method call into the class attribute.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        # We're wrapping a method, so args[0] is always self
        type(args[0]).log.append((args[0], func.__name__, args[1:], kwargs))
        return func(*args, **kwargs)

    return wrapped


class AlgebraElement:
    log = []

    def __init__(self, name):
        """An abstract algebra element.

        Parameters
        ----------
        name : str
            Name of the element.

        Attributes
        ----------
        log : list
            List of all algebraic method calls with the format ``(self,
            method_name, args, kwargs)``.
        name : str
            Name of the element. For a result of a calculation will contain a
            sympy-fiable formula.

        Notes
        -----
        The two main uses of this class are:

        - Obtain a symbolic expression of all computations that are performed
          with its instances.
        - Log all of its method calls into a class attribute.
        """
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"AlgebraElement({self.name})"

    @_log_call
    def __mul__(self, other):
        return type(self)(rf"({self} * {other})")

    @_log_call
    def __rmul__(self, other):
        return type(self)(rf"({other} * {self})")

    @_log_call
    def __add__(self, other):
        return type(self)(rf"({self} + {other})")

    @_log_call
    def adjoint(self):
        return type(self)(rf"adjoint({self})")

    @_log_call
    def __neg__(self):
        return type(self)(f"(-{self})")

    @_log_call
    def __sub__(self, other):
        return self + (-other)

    @_log_call
    def __truediv__(self, other):
        if not isinstance(other, int):
            raise ValueError("Can only divide by integers.")
        if other < 0:
            return type(self)(rf"(-{self} / {-other})")
        return type(self)(rf"({self} / {other})")

    @classmethod
    def call_counts(cls):
        return Counter(call[1] for call in cls.log)

    def to_sympy(self):
        # Based on https://stackoverflow.com/a/32169940, CC BY-SA 3.0
        parsed_expr = sympy.parsing.sympy_parser.parse_expr(self.name, evaluate=False)

        new_locals = {
            sym.name: sympy.Symbol(sym.name, commutative=False)
            for sym in parsed_expr.atoms(sympy.Symbol)
        }

        return sympy.sympify(self.name, locals=new_locals)
