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

# %%
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
    np.index_exp[6, 3, 3],
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
    try:
        a = BlockOperatorSeries(lambda x: x, shape=(5,), n_infinite=2)
        assert shape == a.evaluated[key].shape
    except Exception as e:
        assert type(e) == shape


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


def generate_volume(wanted_orders):
    """
    Generate ordered array with tinyarrays in volume of wanted_orders.

    wanted_orders : list of tinyarrays containing the wanted order of each perturbation.

    Returns:
    List of sorted tinyarrays contained in the volume of required orders to compute the wanted orders.
    """
    wanted_orders = np.array(wanted_orders)
    N_o, N_p = wanted_orders.shape
    max_order = np.max(wanted_orders, axis=0)
    possible_orders = np.array(
        np.meshgrid(*(np.arange(order + 1) for order in max_order))
    ).reshape(len(max_order), -1)
    indices = np.any(
        np.all(
            possible_orders.reshape(N_p, -1, 1) <= wanted_orders.T.reshape(N_p, 1, -1),
            axis=0,
        ),
        axis=1,
    )
    keep_arrays = possible_orders.T[indices]
    return (ta.array(i) for i in sorted(keep_arrays, key=sum) if any(i))


def product_by_order(order, *terms, op=None, hermitian=False):
    """
    Compute sum of all product of terms of wanted order.

    order : int or tinyarray containing the order of the product.
    terms : list of dictionaries of terms to multiply.
    op : (optional) callable for multiplying terms.
    hermitian : (optional) bool for whether to compute hermitian conjugate.

    Returns:
    Sum of all products of terms of wanted order.
    """
    if op is None:
        op = matmul
    contributing_products = []
    for combination in product(*(term.items() for term in terms)):
        key = tuple(key for key, _ in combination)
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

# %%
def compute_next_orders(
    H_0_AA,
    H_0_BB,
    H_p_AA,
    H_p_BB,
    H_p_AB,
    wanted_orders,
    divide_energies=None,
    *,
    op=None
):
    """
    Computes transformation to diagonalized Hamiltonian with multivariate perturbation.

    H_0_AA : np.array of the unperturbed Hamiltonian of subspace AA
    H_0_BB : np.array of the unperturbed Hamiltonian of subspace BB
    H_p_AA : dictionary of perturbation terms of subspace AA
    H_p_BB : dictionary of perturbation terms of subspace BB
    H_p_AB : dictionary of perturbation terms of subspace AB
    wanted_orders : list of tinyarrays containing the wanted order of each perturbation
    divide_energies : (optional) callable for solving Sylvester equation
    op: callable for multiplying terms

    Returns:
    exp_S : np.array of the transformation to diagonalized Hamiltonian
    """

    zero_index = ta.zeros([len(wanted_orders[0])], int)
    if any(zero_index in pert for pert in (H_p_AA, H_p_AB, H_p_BB)):
        raise ValueError("Perturbation terms may not contain zeroth order")

    H_p_BA = {key: Dagger(value) for key, value in H_p_AB.items()}
    H = np.array(
        [
            [{zero_index: H_0_AA, **H_p_AA}, H_p_AB],
            [H_p_BA, {zero_index: H_0_BB, **H_p_BB}],
        ],
        dtype=object,
    )

    # We use None as a placeholder for identity.
    exp_S = np.array([[{zero_index: None}, {}], [{}, {zero_index: None}]], dtype=object)

    if divide_energies is None:
        # The Hamiltonians must already be diagonalized
        E_A = np.diag(H_0_AA)
        E_B = np.diag(H_0_BB)
        if not np.allclose(H_0_AA, np.diag(E_A)):
            raise ValueError("H_0_AA must be diagonal")
        if not np.allclose(H_0_BB, np.diag(E_B)):
            raise ValueError("H_0_BB must be diagonal")
        energy_denominators = 1 / (E_A.reshape(-1, 1) - E_B)

        def divide_energies(Y):
            return Y * energy_denominators

    needed_orders = generate_volume(wanted_orders)

    for order in needed_orders:
        Y = _zero_sum(
            -((-1) ** i)
            * product_by_order(order, exp_S[0, i], H[i, j], exp_S[j, 1], op=op)
            for i in (0, 1)
            for j in (0, 1)
        )
        if not isinstance(Y, Zero):
            exp_S[0, 1][order] = divide_energies(Y)
            exp_S[1, 0][order] = -Dagger(exp_S[0, 1][order])

        for i in (0, 1):
            exp_S[i, i][order] = (
                _zero_sum(
                    (
                        -product_by_order(
                            order, exp_S[i, i], exp_S[i, i], op=op, hermitian=True
                        ),
                        product_by_order(
                            order,
                            exp_S[i, 1 - i],
                            exp_S[1 - i, i],
                            op=op,
                            hermitian=True,
                        ),
                    )
                )
                / 2
            )

    return exp_S

# %%
def H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, exp_S, op=None):
    if op is None:
        op = matmul

    try:
        n_infinite = len(list(H_p_AA.keys())[0])
    except IndexError:
        try:
            n_infinite = len(list(H_p_BB.keys())[0])
        except IndexError:
            n_infinite = len(list(H_p_AB.keys())[0])

    def eval(entry):
        order = ta.array(entry[-n_infinite:])
        zero_index = ta.zeros([len(order)], int)
        H_p_BA = {key: Dagger(value) for key, value in H_p_AB.items()}
        H = np.array(
            [
                [{zero_index: H_0_AA, **H_p_AA}, H_p_AB],
                [H_p_BA, {zero_index: H_0_BB, **H_p_BB}],
            ],
            dtype=object,
        )
        k, l = entry[:-n_infinite]

        return _zero_sum(
            (-1) ** (i != k)
            * product_by_order(
                order,
                exp_S[k, i],
                H[i, j],
                exp_S[j, l],
                hermitian=(i == j and k == l),
                op=op,
            )
            for i in (0, 1)
            for j in (0, 1)
        )

    H_tilde = BlockOperatorSeries(eval, shape=(2, 2), n_infinite=n_infinite)
    return H_tilde
# %%
H_0_AA = np.diag([1, 2, 3.2])
H_0_BB = np.diag([4, 5, 6])

H_p_AA = {ta.array([1]): np.ones((3, 3))}
H_p_BB = {ta.array([1]): np.ones((3, 3))}
H_p_AB = {ta.array([1]): np.ones((3, 3))}

wanted_orders = [ta.array([2])]

exp_S = compute_next_orders(
    H_0_AA,
    H_0_BB,
    H_p_AA,
    H_p_BB,
    H_p_AB,
    wanted_orders,
    divide_energies=None,
    op=None
)
# %%
H_tilde = H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, exp_S)

# %%
H_tilde_AA_1 = H_tilde.evaluated[0, 0, 0]
H_tilde_AA_1
