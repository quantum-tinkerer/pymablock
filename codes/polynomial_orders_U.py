# # The polynomial alternative to Lowdin perturbation theory
#
# See [this hackmd](https://hackmd.io/Rpt2C8oOQ2SGkGS9OYrlfQ?view) for the motivation and the expressions

# +
from itertools import count, product
from functools import reduce
from operator import matmul

import numpy as np
import sympy
from sympy import (
    symbols, Symbol, MatrixSymbol, Matrix,
    diff, BlockMatrix, BlockDiagMatrix,
    ZeroMatrix, Identity, diag, eye, zeros
)
from sympy.physics.quantum import Dagger
from IPython.display import display_latex
import matplotlib.pyplot as plt
import tinyarray as ta
# -

sympy.init_printing()


# ### Auxiliary classes

# +
class Zero(np.ndarray):
    """A class that skips itself in additions

    It is derived from a numpy array for its implementation of right addition
    and subtraction to take priority.
    """

    def __add__(self, other):
        return other

    __radd__ = __rsub__ = __add__

    def __sub__(self, other):
        return -other

    def __neg__(self):
        return self

    def __truediv__(self, other):
        return self

    def __matmul__(self, other):
        return self

    __rmatmul__ = __matmul__


_zero = Zero(0)


class One(np.ndarray):
    """A class that skips itself in matrix multiplications

    It is derived from a numpy array for its implementation of right
    multiplication to take priority.
    """

    def __add__(self, other):
        raise NotImplementedError

    __radd__ = __rsub__ = __sub__ = __neg__ = __add__
    __div__ = __rdiv__ = __mul__ = __rmul__ = __add__

    def __matmul__(self, other):
        return other

    __rmatmul__ = __matmul__

_one = One(1)


# -

# ### Computing $U_n$ and $V_n$

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
        np.meshgrid(*(np.arange(order+1) for order in max_order))
    ).reshape(len(max_order), -1)
    indices = np.any(np.all(possible_orders.reshape(N_p, -1, 1)
                            <= wanted_orders.T.reshape(N_p, 1, -1), axis=0), axis=1)
    keep_arrays = possible_orders.T[indices]
    return (ta.array(i) for i in sorted(keep_arrays, key=sum) if any(i))


def product_by_order(order, *terms, op=None):
    """
    Compute sum of all product of terms of wanted order.

    wanted_orders : list of tinyarrays containing the wanted order of each perturbation.

    Returns:
    Sum of all contributing products.
    """
    if op is None:
        op = matmul
    contributing_products = []
    for combination in product(*(term.items() for term in terms)):
        if sum(key for key, _ in combination) == order:
            values = [value for _, value in combination if value is not _one]
            if _zero in values:
                continue
            contributing_products.append(reduce(op, values))
    return sum(contributing_products, start=_zero)


def compute_next_orders(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, divide_energies=None, *, op=None):
    """
    Computes transformation to diagonalized Hamiltonian with multivariate perturbation.

    H_0_AA : unperturbed Hamiltonian A block in eigenbasis and ordered by eigenenergy.
    H_0_BB : unperturbed Hamiltonian B block in eigenbasis and ordered by eigenenergy.
    H_p_AA : dictionary of perturbations A blocks in eigenbasis of H_0
    H_p_BB : dictionary of perturbations B blocks in eigenbasis of H_0
    H_p_AB : dictionary of perturbations AB blocks in eigenbasis of H_0
    wanted_orders : list of tinyarrays containing the wanted order of each perturbation
    divide_energies : (optional) callable for solving Sylvester equation
    op: callable

    Returns:
    exp_S : 2x2 np.array of dictionaries of transformations
    """
    if divide_energies is None:
        E_A = np.diag(H_0_AA)
        E_B = np.diag(H_0_BB)
        energy_denominators = 1/(E_A.reshape(-1, 1) - E_B)

        def divide_energies(Y):
            return Y * energy_denominators

    H_p_BA = {key: Dagger(value) for key, value in H_p_AB.items()}
    zero_index = ta.zeros([len(wanted_orders[0])], int)
    H = np.array([
        [{zero_index: H_0_AA, **H_p_AA}, H_p_AB],
        [H_p_BA, {zero_index: H_0_BB, **H_p_BB}]
    ])

    exp_S = np.array([
        [{zero_index: _one}, {}],
        [{}, {zero_index: _one}]
    ])
    needed_orders = generate_volume(wanted_orders)

    for order in needed_orders:
        Y = sum(
            (
                -(-1)**i * product_
                for i in (0, 1) for j in (0, 1)
                if not isinstance(
                    (
                        product_ := product_by_order(
                            order, exp_S[0, i], H[i, j], exp_S[j, 1],
                            op=op
                        )
                    ),
                    Zero
                )
            ),
            start=_zero
        )
        if not isinstance(Y, Zero):
            exp_S[0, 1][order] = divide_energies(Y)
            exp_S[1, 0][order] = -Dagger(exp_S[0, 1][order])

        for i in (0, 1):
            exp_S[i, i][order] = (
                - product_by_order(order, exp_S[i, i], exp_S[i, i], op=op)
                + product_by_order(order, exp_S[i, 1-i], exp_S[1-i, i], op=op)
            )/2

    return exp_S


def H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, exp_S, compute_AB=False):
    """
    Computes block-diagonal form of Hamiltonian with multivariate perturbation.

    H_0_AA : unperturbed Hamiltonian A block in eigenbasis and ordered by eigenenergy.
    H_0_BB : unperturbed Hamiltonian B block in eigenbasis and ordered by eigenenergy.
    H_p_AA : dictionary of perturbations A blocks in eigenbasis of H_0
    H_p_BB : dictionary of perturbations B blocks in eigenbasis of H_0
    H_p_AB : dictionary of perturbations AB blocks in eigenbasis of H_0
    wanted_orders : list of tinyarrays containing the wanted order of each perturbation.
    exp_S : 2x2 np.array of dictionaries of transformations

    Returns:
    H_AA : dictionary of orders of transformed perturbed Hamiltonian A block
    H_BB : dictionary of orders of transformed perturbed Hamiltonian A block
    """
    zero_index = ta.zeros([len(wanted_orders[0])], int)
    H_p_BA = {key: Dagger(value) for key, value in H_p_AB.items()}
    H = np.array([
        [{zero_index: H_0_AA, **H_p_AA}, H_p_AB],
        [H_p_BA, {zero_index: H_0_BB, **H_p_BB}]
    ])

    if compute_AB:
        H_tilde = ({}, {}, {}) # AA BB AB
        indices = ((0, 1, 0), (0, 1, 1), (0, 1, 2))
    else:
        H_tilde = ({}, {}) # AA BB
        indices = ((0, 1), (0, 1), (0, 1))

    needed_orders = generate_volume(wanted_orders)
    for order in needed_orders:
        for k, l, block in zip(*indices):
            H_tilde[block][order] = sum(
                (
                    (-1)**(i != k) * product_
                    for i in (0, 1) for j in (0, 1)
                    if not isinstance(
                        (
                            product_ := product_by_order(
                                order, exp_S[k, i], H[i, j], exp_S[j, l],
                            )
                        ),
                        Zero
                    )
                ),
                start=_zero
            )
    H_tilde = tuple(
        {order: value for order, value in term.items() if value is not _zero}
        for term in H_tilde
    )

    return H_tilde


# -

# ### Testing

# +
N_A = 2
N_B = 2
N = N_A + N_B
H_0 = np.diag(np.sort(np.random.randn(N)))

N_p = 1
wanted_orders = [ta.array([5], int)]
H_ps = []
for perturbation in range(N_p):
    H_p = np.random.random(size=(N, N)) + 1j * np.random.random(size=(N, N))
    H_p += H_p.conj().T
    H_ps.append(H_p)

H_0_AA = H_0[:N_A, :N_A]
H_0_BB = H_0[N_A:, N_A:]

orders = ta.array(np.eye(N_p))
H_p_AA = {
    order: value[:N_A, :N_A]
    for order, value in zip(orders, H_ps)
}

H_p_BB = {
    order: value[N_A:, N_A:]
    for order, value in zip(orders, H_ps)
}

H_p_AB = {
    order: value[:N_A, N_A:]
    for order, value in zip(orders, H_ps)
}
# -

exp_S = compute_next_orders(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders=wanted_orders)

H_AA, H_BB, H_AB = H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, exp_S, compute_AB=True, precision_tol=1e-10)

# H_AA, H_BB, H_AB = H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, exp_S, compute_AB=True, precision_tol=1e-10)
# H_AA, H_BB = H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, exp_S)

# H_AA

# H_BB

# for key, values in H_AB.items():
#     assert np.allclose(values, 0, atol=10-8)
