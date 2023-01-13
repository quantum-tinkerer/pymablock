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

from poly_kpm import SumOfOperatorProducts


# ### Auxiliary functions
# -


# +
class Zero:
    """A class that skips itself in additions

    It is derived from a numpy array for its implementation of right addition
    and subtraction to take priority.
    """
    def __mul__(self, other=None):
        return self

    def __add__(self, other):
        return other

    __neg__ = __truediv__ = __rmul__ = __mul__


_zero = Zero()


def _zero_sum(terms):
    """Sum that returns a singleton _zero if empty and omits _zero terms"""
    return sum(
        (term for term in terms if not isinstance(term, Zero)),
        start=_zero
    )


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
            values = [value for _, value in combination if value is not None]
            if any(isinstance(value, Zero) for value in values):
                continue
            contributing_products.append(reduce(op, values))
    return _zero_sum(contributing_products)


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

    zero_index = ta.zeros([len(wanted_orders[0])], int)
    if any(zero_index in pert for pert in (H_p_AA, H_p_AB, H_p_BB)):
        raise ValueError("Perturbation terms may not contain zeroth order")

    H_p_BA = {key: Dagger(value) for key, value in H_p_AB.items()}
    H = np.array([
        [{zero_index: H_0_AA, **H_p_AA}, H_p_AB],
        [H_p_BA, {zero_index: H_0_BB, **H_p_BB}]
    ])

    # We use None as a placeholder for identity.
    exp_S = np.array([
        [{zero_index: None}, {}],
        [{}, {zero_index: None}]
    ])

    if divide_energies is None:
        E_A = np.diag(H_0_AA)
        E_B = np.diag(H_0_BB)
        energy_denominators = 1/(E_A.reshape(-1, 1) - E_B)

        def divide_energies(Y):
            return Y * energy_denominators

    needed_orders = generate_volume(wanted_orders)

    for order in needed_orders:
        Y = _zero_sum(
            -(-1)**i * product_by_order(
                order, exp_S[0, i], H[i, j], exp_S[j, 1],
                op=op
            )
            for i in (0, 1) for j in (0, 1)
        )
        if not isinstance(Y, Zero):
            exp_S[0, 1][order] = divide_energies(Y)
            exp_S[1, 0][order] = -Dagger(exp_S[0, 1][order])

        for i in (0, 1):
            exp_S[i, i][order] = _zero_sum((
                -product_by_order(order, exp_S[i, i], exp_S[i, i], op=op),
                product_by_order(order, exp_S[i, 1-i], exp_S[1-i, i], op=op)
            ))/2

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
            H_tilde[block][order] = _zero_sum(
                (-1)**(i != k) * product_by_order(
                    order, exp_S[k, i], H[i, j], exp_S[j, l],
                )
                for i in (0, 1) for j in (0, 1)
            )
    H_tilde = tuple(
        {order: value for order, value in term.items() if value is not _zero}
        for term in H_tilde
    )

    return H_tilde


