# %%
# # The polynomial alternative to Lowdin perturbation theory
#
# See [this hackmd](https://hackmd.io/Rpt2C8oOQ2SGkGS9OYrlfQ?view) for the motivation and the expressions

# %%
from operator import matmul, mul

import numpy as np
import numpy.ma as ma
import sympy
from sympy.physics.quantum import Dagger, Operator, HermitianOperator
import tinyarray as ta

from lowdin.series import (
    BlockOperatorSeries,
    _zero,
    cauchy_dot_product,
    generate_orders,
)


def general(H, divide_energies=None, *, op=None):
    """
    Computes the block diagonalization of a BlockOperatorSeries.

    H : BlockOperatorSeries
    divide_energies : (optional) function to use for dividing energies
    op : (optional) function to use for matrix multiplication

    Returns:
    H_tilde : BlockOperatorSeries
    U : BlockOperatorSeries
    U_adjoint : BlockOperatorSeries
    """
    if op is None:
        op = matmul

    if divide_energies is None:
        divide_energies = _divide_energies(H)

    U = initialize_U(H.n_infinite)
    U_adjoint = BlockOperatorSeries(
        eval=(
            lambda index: U.evaluated[index]
            if index[0] == index[1]
            else -U.evaluated[index]
        ),
        data=None,
        shape=(2, 2),
        n_infinite=H.n_infinite,
    )

    # Identity and temporary H_tilde for the recursion
    identity = cauchy_dot_product(U_adjoint, U, op=op, hermitian=True, recursive=True)
    H_tilde_rec = cauchy_dot_product(
        U_adjoint, H, U, op=op, hermitian=True, recursive=True
    )

    def eval(index):
        if index[0] == index[1]:  # diagonal block
            return -identity.evaluated[index] / 2
        elif index[:2] == (0, 1):  # off-diagonal block
            return -divide_energies(H_tilde_rec.evaluated[index])
        elif index[:2] == (1, 0):  # off-diagonal block
            return -Dagger(U.evaluated[(0, 1) + tuple(index[2:])])

    U.eval = eval

    H_tilde = cauchy_dot_product(
        U_adjoint, H, U, op=op, hermitian=True, recursive=False
    )
    return H_tilde, U, U_adjoint


def _divide_energies(H):
    """
    Returns a function that divides a matrix by the difference of its diagonal elements.

    H : BlockOperatorSeries

    Returns:
    divide_energies : function
    """
    H_0_AA = H.evaluated[(0, 0) + (0,) * H.n_infinite]
    H_0_BB = H.evaluated[(1, 1) + (0,) * H.n_infinite]

    E_A = np.diag(H_0_AA)
    E_B = np.diag(H_0_BB)

    # The Hamiltonians must already be diagonalized
    if not np.allclose(H_0_AA, np.diag(E_A)):
        raise ValueError("H_0_AA must be diagonal")
    if not np.allclose(H_0_BB, np.diag(E_B)):
        raise ValueError("H_0_BB must be diagonal")

    energy_denominators = 1 / (E_A.reshape(-1, 1) - E_B)

    def divide_energies(Y):
        return Y * energy_denominators

    return divide_energies


def initialize_U(n_infinite=1):
    """
    Initializes the BlockOperatorSeries for the transformation to diagonalized Hamiltonian.

    n_infinite : (optional) number of infinite indices

    Returns:
    U : BlockOperatorSeries
    """
    zero_order = (0,) * n_infinite
    U = BlockOperatorSeries(
        data={
            **{(0, 0) + zero_order: None},
            **{(1, 1) + zero_order: None},
            **{(0, 1) + zero_order: _zero},
            **{(1, 0) + zero_order: _zero},
        },
        shape=(2, 2),
        n_infinite=n_infinite,
    )
    return U


def to_BlockOperatorSeries(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, n_infinite=1):
    """
    Creates a BlockOperatorSeries from a dictionary of perturbation terms.

    H_0_AA : np.array of the unperturbed Hamiltonian of subspace AA
    H_0_BB : np.array of the unperturbed Hamiltonian of subspace BB
    H_p_AA : dictionary of perturbation terms of subspace AA
    H_p_BB : dictionary of perturbation terms of subspace BB
    H_p_AB : dictionary of perturbation terms of subspace AB
    n_infinite : (optional) number of infinite indices

    Returns:
    H : BlockOperatorSeries
    """
    zeroth_order = (0,) * n_infinite
    H = BlockOperatorSeries(
        data={
            **{(0, 0) + zeroth_order: H_0_AA},
            **{(1, 1) + zeroth_order: H_0_BB},
            **{(0, 0) + tuple(key): value for key, value in H_p_AA.items()},
            **{(0, 1) + tuple(key): value for key, value in H_p_AB.items()},
            **{(1, 0) + tuple(key): Dagger(value) for key, value in H_p_AB.items()},
            **{(1, 1) + tuple(key): value for key, value in H_p_BB.items()},
        },
        shape=(2, 2),
        n_infinite=n_infinite,
    )
    return H

# %%
