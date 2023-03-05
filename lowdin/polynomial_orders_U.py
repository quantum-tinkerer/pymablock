# # The polynomial alternative to Lowdin perturbation theory
#
# See [this hackmd](https://hackmd.io/Rpt2C8oOQ2SGkGS9OYrlfQ?view) for the motivation and the expressions

# %%
from itertools import permutations
from operator import matmul

import numpy as np
from sympy.physics.quantum import Dagger

from lowdin.series import (
    BlockOperatorSeries,
    Zero,
    _zero,
    cauchy_dot_product,
)


# -
def compute_next_orders(
    H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, divide_energies=None, *, op=None
):
    """
    Computes transformation to diagonalized Hamiltonian with multivariate perturbation.

    H_0_AA : np.array of the unperturbed Hamiltonian of subspace AA
    H_0_BB : np.array of the unperturbed Hamiltonian of subspace BB
    H_p_AA : dictionary of perturbation terms of subspace AA
    H_p_BB : dictionary of perturbation terms of subspace BB
    H_p_AB : dictionary of perturbation terms of subspace AB
    divide_energies : (optional) callable for solving Sylvester equation
    op: callable for multiplying terms

    Returns:
    exp_S : BlockOperatorSeries of the transformation to diagonalized Hamiltonian
    """
    keys = [
        key for hamiltonian in [H_p_AA, H_p_AB, H_p_BB] for key in hamiltonian.keys()
    ]
    if len(keys) == 0:
        n_infinite = 0
    else:
        n_infinite = len(next(iter(keys)))
    zero_index = (0,) * n_infinite
    if any(zero_index in pert for pert in (H_p_AA, H_p_AB, H_p_BB)):
        raise ValueError("Perturbation terms may not contain zeroth order")
    H = H_from_dict(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, n_infinite)
    H.name = "H"

    exp_S = exp_S_initialize(H_0_AA.shape[0], H_0_BB.shape[1], n_infinite)
    exp_S.name = "exp_S"
    exp_S_dagger = BlockOperatorSeries(
        eval=(
            lambda entry: exp_S.evaluated[entry]
            if entry[0] == entry[1]
            else -exp_S.evaluated[entry]
        ),
        data=None,
        shape=(2, 2),
        n_infinite=n_infinite,
    )
    exp_S_dagger.name = "exp_S_dagger"

    identity = cauchy_dot_product(exp_S_dagger, exp_S, op=op, hermitian=True)
    identity.name = "I"
    H_tilde = cauchy_dot_product(exp_S_dagger, H, exp_S, op=op, hermitian=True)
    H_tilde.name = "Y"

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

    def eval(index):
        if index[0] == index[1]:  # U
            return -identity.evaluated[index] / 2
        elif index[:2] == (0, 1):  # V
            return -divide_energies(H_tilde.evaluated[index])
        elif index[:2] == (1, 0):  # V
            return -Dagger(exp_S.evaluated[(0, 1) + tuple(index[2:])])
    exp_S.eval = eval
    return exp_S


def H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, exp_S, op=None):
    """
    Computes perturbed Hamiltonian in eigenbasis of unperturbed Hamiltonian.

    H_0_AA : np.array of the unperturbed Hamiltonian of subspace AA
    H_0_BB : np.array of the unperturbed Hamiltonian of subspace BB
    H_p_AA : dictionary of perturbation terms of subspace AA
    H_p_BB : dictionary of perturbation terms of subspace BB
    H_p_AB : dictionary of perturbation terms of subspace AB
    exp_S : BlockOperatorSeries of the transformation to diagonalized Hamiltonian

    Returns:
    H_tilde : BlockOperatorSeries
    """

    if op is None:
        op = matmul

    n_infinite = exp_S.n_infinite
    H = H_from_dict(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, n_infinite)
    H.name = "H"
    exp_S_dagger = BlockOperatorSeries(
        eval=(
            lambda entry: exp_S.evaluated[entry]
            if entry[0] == entry[1]
            else -exp_S.evaluated[entry]
        ),
        data=None,
        shape=(2, 2),
        n_infinite=n_infinite,
    )
    exp_S_dagger.name = "exp_S_dagger"
    result = cauchy_dot_product(exp_S_dagger, H, exp_S, op=op, hermitian=True)
    result.name = "H_tilde"
    return result


def H_from_dict(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, n_infinite=1):
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

def exp_S_initialize(N_A, N_B, n_infinite=1):
    zero_index = (0,) * n_infinite
    exp_S = BlockOperatorSeries(
        data=
        {
            **{(0, 0) + zero_index: None},
            **{(1, 1) + zero_index: None},
            **{(0, 1) + zero_index: _zero},
            **{(1, 0) + zero_index: _zero},
        },
        shape=(2, 2),
        n_infinite=n_infinite,
    )
    exp_S.name = "exp_S"
    return exp_S

