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


def general(H, divide_by_energies=None, *, op=None):
    """
    Computes the block diagonalization of a BlockOperatorSeries.

    H : BlockOperatorSeries
    divide_by_energies : (optional) function to use for dividing energies
    op : (optional) function to use for matrix multiplication

    Returns:
    H_tilde : BlockOperatorSeries
    U : BlockOperatorSeries
    U_adjoint : BlockOperatorSeries
    """
    if op is None:
        op = matmul

    if divide_by_energies is None:
        divide_by_energies = _default_divide_by_energies(H)

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
            result = -divide_by_energies(H_tilde_rec.evaluated[index])
            if isinstance(result.all(), int) or isinstance(result.all(), float):
                if result == 0: # Dagger fails on 0
                    return _zero
            return -divide_by_energies(H_tilde_rec.evaluated[index])
        elif index[:2] == (1, 0):  # off-diagonal block
            return -Dagger(U.evaluated[(0, 1) + tuple(index[2:])])

    U.eval = eval

    H_tilde = cauchy_dot_product(
        U_adjoint, H, U, op=op, hermitian=True, recursive=False
    )
    return H_tilde, U, U_adjoint


def _default_divide_by_energies(H):
    """
    Returns a function that divides a matrix by the difference of its diagonal elements.

    H : BlockOperatorSeries

    Returns:
    divide_by_energies : function
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

    def divide_by_energies(Y):
        return Y * energy_denominators

    return divide_by_energies


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


def to_BlockOperatorSeries(H_0_AA=None, H_0_BB=None, H_p_AA=None, H_p_BB=None, H_p_AB=None, n_infinite=1):
    """
    TEMPORARY, WILL DELETE WHEN USER API IS READY
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
    if H_0_AA is None:
        H_0_AA = _zero
    if H_0_BB is None:
        H_0_BB = _zero
    if H_p_AA is None:
        H_p_AA = {}
    if H_p_BB is None:
        H_p_BB = {}
    if H_p_AB is None:
        H_p_AB = {}

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


def _commute_H0_away(expr, H_0_AA, H_0_BB, Y_data, n_times):
    """
    Simplify expression by commmuting H_0 and V using Sylvester's Equation relations.

    expr : sympy expression to simplify
    H_0_AA : np.array of the unperturbed Hamiltonian of subspace AA
    H_0_BB : np.array of the unperturbed Hamiltonian of subspace BB
    Y_data : np.array of the Y perturbation terms

    Returns:
    expr : sympy expression
    """
    subs = {
        **{H_0_AA * V: rhs + V * H_0_BB for V, rhs in Y_data.items()},
        **{
            H_0_BB * Dagger(V): -Dagger(rhs) + Dagger(V) * H_0_AA
            for V, rhs in Y_data.items()
        },
    }
    if _zero == expr:
        return expr

    while (H_0_AA in expr.free_symbols and H_0_BB in expr.free_symbols):
        expr = sympy.expand(expr.subs(subs))
    return expr


def general_symbolic(n_infinite=1):
    """
    General symbolic algorithm for diagonalizing a Hamiltonian.

    Returns:
    H_tilde_s : BlockOperatorSeries
    U_s : BlockOperatorSeries
    U_adjoint_s : BlockOperatorSeries
    """
    H_0_AA = HermitianOperator("{H_{(0,)}^{AA}}")
    H_0_BB = HermitianOperator("{H_{(0,)}^{BB}}")

    H_p_BB = {(1,): HermitianOperator("{H_{(1,)}^{BB}}")}
    H_p_AA = {(1,): HermitianOperator("{H_{(1,)}^{AA}}")}
    H_p_AB = {(1,): Operator("{H_{(1,)}^{AB}}")}

    H = to_BlockOperatorSeries(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, n_infinite)

    H_tilde, U, U_adjoint = general(H, divide_by_energies=(lambda x: x), op=mul)

    Y_data = {}

    old_U_eval = U.eval

    def U_eval(index):
        if index[:2] == (0, 1):
            V = Operator(f"V_{{{index[2:]}}}")
            Y_data[V] = _commute_H0_away(old_U_eval(index), H_0_AA, H_0_BB, Y_data, np.max(index[2:]))
            return V
        return old_U_eval(index)

    U.eval = U_eval

    old_H_tilde_eval = H_tilde.eval

    def H_eval(index):
        return _commute_H0_away(old_H_tilde_eval(index), H_0_AA, H_0_BB, Y_data, np.max(index[2:]))

    H_tilde.eval = H_eval

    return H_tilde, U, U_adjoint, Y_data, H


def expand(H, divide_by_energies=None, *, op=None):
    """
    Replace specifics of the Hamiltonian in the general symbolic algorithm.

    H : BlockOperatorSeries
    divide_by_energies : (optional) function to use for dividing energies
    op : (optional) function to use for matrix multiplication

    Returns:
    H_tilde : BlockOperatorSeries
    U : BlockOperatorSeries
    U_adjoint : BlockOperatorSeries
    """
    if op is None:
        op = matmul

    if divide_by_energies is None:
        divide_by_energies = _default_divide_by_energies(H)

    zero_orders = list(H.data.keys())
    # Solve completely symbolic problem first
    H_tilde_s, U_s, U_adjoint_s, Y_data, H_s = general_symbolic(H.n_infinite)
    H_tilde, U, U_adjoint = general(H, divide_by_energies=divide_by_energies, op=op)

    def eval(index):
        H_tilde = H_tilde_s.evaluated[index]
        for V, rhs in Y_data.items():
            while any(V in rhs.free_symbols for V in Y_data.keys()):
                rhs = rhs.subs({key: divide_by_energies(value) for key, value in Y_data.items()}).expand()
                Y_data.update({V: rhs})
        H_tilde = H_tilde.subs({V: divide_by_energies(rhs) for V, rhs in Y_data.items()})
        H_tilde = H_tilde.subs({H_s.evaluated[id]: H.evaluated[id] for id in zero_orders})
        return H_tilde.expand()

    H_tilde.eval = eval

    return H_tilde, U, U_adjoint