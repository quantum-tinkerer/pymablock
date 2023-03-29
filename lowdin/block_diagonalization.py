# %%
# # The polynomial alternative to Lowdin perturbation theory
#
# See [this hackmd](https://hackmd.io/Rpt2C8oOQ2SGkGS9OYrlfQ?view) for the motivation and the expressions

# %%
from operator import matmul, mul
from functools import reduce
from copy import copy

import numpy as np
import sympy
from sympy.physics.quantum import Dagger, Operator, HermitianOperator

from lowdin.series import (
    BlockSeries,
    zero,
    one,
    cauchy_dot_product,
    _zero_sum,
)


def general(H, solve_sylvester=None, *, op=None):
    """
    Computes the block diagonalization of a BlockSeries.

    H : BlockSeries of original Hamiltonian
    solve_sylvester : (optional) function that solves the Sylvester equation
    op : (optional) function to use for matrix multiplication

    Returns:
    H_tilde : BlockSeries
    U : BlockSeries
    U_adjoint : BlockSeries
    """
    if op is None:
        op = matmul

    if solve_sylvester is None:
        solve_sylvester = _default_solve_sylvester(H)

    # Initialize the transformation as the identity operator
    U = BlockSeries(
        data={
            **{block + (0,) * H.n_infinite: one for block in ((0, 0), (1, 1))},
            **{block + (0,) * H.n_infinite: zero for block in ((0, 1), (1, 0))},
        },
        shape=(2, 2),
        n_infinite=H.n_infinite,
    )
    U_adjoint = BlockSeries(
        eval=(
            lambda index: U.evaluated[index]
            if index[0] == index[1]
            else -U.evaluated[index]
        ),
        data=None,
        shape=(2, 2),
        n_infinite=H.n_infinite,
    )

    # Uncorrected identity and H_tilde to compute U
    identity = cauchy_dot_product(
        U_adjoint, U, op=op, hermitian=True, exclude_last=[True, True]
    )
    H_tilde_rec = cauchy_dot_product(
        U_adjoint, H, U, op=op, hermitian=True, exclude_last=[True, False, True]
    )

    def eval(index):
        if index[0] == index[1]:
            # diagonal is constrained by unitarity
            return -identity.evaluated[index] / 2
        elif index[:2] == (0, 1):
            # off-diagonal block nullifies the off-diagonal part of H_tilde
            return -solve_sylvester(H_tilde_rec.evaluated[index])
        elif index[:2] == (1, 0):
            # off-diagonal of U is anti-Hermitian
            return -Dagger(U.evaluated[(0, 1) + tuple(index[2:])])

    U.eval = eval

    H_tilde = cauchy_dot_product(U_adjoint, H, U, op=op, hermitian=True)
    return H_tilde, U, U_adjoint


def _default_solve_sylvester(H):
    """
    Returns a function that divides a matrix by the difference of a diagonal unperturbed Hamiltonian.

    H : BlockSeries of original Hamiltonian

    Returns:
    solve_sylvester : function
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

    def solve_sylvester(Y):
        return Y * energy_denominators

    return solve_sylvester


def to_BlockSeries(H_0_AA=None, H_0_BB=None, H_p_AA=None, H_p_BB=None, H_p_AB=None, n_infinite=1):
    """
    TEMPORARY, WILL DELETE WHEN USER API IS READY
    Creates a BlockSeries from a dictionary of perturbation terms.

    H_0_AA : np.array of the unperturbed Hamiltonian of subspace AA
    H_0_BB : np.array of the unperturbed Hamiltonian of subspace BB
    H_p_AA : dictionary of perturbation terms of subspace AA
    H_p_BB : dictionary of perturbation terms of subspace BB
    H_p_AB : dictionary of perturbation terms of subspace AB
    n_infinite : (optional) number of infinite indices

    Returns:
    H : BlockSeries of the Hamiltonian
    """
    if H_0_AA is None:
        raise ValueError("H_0_AA must be specified")
    if H_0_BB is None:
        raise ValueError("H_0_BB must be specified")
    if H_p_AA is None:
        H_p_AA = {}
    if H_p_BB is None:
        H_p_BB = {}
    if H_p_AB is None:
        H_p_AB = {}

    zeroth_order = (0,) * n_infinite
    H = BlockSeries(
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


def _commute_H0_away(expr, H_0_AA, H_0_BB, Y_data):
    """
    Simplify expression by commmuting H_0 and V using Sylvester's Equation relations.

    expr : zero or sympy expression to simplify
    H_0_AA : np.array of the unperturbed Hamiltonian of subspace AA
    H_0_BB : np.array of the unperturbed Hamiltonian of subspace BB
    Y_data : np.array of the Y perturbation terms

    Returns:
    expr : zero or sympy expression without H_0 in it.
    """
    if zero == expr:
        return expr

    subs = {
        **{H_0_AA * V: rhs + V * H_0_BB for V, rhs in Y_data.items()},
        **{
            H_0_BB * Dagger(V): -Dagger(rhs) + Dagger(V) * H_0_AA
            for V, rhs in Y_data.items()
        },
    }

    while any(H in expr.free_symbols for H in (H_0_AA, H_0_BB)) and len(expr.free_symbols) > 1:
        expr = expr.subs(subs).expand()

    return expr or zero


def general_symbolic(initial_indices):
    """
    General symbolic algorithm for diagonalizing a Hamiltonian.

    initial_indices : list of tuples of the indices of the nonzero terms of the Hamiltonian.

    Returns:
    H_tilde_s : BlockSeries of the diagonalized Hamiltonian
    U_s : BlockSeries of the transformation to the diagonalized Hamiltonian
    U_adjoint_s : BlockSeries of the adjoint of U_s
    Y_data : dictionary of the right hand side of Sylvester Equation
    H : BlockSeries of the original Hamiltonian
    """
    initial_indices = tuple(initial_indices)
    H = BlockSeries(
        data={
            **{index: HermitianOperator(f"H_{index}") for index in initial_indices if index[0] == index[1]},
            **{index: Operator(f"H_{index}") for index in initial_indices if index[0] != index[1]},
        },
        shape=(2, 2),
        n_infinite=len(initial_indices[0]) - 2,
    )
    H_symbols = copy(H.data)
    H_0_AA = H_symbols[(0, 0) + (0,) * H.n_infinite]
    H_0_BB = H_symbols[(1, 1) + (0,) * H.n_infinite]

    H_tilde, U, U_adjoint = general(H, solve_sylvester=(lambda x: x), op=mul)

    Y_data = {}

    old_U_eval = U.eval

    def U_eval(index):
        if index[:2] == (0, 1):
            V = Operator(f"V_{{{index[2:]}}}")
            Y = _commute_H0_away(old_U_eval(index), H_0_AA, H_0_BB, Y_data)
            if zero == Y:
                return zero
            Y_data[V] = Y
            return V
        return old_U_eval(index)

    U.eval = U_eval

    old_H_tilde_eval = H_tilde.eval

    def H_tilde_eval(index):
        return _commute_H0_away(old_H_tilde_eval(index), H_0_AA, H_0_BB, Y_data)

    H_tilde.eval = H_tilde_eval

    return H_tilde, U, U_adjoint, Y_data, H_symbols


def expanded(H, solve_sylvester=None, *, op=None):
    """
    Replace specifics of the Hamiltonian in the general symbolic algorithm.

    H : BlockSeries of the Hamiltonian
    solve_sylvester : (optional) function to use for solving Sylvester's equation
    op : (optional) function to use for matrix multiplication

    Returns:
    H_tilde : BlockSeries of the diagonalized Hamiltonian
    U : BlockSeries of the transformation to the diagonalized Hamiltonian
    U_adjoint : BlockSeries of the adjoint of U
    """
    if op is None:
        op = matmul

    if solve_sylvester is None:
        solve_sylvester = _default_solve_sylvester(H)

    H_tilde_s, U_s, _, Y_data, H_symbols = general_symbolic(H.data.keys())
    _, U, U_adjoint = general(H, solve_sylvester=solve_sylvester, op=op)

    subs = {symbol: H.evaluated[index] for index, symbol in H_symbols.items()}

    def H_tilde_eval(index):
        H_tilde = H_tilde_s.evaluated[index]
        _update_subs(Y_data, subs, solve_sylvester, op)
        return _replace(H_tilde, subs, op)

    H_tilde = BlockSeries(eval=H_tilde_eval, shape=(2, 2), n_infinite=H.n_infinite)

    old_U_eval = U.eval
    def U_eval(index):
        if index[:2] == (0, 1):
            U_s.evaluated[index] # Update Y_data
            _update_subs(Y_data, subs, solve_sylvester, op)
            return subs.get(Operator(f"V_{{{index[2:]}}}"), zero)
        return old_U_eval(index)
    U.eval = U_eval

    return H_tilde, U, U_adjoint

def _update_subs(Y_data, subs, solve_sylvester, op):
    """
    Update the solutions to the Sylvester equation in subs.

    Y_data : dictionary of the right hand side of Sylvester Equation
    subs : dictionary of substitutions to make
    solve_sylvester : function to use for solving Sylvester's equation
    op : function to use for matrix multiplication
    """
    for V, rhs in Y_data.items():
        if V not in subs:
            rhs = _replace(rhs, subs, op) # No general symbols left
            subs[V] = solve_sylvester(rhs)

def _replace(expr, subs, op):
    """
    Substitute terms in an expression and multiply them accordingly.
    Numerical prefactors are factored out of the matrix multiplication.

    expr : sympy expression to replace
    subs : dictionary of substitutions to make
    op : function to use to multiply the substituted terms

    Return:
    result : zero or sympy expression with replacements such that general symbols are not present
    """
    if zero == expr:
        return expr
    subs = {
        **subs,
        **{Dagger(symbol): Dagger(expression) for symbol, expression in subs.items()},
    }

    result = []
    for term in expr.as_ordered_terms():
        if term.is_Mul:
            prefactor, term = term.as_coeff_Mul()
            numerator, denominator = sympy.fraction(prefactor)
        else:
            numerator = denominator = 1
        substituted_factors = [subs[factor] for factor in term.as_ordered_factors()]
        if any(zero == factor for factor in substituted_factors):
            continue
        result.append(int(numerator) * reduce(op, substituted_factors) / int(denominator))

    return _zero_sum(result)
