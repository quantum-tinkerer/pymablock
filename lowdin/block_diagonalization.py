# %%
# # The polynomial alternative to Lowdin perturbation theory
#
# See [this hackmd](https://hackmd.io/Rpt2C8oOQ2SGkGS9OYrlfQ?view)
# for the motivation and the expressions

# %%
from operator import matmul, mul
from functools import reduce
from copy import copy
from typing import Any, Optional, Callable

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

__all__ = ["general", "expanded", "general_symbolic", "to_BlockSeries"]


def general(
    H: BlockSeries,
    solve_sylvester: Optional[Callable] = None,
    *,
    op: Optional[Callable] = None,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Computes the block diagonalization of a Hamiltonian.

    Parameters
    ----------
    H :
        Initial Hamiltonian, unperturbed and perturbation.
    solve_sylvester :
        (optional) function that solves the Sylvester equation.
        Defaults to a function that works for numerical diagonal unperturbed Hamiltonians.
    op :
        (optional) function to use for matrix multiplication. Defaults to matmul.

    Returns
    -------
    H_tilde : `~lowdin.series.BlockSeries`
        Block diagonalized Hamiltonian.
    U : `~lowdin.series.BlockSeries`
        Unitary transformation that block diagonalizes H such that H_tilde = U^H H U.
    U_adjoint : `~lowdin.series.BlockSeries`
        Adjoint of U.
    """
    if op is None:
        op = matmul

    if solve_sylvester is None:
        solve_sylvester = _default_solve_sylvester(H)

    # Initialize the transformation as the identity operator
    U = BlockSeries(
        data={block + (0,) * H.n_infinite: one for block in ((0, 0), (1, 1))},
        shape=(2, 2),
        n_infinite=H.n_infinite,
    )

    U_adjoint = BlockSeries(
        eval=(
            lambda *index: U.evaluated[index]  # diagonal block is Hermitian
            if index[0] == index[1]
            else -U.evaluated[index]  # off-diagonal block is anti-Hermitian
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

    def eval(*index: tuple[int, ...]) -> Any:
        if index[0] == index[1]:
            # diagonal is constrained by unitarity
            return -identity.evaluated[index] / 2
        elif index[:2] == (0, 1):
            # off-diagonal block nullifies the off-diagonal part of H_tilde
            Y = H_tilde_rec.evaluated[index]
            return -solve_sylvester(Y) if zero != Y else zero
        elif index[:2] == (1, 0):
            # off-diagonal of U is anti-Hermitian
            return -Dagger(U.evaluated[(0, 1) + tuple(index[2:])])

    U.eval = eval

    H_tilde = cauchy_dot_product(U_adjoint, H, U, op=op, hermitian=True)
    return H_tilde, U, U_adjoint


def _default_solve_sylvester(H: BlockSeries) -> Callable:
    """
    Returns a function that divides a matrix by the difference
    of a numerical diagonal unperturbed Hamiltonian.

    Parameters
    ----------
    H : Initial Hamiltonian, unperturbed and perturbation.

    Returns
    -------:
    solve_sylvester : Function that solves the Sylvester equation.
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

    def solve_sylvester(Y: Any) -> Any:
        return Y * energy_denominators

    return solve_sylvester


def to_BlockSeries(
    H_0_AA: Any,
    H_0_BB: Any,
    H_p_AA: Optional[dict[tuple[int, ...], Any]] = None,
    H_p_BB: Optional[dict[tuple[int, ...], Any]] = None,
    H_p_AB: Optional[dict[tuple[int, ...], Any]] = None,
    n_infinite: int = 1,
) -> BlockSeries:
    """
    TEMPORARY, WILL DELETE WHEN USER API IS READY
    Creates a BlockSeries from a dictionary of perturbation terms.

    Parameters
    ----------
    H_0_AA :
        Unperturbed Hamiltonian of subspace AA
    H_0_BB :
        Unperturbed Hamiltonian of subspace BB
    H_p_AA :
        dictionary of perturbation terms of subspace AA
    H_p_BB :
        dictionary of perturbation terms of subspace BB
    H_p_AB :
        dictionary of perturbation terms of subspace AB
    n_infinite :
        (optional) number of infinite indices

    Returns
    -------
    H : `~lowdin.series.BlockSeries`
        BlockSeries of the Hamiltonian
    """
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


def _commute_H0_away(
    expr: Any, H_0_AA: Any, H_0_BB: Any, Y_data: dict[Operator, Any]
) -> Any:
    """
    Simplify expression by commmuting H_0 and V using Sylvester's Equation relations.

    Parameters
    ----------
    expr : (zero or sympy) expression to simplify.
    H_0_AA : Unperturbed Hamiltonian in subspace AA.
    H_0_BB : Unperturbed Hamiltonian in subspace BB.
    Y_data : dictionary of {V: rhs} such that H_0_AA * V - V * H_0_BB = rhs.

    Returns
    -------
    expr : (zero or sympy) expression without H_0_AA or H_0_BB in it.
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

    while (
        any(H in expr.free_symbols for H in (H_0_AA, H_0_BB))
        and len(expr.free_symbols) > 1
    ):
        expr = expr.subs(subs).expand()

    return expr or zero


def general_symbolic(
    initial_indices: list[tuple[int, ...]],
) -> tuple[BlockSeries, BlockSeries, BlockSeries, dict[Operator, Any], BlockSeries]:
    """
    General symbolic algorithm for diagonalizing a Hamiltonian.

    Parameters
    ----------
    initial_indices :
        indices of nonzero terms of the Hamiltonian to be diagonalized.

    Returns
    -------
    H_tilde_s : `~lowdin.series.BlockSeries`
        Symbolic diagonalized Hamiltonian.
    U_s : `~lowdin.series.BlockSeries`
        Symbolic unitary matrix that block diagonalizes H such that
        U_s * H * U_s^H = H_tilde_s.
    U_adjoint_s : `~lowdin.series.BlockSeries`
        Symbolic adjoint of U_s.
    Y_data : dict
        dictionary of {V: rhs} such that H_0_AA * V - V * H_0_BB = rhs.
        It is updated whenever new terms of `H_tilde_s` or `U_s` are evaluated.
    H : `~lowdin.series.BlockSeries`
        Symbolic initial Hamiltonian, unperturbed and perturbation.
    """
    initial_indices = tuple(initial_indices)
    H = BlockSeries(
        data={
            **{
                index: HermitianOperator(f"H_{index}")
                for index in initial_indices
                if index[0] == index[1]
            },
            **{
                index: Operator(f"H_{index}")
                for index in initial_indices
                if index[0] != index[1]
            },
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

    def U_eval(*index):
        if index[:2] == (0, 1):
            V = Operator(f"V_{{{index[2:]}}}")
            Y = _commute_H0_away(old_U_eval(*index), H_0_AA, H_0_BB, Y_data)
            if zero == Y:
                return zero
            Y_data[V] = Y
            return V
        return old_U_eval(*index)

    U.eval = U_eval

    old_H_tilde_eval = H_tilde.eval

    def H_tilde_eval(*index):
        return _commute_H0_away(old_H_tilde_eval(*index), H_0_AA, H_0_BB, Y_data)

    H_tilde.eval = H_tilde_eval

    return H_tilde, U, U_adjoint, Y_data, H_symbols


def expanded(
    H: BlockSeries,
    solve_sylvester: Optional[Callable] = None,
    *,
    op: Optional[Callable] = None,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Diagonalize a Hamiltonian using the general_symbolic algorithm and
    replacing the inputs.

    Parameters
    ----------
    H :
        Initial Hamiltonian, unperturbed and perturbation.
    solve_sylvester :
        Function to use for solving Sylvester's equation.
    op :
        Function to use for matrix multiplication.

    Returns
    -------
    H_tilde : `~lowdin.series.BlockSeries`
        Diagonalized Hamiltonian.
    U : `~lowdin.series.BlockSeries`
        Unitary matrix that block diagonalizes H such that U * H * U^H = H_tilde.
    U_adjoint : `~lowdin.series.BlockSeries`
        Adjoint of U.
    """
    if op is None:
        op = matmul

    if solve_sylvester is None:
        solve_sylvester = _default_solve_sylvester(H)

    H_tilde_s, U_s, _, Y_data, H_symbols = general_symbolic(H.data.keys())
    _, U, U_adjoint = general(H, solve_sylvester=solve_sylvester, op=op)

    subs = {symbol: H.evaluated[index] for index, symbol in H_symbols.items()}

    def H_tilde_eval(*index):
        H_tilde = H_tilde_s.evaluated[index]
        _update_subs(Y_data, subs, solve_sylvester, op)
        return _replace(H_tilde, subs, op)

    H_tilde = BlockSeries(eval=H_tilde_eval, shape=(2, 2), n_infinite=H.n_infinite)

    old_U_eval = U.eval

    def U_eval(*index):
        if index[:2] == (0, 1):
            U_s.evaluated[index]  # Update Y_data
            _update_subs(Y_data, subs, solve_sylvester, op)
            return subs.get(Operator(f"V_{{{index[2:]}}}"), zero)
        return old_U_eval(*index)

    U.eval = U_eval

    return H_tilde, U, U_adjoint


def _update_subs(
    Y_data: dict[Operator, Any],
    subs: dict[Operator | HermitianOperator, Any],
    solve_sylvester: Callable,
    op: Callable,
) -> None:
    """
    Store the solutions to the Sylvester equation in subs.

    Parameters
    ----------
    Y_data : dictionary of {V: rhs} such that H_0_AA * V - V * H_0_BB = rhs.
    subs : dictionary of substitutions to make.
    solve_sylvester : function to use for solving Sylvester's equation.
    op : function to use for matrix multiplication.
    """
    for V, rhs in Y_data.items():
        if V not in subs:
            rhs = _replace(rhs, subs, op)  # No general symbols left
            subs[V] = solve_sylvester(rhs)


def _replace(
    expr: Any, subs: dict[Operator | HermitianOperator, Any], op: Callable
) -> Any:
    """
    Substitute terms in an expression and multiply them accordingly.
    Numerical prefactors are factored out of the matrix multiplication.

    Parameters
    ----------
    expr : (zero or sympy) expression in which to replace general symbols.
    subs : dictionary {symbol: value} of substitutions to make.
    op : function to use to multiply the substituted terms.

    Return
    ------
    zero or expr with replacements such that general symbols are not present.
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
        result.append(
            int(numerator) * reduce(op, substituted_factors) / int(denominator)
        )

    return _zero_sum(result)
