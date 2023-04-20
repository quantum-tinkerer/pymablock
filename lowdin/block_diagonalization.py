# %%
# # The polynomial alternative to Lowdin perturbation theory
#
# See [this hackmd](https://hackmd.io/Rpt2C8oOQ2SGkGS9OYrlfQ?view)
# for the motivation and the expressions

# %%
from operator import matmul, mul
from functools import reduce
from typing import Any, Optional, Callable

import numpy as np
import sympy
import qsymm
from scipy import sparse
from sympy.physics.quantum import Dagger, Operator, HermitianOperator

from lowdin.linalg import ComplementProjector, aslinearoperator
from lowdin.kpm_funcs import greens_function
from lowdin.series import (
    BlockSeries,
    zero,
    one,
    cauchy_dot_product,
    _zero_sum,
    safe_divide,
)

__all__ = ["general", "expanded", "general_symbolic", "numerical", "to_BlockSeries"]


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
            return safe_divide(-identity.evaluated[index], 2)
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
    expr :
        (zero or sympy) expression to simplify.
    H_0_AA :
        Unperturbed Hamiltonian in subspace AA.
    H_0_BB :
        Unperturbed Hamiltonian in subspace BB.
    Y_data :
        dictionary of {V: rhs} such that H_0_AA * V - V * H_0_BB = rhs.

    Returns
    -------
    expr : zero or sympy.expr
        (zero or sympy) expression without H_0_AA or H_0_BB in it.
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

    return expr.expand() or zero


def general_symbolic(
    H: BlockSeries,
) -> tuple[
    BlockSeries, BlockSeries, BlockSeries, dict[Operator, Any], dict[Operator, Any]
]:
    """
    General symbolic algorithm for diagonalizing a Hamiltonian.

    Parameters
    ----------
    H :
        The Hamiltonian. The algorithm only checks which terms are present in
        the Hamiltonian, but does not substitute them.

    Returns
    -------
    H_tilde_s : `~lowdin.series.BlockSeries`
        Symbolic diagonalized Hamiltonian.
    U_s : `~lowdin.series.BlockSeries`
        Symbolic unitary matrix that block diagonalizes H such that
        U_s * H * U_s^H = H_tilde_s.
    U_adjoint_s : `~lowdin.series.BlockSeries`
        Symbolic adjoint of U_s.
    Y_data : `dict`
        dictionary of {V: rhs} such that H_0_AA * V - V * H_0_BB = rhs.
        It is updated whenever new terms of `H_tilde_s` or `U_s` are evaluated.
    subs : `dict`
        Dictionary with placeholder symbols as keys and original Hamiltonian terms as
        values.
    """
    subs = {}
    def placeholder_eval(*index):
        if zero == (actual_value := H.evaluated[index]):
            return zero
        operator_type = HermitianOperator if index[0] == index[1] else Operator
        placeholder = operator_type(f"H_{index}")
        subs[placeholder] = actual_value
        return placeholder

    H_placeholder = BlockSeries(
        eval=placeholder_eval,
        shape=H.shape,
        n_infinite=H.n_infinite,
    )
    H_0_AA = H_placeholder.evaluated[(0, 0) + (0,) * H.n_infinite]
    H_0_BB = H_placeholder.evaluated[(1, 1) + (0,) * H.n_infinite]

    H_tilde, U, U_adjoint = general(H_placeholder, solve_sylvester=(lambda x: x), op=mul)

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

    return H_tilde, U, U_adjoint, Y_data, subs


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

    H_tilde_s, U_s, _, Y_data, subs = general_symbolic(H)
    _, U, U_adjoint = general(H, solve_sylvester=solve_sylvester, op=op)

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


def numerical(
    H: BlockSeries,
    vecs_a: np.ndarray,
    eigs_a: np.ndarray,
    vecs_b: Optional[np.ndarray] = None,
    eigs_b: Optional[np.ndarray] = None,
    kpm_params: Optional[dict] = None,
    precalculate_moments: Optional[bool] = False,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Diagonalize a Hamiltonian using the hybrid KPM algorithm.

    Parameters
    ----------
    H :
        Full Hamiltonian of the system.
    vecs_a :
        Eigenvectors of the A (effective) subspace of the known Hamiltonian.
    eigs_a :
        Eigenvalues to the aforementioned eigenvectors.
    vecs_b :
        Explicit parts of the B (auxilliary) space. Need to be eigenvectors of the
        unperturbed Hamiltonian.
    eigs_b :
        Eigenvalues to the aforementioned explicit B space eigenvectors.
    kpm_params :
        Dictionary containing the parameters to pass to the `~kwant.kpm` module.
        'num_vectors' will be overwritten to match the number of vectors, and the
        'operator' key will be deleted.
    precalculate_moments :
        Whether to precalculate and store all the KPM moments of ``vectors``. This
        is useful if the Green's function is evaluated at a large number of
        energies, but uses a large amount of memory. If False, the KPM expansion
        is performed every time the Green's function is called, which minimizes
        memory use.

    Returns
    -------
    H_tilde : `~lowdin.series.BlockSeries`
        Full block-diagonalized Hamiltonian of the problem. The ``(0, 0)`` block
        (A subspace) is a numpy array, while the ``(1, 1)`` block (B subspace)
        is a ``LinearOperator``.
    U : `~lowdin.series.BlockSeries`
        Unitary transformation that block diagonalizes the initial Hamiltonian.
    U_adjoint : `~lowdin.series.BlockSeries`
        Adjoint of ``U``.
    """
    H_input = H
    p_b = ComplementProjector(vecs_a)

    def H_eval(*index):
        original = H_input.evaluated[index[2:]]
        if zero == original:
            return zero
        if index[:2] == (0, 0):
            return Dagger(vecs_a) @ original @ vecs_a
        if index[:2] == (0, 1):
            return Dagger(vecs_a) @ original @ p_b
        if index[:2] == (1, 0):
            return Dagger(H.evaluated[(0, 1) + tuple(index[2:])])
        if index[:2] == (1, 1):
            return p_b @ aslinearoperator(original) @ p_b

    H = BlockSeries(eval=H_eval, shape=(2, 2), n_infinite=H_input.n_infinite)

    solve_sylvester = solve_sylvester_KPM(
        H_input.evaluated[(0,) * H_input.n_infinite],
        vecs_a,
        eigs_a,
        vecs_b,
        eigs_b,
        kpm_params,
        precalculate_moments,
    )
    H_tilde, U, U_adjoint = general(H, solve_sylvester=solve_sylvester)

    # Create series wrapped in linear operators to avoid forming explicit matrices
    def linear_operator_wrapped(original):
        return lambda *index: aslinearoperator(original[index])

    H_operator, U_operator, U_adjoint_operator = (
        BlockSeries(
            eval=linear_operator_wrapped(original),
            shape=(2, 2),
            n_infinite=H.n_infinite,
        )
        for original in (H.evaluated, U.evaluated, U_adjoint.evaluated)
    )
    identity = cauchy_dot_product(
        U_operator, U_adjoint_operator, hermitian=True, exclude_last=[True, True]
    )

    old_U_eval = U.eval

    def U_eval(*index):
        if index[:2] == (1, 1):
            return safe_divide(-identity.evaluated[index], 2)
        return old_U_eval(*index)

    U.eval = U_eval

    H_tilde_operator = cauchy_dot_product(
        U_adjoint_operator, H_operator, U_operator, hermitian=True
    )

    def H_tilde_eval(*index):
        if index[:2] == (1, 1):
            return H_tilde_operator.evaluated[index]
        return H_tilde.evaluated[index]

    result_H_tilde = BlockSeries(eval=H_tilde_eval, shape=(2, 2), n_infinite=H.n_infinite)

    return result_H_tilde, U, U_adjoint


def solve_sylvester_KPM(
    h_0: Any,
    vecs_a: np.ndarray,
    eigs_a: np.ndarray,
    vecs_b: Optional[np.ndarray] = None,
    eigs_b: Optional[np.ndarray] = None,
    kpm_params: Optional[dict] = None,
    precalculate_moments: Optional[bool] = False,
) -> Callable:
    """
    Solve Sylvester energy division for KPM.

    General energy division for numerical problems through either full knowledge of
    the B-space or application of the KPM Green's function.

    Parameters
    ----------
    h_0 :
        Unperturbed Hamiltonian of the system.
    vecs_a :
        Eigenvectors of the A (effective) subspace of the known Hamiltonian.
    eigs_a :
        Eigenvalues to the aforementioned eigenvectors.
    vecs_b :
        Explicit parts of the B (auxilliary) space. Need to be eigenvectors of the
        unperturbed Hamiltonian.
    eigs_b :
        Eigenvalues to the aforementioned explicit B space eigenvectors.
    kpm_params :
        Dictionary containing the parameters to pass to the `~kwant.kpm` module.
        'num_vectors' will be overwritten to match the number of vectors, and the
        'operator' key will be deleted.
    precalculate_moments :
        Whether to precalculate and store all the KPM moments of ``vectors``. This
        is useful if the Green's function is evaluated at a large number of
        energies, but uses a large amount of memory. If False, the KPM expansion
        is performed every time the Green's function is called, which minimizes
        memory use.

    Returns
    ----------
    solve_sylvester: callable
        Function that applies divide by energies to the RHS of the Sylvester equation.
    """
    if vecs_b is None:
        vecs_b = np.empty((vecs_a.shape[0], 0))
    if eigs_b is None:
        eigs_b = np.diag(Dagger(vecs_b) @ h_0 @ vecs_b)
    if kpm_params is None:
        kpm_params = dict()

    need_kpm = len(eigs_a) + len(eigs_b) < h_0.shape[0]
    need_explicit = bool(len(eigs_b))
    if not any((need_kpm, need_explicit)):
        # B subspace is empty
        return lambda Y: Y

    if need_kpm:
        kpm_projector = ComplementProjector(np.concatenate((vecs_a, vecs_b), axis=-1))

        def sylvester_kpm(Y: np.ndarray) -> np.ndarray:
            Y_KPM = Y @ kpm_projector
            vec_G_Y = greens_function(
                h_0,
                params=None,
                vectors=Y_KPM.conj(),
                kpm_params=kpm_params,
                precalculate_moments=precalculate_moments,
            )(eigs_a)
            return np.vstack([vec_G_Y.conj()[:, m, m] for m in range(len(eigs_a))])

    if need_explicit:
        G_ml = 1 / (eigs_a[:, None] - eigs_b[None, :])

        def sylvester_explicit(Y: np.ndarray) -> np.ndarray:
            return ((Y @ vecs_b) * G_ml) @ vecs_b.conj().T

    def solve_sylvester(Y: np.ndarray) -> np.ndarray:
        if need_kpm and need_explicit:
            result = sylvester_kpm(Y) + sylvester_explicit(Y)
        elif need_kpm:
            result = sylvester_kpm(Y)
        elif need_explicit:
            result = sylvester_explicit(Y)

        return result

    return solve_sylvester


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
    Y_data :
        dictionary of {V: rhs} such that H_0_AA * V - V * H_0_BB = rhs.
    subs :
        dictionary of substitutions to make.
    solve_sylvester :
        function to use for solving Sylvester's equation.
    op :
        function to use for matrix multiplication.
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
    expr :
        (zero or sympy) expression in which to replace general symbols.
    subs :
        dictionary {symbol: value} of substitutions to make.
    op :
        function to use to multiply the substituted terms.

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
            safe_divide(
                int(numerator) * reduce(op, substituted_factors), int(denominator)
            )
        )

    return _zero_sum(result)


def block_diagonalize(
        hamiltonian :  list[Any, list] | dict | BlockSeries,
        *,
        algorithm : Optional[Callable] = None,
        solve_sylvester : Optional[Callable] = None,
        subspaces: Optional[tuple[Any, Any]] = None,
        subspaces_indices: Optional[tuple[int, ...]] = None,
        **kwargs
    ) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Parameters
    ----------
    hamiltonian :
        Hamiltonian to diagonalize.

    algorithm :
        Function that block diagonalizes a Hamiltonian.
        Options are `~lowdin.block_diagonalize.general` and
        `~lowdin.block_diagonalize.expanded`.
    subspaces:
        Subspaces to use for block diagonalization.

    Returns
    -------
    H_tilde : `~lowdin.series.BlockSeries`
        Block diagonalized Hamiltonian.
    U : `~lowdin.series.BlockSeries`
        Unitary matrix that block diagonalizes H such that U * H * U^H = H_tilde.
    U_adjoint : `~lowdin.series.BlockSeries`
        Adjoint of U.
    """
    if algorithm is None:
        algorithm = general

    H = hamiltonian_to_BlockSeries(
        hamiltonian,
        subspaces=subspaces,
        subspaces_indices=subspaces_indices,
    )

    # Determine operator to use for matrix multiplication
    if hasattr(H.evaluated[(0,) * H.n_infinite], '__matmul__'):
        op = matmul
    else:
        op = mul

    # Determine function to use for solving Sylvester's equation
    # If H_0 is diagonal, don't provide anything
    # If H_0 is not diagonal, provide a function that solves Sylvester's equation
    if solve_sylvester is None:
        if np.allclose(
            H.evaluated[(0, 0) + (0,) * H.n_infinite],
            np.diag.np.diag(H.evaluated[(0, 0) + (0,) * H.n_infinite])
        ) and np.allclose(
            H.evaluated[(1, 1) + (0,) * H.n_infinite],
            np.diag.np.diag(H.evaluated[(1, 1) + (0,) * H.n_infinite])
        ):
            pass
        else:
            NotImplementedError
    return algorithm(H, solve_sylvester=solve_sylvester, op=op, **kwargs)


def _list_to_dict(hamiltonian: list[Any]) -> dict[int, Any]:
    """
    Parameters
    ----------
    hamiltonian :
        Unperturbed Hamiltonian and 1st order perturbations.

    Returns
    -------
    H : `~lowdin.series.BlockSeries`
    """
    # [H_0, H_1, H_2, ...], 1st order perturbations only
    n_infinite = len(hamiltonian) - 1
    zeroth_order = (0,) * n_infinite

    hamiltonian = {
        zeroth_order: hamiltonian.pop(0),
        **{
            tuple(order): perturbation for order, perturbation in
            zip(np.eye(n_infinite, dtype=int), hamiltonian)
        }
    }
    return hamiltonian


def _dict_to_BlockSeries(hamiltonian: dict[tuple[int, ...], Any]) -> BlockSeries:
    """

    Parameters
    ----------
    hamiltonian :
        Unperturbed Hamiltonian and perturbations.
        The keys are tuples of integers that indicate the order of the perturbation.

    Returns
    -------
    H : `~lowdin.series.BlockSeries`
    """
    # {0: H_0, (1, 0): H_1, (0, 1): H_2, ...}
    n_infinite = len(list(hamiltonian.keys())[0])
    H_temporary = BlockSeries(
        data=hamiltonian,
        shape=(),
        n_infinite=n_infinite,
    )
    return H_temporary


def _qsymm_to_dict(hamiltonian):
    # TODO: Implement by requiring list of perturbative symbols.
    raise NotImplementedError


def _get_subspaces_from_indices(
        subspaces_indices: tuple[int, ...],
    ) -> tuple[Any, Any]:
    """
    
    Parameters
    ----------
    subspaces_indices :
        Indices of the subspaces.
        0 indicates the first subspace, 1 indicates the second subspace.

    Returns
    -------
    subspaces :
        Subspaces to use for block diagonalization. 
    """

    # H_0 is a diagonal array, subspaces always needed
    subspaces_indices = np.array(subspaces_indices)
    max_subspaces = 2
    if np.any(subspaces_indices >= max_subspaces):
        raise ValueError(
            "Only 0 and 1 are allowed as indices for subspaces."
    )
    dim = len(subspaces_indices)
    eigvecs = np.eye(dim, dtype=complex)
    subspaces = tuple(
        eigvecs[:, np.compress(subspaces_indices==block, np.arange(dim))]
        for block in range(max_subspaces)
    )
    return subspaces


def hamiltonian_to_BlockSeries(
        hamiltonian: list[Any, list] | dict | BlockSeries,
        *,
        subspaces: Optional[tuple[Any, Any]] = None,
        subspaces_indices: Optional[tuple[int, ...]] = None,
    ) -> BlockSeries:
    """
    # TODO: change the name once to_BlockSeries is removed
    Converts a Hamiltonian to a `~lowdin.series.BlockSeries`.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian to convert.
        If a list, it is assumed to be of the form
        [H_0, [H_1, H_2, ...]] where H_0 is the zeroth order Hamiltonian and H_1, H_2, ...
        are the first order perturbations.
    subspaces :
        Tuple of eigenvectors of each subspace of the Hamiltonian.

    Returns
    -------
    H : `~lowdin.series.BlockSeries`
        Hamiltonian in the format required by algorithms.
    """
    if isinstance(hamiltonian, list):
        hamiltonian = _list_to_dict(hamiltonian)
    elif isinstance(hamiltonian, qsymm.Model):
        hamiltonian = _qsymm_to_dict(hamiltonian)
    if isinstance(hamiltonian, dict):
        H_temporary = _dict_to_BlockSeries(hamiltonian)
    elif isinstance(hamiltonian, BlockSeries):
        H_temporary = hamiltonian
    else:
        raise NotImplementedError

    if subspaces_indices is not None:
        if subspaces is not None:
            raise ValueError("Only subspaces or subspaces_indices can be provided.")
        if not np.allclose(
            H_temporary.evaluated[(0,) * H_temporary.n_infinite],
            np.diag(np.diag(H_temporary.evaluated[(0,) * H_temporary.n_infinite]))
        ):
            raise ValueError("If subspaces_indices is provided, H_0 must be diagonal.")
        subspaces = _get_subspaces_from_indices(subspaces_indices)

    if subspaces is None:
        if H_temporary.shape == ():
            H = BlockSeries(
                eval=(
                    lambda *index: H_temporary.evaluated[index[:2]][index[0]][index[1]]
                ),
                shape=(2, 2),
                n_infinite=H_temporary.n_infinite,
            )
        elif H_temporary.shape == (2, 2):
            H = H_temporary
        else:
            raise NotImplementedError
    else:
        # TODO: Implement this for non-numerical inputs and KPM
        # This only works for numpy arrays so far
        eigvecs_A, eigvecs_B = subspaces[0], subspaces[1]
        if H_temporary.shape==():
            def H_eval(*index):
                if index[:2] == (0, 0):
                    return eigvecs_A.T.conj() @ H_temporary.evaluated[index[2:]] @ eigvecs_A
                elif index[:2] == (1, 1):
                    return eigvecs_B.T.conj() @ H_temporary.evaluated[index[2:]] @ eigvecs_B
                elif index[:2] == (0, 1):
                    return eigvecs_A.T.conj() @ H_temporary.evaluated[index[2:]] @ eigvecs_B
                else:
                    return Dagger(H.evaluated[(0, 1) + tuple(index[2:])])

            H = BlockSeries(
                eval=H_eval,
                shape=(2, 2),
                n_infinite=H_temporary.n_infinite,
            )

        elif H_temporary.shape == (2, 2):
            raise ValueError("Blocks are already separated. Do not provide subspaces.")
    return H