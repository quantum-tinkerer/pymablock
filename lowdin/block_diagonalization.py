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
from scipy import sparse
from sympy.physics.quantum import Dagger, Operator, HermitianOperator

from lowdin.linalg import ComplementProjector, aslinearoperator, is_diagonal
from lowdin.kpm_funcs import greens_function
from lowdin.series import (
    BlockSeries,
    zero,
    one,
    cauchy_dot_product,
    _zero_sum,
    safe_divide,
)

__all__ = ["general", "expanded", "general_symbolic", "numerical"]


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
        h_0_AA = H.evaluated[(0, 0) + (0,) * H.n_infinite]
        h_0_BB = H.evaluated[(1, 1) + (0,) * H.n_infinite]
        if not is_diagonal(h_0_AA) or not is_diagonal(h_0_BB):
            raise ValueError(
                "The unperturbed Hamiltonian must be diagonal if solve_sylvester"
                 " is not provided."
            )

        eigs_a = h_0_AA.diagonal()
        eigs_b = h_0_BB.diagonal()
        solve_sylvester = _solve_sylvester_diagonal(eigs_a, eigs_b)

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


def _solve_sylvester_diagonal(
        eigs_a: Any,
        eigs_b: Any,
        vecs_b: Optional[np.ndarray] = None
    ) -> Callable:
    """
    Returns a function that divides a matrix by the difference
    of a numerical diagonal unperturbed Hamiltonian.

    Parameters
    ----------
    eigs_a :
        Eigenvalues of A subspace of the unperturbed Hamiltonian.
    eigs_b :
        Eigenvalues of B subspace of the unperturbed Hamiltonian.
    vecs_b :
        (optional) Eigenvectors of B subspace of the unperturbed Hamiltonian.

    Returns
    -------:
    solve_sylvester : Function that solves the Sylvester equation.
    """
    def solve_sylvester(
            Y: np.ndarray | sparse.csr_array
    ) -> np.ndarray | sparse.csr_array:
        if vecs_b is not None:
            energy_denominators = 1 / (eigs_a[:, None] - eigs_b[None, :])
            return ((Y @ vecs_b) * energy_denominators) @ Dagger(vecs_b)
        elif isinstance(Y, np.ndarray):
            energy_denominators = 1/ (eigs_a.reshape(-1, 1) - eigs_b)
            return Y * energy_denominators
        elif sparse.issparse(Y):
            Y_coo = Y.tocoo()
            energy_denominators = eigs_a[Y_coo.row] - eigs_b[Y_coo.col]
            new_data = Y_coo.data / energy_denominators
            return sparse.csr_array((new_data, (Y_coo.row, Y_coo.col)), Y_coo.shape)

    return solve_sylvester


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
        h_0_AA = H.evaluated[(0, 0) + (0,) * H.n_infinite]
        h_0_BB = H.evaluated[(1, 1) + (0,) * H.n_infinite]
        if not is_diagonal(h_0_AA) or not is_diagonal(h_0_BB):
            raise ValueError(
                "The unperturbed Hamiltonian must be diagonal if solve_sylvester"
                 " is not provided."
            )

        eigs_a = h_0_AA.diagonal()
        eigs_b = h_0_BB.diagonal()
        solve_sylvester = _solve_sylvester_diagonal(eigs_a, eigs_b)

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
    solve_sylvester: Callable,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Diagonalize a Hamiltonian using the hybrid KPM algorithm.

    Parameters
    ----------
    H :
        Full Hamiltonian of the system.
    solve_sylvester :
        Function to use for solving Sylvester's equation.

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
    subspace_vectors: tuple[Any, Any],
    eigenvalues: np.ndarray,
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
    subspace_vectors :
        Subspaces to project the unperturbed Hamiltonian and separate it into blocks.
        The first element of the tuple contains the effective subspace,
        and the second element contains the (partial) auxiliary subspace.
    eigenvalues :
        Eigenvalues of the unperturbed Hamiltonian. The first element of the tuple
        contains the full eigenvalues of the effective subspace. The second element is
        optional, and it contains the (partial) eigenvalues of the auxiliary subspace.
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
    if len(subspace_vectors) == 2:
        vecs_a, vecs_b = subspace_vectors
        eigs_a, eigs_b = eigenvalues

    elif len(subspace_vectors) == 1:
        vecs_a = subspace_vectors[0]
        eigs_a = eigenvalues[0]
        vecs_b = np.empty((vecs_a.shape[0], 0))
        eigs_b = np.diagonal((Dagger(vecs_b) @ h_0 @ vecs_b))
    else:
        raise NotImplementedError("Too many subspace_vectors")

    if not isinstance(eigs_a, np.ndarray) or not isinstance(eigs_b, np.ndarray):
        raise TypeError("Eigenvalues must be a numpy array")

    if kpm_params is None:
        kpm_params = dict()

    need_kpm = len(eigs_a) + len(eigs_b) < h_0.shape[0]
    need_explicit = bool(len(eigs_b))
    if not any((need_kpm, need_explicit)):
        # B subspace is empty
        return lambda Y: Y

    if need_kpm:
        kpm_projector = ComplementProjector(np.hstack([vecs_a, vecs_b]))

        def solve_sylvester_kpm(Y: np.ndarray)-> np.ndarray:
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
        solve_sylvester_explicit = _solve_sylvester_diagonal(eigs_a, eigs_b, vecs_b)

    def solve_sylvester(Y: np.ndarray)-> np.ndarray:
        if need_kpm and need_explicit:
            result = solve_sylvester_kpm(Y) + solve_sylvester_explicit(Y)
        elif need_kpm:
            result = solve_sylvester_kpm(Y)
        elif need_explicit:
            result = solve_sylvester_explicit(Y)

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
        algorithm : Optional[str] = None,
        solve_sylvester : Optional[Callable] = None,
        subspace_vectors: Optional[tuple[Any, Any]] = None,
        subspace_indices: Optional[tuple[int, ...]] = None,
        eigenvalues: Optional[tuple[np.ndarray, np.ndarray]] = None,
        kpm_params: Optional[dict] = None,
        precalculate_moments: Optional[bool] = False,
        symbols: list[sympy.Symbol] = None,
    ) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Parameters
    ----------
    hamiltonian :
        Hamiltonian to diagonalize. Several formats are accepted:
        - `~lowdin.series.BlockSeries` object.
        - `dict` of {index: matrix} where index is a tuple of integers and
            matrix is a `~numpy.ndarray`.
        - `list` of [unperturbed, perturbation] where unperturbed and
            perturbation are `~numpy.ndarray`.
    algorithm :
        Name of the function that block diagonalizes a Hamiltonian.
        Options are "general", "expanded", "numerical", and "general_symbolic".
    solve_sylvester :
        Function to use for solving Sylvester's equation.
        If None, the default function is used for a diagonal Hamiltonian.
    subspace_vectors :
        Subspaces to project the unperturbed Hamiltonian and separate it into
        blocks. The first element of the tuple contains the orthonormal vectors
        that span the AA subspace, while the second element contains the
        orthonormal vectors that span the BB subspace.
        If None, the unperturbed Hamiltonian must be block diagonal.
        For KPM, the first element contains the effective subspace, and the
        second element contains the (partial) auxiliary subspace.
        Mutually exclusive with `subspace_indices`.
    subspace_indices :
        If the unperturbed Hamiltonian is diagonal, the indices that label the
        diagonal elements according to the subspace_vectors may be provided.
        Indices 0 and 1 are reserved for the AA and BB subspaces, respectively.
        Mutually exclusive with `subspace_vectors`.
    eigenvalues :
        Eigenvalues of the unperturbed Hamiltonian. The first element of the
        tuple contains the full eigenvalues of the effective subspace. The
        second element is optional, and it contains the (partial) eigenvalues
        of the auxiliary subspace. This argument is needed for KPM.
    kpm_params :
        Dictionary containing the parameters to pass to the `~kwant.kpm` module.
        'num_vectors' will be overwritten to match the number of vectors, and
        the 'operator' key will be deleted.
    precalculate_moments :
        Whether to precalculate and store all the KPM moments of ``vectors``.
        This is useful if the Green's function is evaluated at a large number
        of energies, but uses a large amount of memory. If False, the KPM
        expansion is performed every time the Green's function is called, which
        minimizes memory use.
    perturbatitve_parameters :
        List of symbols that label the perturbative parameters. The order of
        the symbols will be used to determine the indices of the Hamiltonian.
        If None, the perturbative parameters are taken from the unperturbed
        Hamiltonian.

    Returns
    -------
    H_tilde : `~lowdin.series.BlockSeries`
        Block diagonalized Hamiltonian.
    U : `~lowdin.series.BlockSeries`
        Unitary matrix that block diagonalizes H such that U * H * U^H = H_tilde.
    U_adjoint : `~lowdin.series.BlockSeries`
        Adjoint of U.
    """
    implicit = False
    if algorithm is None or algorithm == numerical:
        if eigenvalues is not None and subspace_vectors is None:
            raise ValueError("subspace_vectors must be provided if"
                                " eigenvalues is provided.")
        elif eigenvalues is not None and subspace_vectors is not None:
            algorithm = "numerical"
            implicit = True
            if isinstance(hamiltonian, list):
                h_0 = hamiltonian[0]
            elif isinstance(hamiltonian, dict):
                n_infinite = len(list(hamiltonian.keys())[0])
                h_0 = hamiltonian[(0,) * n_infinite]
        elif symbols is not None: #this could be none if H is already symbolic
            algorithm = "expanded"
        else:
            algorithm = "general"

    H = hamiltonian_to_BlockSeries(
        hamiltonian,
        subspace_vectors=subspace_vectors,
        subspace_indices=subspace_indices,
        implicit = implicit,
        symbols=symbols,
    )

    # Determine operator to use for matrix multiplication
    if hasattr(H.evaluated[(0, 0) + (0,) * H.n_infinite], '__matmul__'):
        op = matmul
    else:
        op = mul

    # Determine function to use for solving Sylvester's equation
    if solve_sylvester is None:
        h_0_AA = H.evaluated[(0, 0) + (0,) * H.n_infinite]
        h_0_BB = H.evaluated[(1, 1) + (0,) * H.n_infinite]
        if implicit:
            solve_sylvester = solve_sylvester_KPM(
                h_0,
                subspace_vectors,
                eigenvalues,
                kpm_params,
                precalculate_moments,
            )
        elif all(is_diagonal(h) for h in (h_0_AA, h_0_BB)):
            eigenvalues = tuple(h.diagonal() for h in (h_0_AA, h_0_BB))
            solve_sylvester = _solve_sylvester_diagonal(*eigenvalues)
        else:
            raise ValueError("solve_sylvester must be provided or the"
                             " unperturbed Hamiltonian must be diagonal.")

    if algorithm in ("general", "expanded"):
        return globals()[algorithm](
            H,
            solve_sylvester=solve_sylvester,
            op=op,
        )
    elif algorithm == "numerical":
        return globals()[algorithm](
            H,
            solve_sylvester=solve_sylvester,
        )
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')


def _list_to_dict(hamiltonian: list[Any]) -> dict[int, Any]:
    """
    Parameters
    ----------
    hamiltonian :
        Unperturbed Hamiltonian and 1st order perturbations.
        [H_0, H_1, H_2, ...], where H_0 is the unperturbed Hamiltonian and the
        remaining elements are the 1st order perturbations.
        This method is for first order perturbations only.

    Returns
    -------
    H : `~lowdin.series.BlockSeries`
    """
    n_infinite = len(hamiltonian) - 1 # all the perturbations are 1st order
    zeroth_order = (0,) * n_infinite

    hamiltonian = {
        zeroth_order: hamiltonian[0],
        **{
            tuple(order): perturbation for order, perturbation in
            zip(np.eye(n_infinite, dtype=int), hamiltonian[1:])
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
        The values are the perturbations, and can be either a `~numpy.ndarray` or
        `~scipy.sparse.csr_matrix` or a list with the blocks of the Hamiltonian.
        {(0, 0): H_0, (1, 0): H_1, (0, 1): H_2}

    Returns
    -------
    H : `~lowdin.series.BlockSeries`
    """
    symbols = None
    key_types = set(isinstance(key, sympy.Basic) for key in hamiltonian.keys())
    if any(key_types):
        hamiltonian, symbols = _symbolic_keys_to_orders(hamiltonian)

    n_infinite = len(list(hamiltonian.keys())[0])
    zeroth_order = (0,) * n_infinite
    h_0 = hamiltonian[zeroth_order]

    if isinstance(h_0, np.ndarray):
        if np.allclose(h_0, np.diag(np.diag(h_0))):
            hamiltonian[zeroth_order] = sparse.csr_array(hamiltonian[zeroth_order])
    elif sparse.issparse(h_0):  # normalize sparse matrices for solve_sylvester
        hamiltonian[zeroth_order] = sparse.csr_array(hamiltonian[zeroth_order])

    H_temporary = BlockSeries(
        data=hamiltonian,
        shape=(),
        n_infinite=n_infinite,
    )
    return H_temporary, symbols


def _symbolic_keys_to_orders(hamiltonian):
    """
    Parameters
    ----------
    hamiltonian :
        Dictionary with symbolic keys, each a monomial without numerical
        prefactor. The values can be either a `~numpy.ndarray` or
        `~scipy.sparse.csr_matrix` or a list with the blocks of the Hamiltonian.

    Returns
    -------
    new_hamiltonian :
        Dictionary with the same values as ``hamiltonian``, but with keys that
        indicate the order of the perturbation.
    symbols :
        List of symbols in the order they appear in the keys of ``hamiltonian``.
    """
    symbols = list(set.union(*[key.free_symbols for key in hamiltonian.keys()]))
    symbols = sorted(symbols, key=lambda x: x.name)
    if not all(symbol.is_commutative for symbol in symbols):
        raise ValueError("All symbols must be commutative.")

    new_hamiltonian = {}
    for key, value in hamiltonian.items():
        if value.dtype == np.dtype("O"):
            raise ValueError("Values of the dictionary must be numeric.")
        monomial = key.as_powers_dict()
        basis = monomial.keys()
        if len(basis) == 1 and list(basis)[0] not in symbols:
            # unperturbed Hamiltonian
            new_hamiltonian[(0,) * len(symbols)] = value
            continue
        indices = np.array([symbols.index(base, ) for base in basis])
        order = np.zeros(len(symbols), dtype=int)
        exponents = np.zeros(len(symbols), dtype=int)
        order[indices] = 1
        exponents[indices] = [monomial[base] for base in basis]
        new_hamiltonian[tuple(order * exponents)] = value
    return new_hamiltonian, symbols


def _sympy_to_BlockSeries(
        hamiltonian: sympy.MatrixBase,
        symbols: list[sympy.Symbol] = None,
    ):
    """
    Parameters
    ----------
    hamiltonian :
        Symbolic Hamiltonian.
    symbols :
        List of symbols that are the perturbative coefficients.
        If None, all symbols in the Hamiltonian are assumed to be perturbative
        coefficients.

    Returns
    -------
    H : `~lowdin.series.BlockSeries`
    """
    if symbols is None:
        symbols = list(hamiltonian.free_symbols)
    if any(n not in hamiltonian.free_symbols for n in symbols):
        raise ValueError("Not all perturbative parameters are in the Hamiltonian.")

    hamiltonian = hamiltonian.expand()

    def H_eval(*index):  # TODO: write this recursively, more efficient
        expr = sympy.diff(
            hamiltonian, *((n, i) for n, i in zip(symbols, index))
        )
        expr = expr.subs({n: 0 for n in symbols})
        expr = expr / reduce(mul, [sympy.factorial(i) for i in index])
        expr = expr * reduce(mul, [n**i for n, i in zip(symbols, index)])
        if expr.is_hermitian is False:  # sympy three-valued logic
            raise ValueError("Hamiltonian must be Hermitian at every order.")
        return _convert_if_zero(expr)

    H = BlockSeries(
        eval=H_eval,
        shape=(),
        n_infinite = len(symbols),
    )
    return H


def _subspaces_from_indices(
        subspace_indices: tuple[int, ...] | np.ndarray,
        symbolic: Optional[bool] = False,
    ) -> tuple[sparse.csr_array, sparse.csr_array]:
    """
    Returns the subspace_vectors projection from the indices of the elements
    of the diagonal.

    Parameters
    ----------
    subspace_indices :
        Indices of the subspace_vectors.
        0 indicates the first subspace, 1 indicates the second subspace.

    Returns
    -------
    subspace_vectors :
        Subspaces to use for block diagonalization.
    """
    subspace_indices = np.array(subspace_indices)
    max_subspaces = 2
    if np.any(subspace_indices >= max_subspaces):
        raise ValueError(
            "Only 0 and 1 are allowed as indices for subspace_vectors."
    )
    dim = len(subspace_indices)
    eigvecs = sparse.csr_array(sparse.identity(dim, dtype=int, format="csr"))
    subspace_vectors = tuple(
        eigvecs[:, np.compress(subspace_indices==block, np.arange(dim))]
        for block in range(max_subspaces)
    )
    if symbolic:
        return tuple(subspace.toarray() for subspace in subspace_vectors)
    return subspace_vectors


def hamiltonian_to_BlockSeries(
        hamiltonian: list[Any, list] | dict | BlockSeries,
        *,
        subspace_vectors: Optional[tuple[Any, Any]] = None,
        subspace_indices: Optional[tuple[int, ...]] = None,
        implicit : Optional[bool] = False,
        symbols: Optional[list[sympy.Symbol]] = None,
    ) -> BlockSeries:
    """
    Converts a Hamiltonian to a `~lowdin.series.BlockSeries`.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian to convert to a `~lowdin.series.BlockSeries`.
        If a list, it is assumed to be of the form [H_0, H_1, H_2, ...] where
        H_0 is the zeroth order Hamiltonian and H_1, H_2, ... are the first
        order perturbations.
        If a dictionary, it is assumed to be of the form
        {(0, 0): H_0, (1, 0): H_1, (0, 1): H_2}.
        If a `~lowdin.series.BlockSeries`, it is returned unchanged.
        If a sympy matrix, it is tranformed to a `~lowdin.series.BlockSeries`
        by Taylor expanding the matrix with respect to the perturbative
        parameters.
    subspace_vectors :
        Subspaces to project the unperturbed Hamiltonian and separate it into
        blocks. The first element of the tuple contains the orthonormal vectors
        that span the AA subspace, while the second element contains the
        orthonormal vectors that span the BB subspace.
        If None, the unperturbed Hamiltonian must be block diagonal.
        For KPM, the first element contains the effective subspace, and the
        second element contains the (partial) auxiliary subspace.
        Mutually exclusive with `subspace_indices`.
    subspace_indices :
        If the unperturbed Hamiltonian is diagonal, the indices that label the
        diagonal elements according to the subspace_vectors may be provided.
        Indices 0 and 1 are reserved for the AA and BB subspaces, respectively.
        Mutually exclusive with `subspace_vectors`.
    implicit :
        Whether to use KPM to solve the Sylvester equation. If True, the first
        element of `subspace_vectors` must be the effective subspace, and the
        second element must be the (partial) auxiliary subspace.
    symbols :
        List of symbols that label the perturbative parameters. The order of
        the symbols will be used to determine the indices of the Hamiltonian.
        If None, the perturbative parameters are taken from the unperturbed
        Hamiltonian.

    Returns
    -------
    H : `~lowdin.series.BlockSeries`
        Initial Hamiltonian in the format required by algorithms.
    """
    if isinstance(hamiltonian, list):
        hamiltonian = _list_to_dict(hamiltonian)
    elif isinstance(hamiltonian, sympy.MatrixBase):
        hamiltonian = _sympy_to_BlockSeries(hamiltonian, symbols)
    if isinstance(hamiltonian, dict):
        hamiltonian, symbols = _dict_to_BlockSeries(hamiltonian)
    elif isinstance(hamiltonian, BlockSeries):
        pass
    else:
        raise NotImplementedError

    if symbols is None:
        symbols = []

    if hamiltonian.shape == (2, 2):
        if subspace_vectors is not None or subspace_indices is not None:
            raise ValueError("H is already separated but subspace_vectors"
                                " are provided.")
        return hamiltonian
    if hamiltonian.shape != ():
        raise NotImplementedError("Only two subspace_vectors are supported.")

    # Separation into subspace_vectors
    if subspace_vectors is None and subspace_indices is None:
        def H_eval(*index):
            monomial = reduce(mul, [n**i for n, i in zip(symbols, index[2:])], 1)
            h = hamiltonian.evaluated[index[2:]]
            if zero == h:
                return zero
            try: # Hamiltonians come in blocks of 2x2
                result = _convert_if_zero(h[index[0]][index[1]])
                return result * monomial
            except (TypeError, NotImplementedError):  # scipy.sparse lacks slices
                raise ValueError("`subspace_vectors` or `subspace_indices`"
                                    " must be provided.")

        H = BlockSeries(
            eval=H_eval,
            shape=(2, 2),
            n_infinite=hamiltonian.n_infinite,
        )
        return H

    if subspace_indices is not None:
        if subspace_vectors is not None:
            raise ValueError("Only subspace_vectors or subspace_indices can"
                                " be provided.")
        h_0 = hamiltonian.evaluated[(0,) * hamiltonian.n_infinite]
        if not is_diagonal(h_0):
            raise ValueError("If subspace_indices is provided, H_0 must be"
                                " diagonal.")
        if issubclass(type(h_0), sympy.MatrixBase):
            symbolic = True
        else:
            symbolic = False
        subspace_vectors = _subspaces_from_indices(subspace_indices, symbolic=symbolic)

    if implicit:
        # Define subspace_vectors for KPM
        vecs_a = subspace_vectors[0]
        subspace_vectors = (vecs_a, ComplementProjector(vecs_a))
    # Separation into subspace_vectors
    def H_eval(*index):
        monomial = reduce(mul, [n**i for n, i in zip(symbols, index[2:])], 1)
        original = hamiltonian.evaluated[index[2:]]
        if zero == original:
            return zero
        left, right = index[:2]
        if left <= right:
            if implicit and left == right == 1:
                # compatibility with KPM
                result = (
                    subspace_vectors[left] @ aslinearoperator(original)
                    @ subspace_vectors[right]
                )
            else:
                result = (
                    Dagger(subspace_vectors[left]) @ original @ subspace_vectors[right]
                )
            return monomial * _convert_if_zero(result)
        return monomial * Dagger(H.evaluated[(right, left) + tuple(index[2:])])

    H = BlockSeries(
        eval=H_eval,
        shape=(2, 2),
        n_infinite=hamiltonian.n_infinite,
    )

    return H


def _convert_if_zero(value: Any):
    """
    Converts a value to zero if it is close enough.

    Parameters
    ----------
    value :
        Value to convert to zero.

    Returns
    -------
    zero :
        Zero if value is close enough to zero, otherwise value.
    """
    if isinstance(value, np.ndarray):
        if not np.any(value):
            return zero
    elif sparse.issparse(value):
        if value.count_nonzero() == 0:
            return zero
    if isinstance(value, sympy.MatrixBase):
        if value.is_zero_matrix:
            return zero
    return value
