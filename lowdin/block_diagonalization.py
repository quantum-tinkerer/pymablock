# %%
# # The polynomial alternative to Lowdin perturbation theory
#
# See [this hackmd](https://hackmd.io/Rpt2C8oOQ2SGkGS9OYrlfQ?view)
# for the motivation and the expressions

# %%
from operator import matmul, mul
from functools import reduce
from typing import Any, Optional, Callable
from copy import copy

import numpy as np
import sympy
from scipy import sparse
from sympy.physics.quantum import Dagger, Operator, HermitianOperator

from lowdin.linalg import (
    ComplementProjector,
    aslinearoperator,
    is_diagonal,
    direct_greens_function,
)
from lowdin.kpm_funcs import greens_function
from lowdin.series import (
    BlockSeries,
    zero,
    one,
    cauchy_dot_product,
    _zero_sum,
    safe_divide,
)

__all__ = ["general", "expanded", "general_symbolic", "implicit"]


### The main function for end-users.
def block_diagonalize(
    hamiltonian: list[Any, list] | dict | BlockSeries,
    *,
    algorithm: Optional[str] = None,
    solve_sylvester: Optional[Callable] = None,
    subspace_vectors: Optional[tuple[Any, Any]] = None,
    subspace_indices: Optional[tuple[int, ...]] = None,
    eigenvalues: Optional[tuple[np.ndarray, np.ndarray]] = None,
    direct_solver: bool = True,
    solver_options: Optional[dict] = None,
    symbols: list[sympy.Symbol] = None,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Parameters
    ----------
    hamiltonian :
        Hamiltonian to convert to a `~lowdin.series.BlockSeries`.
        If a list, it is assumed to be of the form [h_0, h_1, h_2, ...] where
        h_0 is the zeroth order Hamiltonian and h_1, h_2, ... are the first
        order perturbations.
        If a dictionary, it is assumed to be of the form
        {(0, 0): h_0, (1, 0): h_1, (0, 1): h_2}, or
        {1: h_0, x: h_1, y: h_2} for symbolic Hamiltonians.
        If a `~lowdin.series.BlockSeries`, it is returned unchanged.
        If a sympy matrix, it is tranformed to a `~lowdin.series.BlockSeries`
        by Taylor expanding the matrix with respect to the perturbative
        parameters.
    algorithm :
        Name of the function that block diagonalizes a Hamiltonian.
        Options are "general" and "expanded".
    solve_sylvester :
        Function to use for solving Sylvester's equation.
        If None, the default function is used for a diagonal Hamiltonian.
    subspace_vectors :
        Subspaces to project the unperturbed Hamiltonian and separate it into
        blocks. The first element of the tuple contains the orthonormal vectors
        that span the A subspace, while the second element contains the
        orthonormal vectors that span the B subspace.
        If None, the unperturbed Hamiltonian must be block diagonal.
        For KPM, the first element contains the effective subspace, and the
        second element contains the (partial) auxiliary subspace.
        Mutually exclusive with `subspace_indices`.
    subspace_indices :
        If the unperturbed Hamiltonian is diagonal, the indices that label the
        diagonal elements according to the subspace_vectors may be provided.
        Indices 0 and 1 are reserved for the A and B subspaces, respectively.
        Mutually exclusive with `subspace_vectors`.
    eigenvalues :
        Eigenvalues of the unperturbed Hamiltonian. The first element of the
        tuple contains the full eigenvalues of the effective subspace. The
        second element is optional, and it contains the (partial) eigenvalues
        of the auxiliary subspace. This argument is needed for KPM.
    solver_options :
        Dictionary containing the options to pass to the solver.
        Relevant keys are:
            num_moments : int
                Number of moments to use for the KPM expansion.
            num_vectors : int
                Number of vectors to use for the KPM expansion.
            precalculate_moments : bool
                Whether to precalculate and store all the KPM moments of ``vectors``.
                This is useful if the Green's function is evaluated at a large number
                of energies, but uses a large amount of memory. If False, the KPM
                expansion is performed every time the Green's function is called, which
                minimizes memory use.
    symbols :
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
    if (use_implicit := eigenvalues is not None):
        # Build solve_sylvester
        if subspace_vectors is None:
            raise ValueError(
                "`subspace_vectors` must be provided if `eigenvalues` is provided."
            )
        if isinstance(hamiltonian, list):
            h_0 = hamiltonian[0]
        elif isinstance(hamiltonian, dict):
            n_infinite = len(list(hamiltonian.keys())[0])
            h_0 = hamiltonian[(0,) * n_infinite]
        elif isinstance(hamiltonian, BlockSeries):
            h_0 = hamiltonian.evaluated[(0,) * hamiltonian.n_infinite]
        else:
            raise TypeError(
                "`hamiltonian` must be a list, dictionary, or BlockSeries."
            )
        if solve_sylvester is None:
            if direct_solver:
                solve_sylvester = solve_sylvester_direct(
                    h_0,
                    subspace_vectors[0],
                    eigenvalues[0],
                )
            else:
                solve_sylvester = solve_sylvester_KPM(
                    h_0,
                    subspace_vectors,
                    eigenvalues,
                    solver_options=solver_options,
                )

    # Normalize the Hamiltonian
    H = hamiltonian_to_BlockSeries(
        hamiltonian,
        subspace_vectors=subspace_vectors,
        subspace_indices=subspace_indices,
        implicit=use_implicit,
        symbols=symbols,
    )

    # Determine operator to use for matrix multiplication.
    if hasattr(H.evaluated[(0, 0) + (0,) * H.n_infinite], "__matmul__"):
        operator = matmul
    else:
        operator = mul

    # If solve_sylvester is not yet defined, use the diagonal one.
    if solve_sylvester is None:
        solve_sylvester = solve_sylvester_diagonal(*_extract_diagonal(H))

    if algorithm is None:
        # symbolic expressions benefit from no H_0 in numerators
        algorithm = "expanded" if symbols is not None else "general"
    if algorithm not in ("general", "expanded"):
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if use_implicit:
        return implicit(
            H,
            solve_sylvester=solve_sylvester,
            algorithm=algorithm,
        )
    return globals()[algorithm](
        H,
        solve_sylvester=solve_sylvester,
        operator=operator,
    )


### Converting different formats to BlockSeries
def hamiltonian_to_BlockSeries(
    hamiltonian: list[Any, list] | dict | BlockSeries,
    *,
    subspace_vectors: Optional[tuple[Any, Any]] = None,
    subspace_indices: Optional[tuple[int, ...]] = None,
    implicit: Optional[bool] = False,
    symbols: Optional[list[sympy.Symbol]] = None,
) -> BlockSeries:
    """
    Converts a Hamiltonian to a `~lowdin.series.BlockSeries`.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian to convert to a `~lowdin.series.BlockSeries`.
        If a list, it is assumed to be of the form [h_0, h_1, h_2, ...] where
        h_0 is the zeroth order Hamiltonian and h_1, h_2, ... are the first
        order perturbations.
        If a dictionary, it is assumed to be of the form
        {(0, 0): h_0, (1, 0): h_1, (0, 1): h_2}, or
        {1: h_0, x: h_1, y: h_2} for symbolic Hamiltonians.
        If a `~lowdin.series.BlockSeries`, it is returned unchanged.
        If a sympy matrix, it is tranformed to a `~lowdin.series.BlockSeries`
        by Taylor expanding the matrix with respect to the perturbative
        parameters.
    subspace_vectors :
        Subspaces to project the unperturbed Hamiltonian and separate it into
        blocks. The first element of the tuple contains the orthonormal vectors
        that span the A subspace, while the second element contains the
        orthonormal vectors that span the B subspace.
        If None, the unperturbed Hamiltonian must be block diagonal.
        For KPM, the first element contains the effective subspace, and the
        second element contains the (partial) auxiliary subspace.
        Mutually exclusive with `subspace_indices`.
    subspace_indices :
        If the unperturbed Hamiltonian is diagonal, the indices that label the
        diagonal elements according to the subspace_vectors may be provided.
        Indices 0 and 1 are reserved for the A and B subspaces, respectively.
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
    if subspace_vectors is not None and subspace_indices is not None:
        raise ValueError("subspace_vectors and subspace_indices are mutually exclusive.")
    to_split = subspace_vectors is not None or subspace_indices is not None

    # Convert anything to BlockSeries
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

    if hamiltonian.shape and to_split:
        raise ValueError(
            "H is already separated but subspace_vectors" " are provided."
        )

    if hamiltonian.shape == (2, 2):
        return hamiltonian
    elif hamiltonian.shape:
        raise NotImplementedError("Only two subspace_vectors are supported.")

    # Separation into subspace_vectors
    if not to_split:
        # Hamiltonian must have 2x2 entries in each block
        def H_eval(*index):
            h = _convert_if_zero(hamiltonian.evaluated[index[2:]])
            if zero == h:
                return zero
            try:  # Hamiltonians come in blocks of 2x2
                return _convert_if_zero(h[index[0]][index[1]])
            except Exception as e:
                raise ValueError(
                    "Without `subspace_vectors` or `subspace_indices`"
                    " H must have a 2x2 block structure."
                ) from e

        H = BlockSeries(
            eval=H_eval,
            shape=(2, 2),
            n_infinite=hamiltonian.n_infinite,
        )
        return H

    # Define subspace_vectors
    if subspace_indices is not None:
        h_0 = hamiltonian.evaluated[(0,) * hamiltonian.n_infinite]
        if not is_diagonal(h_0):
            raise ValueError(
                "If `subspace_indices` is provided, the unperturbed Hamiltonian"
                 " must be diagonal."
            )
        symbolic = isinstance(h_0, sympy.MatrixBase)
        subspace_vectors = _subspaces_from_indices(subspace_indices, symbolic=symbolic)

    if implicit:
        # Define subspace_vectors for KPM
        vecs_a = subspace_vectors[0]
        subspace_vectors = (vecs_a, ComplementProjector(vecs_a))

    # Separation into subspace_vectors
    def H_eval(*index):
        original = hamiltonian.evaluated[index[2:]]
        if zero == original:
            return zero
        left, right = index[:2]
        if left > right:
            return Dagger(H.evaluated[(right, left) + tuple(index[2:])])
        if implicit and left == right == 1:
            result = (
                subspace_vectors[left]
                @ aslinearoperator(original)
                @ subspace_vectors[right]
            )
        else:
            result = Dagger(subspace_vectors[left]) @ original @ subspace_vectors[right]
        return _convert_if_zero(result)

    H = BlockSeries(
        eval=H_eval,
        shape=(2, 2),
        n_infinite=hamiltonian.n_infinite,
    )

    return H


### Block diagonalization algorithms
def general(
    H: BlockSeries,
    solve_sylvester: Optional[Callable] = None,
    *,
    operator: Optional[Callable] = None,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Computes the block diagonalization of a Hamiltonian.

    Parameters
    ----------
    H :
        Initial Hamiltonian, unperturbed and perturbation.
    solve_sylvester :
        (optional) function that solves the Sylvester equation.
        Defaults to a function that works for numerical diagonal unperturbed
        Hamiltonians.
    operator :
        (optional) function to use for matrix multiplication.
        Defaults to matmul.

    Returns
    -------
    H_tilde : `~lowdin.series.BlockSeries`
        Block diagonalized Hamiltonian.
    U : `~lowdin.series.BlockSeries`
        Unitary that block diagonalizes H such that H_tilde = U^H H U.
    U_adjoint : `~lowdin.series.BlockSeries`
        Adjoint of U.
    """
    if operator is None:
        operator = matmul

    if solve_sylvester is None:
        solve_sylvester = solve_sylvester_diagonal(*_extract_diagonal(H))

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
        U_adjoint, U, operator=operator, hermitian=True, exclude_last=[True, True]
    )
    H_tilde_rec = cauchy_dot_product(
        U_adjoint,
        H,
        U,
        operator=operator,
        hermitian=True,
        exclude_last=[True, False, True],
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

    H_tilde = cauchy_dot_product(U_adjoint, H, U, operator=operator, hermitian=True)
    return H_tilde, U, U_adjoint


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
        dictionary of {V: rhs} such that h_0_AA * V - V * h_0_BB = rhs.
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
    h_0_AA = H_placeholder.evaluated[(0, 0) + (0,) * H.n_infinite]
    h_0_BB = H_placeholder.evaluated[(1, 1) + (0,) * H.n_infinite]

    H_tilde, U, U_adjoint = general(
        H_placeholder, solve_sylvester=(lambda x: x), operator=mul
    )

    Y_data = {}

    old_U_eval = U.eval

    def U_eval(*index):
        if index[:2] == (0, 1):
            V = Operator(f"V_{{{index[2:]}}}")
            Y = _commute_h0_away(old_U_eval(*index), h_0_AA, h_0_BB, Y_data)
            if zero == Y:
                return zero
            Y_data[V] = Y
            return V
        return old_U_eval(*index)

    U.eval = U_eval

    old_H_tilde_eval = H_tilde.eval

    def H_tilde_eval(*index):
        return _commute_h0_away(old_H_tilde_eval(*index), h_0_AA, h_0_BB, Y_data)

    H_tilde.eval = H_tilde_eval

    return H_tilde, U, U_adjoint, Y_data, subs


def expanded(
    H: BlockSeries,
    solve_sylvester: Optional[Callable] = None,
    *,
    operator: Optional[Callable] = None,
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
    operator :
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
    if operator is None:
        operator = matmul

    if solve_sylvester is None:
        solve_sylvester = solve_sylvester_diagonal(*_extract_diagonal(H))

    H_tilde_s, U_s, _, Y_data, subs = general_symbolic(H)
    _, U, U_adjoint = general(H, solve_sylvester=solve_sylvester, operator=operator)

    def H_tilde_eval(*index):
        H_tilde = H_tilde_s.evaluated[index]
        _update_subs(Y_data, subs, solve_sylvester, operator)
        return _replace(H_tilde, subs, operator)

    H_tilde = BlockSeries(eval=H_tilde_eval, shape=(2, 2), n_infinite=H.n_infinite)

    old_U_eval = U.eval

    def U_eval(*index):
        if index[:2] == (0, 1):
            U_s.evaluated[index]  # Update Y_data
            _update_subs(Y_data, subs, solve_sylvester, operator)
            return subs.get(Operator(f"V_{{{index[2:]}}}"), zero)
        return old_U_eval(*index)

    U.eval = U_eval

    return H_tilde, U, U_adjoint


def implicit(
    H: BlockSeries,
    solve_sylvester: Callable,
    algorithm : str = "general",
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Block diagonalize a Hamiltonian without explicitly forming BB matrices.

    Instead these matrices are wrapped in `scipy.sparse.LinearOperator` and
    combined to keep them low rank.

    Parameters
    ----------
    H :
        Full Hamiltonian of the system.
    solve_sylvester :
        Function to use for solving Sylvester's equation.
    algorithm :
        Algorithm to use for diagonalization. One of "general", "expanded".
        The "expanded" (default) is faster in lower orders.

    Returns
    -------
    H_tilde : `~lowdin.series.BlockSeries`
        Full block-diagonalized Hamiltonian of the problem. The ``(0, 0)`` block
        (A subspace) is a numpy array, while the ``(1, 1)`` block (B subspace)
        is a ``LinearOperator``.
    U : `~lowdin.series.BlockSeries`
        Unitary that block diagonalizes the initial Hamiltonian.
    U_adjoint : `~lowdin.series.BlockSeries`
        Adjoint of ``U``.
    """
    if algorithm not in ("general", "expanded"):
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    algorithm = globals()[algorithm]
    H_tilde, U, U_adjoint = algorithm(H, solve_sylvester=solve_sylvester)

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

    result_H_tilde = BlockSeries(
        eval=H_tilde_eval, shape=(2, 2), n_infinite=H.n_infinite
    )

    return result_H_tilde, U, U_adjoint


### Different formats and algorithms of solving Sylvester equation.
def solve_sylvester_diagonal(
    eigs_A: Any, eigs_B: Any, vecs_b: Optional[np.ndarray] = None
) -> Callable:
    """
    Define a function for solving a Sylvester's equation with diagonal matrices

    Optionally, this function also applies the eigenvectors of the second matrix
    to the solution.

    Parameters
    ----------
    eigs_A :
        Eigenvalues of A subspace of the unperturbed Hamiltonian.
    eigs_B :
        Eigenvalues of B subspace of the unperturbed Hamiltonian.
    vecs_b :
        (optional) Eigenvectors of B subspace of the unperturbed Hamiltonian.

    Returns
    -------
    solve_sylvester : Function that solves the Sylvester equation.
    """

    def solve_sylvester(
        Y: np.ndarray | sparse.csr_array | sympy.MatrixBase,
    ) -> np.ndarray | sparse.csr_array | sympy.MatrixBase:
        if vecs_b is not None:
            energy_denominators = 1 / (eigs_A[:, None] - eigs_B[None, :])
            return ((Y @ vecs_b) * energy_denominators) @ Dagger(vecs_b)
        elif isinstance(Y, np.ndarray):
            energy_denominators = 1 / (eigs_A.reshape(-1, 1) - eigs_B)
            return Y * energy_denominators
        elif sparse.issparse(Y):
            Y_coo = Y.tocoo()
            energy_denominators = 1 / (eigs_A[Y_coo.row] - eigs_B[Y_coo.col])
            new_data = Y_coo.data * energy_denominators
            return sparse.csr_array((new_data, (Y_coo.row, Y_coo.col)), Y_coo.shape)
        elif isinstance(Y, sympy.MatrixBase):
            array_eigs_a = np.array(eigs_A, dtype=object)  # Use numpy to reshape
            array_eigs_b = np.array(eigs_B, dtype=object)
            energy_denominators = sympy.Matrix(
                1 / (array_eigs_a.reshape(-1, 1) - array_eigs_b)
            )
            return energy_denominators.multiply_elementwise(Y)
        else:
            NotImplementedError

    return solve_sylvester


def solve_sylvester_KPM(
    h_0: Any,
    subspace_vectors: tuple[Any, Any],
    eigenvalues: np.ndarray,
    solver_options: Optional[dict] = None,
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
    solver_options :
        Dictionary containing the options to pass to the solver.
        Relevant keys are:
            num_moments : int
                Number of moments to use for the KPM expansion.
            num_vectors : int
                Number of vectors to use for the KPM expansion.
            precalculate_moments : bool
                Whether to precalculate and store all the KPM moments of ``vectors``.
                This is useful if the Green's function is evaluated at a large number
                of energies, but uses a large amount of memory. If False, the KPM
                expansion is performed every time the Green's function is called, which
                minimizes memory use.

    Returns
    ----------
    solve_sylvester: callable
        Function that applies divide by energies to the RHS of the Sylvester equation.
    """
    # Full A subspace and partial/full B subspace provided
    if len(subspace_vectors) == 2:
        vecs_a, vecs_b = subspace_vectors
        eigs_A, eigs_B = eigenvalues

    # Full A subspace and no B subspace provided
    elif len(subspace_vectors) == 1:
        vecs_a = subspace_vectors[0]
        eigs_A = eigenvalues[0]
        vecs_b = np.empty((vecs_a.shape[0], 0))
        eigs_B = np.diagonal((Dagger(vecs_b) @ h_0 @ vecs_b))
    else:
        raise NotImplementedError("Too many subspace_vectors")

    if not isinstance(eigs_A, np.ndarray) or not isinstance(eigs_B, np.ndarray):
        raise TypeError("Eigenvalues must be a numpy array")

    if solver_options is None:
        solver_options = dict()

    precalculate_moments = solver_options.get("precalculate_moments", False)

    need_kpm = len(eigs_A) + len(eigs_B) < h_0.shape[0]
    need_explicit = bool(len(eigs_B))
    if not any((need_kpm, need_explicit)):
        # B subspace is empty
        return lambda Y: Y

    if need_kpm:
        kpm_projector = ComplementProjector(np.hstack([vecs_a, vecs_b]))

        def solve_sylvester_kpm(Y: np.ndarray) -> np.ndarray:
            Y_KPM = Y @ kpm_projector
            vec_G_Y = greens_function(
                h_0,
                params=None,
                vectors=Y_KPM.conj(),
                kpm_params=solver_options,
                precalculate_moments=precalculate_moments,
            )(eigs_A)
            return np.vstack([vec_G_Y.conj()[:, m, m] for m in range(len(eigs_A))])

    if need_explicit:
        solve_sylvester_explicit = solve_sylvester_diagonal(eigs_A, eigs_B, vecs_b)

    def solve_sylvester(Y: np.ndarray) -> np.ndarray:
        if need_kpm and need_explicit:
            result = solve_sylvester_kpm(Y) + solve_sylvester_explicit(Y)
        elif need_kpm:
            result = solve_sylvester_kpm(Y)
        elif need_explicit:
            result = solve_sylvester_explicit(Y)

        return result

    return solve_sylvester


def solve_sylvester_direct(
    h_0: sparse.spmatrix,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Solve Sylvester equation using a direct sparse solver.

    Parameters
    ----------
    h_0 :
        Unperturbed Hamiltonian of the system.
    eigenvectors :
        Eigenvectors of the relevant subspace of the unperturbed Hamiltonian.
    eigenvalues :
        Corresponding eigenvalues.

    Returns
    -------
    solve_sylvester : `Callable[[np.ndarray], np.ndarray]`
        Function that solves the corresponding Sylvester equation.
    """
    projector = ComplementProjector(eigenvectors)
    # Compute the Green's function of the transposed Hamiltonian because we are
    # solving the equation from the right.
    greens_functions = [
        direct_greens_function(h_0.T, E) for E in eigenvalues
    ]

    def solve_sylvester(Y: np.ndarray) -> np.ndarray:
        Y = Y @ projector
        result = np.vstack([-gf(vec) for gf, vec in zip(greens_functions, Y)])
        return result @ projector

    return solve_sylvester


### Auxiliary functions.
def _commute_h0_away(
    expr: Any, h_0_AA: Any, h_0_BB: Any, Y_data: dict[Operator, Any]
) -> Any:
    """
    Simplify expression by commmuting h_0 and V using Sylvester's Equation relations.

    Parameters
    ----------
    expr :
        (zero or sympy) expression to simplify.
    h_0_AA :
        Unperturbed Hamiltonian in subspace A.
    h_0_BB :
        Unperturbed Hamiltonian in subspace B.
    Y_data :
        dictionary of {V: rhs} such that h_0_AA * V - V * h_0_BB = rhs.

    Returns
    -------
    expr : zero or sympy.expr
        (zero or sympy) expression without h_0_AA or h_0_BB in it.
    """
    if zero == expr:
        return expr

    subs = {
        **{h_0_AA * V: rhs + V * h_0_BB for V, rhs in Y_data.items()},
        **{
            h_0_BB * Dagger(V): -Dagger(rhs) + Dagger(V) * h_0_AA
            for V, rhs in Y_data.items()
        },
    }

    while (
        any(H in expr.free_symbols for H in (h_0_AA, h_0_BB))
        and len(expr.free_symbols) > 1
    ):
        expr = expr.subs(subs).expand()

    return expr.expand() or zero


def _update_subs(
    Y_data: dict[Operator, Any],
    subs: dict[Operator | HermitianOperator, Any],
    solve_sylvester: Callable,
    operator: Callable,
) -> None:
    """
    Store the solutions to the Sylvester equation in subs.

    Parameters
    ----------
    Y_data :
        dictionary of {V: rhs} such that h_0_AA * V - V * h_0_BB = rhs.
    subs :
        dictionary of substitutions to make.
    solve_sylvester :
        function to use for solving Sylvester's equation.
    operator :
        function to use for matrix multiplication.
    """
    for V, rhs in Y_data.items():
        if V not in subs:
            rhs = _replace(rhs, subs, operator)  # No general symbols left
            subs[V] = solve_sylvester(rhs)


def _replace(
    expr: Any, subs: dict[Operator | HermitianOperator, Any], operator: Callable
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
    operator :
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
                int(numerator) * reduce(operator, substituted_factors), int(denominator)
            )
        )

    return _zero_sum(result)


def _list_to_dict(hamiltonian: list[Any]) -> dict[int, Any]:
    """
    Parameters
    ----------
    hamiltonian :
        Unperturbed Hamiltonian and 1st order perturbations.
        [h_0, h_1, h_2, ...], where h_0 is the unperturbed Hamiltonian and the
        remaining elements are the 1st order perturbations.
        This method is for first order perturbations only.

    Returns
    -------
    H : `~lowdin.series.BlockSeries`
    """
    n_infinite = len(hamiltonian) - 1  # All the perturbations are 1st order
    zeroth_order = (0,) * n_infinite

    hamiltonian = {
        zeroth_order: hamiltonian[0],
        **{
            tuple(order): perturbation
            for order, perturbation in zip(
                np.eye(n_infinite, dtype=int), hamiltonian[1:]
            )
        },
    }
    return hamiltonian


def _dict_to_BlockSeries(hamiltonian: dict[tuple[int, ...], Any]) -> BlockSeries:
    """
    Parameters
    ----------
    hamiltonian :
        Unperturbed Hamiltonian and perturbations.
        The keys can be tuples of integers or symbolic monomials. They
        indicate the order of the perturbation in its respective value.
        The values are the perturbations, and can be either a `~numpy.ndarray`,
        `~scipy.sparse.csr_matrix` or a list with the blocks of the Hamiltonian.
        For example, {(0, 0): h_0, (1, 0): h_1, (0, 1): h_2} or
        {1: h_0, x: h_1, y: h_2}.

    Returns
    -------
    H : `~lowdin.series.BlockSeries`
    """
    symbols = None
    key_types = set(isinstance(key, sympy.Basic) for key in hamiltonian.keys())
    if any(key_types):
        hamiltonian, symbols = _symbolic_keys_to_tuples(hamiltonian)

    n_infinite = len(list(hamiltonian.keys())[0])
    zeroth_order = (0,) * n_infinite
    h_0 = hamiltonian[zeroth_order]

    if isinstance(h_0, np.ndarray):
        if is_diagonal(h_0):
            hamiltonian[zeroth_order] = sparse.csr_array(hamiltonian[zeroth_order])
    elif sparse.issparse(h_0):  # Normalize sparse matrices for solve_sylvester
        hamiltonian[zeroth_order] = sparse.csr_array(hamiltonian[zeroth_order])

    H_temporary = BlockSeries(
        data=copy(hamiltonian),
        shape=(),
        n_infinite=n_infinite,
    )
    return H_temporary, symbols


def _symbolic_keys_to_tuples(
    hamiltonian: dict[sympy.Basic, Any]
) -> tuple[dict[tuple[int, ...], Any], list[sympy.Basic]]:
    """
    Parameters
    ----------
    hamiltonian :
        Dictionary with symbolic keys, each a monomial without numerical
        prefactor. The values can be either a `~numpy.ndarray`,
        `~scipy.sparse.csr_matrix`, or a list with the blocks of the Hamiltonian.

    Returns
    -------
    new_hamiltonian :
        Dictionary with the same values as `hamiltonian`, but with keys that
        indicate the order of the perturbation in a tuple.
    symbols :
        List of symbols in the order they appear in the keys of `hamiltonian`.
        The tuple keys of `new_hamiltonian` are ordered according to this list.
    """
    # Collect all symbols from the keys
    symbols = list(set.union(*[key.free_symbols for key in hamiltonian.keys()]))
    symbols = sorted(symbols, key=lambda x: x.name)
    if not all(symbol.is_commutative for symbol in symbols):
        raise ValueError("All symbols must be commutative.")

    # Convert symbolic keys to orders of the perturbation
    new_hamiltonian = {}
    for key, value in hamiltonian.items():
        monomial = key.as_powers_dict()
        if monomial.keys() - set(symbols) - {1}:
            raise ValueError("The Hamiltonian keys must be monomials of symbols")
        new_hamiltonian[tuple(monomial[s] for s in symbols)] = value
    return new_hamiltonian, symbols


def _sympy_to_BlockSeries(
    hamiltonian: sympy.MatrixBase,
    symbols: list[sympy.Symbol] = None,
) -> BlockSeries:
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
        symbols = list(hamiltonian.free_symbols)  # All symbols are perturbative
    if any(n not in hamiltonian.free_symbols for n in symbols):
        raise ValueError("Not all perturbative parameters are in `hamiltonian`.")

    hamiltonian = hamiltonian.expand()

    def H_eval(*index):  # TODO: write this recursively, more efficient
        # Get order of perturbation by Taylor expanding
        expr = sympy.diff(hamiltonian, *((n, i) for n, i in zip(symbols, index)))
        expr = expr.subs({n: 0 for n in symbols})
        expr = expr / reduce(mul, [sympy.factorial(i) for i in index])
        # Multiply by perturbative coefficients
        expr = expr * reduce(mul, [n**i for n, i in zip(symbols, index)])
        if expr.is_hermitian is False:  # Sympy three-valued logic
            raise ValueError("Hamiltonian must be Hermitian at every order.")
        return _convert_if_zero(expr)

    H = BlockSeries(
        eval=H_eval,
        shape=(),
        n_infinite=len(symbols),
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
        0 indicates the first subspace A, 1 indicates the second subspace B.

    Returns
    -------
    subspace_vectors :
        Subspaces to use for block diagonalization.
    """
    subspace_indices = np.array(subspace_indices)
    max_subspaces = 2
    if np.any(subspace_indices >= max_subspaces):
        raise ValueError("Only 0 and 1 are allowed as indices for subspace_vectors.")
    dim = len(subspace_indices)
    eigvecs = sparse.csr_array(sparse.identity(dim, dtype=int, format="csr"))
    # Canonical basis vectors for each subspace
    subspace_vectors = tuple(
        eigvecs[:, np.compress(subspace_indices == block, np.arange(dim))]
        for block in range(max_subspaces)
    )
    if symbolic:
        # Convert to dense arrays, otherwise they cannot multiply sympy matrices.
        return tuple(subspace.toarray() for subspace in subspace_vectors)
    return subspace_vectors


def _extract_diagonal(H: BlockSeries) -> tuple[np.ndarray, np.ndarray]:
    """Extract the diagonal of the zeroth order of the Hamiltonian."""
    h_0_AA = H.evaluated[(0, 0) + (0,) * H.n_infinite]
    h_0_BB = H.evaluated[(1, 1) + (0,) * H.n_infinite]
    if not is_diagonal(h_0_AA) or not is_diagonal(h_0_BB):
        raise ValueError(
            "The unperturbed Hamiltonian must be diagonal if solve_sylvester"
            " is not provided."
        )

    return h_0_AA.diagonal(), h_0_BB.diagonal()


def _convert_if_zero(value: Any):
    """
    Converts an exact zero to sentinel value zero.

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
