"""Algorithms for quasi-degenerate perturbation theory."""

from collections.abc import Sequence
from copy import copy
from functools import reduce
from operator import matmul, mul
from typing import Any, Callable, Optional, Union
from warnings import warn

import numpy as np
import sympy
from scipy import sparse
from sympy.physics.quantum import Dagger

from pymablock.algorithms import main
from pymablock.kpm import greens_function, rescale
from pymablock.linalg import (
    ComplementProjector,
    aslinearoperator,
    direct_greens_function,
    is_diagonal,
)
from pymablock.series import (
    BlockSeries,
    cauchy_dot_product,
    one,
    zero,
)

__all__ = ["block_diagonalize"]

# Common types
Eigenvectors = tuple[Union[np.ndarray, sympy.Matrix], ...]


### The main function for end-users.
def block_diagonalize(
    hamiltonian: Union[list, dict, BlockSeries, sympy.Matrix],
    *,
    solve_sylvester: Optional[Callable] = None,
    subspace_eigenvectors: Optional[Eigenvectors] = None,
    subspace_indices: Optional[Union[tuple[int, ...], np.ndarray]] = None,
    direct_solver: bool = True,
    solver_options: Optional[dict] = None,
    symbols: Optional[Union[sympy.Symbol, Sequence[sympy.Symbol]]] = None,
    atol: float = 1e-12,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """Find the block diagonalization of a Hamiltonian order by order.

    This uses quasi-degenerate perturbation theory known as Lowdin perturbation
    theory, Schrieffer-Wolff transformation, or van Vleck transformation.

    This function does not yet perform the computation. Instead, it defines the
    computation as a `~pymablock.series.BlockSeries` object, which can be
    evaluated at any order.

    This function accepts a Hamiltonian in several formats and it first
    brings it to the eigenbasis of the unperturbed Hamiltonian if the blocks,
    eigenvectors or indices of the eigenvalues are provided, see below.


    For large numerical calculations with a sparse Hamiltonian and a low
    dimensional relevant subspace, the algorithm uses an implicit representation
    of the spectrum and does not require full diagonalization. This is enabled
    if ``subspace_vectors`` do not span the full space of the unperturbed
    Hamiltonian.

    Parameters
    ----------
    hamiltonian :
        Full symbolic or numeric Hamiltonian to block diagonalize.
        The Hamiltonian is normalized to a `~pymablock.series.BlockSeries` by
        separating it into effective and auxiliary subspaces.

        Supported formats:

        - A list,
            of the form ``[h_0, h_1, h_2, ...]`` where ``h_0`` is the
            unperturbed Hamiltonian and ``h_1, h_2, ...`` are the first order
            perturbations. The elements ``h_i`` may be
            `~sympy.matrices.dense.Matrix`, `~numpy.ndarray`,
            `~scipy.sparse.spmatrix`, that require separating the unperturbed
            Hamiltonian into effective and auxiliary subspaces. Otherwise,
            ``h_i`` may be a list of lists with the Hamiltonian blocks.
        - A dictionary,
            of the form ``{(0, 0): h_0, (1, 0): h_1, (0, 1): h_2}``, or ``{1:
            h_0, x: h_1, y: h_2}`` for symbolic Hamiltonians. In the former
            case, the keys must be tuples of integers indicating the order of
            each perturbation. In the latter case, the keys must be monomials
            and the indices are ordered as in `H.dimension_names`. The values
            of the dictionary, ``h_i`` may be `~sympy.matrices.dense.Matrix`,
            `~numpy.ndarray`, `~scipy.sparse.spmatrix`, that require separating
            the unperturbed Hamiltonian into effective and auxiliary subspaces.
            Otherwise, ``h_i`` may be a list of lists with the Hamiltonian
            blocks.
        - A `~sympy.matrices.dense.Matrix`,
            unless a list of ``symbols`` is provided as perturbative parameters,
            all symbols will be treated as perturbative. The normalization to
            `~pymablock.series.BlockSeries` is done by Taylor expanding in
            ``symbols`` to the desired order.
        - A `~pymablock.series.BlockSeries`,
            returned unchanged.
    solve_sylvester :
        A function that solves the Sylvester equation. If not provided,
        it is selected automatically based on the inputs.
    subspace_eigenvectors :
        A tuple with orthonormal eigenvectors to project the Hamiltonian in
        and separate it into the A (effective) and B (auxiliary) blocks.
        The first element of the tuple has the eigenvectors of the A
        subspace, and the second element has the eigenvectors of the B subspace.
        If None, the unperturbed Hamiltonian must be block diagonal.
        For implicit, the (partial) auxiliary subspace may be missing or
        incomplete.
        Mutually exclusive with ``subspace_indices``.
    subspace_indices :
        An array indicating which basis vector belongs to which subspace. The
        labels are 0 for the A (effective) subspace and 1 for the B (auxiliary)
        subspace.
        Only applicable if the unperturbed Hamiltonian is diagonal.
        Mutually exclusive with ``subspace_eigenvectors``.
    solver_options :
        Dictionary containing the options to pass to the Sylvester solver.
        See docstrings of `~pymablock.block_diagonalization.solve_sylvester_KPM`
        and `~pymablock.block_diagonalization.solve_sylvester_direct` for details.
    direct_solver :
        Whether to use the direct solver that relies on MUMPS (default).
        Otherwise, the KPM solver is used. Only applicable if the implicit
        method is used (i.e. `subspace_vectors` is incomplete)
    symbols :
        Symbols that label the perturbative parameters of a symbolic
        Hamiltonian. The order of the symbols is mapped to the indices of the
        Hamiltonian, see `~pymablock.series.BlockSeries`. If None, the
        perturbative parameters are taken from the unperturbed Hamiltonian.
    atol :
        Absolute tolerance to consider matrices as exact zeros. This is used
        to validate that the unperturbed Hamiltonian is block-diagonal.

    Returns
    -------
    H_tilde : `~pymablock.series.BlockSeries`
        Block diagonalized Hamiltonian.
    U : `~pymablock.series.BlockSeries`
        Unitary matrix that block diagonalizes H such that U * H * U^H = H_tilde.
    U_adjoint : `~pymablock.series.BlockSeries`
        Adjoint of U.

    """
    if isinstance(symbols, sympy.Symbol):
        symbols = [symbols]

    use_implicit = False
    if subspace_eigenvectors is not None:
        _check_orthonormality(subspace_eigenvectors, atol=atol)
        num_vectors = sum(vecs.shape[1] for vecs in subspace_eigenvectors)
        use_implicit = num_vectors < subspace_eigenvectors[0].shape[0]

    if use_implicit:
        assert subspace_eigenvectors is not None  # for mypy
        # Build solve_sylvester
        if isinstance(hamiltonian, list):
            h_0 = hamiltonian[0]
        elif isinstance(hamiltonian, dict):
            h_0 = hamiltonian.get(1, hamiltonian.get(1.0))
            if h_0 is None:
                h_0 = hamiltonian[(0,) * len(next(iter(hamiltonian.keys())))]
        elif isinstance(hamiltonian, BlockSeries):
            if hamiltonian.shape:
                raise ValueError(
                    "`hamiltonian` must be a scalar BlockSeries when using an implicit"
                    " solver."
                )
            h_0 = hamiltonian[(0,) * hamiltonian.n_infinite]
        else:
            raise TypeError("`hamiltonian` must be a list, dictionary, or BlockSeries.")
        if any(h_0.shape[0] != vecs.shape[0] for vecs in subspace_eigenvectors):
            raise ValueError("`subspace_eigenvectors` does not match the shape of `h_0`.")
        if solve_sylvester is None:
            if not all(isinstance(vecs, np.ndarray) for vecs in subspace_eigenvectors):
                raise TypeError(
                    "Implicit problem requires numpy arrays for eigenvectors."
                )
            if direct_solver:
                solve_sylvester = solve_sylvester_direct(
                    h_0,
                    subspace_eigenvectors[0],
                )
            else:
                solve_sylvester = solve_sylvester_KPM(
                    h_0,
                    subspace_eigenvectors,
                    solver_options=solver_options,
                )

    # Normalize the Hamiltonian
    H = hamiltonian_to_BlockSeries(
        hamiltonian,
        subspace_eigenvectors=subspace_eigenvectors,
        subspace_indices=subspace_indices,
        implicit=use_implicit,
        symbols=symbols,
        atol=atol,
    )

    if H[(0, 1) + (0,) * H.n_infinite] is not zero:
        raise ValueError(
            "The off-diagonal elements of the unperturbed Hamiltonian must be zero."
        )

    # Determine operator to use for multiplication. We prefer matmul, and use mul if
    # matmul is not available.
    H_0_diag = [
        block for i in range(2) if (block := H[(i, i) + (0,) * H.n_infinite]) is not zero
    ]
    if not H_0_diag:
        raise ValueError("Both blocks of the unperturbed Hamiltonian should not be zero.")
    if all(hasattr(H, "__matmul__") for H in H_0_diag):
        operator = matmul
    elif all(hasattr(H, "__mul__") for H in H_0_diag):
        operator = mul
    else:
        raise ValueError("The unperturbed Hamiltonian is not a valid operator.")

    # If solve_sylvester is not yet defined, use the diagonal one.
    if solve_sylvester is None:
        solve_sylvester = solve_sylvester_diagonal(*_extract_diagonal(H, atol))

    return _block_diagonalize(
        H,
        solve_sylvester=solve_sylvester,
        operator=operator,
    )


### Converting different formats to BlockSeries
def hamiltonian_to_BlockSeries(
    hamiltonian: Union[list, dict, BlockSeries, sympy.Matrix],
    *,
    subspace_eigenvectors: Optional[tuple[Any, Any]] = None,
    subspace_indices: Optional[tuple[int, ...]] = None,
    implicit: Optional[bool] = False,
    symbols: Optional[list[sympy.Symbol]] = None,
    atol: float = 1e-12,
) -> BlockSeries:
    """Normalize a Hamiltonian to be used by the algorithms.

    This function separates the Hamiltonian into a 2x2 block form consisting of
    effective and auxiliary subspaces based on the inputs.

    Parameters
    ----------
    hamiltonian :
        Full symbolic or numeric Hamiltonian to block diagonalize. The
        Hamiltonian is normalized to a `~pymablock.series.BlockSeries` by
        separating it into effective and auxiliary subspaces.

        Supported formats:

        - A list,
            of the form ``[h_0, h_1, h_2, ...]`` where ``h_0`` is the
            unperturbed Hamiltonian and ``h_1, h_2, ...`` are the first order
            perturbations. The elements ``h_i`` may be
            `~sympy.matrices.dense.Matrix`, `~numpy.ndarray`,
            `~scipy.sparse.spmatrix`, that require separating the unperturbed
            Hamiltonian into effective and auxiliary subspaces. Otherwise,
            ``h_i`` may be a list of lists with the Hamiltonian blocks.
        - A dictionary,
            of the form ``{(0, 0): h_0, (1, 0): h_1, (0, 1): h_2}``, or ``{1:
            h_0, x: h_1, y: h_2}`` for symbolic Hamiltonians. In the former
            case, the keys must be tuples of integers indicating the order of
            each perturbation. In the latter case, the keys must be monomials
            and the indices are ordered as in `H.dimension_names`. The values
            of the dictionary, ``h_i`` may be `~sympy.matrices.dense.Matrix`,
            `~numpy.ndarray`, `~scipy.sparse.spmatrix`, that require separating
            the unperturbed Hamiltonian into effective and auxiliary subspaces.
            Otherwise, ``h_i`` may be a list of lists with the Hamiltonian
            blocks.
        - A `~sympy.matrices.dense.Matrix`,
            unless a list of ``symbols`` is provided as perturbative parameters,
            all symbols will be treated as perturbative. The normalization to
            `~pymablock.series.BlockSeries` is done by Taylor expanding on
            ``symbols`` to the desired order.
        - A `~pymablock.series.BlockSeries`,
            returned unchanged.
    subspace_eigenvectors :
        A tuple with orthonormal eigenvectors to project the Hamiltonian on and
        separate it into blocks. The first element of the tuple has the
        eigenvectors of the A (effective) subspace, and the second element has
        the eigenvectors of the B (auxiliary) subspace. If None, the unperturbed
        Hamiltonian must be block diagonal. For implicit, the (partial)
        auxiliary subspace may be missing or incomplete. Mutually exclusive with
        ``subspace_indices``.
    subspace_indices :
        An array indicating which basis vector belongs to which subspace. The
        labels are 0 for the A (effective) subspace and 1 for the B (auxiliary)
        subspace.
        Only applicable if the unperturbed Hamiltonian is diagonal.
        Mutually exclusive with ``subspace_eigenvectors``.
    implicit :
        Whether to wrap the Hamiltonian of the BB subspace into a linear
        operator.
    symbols :
        Symbols that label the perturbative parameters of a symbolic
        Hamiltonian. The order of the symbols is mapped to the indices of the
        Hamiltonian, see `~pymablock.series.BlockSeries`. If None, the
        perturbative parameters are taken from the unperturbed Hamiltonian.
    atol :
        Absolute tolerance to consider matrices as exact zeros. This is used to
        validate that the unperturbed Hamiltonian is block-diagonal.

    Returns
    -------
    H : `~pymablock.series.BlockSeries`
        Initial Hamiltonian in the format required by the algorithms, such that
        the unperturbed Hamiltonian is block diagonal.

    """
    if subspace_eigenvectors is not None and subspace_indices is not None:
        raise ValueError(
            "subspace_eigenvectors and subspace_indices are mutually exclusive."
        )
    to_split = subspace_eigenvectors is not None or subspace_indices is not None

    # Convert anything to BlockSeries
    if isinstance(hamiltonian, list):
        hamiltonian = _list_to_dict(hamiltonian)
    elif isinstance(hamiltonian, sympy.MatrixBase):
        hamiltonian = _sympy_to_BlockSeries(hamiltonian, symbols)
    if isinstance(hamiltonian, dict):
        hamiltonian, symbols = _dict_to_BlockSeries(hamiltonian, symbols, atol)
    elif isinstance(hamiltonian, BlockSeries):
        pass
    else:
        raise TypeError("Unrecognized type for Hamiltonian.")
    assert isinstance(hamiltonian, BlockSeries)  # for mypy type checking

    if hamiltonian.shape and to_split:
        raise ValueError("H is already separated but subspace_eigenvectors are provided.")

    if hamiltonian.shape == (2, 2):
        return hamiltonian
    if hamiltonian.shape:
        raise ValueError("Only 2x2 block Hamiltonians are supported.")

    # Separation into subspace_eigenvectors
    if not to_split:
        # Hamiltonian must have 2x2 entries in each block
        def H_eval(*index):
            h = _convert_if_zero(hamiltonian[index[2:]], atol=atol)
            if h is zero:
                return zero
            try:  # Hamiltonians come in blocks of 2x2
                return _convert_if_zero(h[index[0]][index[1]], atol=atol)
            except Exception as e:
                raise ValueError(
                    "Without `subspace_eigenvectors` or `subspace_indices`"
                    " H must have a 2x2 block structure."
                ) from e

        H = BlockSeries(
            eval=H_eval,
            shape=(2, 2),
            n_infinite=hamiltonian.n_infinite,
            dimension_names=symbols,
            name="H",
        )
        return H

    # Define subspace_eigenvectors
    if subspace_indices is not None:
        h_0 = hamiltonian[(0,) * hamiltonian.n_infinite]
        if not is_diagonal(h_0, atol):
            raise ValueError(
                "If `subspace_indices` is provided, the unperturbed Hamiltonian"
                " must be diagonal."
            )
        symbolic = isinstance(h_0, sympy.MatrixBase)
        subspace_eigenvectors = _subspaces_from_indices(
            subspace_indices, symbolic=symbolic
        )
    if implicit:
        # Define subspace_eigenvectors for implicit
        vecs_A = subspace_eigenvectors[0]
        subspace_eigenvectors = (vecs_A, ComplementProjector(vecs_A))

    # Separation into subspace_eigenvectors
    def H_eval(*index):
        left, right = index[:2]
        if left > right:
            return Dagger(H[(right, left, *tuple(index[2:]))])
        original = hamiltonian[index[2:]]
        if original is zero:
            return zero
        if implicit and left == right == 1:
            original = aslinearoperator(original)
        return _convert_if_zero(
            Dagger(subspace_eigenvectors[left]) @ original @ subspace_eigenvectors[right],
            atol=atol,
        )

    H = BlockSeries(
        eval=H_eval,
        shape=(2, 2),
        n_infinite=hamiltonian.n_infinite,
        dimension_names=symbols,
        name="H",
    )

    return H


### Block diagonalization algorithms
def _block_diagonalize(
    H: BlockSeries,
    solve_sylvester: Optional[Callable] = None,
    *,
    operator: Optional[Callable] = None,
    return_all: bool = False,
) -> tuple[BlockSeries, ...] | tuple[dict[str, BlockSeries], dict[str, BlockSeries]]:
    """Algorithm for computing block diagonalization of a Hamiltonian.

    It parameterizes the unitary transformation as a series of block matrices.
    It computes them order by order by imposing unitarity and the
    block-diagonality of the transformed Hamiltonian.

    If the BB block of the Hamiltonian is a linear operator, the algorithm
    avoids forming the full matrices in the BB space.

    Parameters
    ----------
    H :
        Initial Hamiltonian, unperturbed and perturbation.
        The data in ``H`` can be either numerical or symbolic.
    solve_sylvester :
        (optional) function that solves the Sylvester equation.
        Defaults to a function that works for diagonal unperturbed Hamiltonians.
    operator :
        (optional) function to use for matrix multiplication.
        Defaults to matmul.
    return_all :
        (optional) whether to return all series as a dictionary.
        Defaults to `False`.

    Returns
    -------
    If `return_all` is false, it returns a tuple with the following elements:
    H_tilde : `~pymablock.series.BlockSeries`
        Block diagonalized Hamiltonian.
    U : `~pymablock.series.BlockSeries`
        Unitary that block diagonalizes H such that ``H_tilde = U^H H U``.
    U_adjoint : `~pymablock.series.BlockSeries`
        Adjoint of ``U``.

    If `return_all` is true, it returns a tuple with two dictionaries:
    series : dict[str, BlockSeries]
        Dictionary with all series.
    linear_operator_series : dict[str, BlockSeries]
        Dictionary with all series as linear operators.

    """
    if operator is None:
        operator = matmul

    if solve_sylvester is None:
        solve_sylvester = solve_sylvester_diagonal(*_extract_diagonal(H))

    zeroth_order = (0,) * H.n_infinite
    zero_data = {block + zeroth_order: zero for block in ((0, 0), (0, 1), (1, 0), (1, 1))}
    identity_data = {block + zeroth_order: one for block in ((0, 0), (1, 1))}
    H_0_data = {
        block + zeroth_order: H[block + zeroth_order]
        for block in ((0, 0), (0, 1), (1, 0), (1, 1))
    }
    data = {
        "zero_data": zero_data,
        "identity_data": identity_data,
        "H_0_data": H_0_data,
    }
    # Common series kwargs to avoid some repetition
    series_kwargs = dict(
        shape=(2, 2),
        n_infinite=H.n_infinite,
        dimension_names=H.dimension_names,
    )

    if use_implicit := isinstance(H[(1, 1, *zeroth_order)], sparse.linalg.LinearOperator):
        if operator is not matmul:
            raise ValueError("The implicit method only supports matrix multiplication.")

    series = {
        "H": H,
    }

    def linear_operator_wrapped(original: BlockSeries) -> BlockSeries:
        return BlockSeries(
            eval=(lambda *index: aslinearoperator(original[index])),
            name=original.name,
            **series_kwargs,
        )

    linear_operator_series = {
        "H": linear_operator_wrapped(H),
    }

    def del_(series_name, index: int) -> None:
        series[series_name].pop(index, None)
        linear_operator_series[series_name].pop(index, None)

    eval_scope = {
        # Locals
        "use_implicit": use_implicit,
        "series": series,
        "linear_operator_series": linear_operator_series,
        "solve_sylvester": solve_sylvester,
        "del_": del_,
        # Globals
        "zero": zero,
        "_safe_divide": _safe_divide,
        "_zero_sum": _zero_sum,
        "Dagger": Dagger,
    }

    terms, products, outputs = main

    for term in terms:
        # This defines `series_eval` as the eval function for this term.
        exec(compile(term.definition, filename="<string>", mode="exec"), eval_scope)

        series_data = data.get(term.start, None)

        series[term.name] = BlockSeries(
            eval=eval_scope["series_eval"],
            data=series_data,
            name=term.name,
            **series_kwargs,
        )
        linear_operator_series[term.name] = linear_operator_wrapped(series[term.name])

    for product in products:
        for which in series, linear_operator_series:
            which[product.name] = cauchy_dot_product(
                *(which[term] for term in product.terms),
                operator=operator,
                hermitian=product.hermitian,
            )

    if return_all:
        return series, linear_operator_series

    return tuple(series[output] for output in outputs)


### Different formats and algorithms of solving Sylvester equation.
def solve_sylvester_diagonal(
    eigs_A: Union[np.ndarray, sympy.matrices.MatrixBase],
    eigs_B: Union[np.ndarray, sympy.matrices.MatrixBase],
    vecs_B: Optional[np.ndarray] = None,
) -> Callable:
    """Define a function for solving a Sylvester's equation for diagonal matrices.

    Optionally, this function also applies the eigenvectors of the second
    matrix to the solution.

    Parameters
    ----------
    eigs_A :
        Eigenvalues of the effective (A) subspace of the unperturbed Hamiltonian.
    eigs_B :
        Eigenvalues of auxiliary (B) subspace of the unperturbed Hamiltonian.
    vecs_B :
        Eigenvectors of the auxiliary (B) subspace of the
        unperturbed Hamiltonian.

    Returns
    -------
    solve_sylvester : `Callable`
        Function that solves Sylvester's equation.

    """

    def solve_sylvester(
        Y: Union[np.ndarray, sparse.csr_array, sympy.MatrixBase],
    ) -> Union[np.ndarray, sparse.csr_array, sympy.MatrixBase]:
        if vecs_B is not None:
            energy_denominators = 1 / (eigs_A[:, None] - eigs_B[None, :])
            return ((Y @ vecs_B) * energy_denominators) @ Dagger(vecs_B)
        if isinstance(Y, np.ndarray):
            energy_denominators = 1 / (eigs_A.reshape(-1, 1) - eigs_B)
            return Y * energy_denominators
        if sparse.issparse(Y):
            Y_coo = Y.tocoo()
            # Sometimes eigs_A/eigs_B can be a scalar zero.
            eigs_A_select = eigs_A if not eigs_A.shape else eigs_A[Y_coo.row]
            eigs_B_select = eigs_B if not eigs_B.shape else eigs_B[Y_coo.col]
            energy_denominators = 1 / (eigs_A_select - eigs_B_select)
            new_data = Y_coo.data * energy_denominators
            return sparse.csr_array((new_data, (Y_coo.row, Y_coo.col)), Y_coo.shape)
        if isinstance(Y, sympy.MatrixBase):
            array_eigs_a = np.array(eigs_A, dtype=object)  # Use numpy to reshape
            array_eigs_b = np.array(eigs_B, dtype=object)
            energy_denominators = sympy.Matrix(
                np.resize(1 / (array_eigs_a.reshape(-1, 1) - array_eigs_b), Y.shape)
            )
            return energy_denominators.multiply_elementwise(Y)
        TypeError(f"Unsupported rhs type: {type(Y)}")

    return solve_sylvester


def solve_sylvester_KPM(
    h_0: Union[np.ndarray, sparse.spmatrix],
    subspace_eigenvectors: tuple[np.ndarray, ...],
    solver_options: Optional[dict] = None,
) -> Callable:
    """Solve Sylvester energy division for the Kernel Polynomial Method (KPM).

    General energy division for numerical problems through either full
    knowledge of the B-space or application of the KPM Green's function.

    Parameters
    ----------
    h_0 :
        Unperturbed Hamiltonian of the system.
    subspace_eigenvectors :
        Subspaces to project the unperturbed Hamiltonian and separate it into
        blocks. The first element of the tuple contains the effective subspace,
        and the second element contains the (partial) auxiliary subspace.
    solver_options :
        Dictionary containing any of the following options for KPM.

        - eps: float
            Tolerance for Hamiltonian rescaling.
        - atol: float
            Accepted precision of the Green's function result in 2-norm.
        - max_moments: int
            Maximum number of expansion moments of the Green's function.

    Returns
    -------
    solve_sylvester: Callable
        Function that applies divide by energies to the right hand side of
        Sylvester's equation.

    """
    eigs_A = (
        Dagger(subspace_eigenvectors[0]) @ h_0 @ subspace_eigenvectors[0]
    ).diagonal()
    if len(subspace_eigenvectors) > 2:
        raise ValueError("Invalid number of subspaces")
    if solver_options is None:
        solver_options = {}

    kpm_projector = ComplementProjector(np.hstack(subspace_eigenvectors))
    # Prepare the Hamiltonian for KPM by rescaling to [-1, 1]
    bounds_eigs_A = [np.min(eigs_A), np.max(eigs_A)]
    h_rescaled, (a, b) = rescale(
        h_0, eps=solver_options.get("eps", 0.01), lower_bounds=bounds_eigs_A
    )
    eigs_A_rescaled = (eigs_A - b) / a
    # We need to solve a transposed problem
    h_rescaled_T = h_rescaled.T
    # CSR format has a faster matrix-vector product
    if sparse.issparse(h_rescaled_T):
        h_rescaled_T = h_rescaled_T.tocsr()

    def solve_sylvester_kpm(Y: np.ndarray) -> np.ndarray:
        Y_KPM = Y @ kpm_projector / a  # Keep track of Hamiltonian rescaling
        return np.vstack(
            [
                greens_function(
                    h_rescaled_T,
                    energy,
                    vector,
                    solver_options.get("atol", 1e-5),
                    solver_options.get("max_moments", 1e6),
                )
                for energy, vector in zip(eigs_A_rescaled, Y_KPM)
            ]
        )

    need_explicit = bool(len(subspace_eigenvectors) - 1)
    if need_explicit:
        vecs_B = subspace_eigenvectors[1]
        eigs_B = (
            Dagger(subspace_eigenvectors[1]) @ h_0 @ subspace_eigenvectors[1]
        ).diagonal()
        solve_sylvester_explicit = solve_sylvester_diagonal(eigs_A, eigs_B, vecs_B)

    def solve_sylvester(Y: np.ndarray) -> np.ndarray:
        if need_explicit:
            return solve_sylvester_kpm(Y) + solve_sylvester_explicit(Y)
        return solve_sylvester_kpm(Y)

    return solve_sylvester


def solve_sylvester_direct(
    h_0: sparse.spmatrix,
    eigenvectors: np.ndarray,
    **solver_options: dict,
) -> Callable[[np.ndarray], np.ndarray]:
    """Solve Sylvester equation using a direct sparse solver.

    This function uses MUMPS, which is a parallel direct solver for sparse
    matrices. This solver is very efficient for large sparse matrices.

    Parameters
    ----------
    h_0 :
        Unperturbed Hamiltonian of the system.
    eigenvectors :
        Eigenvectors of the effective subspace of the unperturbed Hamiltonian.
    **solver_options :
        Keyword arguments to pass to the solver ``eps`` and ``atol``, see
        `pymablock.linalg.direct_greens_function`.

    Returns
    -------
    solve_sylvester : `Callable[[numpy.ndarray], numpy.ndarray]`
        Function that solves the corresponding Sylvester equation.

    """
    projector = ComplementProjector(eigenvectors)
    eigenvalues = np.diag(Dagger(eigenvectors) @ h_0 @ eigenvectors)
    # Compute the Green's function of the transposed Hamiltonian because we are
    # solving the equation from the right.
    greens_functions = [
        direct_greens_function(h_0.T, E, **solver_options) for E in eigenvalues
    ]

    def solve_sylvester(Y: np.ndarray) -> np.ndarray:
        Y = Y @ projector
        result = np.vstack([gf(vec) for gf, vec in zip(greens_functions, Y)])
        return result @ projector

    return solve_sylvester


### Auxiliary functions.
def _list_to_dict(hamiltonian: list[Any]) -> dict[int, Any]:
    """Convert a list of perturbations to a dictionary.

    Parameters
    ----------
    hamiltonian :
        Unperturbed Hamiltonian and 1st order perturbations.
        [h_0, h_1, h_2, ...], where h_0 is the unperturbed Hamiltonian and the
        remaining elements are the 1st order perturbations.
        This method is for first order perturbations only.

    Returns
    -------
    H : `~pymablock.series.BlockSeries`

    """
    n_infinite = len(hamiltonian) - 1  # All the perturbations are 1st order
    zeroth_order = (0,) * n_infinite

    hamiltonian = {
        zeroth_order: hamiltonian[0],
        **{
            tuple(order): perturbation
            for order, perturbation in zip(np.eye(n_infinite, dtype=int), hamiltonian[1:])
        },
    }
    return hamiltonian


def _dict_to_BlockSeries(
    hamiltonian: dict[tuple[int, ...], Any],
    symbols: Optional[Sequence[sympy.Symbol]] = None,
    atol: float = 1e-12,
) -> tuple[BlockSeries, Optional[list[sympy.Symbol]]]:
    """Convert a dictionary of perturbations to a BlockSeries.

    Parameters
    ----------
    hamiltonian :
        Unperturbed Hamiltonian and perturbations.
        The keys can be tuples of integers or symbolic monomials. They
        indicate the order of the perturbation in its respective value.
        The values are the perturbations, and can be either a `~numpy.ndarray`,
        `~scipy.sparse.csr_array` or a list with the blocks of the Hamiltonian.
        For example, {(0, 0): h_0, (1, 0): h_1, (0, 1): h_2} or
        {1: h_0, x: h_1, y: h_2}.
    symbols :
        tuple of symbols to use for the BlockSeries.
    atol :
        absolute tolerance for determining if a matrix is diagonal.

    Returns
    -------
    H : `~pymablock.series.BlockSeries`
    symbols :
        List of symbols in the order they appear in the keys of `hamiltonian`.
        The tuple keys of ``new_hamiltonian`` are ordered according to this list.

    """
    key_types = set(isinstance(key, sympy.Basic) for key in hamiltonian.keys())
    if any(key_types):
        hamiltonian, symbols = _symbolic_keys_to_tuples(hamiltonian)

    n_infinite = len(next(iter(hamiltonian.keys())))
    zeroth_order = (0,) * n_infinite
    h_0 = hamiltonian[zeroth_order]

    if isinstance(h_0, np.ndarray):
        if is_diagonal(h_0, atol):
            hamiltonian[zeroth_order] = sparse.csr_array(hamiltonian[zeroth_order])
    elif sparse.issparse(h_0):  # Normalize sparse matrices for solve_sylvester
        hamiltonian[zeroth_order] = sparse.csr_array(hamiltonian[zeroth_order])

    H_temporary = BlockSeries(
        data=copy(hamiltonian),
        shape=(),
        n_infinite=n_infinite,
        dimension_names=symbols,
        name="H",
    )
    return H_temporary, symbols


def _symbolic_keys_to_tuples(
    hamiltonian: dict[sympy.Basic, Any],
) -> tuple[dict[tuple[int, ...], Any], list[sympy.Basic]]:
    """Convert symbolic monomial keys to tuples of integers.

    The key for the unperturbed Hamiltonian is assumed to be 1, and the
    remaining keys are assumed to be symbolic monomials.

    Parameters
    ----------
    hamiltonian :
        Dictionary with symbolic keys, each a monomial without numerical
        prefactor. The values can be either a `~numpy.ndarray`,
        `~scipy.sparse.csr_array`, or a list with the blocks of the Hamiltonian.

    Returns
    -------
    new_hamiltonian :
        Dictionary with the same values as ``hamiltonian``, but with keys that
        indicate the order of the perturbation in a tuple.
    symbols :
        List of symbols in the order they appear in the keys of `hamiltonian`.
        The tuple keys of ``new_hamiltonian`` are ordered according to this list.

    """
    # Collect all symbols from the keys
    symbols = list(set.union(*[key.free_symbols for key in hamiltonian.keys()]))
    symbols = tuple(sorted(symbols, key=lambda x: x.name))
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
    symbols: Optional[Sequence[sympy.Symbol]] = None,
) -> BlockSeries:
    """Convert a symbolic Hamiltonian to a BlockSeries.

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
    H : `~pymablock.series.BlockSeries`

    """
    if symbols is None:
        symbols = tuple(list(hamiltonian.free_symbols))  # All symbols are perturbative
    if any(n not in hamiltonian.free_symbols for n in symbols):
        raise ValueError("Not all perturbative parameters are in `hamiltonian`.")

    hamiltonian = hamiltonian.expand()

    def H_eval(*index):
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
        dimension_names=symbols,
        name="H",
    )
    return H


def _subspaces_from_indices(
    subspace_indices: Union[tuple[int, ...], np.ndarray],
    symbolic: Optional[bool] = False,
) -> tuple[sparse.csr_array, sparse.csr_array]:
    """Compute subspace eigenvectors from indices of diagonal elements.

    Parameters
    ----------
    subspace_indices :
        Indices of the ``subspace_eigenvectors``.
        0 indicates the effective subspace A, 1 indicates the auxiliary
        subspace B.
    symbolic :
        True if the Hamiltonian is symbolic, False otherwise.
        If True, the returned subspaces are dense arrays.

    Returns
    -------
    subspace_eigenvectors :
        Subspaces to use for block diagonalization.

    """
    subspace_indices = np.array(subspace_indices)
    max_subspaces = 2
    if np.any(subspace_indices >= max_subspaces):
        raise ValueError(
            "Only 0 and 1 are allowed as indices for ``subspace_eigenvectors``."
        )
    dim = len(subspace_indices)
    eigvecs = sparse.csr_array(sparse.identity(dim, dtype=int, format="csr"))
    # Canonical basis vectors for each subspace
    subspace_eigenvectors = tuple(
        eigvecs[:, np.compress(subspace_indices == block, np.arange(dim))]
        for block in range(max_subspaces)
    )
    if symbolic:
        # Convert to dense arrays, otherwise they cannot multiply sympy matrices.
        return tuple(subspace.toarray() for subspace in subspace_eigenvectors)
    return subspace_eigenvectors


def _extract_diagonal(
    H: BlockSeries,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the diagonal of the zeroth order of the Hamiltonian."""
    diag_indices = np.arange(H.shape[0])
    h_0 = H[(diag_indices, diag_indices) + (0,) * H.n_infinite]
    is_sympy = any(isinstance(block, sympy.MatrixBase) for block in h_0)
    if not all(is_diagonal(h, atol) for h in h_0):
        warn(
            "Cannot confirm that the unperturbed Hamiltonian is diagonal, "
            "which is required if ``solve_sylvester`` is not provided. "
            "The algorithm will assume that it is diagonal.",
            UserWarning,
        )
    diags = []
    for block in h_0:
        if block is zero or block is np.ma.masked:
            diags.append(np.array(0))
            continue
        eigs = block.diagonal()
        if is_sympy:
            eigs = np.array(eigs, dtype=object)
        diags.append(eigs)

    compare = np.equal if is_sympy else np.isclose
    if np.any(compare(diags[0].reshape(-1, 1), diags[1].reshape(1, -1))):
        raise ValueError("The subspaces must not share eigenvalues.")

    return tuple(diags)


def _convert_if_zero(value: Any, atol=1e-12):
    """Convert an exact zero to sentinel value zero.

    Parameters
    ----------
    value :
        Value to convert to zero.
    atol :
        Absolute tolerance for numerical zero.

    Returns
    -------
    zero :
        Zero if value is close enough to zero, otherwise value.

    """
    if isinstance(value, np.ndarray):
        if np.allclose(value, 0, atol=atol):
            return zero
    elif sparse.issparse(value):
        if value.count_nonzero() == 0:
            return zero
    elif isinstance(value, sympy.MatrixBase):
        if value.is_zero_matrix:
            return zero
    elif value == 0:
        return zero
    return value


def _check_orthonormality(subspace_eigenvectors, atol=1e-12):
    """Check that the eigenvectors are orthonormal."""
    if isinstance(subspace_eigenvectors[0], np.ndarray):
        all_vecs = np.hstack(subspace_eigenvectors)
        overlap = Dagger(all_vecs) @ all_vecs
        if not np.allclose(overlap, np.eye(all_vecs.shape[1]), atol=atol):
            raise ValueError("Eigenvectors must be orthonormal.")
    elif isinstance(subspace_eigenvectors[0], sympy.MatrixBase):
        all_vecs = sympy.Matrix.hstack(*subspace_eigenvectors)
        overlap = Dagger(all_vecs) @ all_vecs
        # Use sympy three-valued logic
        if sympy.Eq(overlap, sympy.eye(all_vecs.shape[1])) == False:  # noqa: E712
            raise ValueError("Eigenvectors must be orthonormal.")


def _zero_sum(*terms: Any) -> Any:
    """Sum that returns a singleton zero if empty and omits zero terms.

    Parameters
    ----------
    terms :
        Terms to sum over with zero as default value.

    Returns
    -------
    Sum of terms, or zero if terms is empty.

    """
    return sum((term for term in terms if term is not zero), start=zero)


def _safe_divide(numerator, denominator):
    """Divide unless it's impossible, then multiply by inverse."""
    try:
        return numerator / denominator
    except TypeError:
        return numerator * (1 / denominator)
