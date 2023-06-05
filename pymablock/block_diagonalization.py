# Algorithms for quasi-degenerate perturbation theory
from operator import matmul, mul
from functools import reduce
from typing import Any, Optional, Callable, Union
from collections.abc import Sequence
from copy import copy

import numpy as np
import sympy
from scipy import sparse
from sympy.physics.quantum import Dagger, Operator, HermitianOperator

from pymablock.linalg import (
    ComplementProjector,
    aslinearoperator,
    is_diagonal,
    direct_greens_function,
)
from pymablock.kpm import greens_function, rescale
from pymablock.series import (
    BlockSeries,
    zero,
    one,
    cauchy_dot_product,
    _zero_sum,
    safe_divide,
)

__all__ = ["block_diagonalize", "general", "expanded", "symbolic", "implicit"]

# Common types
Eigenvectors = tuple[Union[np.ndarray, sympy.Matrix], ...]


### The main function for end-users.
def block_diagonalize(
    hamiltonian: Union[list, dict, BlockSeries, sympy.Matrix],
    *,
    algorithm: Optional[str] = None,
    solve_sylvester: Optional[Callable] = None,
    subspace_eigenvectors: Optional[Eigenvectors] = None,
    subspace_indices: Optional[Union[tuple[int, ...], np.ndarray]] = None,
    direct_solver: bool = True,
    solver_options: Optional[dict] = None,
    symbols: Optional[Union[sympy.Symbol, Sequence[sympy.Symbol]]] = None,
    atol: float = 1e-12,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Find the block diagonalization of a Hamiltonian order by order.

    This uses quasi-degenerate perturbation theory known as Lowdin perturbation
    theory, Schrieffer-Wolff transformation, or van Vleck transformation.

    This function does not yet perform the computation. Instead, it defines the
    computation as a `~pymablock.series.BlockSeries` object, which can be
    evaluated at any order.

    This function accepts a Hamiltonian in several formats and it first
    brings it to the eigenbasis of the unperturbed Hamiltonian if the blocks,
    eigenvectors or indices of the eigenvalues are provided, see below.

    The block diagonalization is performed using the `expanded` or `general`
    algorithm. The former is better suited for lower order numerical
    calculations and symbolic ones. The latter is better suited for higher
    order or numerical calculations.

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
            `~pymablock.series.BlockSeries` is done by Taylor expanding on
            ``symbols`` to the desired order.
        - A `~pymablock.series.BlockSeries`,
            returned unchanged.
    algorithm :
        Name of the function that block diagonalizes the Hamiltonian.
        Options are "general" and "expanded".
    solve_sylvester :
        A function that solves the Sylvester equation. If not provided,
        it is selected automatically based on the inputs.
    subspace_eigenvectors :
        A tuple with orthonormal eigenvectors to project the Hamiltonian on
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
        Otherwise, the an experimental KPM solver is used. Only applicable if
        the implicit method is used (i.e. `subspace_vectors` is incomplete)
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
            raise ValueError(
                "`subspace_eigenvectors` does not match the shape of `h_0`."
            )
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

    if zero != H[(0, 1) + (0,) * H.n_infinite]:
        raise ValueError(
            "The off-diagonal elements of the unperturbed Hamiltonian must be zero."
        )

    # Determine operator to use for matrix multiplication.
    if hasattr(H[(0, 0) + (0,) * H.n_infinite], "__matmul__"):
        operator = matmul
    else:
        operator = mul

    # If solve_sylvester is not yet defined, use the diagonal one.
    if solve_sylvester is None:
        solve_sylvester = solve_sylvester_diagonal(*_extract_diagonal(H, atol))

    if algorithm is None:
        # symbolic expressions benefit from no H_0 in numerators
        algorithm = "expanded" if H.dimension_names != () else "general"
    if algorithm not in ("general", "expanded"):
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if use_implicit:
        return implicit(
            H,
            solve_sylvester=solve_sylvester,
            algorithm=algorithm,
        )
    return _algorithms[algorithm](
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
    """
    Normalize a Hamiltonian to be used by the algorithms.

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
        raise ValueError(
            "H is already separated but subspace_eigenvectors are provided."
        )

    if hamiltonian.shape == (2, 2):
        return hamiltonian
    elif hamiltonian.shape:
        raise ValueError("Only 2x2 block Hamiltonians are supported.")

    # Separation into subspace_eigenvectors
    if not to_split:
        # Hamiltonian must have 2x2 entries in each block
        def H_eval(*index):
            h = _convert_if_zero(hamiltonian[index[2:]], atol=atol)
            if zero == h:
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
            return Dagger(H[(right, left) + tuple(index[2:])])
        original = hamiltonian[index[2:]]
        if zero == original:
            return zero
        if implicit and left == right == 1:
            original = aslinearoperator(original)
        return _convert_if_zero(
            Dagger(subspace_eigenvectors[left])
            @ original
            @ subspace_eigenvectors[right],
            atol=atol,
        )

    H = BlockSeries(
        eval=H_eval,
        shape=(2, 2),
        n_infinite=hamiltonian.n_infinite,
        dimension_names=symbols,
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
    Algorithm for computing block diagonalization of a Hamiltonian.

    It parameterizes the unitary transformation as a series of block matrices.
    It computes them order by order by imposing unitarity and the
    block-diagonality of the transformed Hamiltonian.

    The computational cost of this algorithm scales favorably with the order
    of the perturbation. However, it performs unnecessary matrix products at
    lowest orders, and keeps the unperturbed Hamiltonian in the numerator. This
    makes this algorithm better suited for higher order numerical calculations.

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

    Returns
    -------
    H_tilde : `~pymablock.series.BlockSeries`
        Block diagonalized Hamiltonian.
    U : `~pymablock.series.BlockSeries`
        Unitary that block diagonalizes H such that ``H_tilde = U^H H U``.
    U_adjoint : `~pymablock.series.BlockSeries`
        Adjoint of ``U``.
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
        dimension_names=H.dimension_names,
    )

    U_adjoint = BlockSeries(
        eval=(
            lambda *index: U[index]  # diagonal block is Hermitian
            if index[0] == index[1]
            else -U[index]  # off-diagonal block is anti-Hermitian
        ),
        data=None,
        shape=(2, 2),
        n_infinite=H.n_infinite,
        dimension_names=H.dimension_names,
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

    def eval(*index: int) -> Any:
        if index[0] == index[1]:
            # diagonal is constrained by unitarity
            return safe_divide(-identity[index], 2)
        elif index[:2] == (0, 1):
            # off-diagonal block nullifies the off-diagonal part of H_tilde
            Y = H_tilde_rec[index]
            return -solve_sylvester(Y) if zero != Y else zero
        elif index[:2] == (1, 0):
            # off-diagonal of U is anti-Hermitian
            return -Dagger(U[(0, 1) + tuple(index[2:])])

    U.eval = eval

    H_tilde = cauchy_dot_product(U_adjoint, H, U, operator=operator, hermitian=True)
    return H_tilde, U, U_adjoint


def symbolic(
    H: BlockSeries,
) -> tuple[
    BlockSeries, BlockSeries, BlockSeries, dict[Operator, Any], dict[Operator, Any]
]:
    """
    General symbolic algorithm for block diagonalizing a Hamiltonian.

    This function uses symbolic algebra to compute the block diagonalization,
    producing formulas that contain the orders of the perturbation and the
    off-diagonal blocks of the unitary transformation ``U``.

    This function is general, therefore the solutions to the Sylvester equation
    are not computed. Instead, the solutions are stored in a dictionary that
    is updated whenever new terms of the Hamiltonian are evaluated.

    Parameters
    ----------
    H :
        The perturbed Hamiltonian. The algorithm only checks which terms are present in
        the Hamiltonian, but does not substitute them.

    Returns
    -------
    H_tilde : `~pymablock.series.BlockSeries`
        Symbolic diagonalized Hamiltonian.
    U : `~pymablock.series.BlockSeries`
        Symbolic unitary matrix that block diagonalizes H such that
        ``U * H * U^H = H_tilde``.
    U_adjoint : `~pymablock.series.BlockSeries`
        Symbolic adjoint of ``U``. Its diagonal blocks are Hermitian and its
        off-diagonal blocks (``V``) are anti-Hermitian.
    Y_data : `dict`
        dictionary of ``{V: rhs}`` such that ``h_0_AA * V - V * h_0_BB = rhs``.
        It is updated whenever new terms of ``H_tilde`` or ``U`` are evaluated.
    subs : `dict`
        Dictionary with placeholder symbols as keys and original Hamiltonian
        terms as values.
    """
    subs = {}

    # Initialize symbols for terms in H. Ensure that H_0 is there.
    h_0_indices = [(i, i) + (0,) * H.n_infinite for i in range(2)]

    def placeholder_eval(*index):
        if zero == (actual_value := H[index]) and index not in h_0_indices:
            return zero
        operator_type = HermitianOperator if index[0] == index[1] else Operator
        placeholder = operator_type(f"H_{{{index}}}")
        subs[placeholder] = actual_value
        return placeholder

    H_placeholder = BlockSeries(
        eval=placeholder_eval,
        shape=H.shape,
        n_infinite=H.n_infinite,
        dimension_names=H.dimension_names,
    )
    h_0 = [H_placeholder[index] for index in h_0_indices]

    # Solve for symbols representing H and V (off-diagonal block of U)
    H_tilde, U, U_adjoint = general(
        H_placeholder, solve_sylvester=(lambda x: x), operator=mul
    )

    Y_data = {}

    old_U_eval = U.eval

    def U_eval(*index):
        if index[:2] == (0, 1):
            V = Operator(f"V_{{{index[2:]}}}")
            # Apply h_0_AA * V - V * h_0_BB = rhs to eliminate h_0 terms
            Y = _commute_h0_away(old_U_eval(*index), *h_0, Y_data)
            if zero == Y:
                return zero
            Y_data[V] = Y
            return V
        return old_U_eval(*index)

    U.eval = U_eval

    old_H_tilde_eval = H_tilde.eval

    def H_tilde_eval(*index):
        return _commute_h0_away(old_H_tilde_eval(*index), *h_0, Y_data)

    H_tilde.eval = H_tilde_eval

    return H_tilde, U, U_adjoint, Y_data, subs


def expanded(
    H: BlockSeries,
    solve_sylvester: Optional[Callable] = None,
    *,
    operator: Optional[Callable] = None,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Algorithm for computing block diagonalization of a Hamiltonian.

    Unlike the `general` algorithm, this algorithm does not perform
    multiplications by the unperturbed Hamiltonian. This comes at the cost of
    needing exponentially many matrix multiplications at higher orders.
    This makes this algorithm better suited for lower orders numerical
    calculations and symbolic ones.

    Parameters
    ----------
    H :
        Initial Hamiltonian, unperturbed and perturbation.
    solve_sylvester :
        Function to use for solving Sylvester's equation.
        Defaults to a function that works for diagonal unperturbed Hamiltonians.
    operator :
        Function to use for matrix multiplication.
        Defaults to ``matmul``.

    Returns
    -------
    H_tilde : `~pymablock.series.BlockSeries`
        Diagonalized Hamiltonian.
    U : `~pymablock.series.BlockSeries`
        Unitary matrix that block diagonalizes H such that
        ``U * H * U^H = H_tilde``.
    U_adjoint : `~pymablock.series.BlockSeries`
        Adjoint of ``U``.
    """
    if operator is None:
        operator = matmul

    if solve_sylvester is None:
        solve_sylvester = solve_sylvester_diagonal(*_extract_diagonal(H))

    H_tilde_s, U_s, _, Y_data, subs = symbolic(H)
    _, U, U_adjoint = general(H, solve_sylvester=solve_sylvester, operator=operator)

    def H_tilde_eval(*index):
        H_tilde = H_tilde_s[index]
        _update_subs(Y_data, subs, solve_sylvester, operator)
        return _replace(H_tilde, subs, operator)

    H_tilde = BlockSeries(
        eval=H_tilde_eval,
        shape=(2, 2),
        n_infinite=H.n_infinite,
        dimension_names=H.dimension_names,
    )

    old_U_eval = U.eval

    def U_eval(*index):
        if index[:2] == (0, 1):
            U_s[index]  # Update Y_data
            _update_subs(Y_data, subs, solve_sylvester, operator)
            return subs.get(Operator(f"V_{{{index[2:]}}}"), zero)
        return old_U_eval(*index)

    U.eval = U_eval

    return H_tilde, U, U_adjoint


_algorithms = {"general": general, "expanded": expanded}


def implicit(
    H: BlockSeries,
    solve_sylvester: Callable,
    algorithm: str = "general",
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """
    Block diagonalize a Hamiltonian without explicitly forming BB matrices.

    This function uses either the `general` or `expanded` algorithm to block
    diagonalize, but does not compute products within the B (auxiliary)
    subspace. Instead these matrices are wrapped in
    `~scipy.sparse.linalg.LinearOperator` and combined to keep them low rank.

    This function is useful for large numeric Hamiltonians where the effective
    subspace is small, but the full Hamiltonian is large.

    Parameters
    ----------
    H :
        Full Hamiltonian of the system.
    solve_sylvester :
        Function to use for solving Sylvester's equation.
    algorithm :
        Algorithm to use for diagonalization. One of "general", "expanded".
        The "general" (default) is faster in higher orders.

    Returns
    -------
    H_tilde : `~pymablock.series.BlockSeries`
        Full block-diagonalized Hamiltonian of the problem. The ``(0, 0)`` block
        (A subspace) is a numpy array, while the ``(1, 1)`` block (B subspace)
        is a `~scipy.sparse.linalg.LinearOperator`.
    U : `~pymablock.series.BlockSeries`
        Unitary that block diagonalizes the initial Hamiltonian.
    U_adjoint : `~pymablock.series.BlockSeries`
        Adjoint of ``U``.
    """
    if algorithm not in _algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    algorithm = _algorithms[algorithm]
    H_tilde_temporary, U, U_adjoint = algorithm(H, solve_sylvester=solve_sylvester)

    # Create series wrapped in linear operators to avoid forming explicit matrices
    def linear_operator_wrapped(original):
        return lambda *index: aslinearoperator(original[index])

    H_operator, U_operator, U_adjoint_operator = (
        BlockSeries(
            eval=linear_operator_wrapped(original),
            shape=(2, 2),
            n_infinite=H.n_infinite,
            dimension_names=original.dimension_names,
        )
        for original in (H, U, U_adjoint)
    )
    identity = cauchy_dot_product(
        U_operator, U_adjoint_operator, hermitian=True, exclude_last=[True, True]
    )

    old_U_eval = U.eval

    def U_eval(*index):
        if index[:2] == (1, 1):
            return safe_divide(-identity[index], 2)
        return old_U_eval(*index)

    U.eval = U_eval

    H_tilde_operator = cauchy_dot_product(
        U_adjoint_operator, H_operator, U_operator, hermitian=True
    )

    def H_tilde_eval(*index):
        if index[:2] == (1, 1):
            return H_tilde_operator[index]
        return H_tilde_temporary[index]

    H_tilde = BlockSeries(
        eval=H_tilde_eval,
        shape=(2, 2),
        n_infinite=H.n_infinite,
        dimension_names=H.dimension_names,
    )

    return H_tilde, U, U_adjoint


### Different formats and algorithms of solving Sylvester equation.
def solve_sylvester_diagonal(
    eigs_A: Union[np.ndarray, sympy.MatrixBase],
    eigs_B: Union[np.ndarray, sympy.MatrixBase],
    vecs_B: Optional[np.ndarray] = None,
) -> Callable:
    """
    Define a function for solving a Sylvester's equation for diagonal matrices.

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
    solve_sylvester : Function that solves Sylvester's equation.
    """

    def solve_sylvester(
        Y: Union[np.ndarray, sparse.csr_array, sympy.MatrixBase],
    ) -> Union[np.ndarray, sparse.csr_array, sympy.MatrixBase]:
        if vecs_B is not None:
            energy_denominators = 1 / (eigs_A[:, None] - eigs_B[None, :])
            return ((Y @ vecs_B) * energy_denominators) @ Dagger(vecs_B)
        elif isinstance(Y, np.ndarray):
            energy_denominators = 1 / (eigs_A.reshape(-1, 1) - eigs_B)
            return Y * energy_denominators
        elif sparse.issparse(Y):
            Y_coo = Y.tocoo()
            # Sometimes eigs_A/eigs_B can be a scalar zero.
            eigs_A_select = eigs_A if not eigs_A.shape else eigs_A[Y_coo.row]
            eigs_B_select = eigs_B if not eigs_B.shape else eigs_B[Y_coo.col]
            energy_denominators = 1 / (eigs_A_select - eigs_B_select)
            new_data = Y_coo.data * energy_denominators
            return sparse.csr_array((new_data, (Y_coo.row, Y_coo.col)), Y_coo.shape)
        elif isinstance(Y, sympy.MatrixBase):
            array_eigs_a = np.array(eigs_A, dtype=object)  # Use numpy to reshape
            array_eigs_b = np.array(eigs_B, dtype=object)
            energy_denominators = sympy.Matrix(
                np.resize(1 / (array_eigs_a.reshape(-1, 1) - array_eigs_b), Y.shape)
            )
            return energy_denominators.multiply_elementwise(Y)
        else:
            TypeError(f"Unsupported rhs type: {type(Y)}")

    return solve_sylvester


def solve_sylvester_KPM(
    h_0: Union[np.ndarray, sparse.spmatrix],
    subspace_eigenvectors: tuple[np.ndarray, ...],
    solver_options: Optional[dict] = None,
) -> Callable:
    """
    Solve Sylvester energy division for the Kernel Polynomial Method (KPM).

    General energy division for numerical problems through either full
    knowledge of the B-space or application of the KPM Green's function.

    This is an experimental feature and is not yet fully supported.

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
        - bounds: tuple[float, float]
            ``(E_min, E_max)`` spectral bounds of the Hamiltonian, used to rescale
            inside an interval ``[-1, 1]``.
        - num_moments: int
            Number of moments to use for KPM.
        - energy_resolution: float
            Relative energy resolution of KPM. If provided, overrides ``num_moments``.

    Returns
    ----------
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
    h_rescaled, (a, b) = rescale(h_0, eps=solver_options.get("eps", 0.01))
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
                    solver_options.get("num_moments", 100),
                    solver_options.get("energy_resolution"),
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
    """
    Solve Sylvester equation using a direct sparse solver.

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
        dictionary of ``{V: rhs}`` such that ``h_0_AA * V - V * h_0_BB = rhs``.

    Returns
    -------
    expr : zero or sympy.expr
        (zero or sympy) expression without ``h_0_AA`` or ``h_0_BB`` in it.
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
    subs: dict[Union[Operator, HermitianOperator], Any],
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
    expr: Any, subs: dict[Union[Operator, HermitianOperator], Any], operator: Callable
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
    Convert a list of perturbations to a dictionary.

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
            for order, perturbation in zip(
                np.eye(n_infinite, dtype=int), hamiltonian[1:]
            )
        },
    }
    return hamiltonian


def _dict_to_BlockSeries(
    hamiltonian: dict[tuple[int, ...], Any],
    symbols: Optional[Sequence[sympy.Symbol]] = None,
    atol: float = 1e-12,
) -> BlockSeries:
    """
    Convert a dictionary of perturbations to a BlockSeries.

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
    symbols :
        tuple of symbols to use for the BlockSeries.
    atol :
        absolute tolerance for determining if a matrix is diagonal.

    Returns
    -------
    H : `~pymablock.series.BlockSeries`
    """
    key_types = set(isinstance(key, sympy.Basic) for key in hamiltonian.keys())
    if any(key_types):
        hamiltonian, symbols = _symbolic_keys_to_tuples(hamiltonian)

    n_infinite = len(list(hamiltonian.keys())[0])
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
    )
    return H_temporary, symbols


def _symbolic_keys_to_tuples(
    hamiltonian: dict[sympy.Basic, Any]
) -> tuple[dict[tuple[int, ...], Any], list[sympy.Basic]]:
    """
    Convert symbolic monomial keys to tuples of integers.

    The key for the unperturbed Hamiltonian is assumed to be 1, and the
    remaining keys are assumed to be symbolic monomials.

    Parameters
    ----------
    hamiltonian :
        Dictionary with symbolic keys, each a monomial without numerical
        prefactor. The values can be either a `~numpy.ndarray`,
        `~scipy.sparse.csr_matrix`, or a list with the blocks of the Hamiltonian.

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
    )
    return H


def _subspaces_from_indices(
    subspace_indices: Union[tuple[int, ...], np.ndarray],
    symbolic: Optional[bool] = False,
) -> tuple[sparse.csr_array, sparse.csr_array]:
    """
    Returns the subspace_eigenvectors projection from the indices of the
    elements of the diagonal.

    Parameters
    ----------
    subspace_indices :
        Indices of the ``subspace_eigenvectors``.
        0 indicates the effective subspace A, 1 indicates the auxiliary
        subspace B.

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
        raise ValueError(
            "The unperturbed Hamiltonian must be diagonal if ``solve_sylvester``"
            " is not provided."
        )
    diags = []
    for block in h_0:
        if zero == block or block is np.ma.masked:
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
    atol :
        Absolute tolerance for numerical zero.
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
