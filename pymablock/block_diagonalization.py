"""Algorithms for quasi-degenerate perturbation theory."""

from collections.abc import Callable, Sequence
from copy import copy
from functools import partial, reduce
from inspect import signature
from operator import matmul, mul
from typing import Any
from warnings import warn

import numpy as np
import sympy
from scipy import sparse
from sympy.physics.quantum import Dagger, Operator

from pymablock import second_quantization
from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main
from pymablock.kpm import greens_function, rescale
from pymablock.linalg import (
    ComplementProjector,
    aslinearoperator,
    direct_greens_function,
    is_diagonal,
)
from pymablock.number_ordered_form import NumberOrderedForm, find_operators
from pymablock.series import (
    BlockSeries,
    zero,
)

__all__ = ["block_diagonalize", "operator_to_BlockSeries"]

# Common types
Eigenvectors = tuple[np.ndarray | sympy.Matrix, ...]
Mask = second_quantization.Mask


### The main function for end-users.
def block_diagonalize(
    hamiltonian: list | dict | BlockSeries | sympy.Matrix,
    *,
    solve_sylvester: Callable | None = None,
    subspace_eigenvectors: Eigenvectors | None = None,
    subspace_indices: tuple[int, ...] | np.ndarray | None = None,
    direct_solver: bool = True,
    solver_options: dict | None = None,
    symbols: sympy.Symbol | Sequence[sympy.Symbol] | None = None,
    atol: float = 1e-12,
    fully_diagonalize: (
        tuple[int] | dict[int, np.ndarray | Mask] | np.ndarray | Mask
    ) = (),
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """Find the block diagonalization of a Hamiltonian order by order.

    This uses a generalization of quasi-degenerate perturbation theory known as
    Lowdin perturbation theory, Schrieffer-Wolff transformation, or van Vleck
    transformation to the case of multiple blocks. Some blocks of the resulting
    Hamiltonian can be fully diagonalized, reproducing the usual
    Rayleigh-Schrodinger perturbation theory. Alternatively, the algorithm can
    perturbatively eliminate any subset of the off-diagonal elements of the
    Hamiltonian.

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
        The Hamiltonian is normalized to a `~pymablock.series.BlockSeries`.

        Supported formats:

        - A list,
            of the form ``[h_0, h_1, h_2, ...]`` where ``h_0`` is the unperturbed
            Hamiltonian and ``h_1, h_2, ...`` are the first order perturbations. The
            elements ``h_i`` may be `~sympy.matrices.dense.Matrix`, `~numpy.ndarray`,
            `~scipy.sparse.spmatrix`, that require separating into blocks. Otherwise,
            ``h_i`` may be a list of lists with the Hamiltonian blocks.
        - A dictionary,
            of the form ``{(0, 0): h_0, (1, 0): h_1, (0, 1): h_2}``, or ``{1:
            h_0, x: h_1, y: h_2}`` for symbolic Hamiltonians. In the former
            case, the keys must be tuples of integers indicating the order of
            each perturbation. In the latter case, the keys must be monomials
            and the indices are ordered as in `H.dimension_names`. The values
            of the dictionary, ``h_i`` have the same format as in the list case.
        - A `~sympy.matrices.dense.Matrix`,
            unless a list of ``symbols`` is provided as perturbative parameters,
            all symbols will be treated as perturbative. The normalization to
            `~pymablock.series.BlockSeries` is done by Taylor expanding in
            ``symbols`` to the desired order.
        - A `~pymablock.series.BlockSeries`,
            used directly.
    solve_sylvester :
        A function that solves the Sylvester equation. If not provided,
        it is selected automatically based on the inputs.
    subspace_eigenvectors :
        A tuple with sets of orthonormal eigenvectors to project the Hamiltonian onto
        and separate it into blocks. If None, the unperturbed Hamiltonian must be block
        diagonal. If some vectors are missing, the implicit method is used. Mutually
        exclusive with ``subspace_indices``. If neither
        ``subspace_eigenvectors`` nor ``subspace_indices`` are provided, the
        BlockSeries is defined with a single block.
    subspace_indices :
        An array indicating which state belongs to which subspace. If there are
        two blocks, the labels are 0 for the A (effective) subspace and 1 for
        the B (auxiliary) subspace. Only applicable if the unperturbed
        Hamiltonian is diagonal. Mutually exclusive with
        ``subspace_eigenvectors``.
        If neither ``subspace_eigenvectors`` nor ``subspace_indices`` are provided,
        the BlockSeries is defined with a single block.
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
    fully_diagonalize :
        Indices of the blocks that should be fully diagonalized. If the Hamiltonian only
        has one block, it is fully diagonalized by default. Alternatively can be a
        dictionary with the indices of the diagonal blocks as keys and as values
        appropriately shaped numpy boolean arrays marking which matrix elements should
        be eliminated by the diagonalization. If there is only one block, the boolean
        array may be provided directly without a dictionary. Must be symmetric, and may
        not have any True values corresponding to matrix elements coupling degenerate
        eigenvalues.

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

    if solve_sylvester is not None and fully_diagonalize:
        raise NotImplementedError(
            "Full diagonalization is not yet supported with custom Sylvester solvers."
        )

    # We may not use the full operator_to_BlockSeries here because we need
    # access to the unperturbed Hamiltonian in the implicit mode.
    #
    # Because both functions below are idempotent, this does not lead to inconsistencies
    # with operator_to_BlockSeries in the code below.
    hamiltonian = _unpack_blocks(_to_scalar_BlockSeries(hamiltonian, symbols, atol), atol)

    use_implicit = False
    if subspace_eigenvectors is not None:
        _check_orthonormality(subspace_eigenvectors, atol=atol)
        num_vectors = sum(vecs.shape[1] for vecs in subspace_eigenvectors)
        dim = subspace_eigenvectors[0].shape[0]
        use_implicit = num_vectors < dim

    if use_implicit:
        if hamiltonian.shape:
            raise ValueError("Implicit mode requires an input not separated into blocks")
        h_0 = hamiltonian[(0,) * hamiltonian.n_infinite]
        if isinstance(h_0, sympy.MatrixBase):
            raise ValueError("Implicit mode is not supported with symbolic Hamiltonian.")
        assert subspace_eigenvectors is not None  # for mypy
        # Build solve_sylvester
        if hamiltonian.shape:
            raise ValueError("Implicit mode requires an input not separated into blocks")
        if h_0.shape[0] != subspace_eigenvectors[0].shape[0]:
            raise ValueError("`subspace_eigenvectors` does not match the shape of `h_0`.")
        if solve_sylvester is None:
            if not all(isinstance(vecs, np.ndarray) for vecs in subspace_eigenvectors):
                raise TypeError(
                    "Implicit problem requires numpy arrays for eigenvectors."
                )
            if direct_solver:
                solve_sylvester = solve_sylvester_direct(h_0, subspace_eigenvectors)
            else:
                solve_sylvester = solve_sylvester_KPM(
                    h_0,
                    subspace_eigenvectors,
                    solver_options=solver_options,
                )

    # Normalize the Hamiltonian
    H = operator_to_BlockSeries(
        hamiltonian,
        name="H",
        subspace_eigenvectors=subspace_eigenvectors,
        subspace_indices=subspace_indices,
        implicit=use_implicit,
        symbols=symbols,
        atol=atol,
        hermitian=True,
    )

    if H.shape[0] == 1:
        if not len(fully_diagonalize):
            fully_diagonalize = (0,)
        elif isinstance(fully_diagonalize, np.ndarray):
            fully_diagonalize = {0: fully_diagonalize}
    else:
        if isinstance(fully_diagonalize, np.ndarray):
            raise ValueError(
                "If the Hamiltonian has multiple blocks, `fully_diagonalize` may not be an ndarray."
            )

    zero_order = (0,) * H.n_infinite

    for j in range(1, H.shape[0]):
        for i in range(j):
            if H[(i, j, *zero_order)] is not zero:
                raise ValueError(
                    "The off-diagonal elements of the unperturbed Hamiltonian must be zero."
                )

    # Determine operator to use for multiplication. We prefer matmul, and use mul if
    # matmul is not available.
    H_0_diag = [H[(i, i, *zero_order)] for i in range(H.shape[0])]
    nonzero_blocks = [block for block in H_0_diag if block is not zero]
    if not nonzero_blocks:
        raise ValueError("The diagonal of the unperturbed Hamiltonian may not be zero.")
    if all(hasattr(H, "__matmul__") for H in nonzero_blocks):
        operator = matmul
    elif all(hasattr(H, "__mul__") for H in nonzero_blocks):
        operator = mul
    else:
        raise ValueError("The unperturbed Hamiltonian is not a valid operator.")

    # Extract the default boson operators from the Hamiltonian. If operators aren't
    # empty, we're dealing with a second-quantized problem.
    if any(isinstance(block, (sympy.MatrixBase, sympy.Expr)) for block in nonzero_blocks):
        operators = list(
            set().union(*(find_operators(block) for block in nonzero_blocks))
        )
    else:
        operators = ()

    # To handle the second-quantized problem, convert the Hamiltonian to our
    # NumberOrderedForm.
    if operators:
        H_eval_orig = H.eval

        def H_eval(*index):
            result = H_eval_orig(*index)
            if result is zero:
                return zero
            if isinstance(result, sympy.Matrix):
                return result.applyfunc(
                    lambda x: NumberOrderedForm.from_expr(x, operators)
                )
            return NumberOrderedForm.from_expr(result, operators)

        H.eval = H_eval

    # If solve_sylvester is not yet defined, use the diagonal one.
    if solve_sylvester is None or use_implicit:
        diagonal = _extract_diagonal(H, atol, use_implicit, operators)

    if solve_sylvester is None:
        if not operators:
            solve_sylvester = solve_sylvester_diagonal(diagonal, atol=atol)
        else:
            solve_sylvester = second_quantization.solve_sylvester_bosonic(
                diagonal, operators
            )

    # When the input Hamiltonian value is a linear operator, so should be the output.
    use_linear_operator = np.zeros(H.shape, dtype=bool)
    if isinstance(H[(-1, -1, *zero_order)], sparse.linalg.LinearOperator):
        use_linear_operator[-1, -1] = True

        if H.shape[0] - 1 in fully_diagonalize:
            raise ValueError("Fully diagonalizing an implicit block is not supported.")

        if operator is not matmul:
            raise ValueError("Implicit mode requires matmul operator.")

    # Catch the solve_sylvester that uses the old signature without index.
    if len(signature(solve_sylvester).parameters) == 1:
        solve_sylvester = _preprocess_sylvester(solve_sylvester)

    if not isinstance(fully_diagonalize, dict):
        commuting_blocks = [True] * H.shape[0]
    else:
        commuting_blocks = [i not in fully_diagonalize for i in range(H.shape[0])]

    scope = {
        "solve_sylvester": solve_sylvester,
        "use_linear_operator": use_linear_operator,
        "two_block_optimized": H.shape[0] == 2 and not fully_diagonalize,
        "commuting_blocks": commuting_blocks,
    }

    equal_eigs = {
        i: (
            (np.abs(diagonal[i].reshape(-1, 1) - diagonal[i]) < atol).astype(int)
            if diagonal[i].dtype != object  # numerical array, else sympy
            else ((diagonal[i].reshape(-1, 1) == diagonal[i]) == True)  # noqa E712
        )
        for i in set(fully_diagonalize)
    }

    if not fully_diagonalize:
        pass
    elif not operators:
        # Determine degenerate eigensubspaces of the blocks to fully diagonalize.
        if isinstance(fully_diagonalize, dict):
            # Check that `fully_diagonalize` is symmetric.
            for to_eliminate in fully_diagonalize.values():
                # Check that `fully_diagonalize` has the right format. In particular we
                # do not allow `to_eliminate` to have the format made for second
                # quantized Hamiltonians (type Mask) if H_0 has no operators.
                if not isinstance(to_eliminate, np.ndarray):
                    raise ValueError(
                        "The values of fully_diagonalize dictionary must be numpy arrays."
                    )
                if not (to_eliminate == to_eliminate.T).all():
                    raise ValueError(
                        "The values of fully_diagonalize dictionary must be symmetric."
                    )
            # Check that `fully_diagonalize` does not have any True values corresponding
            # to equal eigenvalues.
            for i, to_eliminate in fully_diagonalize.items():
                if (to_eliminate & equal_eigs[i]).any():
                    raise ValueError(
                        "Full diagonalization must not eliminate matrix elements corresponding"
                        " to equal eigenvalues."
                    )
            to_eliminate = fully_diagonalize
            to_keep = {i: 1 - eliminate for i, eliminate in to_eliminate.items()}
        else:
            to_keep = equal_eigs
            to_eliminate = {i: 1 - keep for i, keep in to_keep.items()}

        # Convert numpy arrays to sympy matrices if blocks are symbolic.
        to_eliminate = {
            i: (
                sympy.Matrix(sympy.S.One * eliminate)
                if diagonal[i].dtype == object
                else eliminate
            )
            for i, eliminate in to_eliminate.items()
        }
        to_keep = {
            i: sympy.Matrix(sympy.S.One * keep) if diagonal[i].dtype == object else keep
            for i, keep in to_keep.items()
        }

        def diag(x, index):
            x = x[index] if isinstance(x, BlockSeries) else x
            if index[0] not in to_keep:
                return x
            if isinstance(x, sympy.MatrixBase):
                return x.multiply_elementwise(to_keep[index[0]])
            if sparse.issparse(x):
                return x.multiply(to_keep[index[0]])
            return x * to_keep[index[0]]

        def offdiag(x, index):
            if index[0] not in to_keep:
                return zero
            x = x[index] if isinstance(x, BlockSeries) else x
            if isinstance(x, sympy.MatrixBase):
                return x.multiply_elementwise(to_eliminate[index[0]])
            if sparse.issparse(x):
                return x.multiply(to_eliminate[index[0]])
            return x * to_eliminate[index[0]]

        scope["diag"] = diag
        scope["offdiag"] = offdiag
    else:
        if isinstance(fully_diagonalize, dict):
            # fully_diagonalize is a dictionary of masks
            def diag(x, index):
                x = x[index] if isinstance(x, BlockSeries) else x
                if index[0] not in fully_diagonalize or x is zero:
                    return x
                return x - second_quantization.apply_mask_to_operator(
                    x, fully_diagonalize[index[0]]
                )

            def offdiag(x, index):
                if index[0] not in fully_diagonalize:
                    return zero
                x = x[index] if isinstance(x, BlockSeries) else x
                if x is zero:
                    return zero
                return second_quantization.apply_mask_to_operator(
                    x, fully_diagonalize[index[0]]
                )
        else:
            # fully_diagonalize is a list of block indices to fully diagonalize
            keep = {
                block_idx: (
                    operators,
                    [(([0], None),) * len(operators) + (equal_eigs[block_idx],)],
                )
                for block_idx in fully_diagonalize
            }

            def diag(x, index):
                x = x[index] if isinstance(x, BlockSeries) else x
                if index[0] not in keep or x is zero:
                    return x

                return second_quantization.apply_mask_to_operator(x, keep[index[0]])

            def offdiag(x, index):
                if index[0] not in keep:
                    return zero
                x = x[index] if isinstance(x, BlockSeries) else x
                if x is zero:
                    return zero

                return sympy.expand(
                    x - second_quantization.apply_mask_to_operator(x, keep[index[0]])
                )

        scope["diag"] = diag
        scope["offdiag"] = offdiag

    outputs, _ = series_computation(
        {"H": H},
        algorithm=main,
        scope=scope,
        operator=operator,
    )
    if operators:
        # Pass the results through number-ordering with simplification
        def simplified_eval(*index, name):
            result = outputs[name][index]
            if result is zero:
                return zero
            return result.applyfunc(lambda x: NumberOrderedForm.from_expr(x))

        return tuple(
            BlockSeries(
                shape=outputs[name].shape,
                n_infinite=outputs[name].n_infinite,
                name=name,
                dimension_names=outputs[name].dimension_names,
                eval=partial(simplified_eval, name=name),
            )
            for name in ("H_tilde", "U", "U†")
        )
    return outputs["H_tilde"], outputs["U"], outputs["U†"]


### Converting different formats to BlockSeries
def operator_to_BlockSeries(
    operator: list | dict | BlockSeries | sympy.Matrix,
    *,
    name: str | None = None,
    subspace_eigenvectors: tuple[Any, Any] | None = None,
    subspace_indices: tuple[int, ...] | None = None,
    implicit: bool = False,
    symbols: list[sympy.Symbol] | None = None,
    atol: float = 1e-12,
    hermitian: bool = False,
) -> BlockSeries:
    """Convert an operator to `~pymablock.series.BlockSeries` format.

    The normalization consists of the following steps:

    - Determine perturbation parameters from the input format,
    - Separate the operator into blocks based on eigenvectors or subspace indices,
    - Form a `~pymablock.series.BlockSeries`.

    Parameters
    ----------
    operator :
        Full symbolic or numeric operator. It is normalized to a
        `~pymablock.series.BlockSeries` and separated into blocks corresponding to
        different subspaces.

        Supported formats:

        - A list,
            of the form ``[h_0, h_1, h_2, ...]`` where ``h_0`` is the unperturbed
            operator and ``h_1, h_2, ...`` are the first order perturbations. The
            elements ``h_i`` may be `~sympy.matrices.dense.Matrix`, `~numpy.ndarray`,
            `~scipy.sparse.spmatrix`, that require separating into blocks. Otherwise,
            ``h_i`` may be a list of lists with the operator blocks.
        - A dictionary,
            of the form ``{(0, 0): h_0, (1, 0): h_1, (0, 1): h_2}``, or ``{1: h_0, x:
            h_1, y: h_2}`` for symbolic operators. In the former case, the keys must be
            tuples of integers indicating the order of each perturbation. In the latter
            case, the keys must be monomials and the indices are ordered as in
            `operator.dimension_names`. ``h_i`` have the same format as in the list
            case.
        - A `~sympy.matrices.dense.Matrix`,
            unless a list of ``symbols`` is provided as perturbative parameters, all
            symbols will be treated as perturbative. The normalization to
            `~pymablock.series.BlockSeries` is done by Taylor expanding in ``symbols``
            to the desired order.
        - A `~pymablock.series.BlockSeries`,
            used directly.
    name:
        Name of the operator.
    subspace_eigenvectors :
        A tuple with sets of orthonormal eigenvectors to project the operator onto and
        separate it into blocks. If some vectors are missing, the last block is defined
        implicitly. Mutually exclusive with ``subspace_indices``. If neither
        ``subspace_eigenvectors`` nor ``subspace_indices`` are provided, the
        BlockSeries is defined with a single block.
    subspace_indices :
        An array indicating which state belongs to which subspace. If there are two
        blocks, the labels are 0 for the A (effective) subspace and 1 for the B
        (auxiliary) subspace. Mutually exclusive with ``subspace_eigenvectors``.
        If neither ``subspace_eigenvectors`` nor ``subspace_indices`` are provided, the
        BlockSeries is defined with a single block.
    implicit :
        Whether to wrap the operator of the last subspace into a linear operator.
    symbols :
        Symbols that label the perturbative parameters of a symbolic operator. The order
        of the symbols is mapped to the indices of the operator, see
        `~pymablock.series.BlockSeries`. If None, the perturbative parameters are taken
        from the unperturbed operator.
    atol :
        Absolute tolerance to consider matrices as exact zeros. This is used to validate
        that the unperturbed operator is block-diagonal.
    hermitian :
        Whether the operator may be assumed to be Hermitian (for performance).

    Returns
    -------
    operator : `~pymablock.series.BlockSeries`
        Transformed operator.

    """
    if subspace_eigenvectors is not None and subspace_indices is not None:
        raise ValueError(
            "subspace_eigenvectors and subspace_indices are mutually exclusive."
        )
    to_split = subspace_eigenvectors is not None or subspace_indices is not None

    operator = _unpack_blocks(_to_scalar_BlockSeries(operator, symbols, atol), atol)
    operator.name = name or operator.name

    if operator.shape:
        if to_split:
            raise ValueError(
                "operator is already separated into blocks but "
                "subspace_eigenvectors are provided."
            )

        if operator.shape[0] != operator.shape[1]:
            raise ValueError("operator must be a square block series.")

        return operator

    # Separation into subspace_eigenvectors
    if not to_split:
        zeroth_order = operator[(0,) * operator.n_infinite]
        if sparse.issparse(zeroth_order) or isinstance(
            zeroth_order, (np.ndarray, sympy.MatrixBase)
        ):
            subspace_indices = np.zeros(zeroth_order.shape[0], dtype=int)

    # Define subspace_eigenvectors
    if subspace_indices is not None:
        h_0 = operator[(0,) * operator.n_infinite]
        symbolic = isinstance(h_0, sympy.MatrixBase)
        subspace_eigenvectors = _subspaces_from_indices(
            subspace_indices, symbolic=symbolic
        )
    subspace_eigenvectors = tuple(subspace_eigenvectors)
    if implicit:
        # Define subspace_eigenvectors for implicit
        subspace_eigenvectors = (
            *subspace_eigenvectors,
            ComplementProjector(np.hstack(subspace_eigenvectors)),
        )

    # Separation into subspace_eigenvectors
    n_blocks = len(subspace_eigenvectors)

    def op_eval(*index):
        left, right = index[:2]
        if left > right and hermitian:
            return Dagger(op[(right, left, *tuple(index[2:]))])
        original = operator[index[2:]]
        if original is zero:
            return zero
        if implicit and left == right == n_blocks - 1:
            original = aslinearoperator(original)
        return _convert_if_zero(
            Dagger(subspace_eigenvectors[left]) @ original @ subspace_eigenvectors[right],
            atol=atol,
        )

    # We must add op to the local scope because op_eval refers to it.
    op = BlockSeries(
        eval=op_eval,
        shape=(n_blocks, n_blocks),
        n_infinite=operator.n_infinite,
        dimension_names=symbols,
        name=name or operator.name,
    )

    return op


### Different formats and algorithms of solving Sylvester equation.
def _preprocess_sylvester(solve_sylvester: Callable) -> Callable:
    """Wrap the solve_sylvester_callable to handle index and zero values."""

    def wrapped(Y: Any, index: tuple[int, ...]) -> Any:
        if index[:2] != (0, 1):
            raise ValueError("Sylvester equation is only defined for (0, 1) blocks.")

        if isinstance(Y, BlockSeries):
            Y = Y[index]

        return solve_sylvester(Y) if Y is not zero else zero

    return wrapped


def solve_sylvester_diagonal(
    eigs: tuple[np.ndarray | sympy.matrices.MatrixBase, ...],
    vecs_implicit: np.ndarray | None = None,
    atol: float = 1e-12,
) -> Callable:
    """Define a function for solving a Sylvester's equation for diagonal matrices.

    Optionally, this function also applies the eigenvectors of the second
    matrix to the solution.

    Parameters
    ----------
    eigs :
        List of arrays of eigenvalues in each subspace. If `vecs_implicit` are provided,
        the last subspace corresponds to the partial set of eigenvalues of the implicit
        subspace.
    vecs_implicit :
        Partial eigenvectors of the implicit subspace. Only used for KPM. If provided,
        the last block is returned in the basis of the original Hilbert space rather
        than eigenbasis of the Hamiltonian.
    atol :
        Absolute tolerance to consider energy differences as exact zeros.

    Returns
    -------
    solve_sylvester : `Callable`
        Function that solves Sylvester's equation.

    """
    index_checked = set()

    def solve_sylvester(
        Y: np.ndarray | sparse.csr_array | sympy.MatrixBase,
        index: tuple[int, ...],
    ) -> np.ndarray | sparse.csr_array | sympy.MatrixBase:
        if Y is zero:
            return zero

        eigs_A, eigs_B = eigs[index[0]], eigs[index[1]]

        if index[0] != index[1] and index[:2] not in index_checked:
            compare = np.equal if isinstance(Y, sympy.MatrixBase) else np.isclose

            if np.any(compare(eigs_A.reshape(-1, 1), eigs_B.reshape(1, -1))):
                raise ValueError("The subspaces must not share eigenvalues.")
            index_checked.add(index[:2])

        if vecs_implicit is not None and index[1] == len(eigs) - 1:
            # Needed for implicit mode with KPM
            energy_denominators = 1 / (eigs_A.reshape(-1, 1) - eigs_B)
            return ((Y @ vecs_implicit) * energy_denominators) @ Dagger(vecs_implicit)
        if isinstance(Y, np.ndarray):
            energy_differences = eigs_A.reshape(-1, 1) - eigs_B
            with np.errstate(divide="ignore", invalid="ignore"):
                energy_denominators = np.where(
                    np.abs(energy_differences) > atol, 1 / energy_differences, 0
                )
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
            ).subs(sympy.zoo, sympy.S.Zero)  # Take care of diagonal elements
            return energy_denominators.multiply_elementwise(Y)
        raise TypeError(f"Unsupported rhs type: {type(Y)}")

    return solve_sylvester


def solve_sylvester_KPM(
    h_0: np.ndarray | sparse.spmatrix,
    subspace_eigenvectors: tuple[np.ndarray, ...],
    solver_options: dict | None = None,
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
        blocks.
    solver_options :
        Dictionary containing any of the following options for KPM.

        - eps: float
            Tolerance for Hamiltonian rescaling.
        - atol: float
            Accepted precision of the Green's function result in 2-norm.
        - max_moments: int
            Maximum number of expansion moments of the Green's function.
        - auxiliary_vectors: np.ndarray
            Partial set of eigenvectors of the auxiliary subspace, used to speed up
            convergence of the KPM solver.

    Returns
    -------
    solve_sylvester: Callable
        Function that applies divide by energies to the right hand side of
        Sylvester's equation.

    """
    if solver_options is None:
        solver_options = {}

    aux_vectors = solver_options.get("auxiliary_vectors", np.zeros((h_0.shape[0], 0)))
    subspace_eigenvectors = (*subspace_eigenvectors, aux_vectors)
    eigs = [
        (Dagger(eigenvectors) @ h_0 @ eigenvectors).diagonal()
        for eigenvectors in subspace_eigenvectors
    ]

    kpm_projector = ComplementProjector(np.hstack(subspace_eigenvectors))
    # Prepare the Hamiltonian for KPM by rescaling to [-1, 1]
    bounds_eigs = [np.min(np.concatenate(eigs[:-1])), np.max(np.concatenate(eigs[:-1]))]
    h_rescaled, (a, b) = rescale(
        h_0, eps=solver_options.get("eps", 0.01), lower_bounds=bounds_eigs
    )
    eigs_rescaled = [(eig - b) / a for eig in eigs[:-1]]
    # We need to solve a transposed problem
    h_rescaled_T = h_rescaled.T
    # CSR format has a faster matrix-vector product
    if sparse.issparse(h_rescaled_T):
        h_rescaled_T = h_rescaled_T.tocsr()

    def solve_sylvester_kpm(Y: np.ndarray, index: tuple[int]) -> np.ndarray:
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
                for energy, vector in zip(eigs_rescaled[index[0]], Y_KPM)
            ]
        )

    vecs_implicit = subspace_eigenvectors[-1]
    solve_sylvester_explicit = solve_sylvester_diagonal(
        eigs, vecs_implicit, atol=solver_options.get("atol")
    )

    def solve_sylvester(Y: np.ndarray, index: tuple[int]) -> np.ndarray:
        if Y is zero:
            return zero
        if index[1] == len(eigs) - 1:
            return solve_sylvester_kpm(Y, index) + solve_sylvester_explicit(Y, index)
        return solve_sylvester_explicit(Y, index)

    return solve_sylvester


def solve_sylvester_direct(
    h_0: sparse.spmatrix,
    eigenvectors: list[np.ndarray],
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
        Eigenvectors of the effective subspaces of the unperturbed Hamiltonian.
    **solver_options :
        Keyword arguments to pass to the solver ``eps`` and ``atol``, see
        `pymablock.linalg.direct_greens_function`.

    Returns
    -------
    solve_sylvester : `Callable[[numpy.ndarray], numpy.ndarray]`
        Function that solves the corresponding Sylvester equation.

    """
    projector = ComplementProjector(np.hstack(eigenvectors))
    eigenvalues = [
        np.diag(Dagger(subspace) @ h_0 @ subspace) for subspace in eigenvectors
    ]
    # Compute the Green's function of the transposed Hamiltonian because we are
    # solving the equation from the right.
    greens_functions = [
        [direct_greens_function(h_0.T, E, **solver_options) for E in subspace_eigenvalues]
        for subspace_eigenvalues in eigenvalues
    ]

    explicit_part = solve_sylvester_diagonal(
        eigenvalues, atol=solver_options.get("atol", 1e-12)
    )

    def solve_sylvester(Y: np.ndarray, index: tuple[int, ...]) -> np.ndarray:
        if Y is zero:
            return zero
        if index[1] < len(eigenvalues):
            return explicit_part(Y, index)

        Y = Y @ projector
        result = np.vstack([gf(vec) for gf, vec in zip(greens_functions[index[0]], Y)])
        return result @ projector

    return solve_sylvester


### Auxiliary functions.
def _list_to_dict(operator: list[Any]) -> dict[int, Any]:
    """Convert a list of perturbations to a dictionary.

    Parameters
    ----------
    operator :
        Unperturbed Hamiltonian and 1st order perturbations.
        [h_0, h_1, h_2, ...], where h_0 is the unperturbed Hamiltonian and the
        remaining elements are the 1st order perturbations.
        This method is for first order perturbations only.

    Returns
    -------
    operator : `~pymablock.series.BlockSeries`

    """
    n_infinite = len(operator) - 1  # All the perturbations are 1st order
    zeroth_order = (0,) * n_infinite

    operator = {
        zeroth_order: operator[0],
        **{
            tuple(order): perturbation
            for order, perturbation in zip(np.eye(n_infinite, dtype=int), operator[1:])
        },
    }
    return operator


def _dict_to_BlockSeries(
    operator: dict[tuple[int, ...], Any],
    symbols: Sequence[sympy.Symbol] | None = None,
    atol: float = 1e-12,
) -> tuple[BlockSeries, list[sympy.Symbol]]:
    """Convert a dictionary of perturbations to a BlockSeries.

    Additionally, convert numpy zeroth order to sparse array if it is diagonal and
    normalize sparse to csr.

    Parameters
    ----------
    operator :
        Unperturbed Hamiltonian and perturbations. The keys can be tuples of integers or
        symbolic monomials. They indicate the order of the perturbation in its
        respective value. The values are the perturbations, and can be either a
        `~numpy.ndarray`, `~scipy.sparse.csr_array` or a list with the blocks of the
        Hamiltonian. For example, {(0, 0): h_0, (1, 0): h_1, (0, 1): h_2} or {1: h_0, x:
        h_1, y: h_2}.
    symbols :
        tuple of symbols to use for the BlockSeries.
    atol :
        absolute tolerance for determining if a matrix is diagonal.

    Returns
    -------
    operator : `~pymablock.series.BlockSeries`

    """
    operator = copy(operator)
    key_types = set(isinstance(key, sympy.Basic) for key in operator.keys())
    if any(key_types):
        operator, symbols = _symbolic_keys_to_tuples(operator)

    n_infinite = len(next(iter(operator.keys())))
    zeroth_order = (0,) * n_infinite
    h_0 = operator[zeroth_order]

    if isinstance(h_0, np.ndarray):
        if is_diagonal(h_0, atol):
            operator[zeroth_order] = sparse.csr_array(operator[zeroth_order])
    elif sparse.issparse(h_0):  # Normalize sparse matrices for solve_sylvester
        operator[zeroth_order] = sparse.csr_array(operator[zeroth_order])

    return BlockSeries(
        data=operator,
        shape=(),
        n_infinite=n_infinite,
        dimension_names=symbols,
    )


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
    operator: sympy.MatrixBase,
    symbols: Sequence[sympy.Symbol] = (),
    check_hermitian: bool = True,
) -> BlockSeries:
    """Convert a symbolic Hamiltonian to a BlockSeries.

    Parameters
    ----------
    operator :
        Symbolic Hamiltonian.
    symbols :
        List of symbols that are the perturbative coefficients.
        If None, all symbols in the Hamiltonian are assumed to be perturbative
        coefficients.
    check_hermitian :
        Whether to check if the operator is Hermitian.

    Returns
    -------
    operator : `~pymablock.series.BlockSeries`

    """
    if not symbols:
        symbols = tuple(list(operator.free_symbols))  # All symbols are perturbative
    if any(n not in operator.free_symbols for n in symbols):
        raise ValueError("Not all perturbative parameters are in `hamiltonian`.")

    operator = operator.expand()
    n_infinite = len(symbols)

    # Define derivatives through lower order derivatives to avoid recomputing.
    # We also divide these by the product of factorials to use in the Taylor expansion.
    operator_derivatives = BlockSeries(
        data={(0,) * n_infinite: operator},
        shape=(),
        n_infinite=n_infinite,
        dimension_names=symbols,
    )

    def derivative_eval(*index):
        # Select an axis along which to take the derivative.
        symbol_number, order = next((i, n) for i, n in enumerate(index) if n)
        previous_index = list(index)
        previous_index[symbol_number] -= 1
        return (
            operator_derivatives[tuple(previous_index)].diff(symbols[symbol_number])
            / order
        )

    operator_derivatives.eval = derivative_eval

    def op_eval(*index):
        expr = operator_derivatives[index].subs({n: 0 for n in symbols})
        # - Sympy three-valued logic requires "is False".
        # - The last check is a workaround for sympy issue #27898. (Matrices
        # with operators are never Hermitian.)
        if check_hermitian and expr.is_hermitian is False and not expr.atoms(Operator):
            raise ValueError("Operator must be Hermitian.")

        expr = expr * reduce(mul, [n**i for n, i in zip(symbols, index)])
        return _convert_if_zero(expr)

    op = BlockSeries(
        eval=op_eval,
        shape=(),
        n_infinite=n_infinite,
        dimension_names=symbols,
    )
    return op


def _to_scalar_BlockSeries(
    operator: list | dict | BlockSeries | sympy.Matrix,
    symbols: Sequence[sympy.Symbol] = (),
    atol: float = 1e-12,
) -> BlockSeries:
    """Normalize input to BlockSeries without transforming values.

    One exception is that numerical zeroth order term is converted to sparse if it is
    diagonal, and sparse is normalized to csr.
    """
    if isinstance(operator, BlockSeries):
        return operator
    if isinstance(operator, sympy.MatrixBase):
        return _sympy_to_BlockSeries(operator, symbols)
    if isinstance(operator, list):
        operator = _list_to_dict(operator)
    if isinstance(operator, dict):
        return _dict_to_BlockSeries(operator, symbols, atol)
    raise TypeError(f"Unsupported input type of Hamiltonian: {type(operator)}.")


def _unpack_blocks(operator: BlockSeries, atol: float = 1e-12) -> BlockSeries:
    """Check if operator values are list of lists and if so, unpack them into blocks."""
    if operator.shape:
        return operator

    if not isinstance(
        (zeroth_order := operator[(0,) * operator.n_infinite]), (tuple, list)
    ):
        return operator

    def op_eval(*index):
        h = _convert_if_zero(operator[index[2:]], atol=atol)
        if h is zero:
            return zero
        try:
            return _convert_if_zero(h[index[0]][index[1]], atol=atol)
        except Exception as e:
            raise ValueError(
                "Without `subspace_eigenvectors` or `subspace_indices`"
                " operator must have an NxN block structure."
            ) from e

    op = BlockSeries(
        eval=op_eval,
        shape=2 * (len(zeroth_order),),
        n_infinite=operator.n_infinite,
        dimension_names=operator.dimension_names,
    )
    return op


def _subspaces_from_indices(
    subspace_indices: tuple[int, ...] | np.ndarray,
    symbolic: bool = False,
) -> tuple[sparse.csr_array, sparse.csr_array]:
    """Compute subspace eigenvectors from indices of diagonal elements.

    Parameters
    ----------
    subspace_indices :
        Subspaces to which the ``subspace_eigenvectors`` belong. Ranges from 0
        to ``n_subspaces-1``.
    symbolic :
        True if the Hamiltonian is symbolic, False otherwise. If True, the
        returned subspaces are dense arrays.

    Returns
    -------
    subspace_eigenvectors :
        Subspaces to use for block diagonalization.

    """
    subspace_indices = np.array(subspace_indices)
    dim = len(subspace_indices)
    eigvecs = sparse.csr_array(sparse.identity(dim, dtype=int, format="csr"))
    # Canonical basis vectors for each subspace
    # TODO: review next statement for readability
    subspace_eigenvectors = tuple(
        eigvecs[:, np.compress(subspace_indices == block, np.arange(dim))]
        for block in range(np.max(subspace_indices) + 1)
    )
    if symbolic:
        # Convert to dense arrays, otherwise they cannot multiply sympy matrices.
        return tuple(subspace.toarray() for subspace in subspace_eigenvectors)
    return subspace_eigenvectors


def _extract_diagonal(
    operator: BlockSeries,
    atol: float = 1e-12,
    implicit: bool = False,
    operators: Sequence[sympy.physics.quantum.Operator] = (),
) -> tuple[np.ndarray, ...]:
    """Extract the diagonal of the zeroth order of the Hamiltonian.

    operator :
        Hamiltonian BlockSeries.
    atol :
        Absolute tolerance for numerical zero.
    implicit :
        Whether the last block is defined as a linear operator.
    operators :
        List of bosonic operators used in the Hamiltonian.
    """
    # If using implicit mode, skip the last block.
    diag_indices = np.arange(operator.shape[0] - implicit)
    h_0 = operator[(diag_indices, diag_indices) + (0,) * operator.n_infinite]
    is_sympy = any(isinstance(block, sympy.MatrixBase) for block in h_0)
    warning = (
        "Cannot confirm that the unperturbed Hamiltonian is diagonal, "
        "which is required if ``solve_sylvester`` is not provided. "
        "The algorithm will assume that it is diagonal."
    )
    if not all(is_diagonal(h, atol) for h in h_0):
        warn(warning, UserWarning)
    diags = []
    for block in h_0:
        if block is zero or block is np.ma.masked:
            diags.append(np.array(0))
            continue
        eigs = block.diagonal()
        if is_sympy:
            # Check if any of the expressions contains sympy.physics.quantum.Operator
            if operators:
                eigs = [NumberOrderedForm.from_expr(eig).simplify() for eig in eigs]
            eigs = np.array(eigs, dtype=object)
        diags.append(eigs)

    return tuple(diags)


def _convert_if_zero(value: Any, atol: float = 1e-12):
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
