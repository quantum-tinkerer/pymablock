# ruff: noqa: N803, N806
"""Linear algebra utilities."""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import sympy
from scipy import sparse
from scipy.linalg import qr
from scipy.sparse import identity, spmatrix
from scipy.sparse.linalg import LinearOperator, factorized
from scipy.sparse.linalg import aslinearoperator as scipy_aslinearoperator

from pymablock.series import one, zero


def _kernel_pivot_rows(kernel_vectors: np.ndarray) -> np.ndarray:
    """Choose pivot rows that fix a gauge for a degenerate kernel."""
    if kernel_vectors.shape[1] == 0:
        return np.array([], dtype=int)

    _, _, pivots = qr(kernel_vectors.T, mode="economic", pivoting=True)
    return np.sort(pivots[: kernel_vectors.shape[1]])


def _constrain_matrix(
    mat: sparse.sparray | spmatrix,
    pivot_rows: np.ndarray,
) -> sparse.csr_array:
    """Replace selected equations with x[row] = 0 constraints."""
    constrained = sparse.csr_array(mat)
    if pivot_rows.size == 0:
        return constrained

    pivot_mask = np.zeros(constrained.shape[0], dtype=bool)
    pivot_mask[pivot_rows] = True

    # Drop all entries on constrained rows, then add back the diagonal ones
    # that enforce x[row] = 0 on those rows.
    constrained_coo = constrained.tocoo(copy=False)
    keep = ~pivot_mask[constrained_coo.row]
    rows = np.concatenate((constrained_coo.row[keep], pivot_rows))
    cols = np.concatenate((constrained_coo.col[keep], pivot_rows))
    data = np.concatenate(
        (
            constrained_coo.data[keep],
            np.ones(len(pivot_rows), dtype=constrained.dtype),
        )
    )

    return sparse.csr_array((data, (rows, cols)), shape=constrained.shape)


def direct_greens_function(
    h: spmatrix,
    E: float,
    kernel_vectors: np.ndarray | None = None,
    atol: float | None = None,
    eps: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Compute the Green's function of a Hamiltonian using a sparse direct solver.

    Parameters
    ----------
    h :
        Hamiltonian matrix.
    E :
        Energy at which to compute the Green's function.
    kernel_vectors :
        Orthonormal basis of the kernel of ``E - H``. When provided, the solver
        fixes the gauge by replacing pivot equations selected with pivoted QR by
        ``x[pivot] = 0`` constraints and projects the kernel away before and
        after the sparse solve. If omitted, an empty kernel basis is used.
    atol :
        Deprecated. Ignored and will be removed in version 2.4.0.
    eps :
        Deprecated. Ignored and will be removed in version 2.4.0.

    Returns
    -------
    greens_function : `Callable[[np.ndarray], np.ndarray]`
        Function that solves :math:`(E - H) sol = vec`.

    """
    mat = E * sparse.csr_array(identity(h.shape[0], dtype=h.dtype, format="csr")) - h
    if kernel_vectors is None:
        kernel_vectors = np.zeros((h.shape[0], 0), dtype=h.dtype)
    deprecated_arguments = [
        name for name, value in (("atol", atol), ("eps", eps)) if value is not None
    ]
    if deprecated_arguments:
        argument_names = ", ".join(f"`{name}`" for name in deprecated_arguments)
        verb = "are" if len(deprecated_arguments) > 1 else "is"
        warnings.warn(
            f"{argument_names} {verb} ignored by `direct_greens_function` and "
            "will be removed in version 2.4.0.",
            DeprecationWarning,
            stacklevel=2,
        )

    pivot_rows = _kernel_pivot_rows(kernel_vectors)
    kernel_projector = ComplementProjector(kernel_vectors)
    mat = _constrain_matrix(mat, pivot_rows)

    is_complex = np.iscomplexobj(mat.data)
    try:
        from mumps import Context as MUMPSContext
    except ImportError:
        solve = factorized(sparse.csc_matrix(mat))
    else:
        ctx = MUMPSContext()
        # MUMPS does not support Hermitian matrices, so we use the symmetric only with real.
        ctx.set_matrix(
            sparse.coo_array(mat),
            overwrite_a=True,
            symmetric=not is_complex and not pivot_rows.size,
        )
        ctx.factor()

        def solve(v: np.ndarray) -> np.ndarray:
            return ctx.solve(v, overwrite_b=True)

    def greens_function(vec: np.ndarray) -> np.ndarray:
        """Apply the Green's function to a vector.

        Parameters
        ----------
        vec :
            Vector to which to apply the Green's function. If the Green's
            function is evaluated at an eigenenergy, this vector must be
            orthogonal to the corresponding eigenvector(s).

        Returns
        -------
        sol :
            Solution of :math:`(E - H) sol = vec`.

        """
        vec = kernel_projector @ vec
        vec[pivot_rows] = 0

        if np.iscomplexobj(vec) and not is_complex:
            vec = (vec.real, vec.imag)
        else:
            vec = (vec,)

        sol = []
        for v in vec:
            sol.append(solve(v))
        result = sol[0] if len(sol) == 1 else sol[0] + 1j * sol[1]
        return kernel_projector @ result

    return greens_function


class ComplementProjector(LinearOperator):
    r"""Projector on the complement of the span of a set of vectors.

    This is used to compute $P_B = I - P_A$ where $P_A$ is the projector on the
    span of the vectors $A$ in the implicit method.
    """

    def __init__(
        self: LinearOperator,
        vecs: np.ndarray,
        left_vecs: np.ndarray | None = None,
    ) -> LinearOperator:
        """Projector on the complement of the span of vecs."""
        self.shape = (vecs.shape[0], vecs.shape[0])
        self._vecs = vecs
        self._hermitian = (
            left_vecs is None or left_vecs is vecs or np.array_equal(left_vecs, vecs)
        )
        self._left_vecs = vecs if self._hermitian else left_vecs
        self.dtype = np.result_type(self._vecs.dtype, self._left_vecs.dtype)
        self._adjoint_operator = self if self._hermitian else None
        self._conjugate_operator = None
        self._transpose_operator = None

    __array_ufunc__ = None

    def _apply(self: LinearOperator, v: np.ndarray) -> np.ndarray:
        return v - self._vecs @ (self._left_vecs.conj().T @ v)

    _matvec = _matmat = _apply

    def _apply_left(self: LinearOperator, v: np.ndarray) -> np.ndarray:
        return v - self._left_vecs.conj() @ (self._vecs.T @ v)

    _rmatvec = _rmatmat = _apply_left

    def _adjoint(self: LinearOperator) -> LinearOperator:
        if self._adjoint_operator is None:
            self._adjoint_operator = self.__class__(
                vecs=self._left_vecs,
                left_vecs=self._vecs,
            )
            self._adjoint_operator._adjoint_operator = self
        return self._adjoint_operator

    def conjugate(self: LinearOperator) -> LinearOperator:
        """Conjugate operator."""
        if self._conjugate_operator is None:
            vecs = self._vecs.conj()
            left_vecs = vecs if self._hermitian else self._left_vecs.conj()
            self._conjugate_operator = self.__class__(
                vecs=vecs,
                left_vecs=left_vecs,
            )
            self._conjugate_operator._conjugate_operator = self
            if self._hermitian:
                self._transpose_operator = self._conjugate_operator
                self._conjugate_operator._transpose_operator = self
        return self._conjugate_operator

    def _transpose(self: LinearOperator) -> LinearOperator:
        if self._transpose_operator is None:
            self._transpose_operator = (
                self.conjugate() if self._hermitian else self.conjugate()._adjoint()
            )
            self._transpose_operator._transpose_operator = self
        return self._transpose_operator


def aslinearoperator(A: Any) -> Any:
    """Construct a linear operator.

    Same as `scipy.sparse.linalg.aslinearoperator`, but with passthrough for
    `~pymablock.series.zero` and `~pymablock.series.one`.
    """
    if A is zero or A is one:
        return A
    return scipy_aslinearoperator(A)


def is_diagonal(A: Any, atol: float = 1e-12) -> bool:
    """Check if A is diagonal."""
    if A is zero or A is np.ma.masked:
        return True
    if isinstance(A, sympy.MatrixBase):
        return A.is_diagonal()
    if isinstance(A, np.ndarray):
        # Create a view of the offdiagonal array elements
        offdiagonal = A.reshape(-1)[:-1].reshape(len(A) - 1, len(A) + 1)[:, 1:]
        return not np.any(np.round(offdiagonal, int(-np.log10(atol))))
    if sparse.issparse(A):
        A = sparse.dia_array(A)
        return not any(A.offsets)
    raise NotImplementedError(f"Cannot extract diagonal from {type(A)}")
