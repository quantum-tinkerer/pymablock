# ruff: noqa: N803, N806
"""Linear algebra utilities."""

from collections.abc import Callable
from typing import Any
from warnings import warn

import numpy as np
import sympy
from scipy import sparse
from scipy.linalg import qr
from scipy.sparse import identity, spmatrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import aslinearoperator as scipy_aslinearoperator

from pymablock.series import one, zero


def _kernel_pivot_rows(kernel_vectors: np.ndarray) -> np.ndarray:
    """Choose pivot rows that fix a gauge for a degenerate kernel."""
    if kernel_vectors.shape[1] == 0:
        return np.array([], dtype=int)

    _, _, pivots = qr(kernel_vectors.T, mode="economic", pivoting=True)
    return np.sort(np.asarray(pivots[: kernel_vectors.shape[1]], dtype=int))


def _constrain_matrix(
    mat: sparse.sparray | spmatrix,
    pivot_rows: np.ndarray,
) -> sparse.csr_array:
    """Replace selected equations with x[row] = 0 constraints."""
    if pivot_rows.size == 0:
        return sparse.csr_array(mat)

    constrained = sparse.lil_array(mat, copy=True)
    constrained[pivot_rows, :] = 0
    for row in pivot_rows:
        constrained[row, row] = 1
    return constrained.tocsr()


def direct_greens_function(
    h: spmatrix,
    E: float,
    kernel_vectors: np.ndarray | None = None,
    atol: float = 1e-7,
    eps: float = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Compute the Green's function of a Hamiltonian using MUMPS solver.

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
        Accepted precision of the desired result in infinity-norm.
    eps :
        Tolerance for the MUMPS solver to identify null pivots. Passed through
        to MUMPS CNTL(3), see MUMPS user guide.

    Returns
    -------
    greens_function : `Callable[[np.ndarray], np.ndarray]`
        Function that solves :math:`(E - H) sol = vec`.

    """
    try:
        from mumps import Context as MUMPSContext
    except ImportError as e:
        raise ImportError(
            "python-mumps is an optional dependency and is not installed."
            "The implicit method of block diagonalization requires it."
            "To install it see https://gitlab.kwant-project.org/kwant/python-mumps"
        ) from e

    mat = E * sparse.csr_array(identity(h.shape[0], dtype=h.dtype, format="csr")) - h
    if kernel_vectors is None:
        kernel_vectors = np.zeros((h.shape[0], 0), dtype=h.dtype)
    pivot_rows = _kernel_pivot_rows(kernel_vectors)
    kernel_projector = ComplementProjector(kernel_vectors)
    mat = _constrain_matrix(mat, pivot_rows)

    is_complex = np.iscomplexobj(mat.data)
    ctx = MUMPSContext()
    # MUMPS does not support Hermitian matrices, so we use the symmetric only with real.
    ctx.set_matrix(mat, symmetric=not is_complex and not pivot_rows.size)
    # Enable null pivot detection
    ctx.mumps_instance.icntl[24] = 1
    # Set tolerance for null pivot detection
    ctx.mumps_instance.cntl[3] = eps
    # Enable computation of the residual
    ctx.mumps_instance.icntl[11] = 2
    ctx.factor()

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
        vec = np.array(kernel_projector @ vec, copy=True)
        vec[pivot_rows] = 0

        if np.iscomplexobj(vec) and not is_complex:
            vec = (vec.real, vec.imag)
        else:
            vec = (vec,)

        residual = 0
        sol = []
        for v in vec:
            sol.append(ctx.solve(v))
            residual += (
                ctx.mumps_instance.rinfog[6]  # Scaled residual
                * ctx.mumps_instance.rinfog[4]  # ||A||_inf
                * ctx.mumps_instance.rinfog[5]  # ||x||_inf
            )
        if residual > atol:
            warn(
                f"Solution only achieved precision {residual} > {atol}."
                " adjust eps or atol.",
                RuntimeWarning,
            )
        result = sol[0] if len(sol) == 1 else sol[0] + 1j * sol[1]
        return kernel_projector @ result

    return greens_function


class ComplementProjector(LinearOperator):
    r"""Projector on the complement of the span of a set of vectors.

    This is used to compute $P_B = I - P_A$ where $P_A$ is the projector on the
    span of the vectors $A$ in the implicit method.
    """

    def __init__(self: LinearOperator, vecs: np.ndarray) -> LinearOperator:
        """Projector on the complement of the span of vecs."""
        self.shape = (vecs.shape[0], vecs.shape[0])
        self._vecs = vecs
        self.dtype = vecs.dtype

    __array_ufunc__ = None

    def _matvec(self: LinearOperator, v: np.ndarray) -> np.ndarray:
        return v - self._vecs @ (self._vecs.conj().T @ v)

    _matmat = _rmatvec = _rmatmat = _matvec

    def _adjoint(self: LinearOperator) -> LinearOperator:
        return self

    def conjugate(self: LinearOperator) -> LinearOperator:
        """Conjugate operator."""
        return self.__class__(vecs=self._vecs.conj())

    _transpose = conjugate


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
