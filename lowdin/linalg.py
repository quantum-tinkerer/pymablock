from typing import Callable, Optional
from warnings import warn

from packaging.version import parse
import numpy as np
from scipy import __version__ as scipy_version
from scipy.sparse import spmatrix, identity
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import aslinearoperator as scipy_aslinearoperator
from scipy import sparse
from kwant.linalg import mumps
import sympy
from sympy.matrices.common import MatrixSpecial, MatrixOperations

from lowdin.series import zero, one

# Monkey-patch LinearOperator to support right multiplication
# TODO: Remove this when we depend on scipy >= 1.11
if parse(scipy_version) < parse("1.11"):
    from scipy.sparse.linalg._interface import (
        _ProductLinearOperator,
        _ScaledLinearOperator,
    )

    def __rmul__(self, x):
        if np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            return self._rdot(x)

    def _rdot(self, x):
        """Matrix-matrix or matrix-vector multiplication from the right.

        Parameters
        ----------
        x : array_like
            1-d or 2-d array, representing a vector or matrix.

        Returns
        -------
        xA : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x from the right.

        Notes
        -----
        This is copied from dot to implement right multiplication.
        """
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(x, self)
        elif np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            x = np.asarray(x)

            if x.ndim == 1 or x.ndim == 2 and x.shape[0] == 1:
                return self.rmatvec(x.T.conj()).T.conj()
            elif x.ndim == 2:
                return self.rmatmat(x.T.conj()).T.conj()
            else:
                raise ValueError("expected 1-d or 2-d array or matrix, got %r" % x)

    LinearOperator.__rmul__ = __rmul__
    LinearOperator._rdot = _rdot
    LinearOperator.__array_ufunc__ = None
    del __rmul__, _rdot


# Monkey-patch scipy adjoint of matrices to support fermionic and bosonic Hamiltonians
def _eval_adjoint(self):
    return self._new(self.cols, self.rows, lambda i, j: self[j, i].adjoint())


MatrixOperations._eval_adjoint = _eval_adjoint


def _eval_is_matrix_hermitian(self, simpfunc):
    mat = self._new(
        self.rows, self.cols, lambda i, j: simpfunc(self[i, j] - self[j, i].adjoint())
    )
    return mat.is_zero_matrix


MatrixSpecial._eval_is_matrix_hermitian = lambda self: _eval_is_matrix_hermitian(
    self, lambda x: x.simplify()
)


def direct_greens_function(
    h: spmatrix,
    E: float,
    atol: float = 1e-7,
    eps: Optional[float] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Compute the Green's function of a Hamiltonian using MUMPS solver.

    Parameters
    ----------
    h :
        Hamiltonian matrix.
    E :
        Energy at which to compute the Green's function.
    atol :
        Accepted precision of the desired result in 2-norm.
    eps :
        Tolerance for the MUMPS solver to identify null pivots. Passed through
        to MUMPS CNTL(3), see MUMPS user guide.

    Returns
    -------
    greens_function : `Callable[[np.ndarray], np.ndarray]`
        Function that computes the Green's function at a given energy.
    """
    original_type = h.dtype
    h_is_real = np.issubdtype(original_type, np.floating)
    h = h.astype(complex)  # Kwant MUMPS wrapper only has complex bindings.
    h = h - E * identity(h.shape[0], dtype=h.dtype, format="csr")
    ctx = mumps.MUMPSContext()
    ctx.analyze(h)
    ctx.mumps_instance.icntl[24] = 1
    if eps is not None:
        ctx.mumps_instance.cntl[3] = eps
    ctx.factor(h)

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
            Solution of :math:`(H - E) sol = vec`.
        """
        is_real = np.issubdtype(vec.dtype, np.floating) and h_is_real
        sol = ctx.solve(vec)
        if (residue := np.linalg.norm(h @ sol - vec)) > atol:
            warn(
                f"Solution only achieved precision {residue} > {atol}."
                " adjust eps or atol.",
                RuntimeWarning,
            )
        if is_real:
            sol = sol.real
        return sol.astype(np.find_common_type([], [original_type, vec.dtype]))

    return greens_function


class ComplementProjector(LinearOperator):
    def __init__(self, vecs):
        """Projector on the complement of the span of vecs"""
        self.shape = (vecs.shape[0], vecs.shape[0])
        self._vecs = vecs
        self.dtype = vecs.dtype

    __array_ufunc__ = None

    def _matvec(self, v):
        return v - self._vecs @ (self._vecs.conj().T @ v)

    _matmat = _rmatvec = _rmatmat = _matvec

    def _adjoint(self):
        return self

    def conjugate(self):
        return self.__class__(vecs=self._vecs.conj())

    _transpose = conjugate


def aslinearoperator(A):
    """
    Same as `scipy.sparse.linalg.aslinearoperator`, but with passthrough for
    `~lowdin.series.zero`.
    """
    if zero == A or A is one:
        return A
    return scipy_aslinearoperator(A)


def is_diagonal(A, atol=1e-12):
    """Check if A is diagonal."""
    if zero == A or A is np.ma.masked:
        return True
    if isinstance(A, sympy.MatrixBase):
        return A.is_diagonal()
    elif isinstance(A, np.ndarray):
        # Create a view of the offdiagonal array elements
        offdiagonal = A.reshape(-1)[:-1].reshape(len(A) - 1, len(A) + 1)[:, 1:]
        return not np.any(np.round(offdiagonal, int(-np.log10(atol))))
    elif sparse.issparse(A):
        A = sparse.dia_array(A)
        return not any(A.offsets)
    raise NotImplementedError(f"Cannot extract diagonal from {type(A)}")
