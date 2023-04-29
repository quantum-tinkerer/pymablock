from typing import Callable

from packaging.version import parse
import numpy as np
from scipy import __version__ as scipy_version
from scipy.sparse import spmatrix, identity
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import aslinearoperator as scipy_aslinearoperator
from kwant.linalg import mumps
from scipy import sparse
import sympy

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


def direct_greens_function(
    h: spmatrix,
    E: float,
    atol: float = 1e-7,
    eps: float = 1e-10,
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
        to MUMPS CNTL(3) with a - sign, see MUMPS user guide.

    Returns
    -------
    greens_function : `Callable[[np.ndarray], np.ndarray]`
        Function that computes the Green's function at a given energy.
    """
    h = h.astype(np.complex128)  # Kwant MUMPS wrapper only has complex bindings.
    h = h - E * identity(h.shape[0], dtype=h.dtype, format="csr")
    ctx = mumps.MUMPSContext()
    ctx.analyze(h)
    ctx.mumps_instance.icntl[24] = 1
    ctx.mumps_instance.cntl[3] = -eps
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
        sol = ctx.solve(vec)
        if np.linalg.norm(h @ sol - vec) > atol:
            raise RuntimeError(
                f"Solution did not achieve required precision of {atol}."
            )
        return sol

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
    """Same as scipy.sparse.linalg.aslinearoperator, but with passthrough."""
    if zero == A or A is one:
        return A
    return scipy_aslinearoperator(A)


def is_diagonal(A):
    """Check if A is diagonal"""
    if isinstance(A, sympy.MatrixBase):  # sympy
        return A.is_diagonal()
    elif isinstance(A, np.ndarray):
        def offdiagonal(B):
            return B.reshape(-1)[:-1].reshape(len(B) - 1, len(B) + 1)[:, 1:]
        return not np.any(offdiagonal(A))
    elif sparse.issparse(A):
        A = sparse.dia_array(A)  # numpy or scipy.sparse
        return not any(A.offsets)
    return False
