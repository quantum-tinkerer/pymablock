from packaging.version import parse
import numpy as np
from scipy import __version__ as scipy_version
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import aslinearoperator as scipy_aslinearoperator

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
