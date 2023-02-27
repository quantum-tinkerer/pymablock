import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._interface import _ProductLinearOperator, _ScaledLinearOperator

# Monkey-patch LinearOperator to support right multiplication
# TODO: Remove this when https://github.com/scipy/scipy/pull/18061
# is merged and released
try:
    _ = np.eye(3) @ aslinearoperator(np.eye(3))
except ValueError:

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


class ComplementProjector(LinearOperator):
    def __init__(self, vec_A):
        """Projector on the complement of the span of vec_A"""
        self.shape = (vec_A.shape[1], vec_A.shape[1])
        self._vec_A = vec_A
        self.dtype = vec_A.dtype

    __array_ufunc__ = None

    def _matvec(self, v):
        return v - self._vec_A.conj().T @ (self._vec_A @ v)

    _matmat = _rmatvec = _rmatmat = _matvec

    def _adjoint(self):
        return self

    def conjugate(self):
        return self.__class__(vec_A=self._vec_A.conj())

    _transpose = conjugate


def complement_projected(operator, vec_A):
    """Project operator on the complement of the span of vec_A"""
    projector = ComplementProjector(vec_A)
    operator = aslinearoperator(operator)
    return projector @ operator @ projector
