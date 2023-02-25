from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import aslinearoperator


class LinearOperator(ScipyLinearOperator):
    __array_ufunc__ = None
    def __init__(self, operator):
        """Wrapper around scipy.sparse.linalg.LinearOperator

        Disables __array_ufunc__ to allow right multiplication with numpy arrays

        Parameters
        ----------
        operator : scipy.sparse.linalg.LinearOperator
        """
        super().__init__(
            dtype=operator.dtype,
            shape=operator.shape,
        )
        self._operator = operator

    def _matvec(self, x):
        return self._operator._matvec(x)

    def _matmat(self, x):
        return self._operator._matmat(x)

    def _rmatvec(self, x):
        return self._operator._rmatvec(x)

    def _rmatmat(self, x):
        return self._operator._rmatmat(x)

    def _adjoint(self):
        return self.__class__(self._operator._adjoint())

    def conjugate(self):
        return self.__class__(self._operator.conjugate())

    def __rmatmul__(self, other):
        try:
            return self._rmatvec(other)
        except ValueError:
            return self._rmatmat(other)


class ComplementProjector(ScipyLinearOperator):
    def __init__(self, vec_A):
        """Projector on the complement of the span of vec_A"""
        self.shape = (vec_A.shape[1], vec_A.shape[1])
        self._vec_A = vec_A
        self.dtype = vec_A.dtype

    __array_ufunc__ = None

    def _matvec(self, v):
        return v - self._vec_A.conj().T @ (self._vec_A @ v)

    _matmat = _matvec

    def _adjoint(self):
        return self

    def conjugate(self):
        return self.__class__(vec_A=self._vec_A.conj())

    _transpose = conjugate


def complement_projected(operator, vec_A):
    """Project operator on the complement of the span of vec_A"""
    projector = ComplementProjector(vec_A)
    operator = aslinearoperator(operator)
    return LinearOperator(projector @ operator @ projector)