import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse.linalg import aslinearoperator

from lowdin import linalg


def test_linear_operator_rmatmul_patched():
    """Test that LinearOperator implement right multiplication"""
    array = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    operator = aslinearoperator(array)
    assert_allclose(array @ operator, array @ array)


def test_complement_projector():
    """Test ComplementProjector against explicit implementation"""
    vec_A = np.random.randn(10, 3) + 1j * np.random.randn(10, 3)
    projector = linalg.ComplementProjector(vec_A)
    explicit = np.eye(10) -  vec_A @ vec_A.conj().T
    assert_allclose(projector @ np.eye(10), explicit)
    assert_allclose(np.eye(10) @ projector, explicit)
