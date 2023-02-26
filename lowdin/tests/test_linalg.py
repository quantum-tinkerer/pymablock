import numpy as np
from numpy.testing import assert_allclose
from lowdin import linalg
from scipy.sparse.linalg import aslinearoperator


def test_linear_operator_rmatmul_patched():
    """Test that LinearOperator implement right multiplication"""
    array = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    operator = aslinearoperator(array)
    assert_allclose(array @ operator, array @ array)


def test_complement_projector():
    """Test ComplementProjector against explicit implementation"""
    vec_A = np.random.randn(3, 10) + 1j * np.random.randn(3, 10)
    projector = linalg.ComplementProjector(vec_A)
    explicit = np.eye(10) - vec_A.conj().T @ vec_A
    assert_allclose(projector @ np.eye(10), explicit)
    assert_allclose(np.eye(10) @ projector, explicit)


def test_complement_projected():
    """Test complement_projected against explicit implementation"""
    vec_A = np.random.randn(3, 10) + 1j * np.random.randn(3, 10)
    array = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
    explicit_projector = np.eye(10) - vec_A.conj().T @ vec_A
    explicit_projected = explicit_projector @ array @ explicit_projector
    assert_allclose(
        np.ones(10) @ linalg.complement_projected(array, vec_A),
        np.ones(10) @ explicit_projected,
    )
    assert_allclose(
        np.eye(10) @ linalg.complement_projected(array, vec_A), explicit_projected
    )
