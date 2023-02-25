import numpy as np
from numpy.testing import assert_allclose
from lowdin import linalg
from scipy.sparse.linalg import aslinearoperator


def test_linear_operator_identity():
    """Test that LinearOperator correctly wraps scipy.sparse.linalg.LinearOperator"""
    array = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    operator = aslinearoperator(array)
    wrapped_operator = linalg.LinearOperator(operator)
    # Various products
    assert_allclose(wrapped_operator @ np.ones((3,)), operator @ np.ones((3,)))
    assert_allclose(wrapped_operator @ np.ones((3, 3)), operator @ np.ones((3, 3)))
    assert_allclose(np.ones((3,)) @ wrapped_operator, operator.rmatvec(np.ones((3,))))
    assert_allclose(
        np.ones((3, 3)) @ wrapped_operator, operator.rmatmat(np.ones((3, 3)))
    )
    # Adjoint
    assert_allclose(
        wrapped_operator.adjoint() @ np.ones((3,)), operator.adjoint() @ np.ones((3,))
    )
    # Transpose
    assert_allclose(
        wrapped_operator.transpose() @ np.ones((3,)),
        operator.transpose() @ np.ones((3,)),
    )
    # Shouldn't raise
    np.ones((3,)) @ wrapped_operator


def test_complement_projector():
    """Test ComplementProjector against explicit implementation"""
    vec_A = np.random.randn(3, 10) + 1j * np.random.randn(3, 10)
    projector = linalg.ComplementProjector(vec_A)
    explicit = np.eye(10) - vec_A.conj().T @ vec_A
    assert_allclose(projector @ np.eye(10), explicit)


def test_complement_projected():
    """Test complement_projected against explicit implementation"""
    vec_A = np.random.randn(3, 10) + 1j * np.random.randn(3, 10)
    array = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
    explicit_projector = np.eye(10) - vec_A.conj().T @ vec_A
    explicit_projected = explicit_projector @ array @ explicit_projector
    assert_allclose(
        linalg.complement_projected(array, vec_A) @ np.eye(10), explicit_projected
    )
