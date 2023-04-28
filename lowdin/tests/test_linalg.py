import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator
from scipy import sparse
import sympy

from lowdin import linalg


def test_linear_operator_rmatmul_patched():
    """Test that LinearOperator implement right multiplication"""
    array = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    operator = aslinearoperator(array)
    assert_allclose(array @ operator, array @ array)


def test_direct_greens_function():
    n = 100
    E = np.random.randn(n)
    t = np.random.rand(n - 1) * np.exp(2j * np.pi * np.random.rand(n - 1))
    h = sparse.diags([t, E, t.conj()], [-1, 0, 1])
    eigvals, eigvecs = np.linalg.eigh(h.toarray())
    G = linalg.direct_greens_function(h, E[n // 3])
    vec = np.random.randn(n) + 1j * np.random.randn(n)
    vec -= (eigvecs[:, n // 3].conj() @ vec) * eigvecs[:, n // 3]
    sol = G(vec)
    assert_allclose(h @ sol - E[n // 3] * sol, vec)


def test_complement_projector():
    """Test ComplementProjector against explicit implementation"""
    vec_A = np.random.randn(10, 3) + 1j * np.random.randn(10, 3)
    projector = linalg.ComplementProjector(vec_A)
    explicit = np.eye(10) - vec_A @ vec_A.conj().T
    assert_allclose(projector @ np.eye(10), explicit)
    assert_allclose(np.eye(10) @ projector, explicit)


def test_is_diagonal():
    array = np.random.randint(0, 4, size=(3, 3))
    assert not linalg.is_diagonal(array)
    assert linalg.is_diagonal(np.diag(np.diag(array)))

    sparse_array = sparse.csr_array(array)
    assert not linalg.is_diagonal(sparse_array)
    assert linalg.is_diagonal(sparse.diags(sparse_array.diagonal()))

    sympy_matrix = sympy.Matrix(array)
    assert not linalg.is_diagonal(sympy_matrix)
    assert linalg.is_diagonal(sympy.Matrix.diag(*sympy_matrix.diagonal()))
