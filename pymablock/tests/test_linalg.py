import builtins

from pytest import raises
import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator
import sympy

from pymablock import linalg


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
    n0 = n // 3
    G = linalg.direct_greens_function(h, E[n0], atol=1e-7)
    vec = np.random.randn(n) + 1j * np.random.randn(n)
    vec -= (eigvecs[:, n0].conj() @ vec) * eigvecs[:, n0]
    sol = G(vec)
    assert_allclose(h @ sol - E[n0] * sol, -vec, atol=1e-7)


def test_direct_greens_function_dtype():
    n = 10
    E = np.random.randn(n).astype(np.float32)
    gf = linalg.direct_greens_function(sparse.diags(E), 0, atol=1e-3)
    assert gf(E).dtype == np.float32
    assert gf(1j * E).dtype == np.complex64


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


def test_no_mumps(monkeypatch):
    def __import__(*args, **kwargs):
        raise ImportError

    with monkeypatch.context() as monkeypatch:
        monkeypatch.setattr(builtins, "__import__", __import__)
        with raises(ImportError):
            linalg.direct_greens_function(sparse.diags([1, 2, 3]))
