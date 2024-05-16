# ruff: noqa: N803, N806
import builtins

import numpy as np
import sympy
from numpy.testing import assert_allclose
from pytest import mark, raises
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator

from pymablock import linalg


def test_linear_operator_rmatmul_patched():
    """Test that LinearOperator implement right multiplication"""
    array = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    operator = aslinearoperator(array)
    assert_allclose(array @ operator, array @ array)


@mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_direct_greens_function(dtype):
    atol = 1e4 * np.finfo(dtype).eps
    n = 100
    E = np.random.randn(n).astype(dtype)
    t = np.random.rand(n - 1).astype(dtype)
    if np.iscomplexobj(E):
        t *= np.exp(2j * np.pi * np.random.rand(n - 1))
    h = sparse.diags([t, E, t.conj()], [-1, 0, 1])
    eigvals, eigvecs = np.linalg.eigh(h.toarray())
    n0 = n // 3
    G = linalg.direct_greens_function(h, E[n0], atol=atol)
    vec = np.random.randn(n).astype(dtype)
    if np.iscomplexobj(vec):
        vec += 1j * np.random.randn(n)
    vec -= (eigvecs[:, n0].conj() @ vec) * eigvecs[:, n0]
    sol = G(vec)
    assert_allclose(h @ sol - E[n0] * sol, -vec, atol=atol)


def test_direct_greens_function_dtype():
    """Test that type promotion works as expected."""
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
    def __import__(*args, **kwargs):  # noqa ARG001
        raise ImportError

    with monkeypatch.context() as monkeypatch:
        monkeypatch.setattr(builtins, "__import__", __import__)
        with raises(ImportError):
            linalg.direct_greens_function(sparse.diags([1, 2, 3]))
