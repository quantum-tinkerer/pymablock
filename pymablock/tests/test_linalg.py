# ruff: noqa: N803, N806

import numpy as np
import pytest
import sympy
from numpy.testing import assert_allclose
from packaging.version import Version
from pytest import mark
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator

from pymablock import __version__, linalg


def test_linear_operator_rmatmul_patched(rng):
    """Test that LinearOperator implement right multiplication"""
    array = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    operator = aslinearoperator(array)
    assert_allclose(array @ operator, array @ array)


@mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_direct_greens_function(dtype, rng):
    atol = 1e4 * np.finfo(dtype).eps
    n = 100
    E = rng.standard_normal(n).astype(dtype)
    t = rng.random(n - 1).astype(dtype)
    if np.iscomplexobj(E):
        t *= np.exp(2j * np.pi * rng.random(n - 1))
    h = sparse.diags([t, E, t.conj()], [-1, 0, 1])
    eigvals, eigvecs = np.linalg.eigh(h.toarray())
    n0 = n // 3
    G = linalg.direct_greens_function(h, E[n0])
    vec = rng.standard_normal(n).astype(dtype)
    if np.iscomplexobj(vec):
        vec += 1j * rng.standard_normal(n)
    vec -= (eigvecs[:, n0].conj() @ vec) * eigvecs[:, n0]
    sol = G(vec)
    assert_allclose(h @ sol - E[n0] * sol, -vec, atol=atol)


def test_direct_greens_function_dtype(rng):
    """Test that type promotion works as expected."""
    n = 10
    E = rng.standard_normal(n).astype(np.float32)
    gf = linalg.direct_greens_function(sparse.diags(E), 0)
    assert gf(E).dtype == np.float32
    assert gf(1j * E).dtype == np.complex64


def test_direct_greens_function_ignored_arguments_warn():
    h = sparse.diags([1.0, 2.0, 3.0])
    if Version(__version__) >= Version("2.4.0"):
        pytest.fail(
            "`atol` and `eps` should be removed from direct_greens_function in 2.4.0"
        )

    with pytest.warns(
        DeprecationWarning,
        match="ignored by `direct_greens_function` and will be removed in version 2.4.0",
    ):
        linalg.direct_greens_function(h, 0.0, atol=1e-3, eps=0.1)


def test_direct_greens_function_degenerate_kernel(rng):
    n = 24
    multiplicity = 3
    spectrum = np.linspace(-2, 2, n)
    spectrum[:multiplicity] = 0.5

    basis, _ = np.linalg.qr(
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    )
    h_dense = basis @ np.diag(spectrum) @ basis.conj().T
    h = sparse.csr_array(h_dense)
    kernel_vectors = basis[:, :multiplicity]

    projector = linalg.ComplementProjector(kernel_vectors)
    vec = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    vec = projector @ vec

    gf = linalg.direct_greens_function(h, spectrum[0], kernel_vectors=kernel_vectors)
    sol = gf(vec)

    expected = basis[:, multiplicity:] @ (
        (basis[:, multiplicity:].conj().T @ vec) / (spectrum[0] - spectrum[multiplicity:])
    )
    assert_allclose(sol, expected, atol=1e-10)
    assert_allclose(kernel_vectors.conj().T @ sol, 0, atol=1e-10)


def test_complement_projector(rng):
    """Test ComplementProjector against explicit implementation"""
    vec_A = rng.standard_normal((10, 3)) + 1j * rng.standard_normal((10, 3))
    projector = linalg.ComplementProjector(vec_A)
    explicit = np.eye(10) - vec_A @ vec_A.conj().T
    assert_allclose(projector @ np.eye(10), explicit)
    assert_allclose(np.eye(10) @ projector, explicit)


def test_complement_projector_biorthogonal(rng):
    vec_right = rng.standard_normal((10, 3)) + 1j * rng.standard_normal((10, 3))
    vec_left = vec_right @ np.linalg.inv(vec_right.conj().T @ vec_right)
    projector = linalg.ComplementProjector(vec_right, vec_left)
    explicit = np.eye(10) - vec_right @ vec_left.conj().T
    assert_allclose(projector @ np.eye(10), explicit)
    assert_allclose(np.eye(10) @ projector, explicit)


def test_complement_projector_cached_transforms(rng):
    vecs = rng.standard_normal((10, 3)) + 1j * rng.standard_normal((10, 3))
    explicit = np.eye(10) - vecs @ vecs.conj().T
    for projector in (
        linalg.ComplementProjector(vecs),
        linalg.ComplementProjector(vecs, vecs),
        linalg.ComplementProjector(vecs, vecs.copy()),
    ):
        assert projector.H is projector
        assert projector.H.H is projector
        assert projector.conjugate() is projector.T
        assert projector.T is projector.T
        assert projector.T.T is projector
        assert_allclose(projector.T @ np.eye(10), explicit.T)

    vec_right = rng.standard_normal((10, 3)) + 1j * rng.standard_normal((10, 3))
    vec_left = vec_right @ np.linalg.inv(vec_right.conj().T @ vec_right)
    projector = linalg.ComplementProjector(vec_right, vec_left)
    explicit = np.eye(10) - vec_right @ vec_left.conj().T

    assert projector.H is projector.H
    assert projector.H.H is projector
    assert projector.T is projector.T
    assert projector.T.T is projector
    assert_allclose(projector.H @ np.eye(10), explicit.conj().T)
    assert_allclose(projector.T @ np.eye(10), explicit.T)


def test_is_diagonal(rng):
    array = rng.integers(0, 4, size=(3, 3))
    assert not linalg.is_diagonal(array)
    assert linalg.is_diagonal(np.diag(np.diag(array)))

    sparse_array = sparse.csr_array(array)
    assert not linalg.is_diagonal(sparse_array)
    assert linalg.is_diagonal(sparse.diags(sparse_array.diagonal()))

    sympy_matrix = sympy.Matrix(array)
    assert not linalg.is_diagonal(sympy_matrix)
    assert linalg.is_diagonal(sympy.Matrix.diag(*sympy_matrix.diagonal()))

    x = sympy.Symbol("x")
    symbolic_array = np.array([[x, 0], [0, x + 1]], dtype=object)
    assert linalg.is_diagonal(symbolic_array)
    symbolic_array[0, 1] = x
    assert not linalg.is_diagonal(symbolic_array)
