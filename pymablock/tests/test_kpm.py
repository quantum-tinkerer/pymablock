import numpy as np
from numpy.testing import assert_allclose

from pymablock import kpm


def test_kpm_greens_function():
    np.random.seed(0)
    n = 10
    n0 = n // 3
    h = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    h += h.conj().T
    h, *_ = kpm.rescale(h)
    eigvals, eigvecs = np.linalg.eigh(h)

    vec = np.random.randn(n) + 1j * np.random.randn(n)
    vec -= (eigvecs[:, n0].conj() @ vec) * eigvecs[:, n0]

    sol = kpm.greens_function(h, eigvals[n0], vec, atol=1e-7)

    assert_allclose(h @ sol - eigvals[n0] * sol, -vec, atol=1e-7)
    assert_allclose(sol.conj() @ eigvecs[:, n0], 0, atol=1e-7)


def test_rescale_lower_bounds():
    n = 10
    h = np.diag(np.linspace(0, 1, n))
    h_rescaled, (a, b) = kpm.rescale(h, lower_bounds=[-5, 0])

    assert np.abs((5.5 - b) / a) > 1
    assert np.abs((0.01 - b) / a) < 1
    assert np.abs((1.1 - b) / a) > 1
    assert np.abs((-4.5 - b) / a) < 1
