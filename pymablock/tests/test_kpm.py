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
    g_vals = 1 / (eigvals - eigvals[n0])
    g_vals[n0] = 0
    G = eigvecs @ np.diag(g_vals) @ eigvecs.T.conj()

    vec = np.random.randn(n) + 1j * np.random.randn(n)
    vec -= (eigvecs[:, n0].conj() @ vec) * eigvecs[:, n0]

    sol = G @ vec
    sol_kpm = kpm.greens_function(h, eigvals[n0], vec, num_moments=100)

    assert_allclose(h @ sol_kpm - eigvals[n0] * sol_kpm, -vec)
    assert_allclose(sol_kpm, -sol)
