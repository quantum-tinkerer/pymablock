import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymablock import kpm


def test_kpm_greens_function(rng):
    n = 10
    n0 = n // 3
    h = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h += h.conj().T
    h, *_ = kpm.rescale(h)
    eigvals, eigvecs = np.linalg.eigh(h)

    vec = rng.standard_normal(n) + 1j * rng.standard_normal(n)
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


@pytest.mark.parametrize("budget", [1, 5, 9, 10, 11, np.int64(5), 5.0])
def test_small_kpm_moment_budgets(budget, monkeypatch):
    counts = []
    jackson = kpm.jackson_kernel

    def record_kernel(count):
        counts.append(count)
        return jackson(count)

    monkeypatch.setattr(kpm, "jackson_kernel", record_kernel)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = kpm.greens_function(
            np.diag([0.1, 0.5]), 0.9, np.array([1.0, 0.0]), max_moments=budget, atol=1e-12
        )
    assert result.shape == (2,)
    assert np.all(np.isfinite(result))
    assert counts[-1] == budget
    assert all(0 < count <= budget for count in counts)
    assert result[1] == 0


@pytest.mark.parametrize("budget", [0, -1, 1.5, np.inf, np.nan, True, "5", None])
def test_invalid_kpm_moment_budgets(budget):
    with pytest.raises(ValueError, match="positive integer"):
        kpm.greens_function(np.eye(2), 0.9, np.ones(2), max_moments=budget)
