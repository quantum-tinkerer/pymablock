import numpy as np
import scipy
import codes.trace_perturbation as pert
from codes.perturbative_model import PerturbativeModel
from ..qsymm.linalg import allclose

def test_simple_model():
    order = 2
    atol = 2e-2
    N = 1000
    kpm_params = dict(num_moments=200, num_vectors=10, rng=0)
    # Simple model
    ev2 = np.array([0] * N + [2] * N)
    mat02 = scipy.sparse.diags(ev2, dtype=complex, format='csr')
    V = scipy.sparse.lil_matrix((2 * N, 2 * N), dtype=complex)
    for i in range(N):
        V[i + N, i + N] = 2
        V[i, i + N] = 1
        V[i + N, i] = 1
    mat12 = {'x': V.tocsr()}

    model = pert.trace_perturbation(mat02, mat12, order=order, kpm_params=kpm_params)

    # Step function
    def func(x):
        return 1 * (x < 1) + 0 * (x >= 1)

    res1 = model(func) / N
    # Density shouldn't change
    exact1 = PerturbativeModel({1: 1})
    assert all([allclose(v, exact1[k], atol=atol) for k, v in res1.items()]), res1 - exact1

    # Energy function
    def func(x):
        return x * (1 * (x < 1) + 0 * (x >= 1))

    res2 = model(func) / N
    # Energy changes at second order
    exact2 = PerturbativeModel({'x**2': -0.5})
    assert all([allclose(v, exact1[k], atol=atol) for k, v in res1.items()]), res2 - exact2
