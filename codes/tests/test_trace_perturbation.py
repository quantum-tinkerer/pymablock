import numpy as np
import scipy
from codes.trace_perturbation import trace_perturbation
from qsymm.model import Model
from ..qsymm.linalg import allclose
from scipy.sparse.linalg import LinearOperator

def test_simple_model():
    order = 2
    atol = 2e-2
    N = 1000
    kpm_params = dict(rng=0)
    # Simple model
    ev2 = np.array([0] * N + [2] * N)
    mat02 = scipy.sparse.diags(ev2, dtype=complex, format='csr')
    V = scipy.sparse.lil_matrix((2 * N, 2 * N), dtype=complex)
    for i in range(N):
        V[i + N, i + N] = 2
        V[i, i + N] = 1
        V[i + N, i] = 1
    mat12 = {'x': V.tocsr()}

    model = trace_perturbation(mat02, mat12, order=order, kpm_params=kpm_params,
                               num_moments=200, num_vectors=10)

    # Step function
    def func(x):
        return 1 * (x < 1) + 0 * (x >= 1)

    res1 = model(func) / N
    # Density shouldn't change
    exact1 = Model({1: 1})
    assert all([allclose(v, exact1[k], atol=atol) for k, v in res1.items()]), res1 - exact1

    # Energy function
    def func(x):
        return x * (1 * (x < 1) + 0 * (x >= 1))

    res2 = model(func) / N
    # Energy changes at second order
    exact2 = Model({'x**2': -0.5})
    assert all([allclose(v, exact2[k], atol=atol) for k, v in res2.items()]), res2 - exact2

    # Test operator
    a = np.array([1] * N + [0] * N) / np.sqrt(N)
    operator = Model({1: LinearOperator((2 * N, 2 * N), lambda v: v.T - a * (a.dot(v)))})
    operator *= 1/(N-1)
    model = trace_perturbation(mat02, mat12, order=2, kpm_params=kpm_params,
                               num_moments=200, num_vectors=10,
                               operator=operator)

    res3 = model(func)
    assert all([allclose(v, exact2[k], atol=atol) for k, v in res3.items()]), res3 - exact2
