from itertools import product, count

import pytest
import numpy as np
import tinyarray as ta
from scipy.linalg import eigh, block_diag

from lowdin.poly_kpm import create_div_energs, numerical
from lowdin.linalg import ComplementProjector


@pytest.fixture(
    scope="module",
    params=[
        [[3]],
        [[2, 2]],
        [[3, 1], [1, 3]],
        [[2, 2, 2], [3, 0, 0]],
    ],
)
def wanted_orders(request):
    """
    Return a list of orders to compute.
    """
    return request.param


@pytest.fixture(scope="module")
def Ns():
    """
    Return a random number of states for each block (A, B).
    """
    return np.random.randint(1, high=5, size=2)


@pytest.fixture(scope="module")
def hamiltonians(Ns, wanted_orders):
    """
    Produce random Hamiltonians to test.

    Ns: dimension of each block (A, B)
    wanted_orders: list of orders to compute

    Returns:
    hams: list of Hamiltonians
    """
    N_p = len(wanted_orders[0])
    orders = ta.array(np.eye(N_p))
    hams = []
    for i in range(2):
        hams.append(np.diag(np.sort(np.random.rand(Ns[i])) - i))

    def matrices_it(N_i, N_j, hermitian):
        """
        Generate random matrices of size N_i x N_j.

        N_i: number of rows
        N_j: number of columns
        hermitian: if True, the matrix is hermitian

        Returns:
        generator of random matrices
        """
        for i in count():
            H = np.random.rand(N_i, N_j) + 1j * np.random.rand(N_i, N_j)
            if hermitian:
                H += H.conj().T
            yield H

    for i, j, hermitian in zip([0, 1, 0], [0, 1, 1], [True, True, False]):
        matrices = matrices_it(Ns[i], Ns[j], hermitian)
        hams.append({order: matrix for order, matrix in zip(orders, matrices)})
    return hams


def assert_almost_zero(a, decimal=5, extra_msg=""):
    """
    Assert that all values in a are almost zero.

    a: array to check
    decimal: number of decimal places to check
    extra_msg: extra message to print if assertion fails
    """
    for key, value in a.items():
        np.testing.assert_almost_equal(
            value, 0, decimal=decimal, err_msg=f"{key=} {extra_msg}"
        )


#############################################################################################

def test_create_div_energs_kpm(hamiltonians):
    n_a = hamiltonians[0].shape[0]
    n_b = hamiltonians[1].shape[0]
    h_0 = block_diag(hamiltonians[0], hamiltonians[1])
    eigs, vecs = eigh(h_0)
    eigs_a, vecs_a = eigs[:n_a], vecs[:, :n_a]
    eigs_b, vecs_b = eigs[n_a:], vecs[:, n_a:]

    Y = []
    for _ in range(5):
        h_ab = np.random.random((n_a + n_b, n_a + n_b)) + 1j * np.random.random(
            (n_a + n_b, n_a + n_b)
        )
        h_ab += h_ab.conj().T
        Y.append(vecs_a.conj().T @ h_ab @ ComplementProjector(vecs_a))

    de_kpm_func = lambda Y: create_div_energs(
        h_0, vecs_a, eigs_a, kpm_params=dict(num_moments=1000)
    )(Y)
    de_exact_func = lambda Y: create_div_energs(h_0, vecs_a, eigs_a, vecs_b, eigs_b)(Y)

    # apply h_ab from left -> Y.conj() since G_0 is hermitian
    applied_exact = [de_exact_func(y.conj()) for y in Y]
    applied_kpm = [de_kpm_func(y.conj()) for y in Y]

    diff_approach = {
        i: np.abs(applied_exact[i] - applied_kpm[i]) for i in range(len(Y))
    }

    assert_almost_zero(diff_approach, decimal=1, extra_msg="")


def test_ab_is_zero():
    max_ord = 4
    n_dim = np.random.randint(low=20, high=100)
    n_a = np.random.randint(low=0, high=int(n_dim / 2))
    h_0 = np.diag(np.sort(np.random.random(n_dim)))
    h_0[n_a:, n_a:] = 15 * h_0[n_a:, n_a:] # increase gap for test to succeed in more cases
    eigs_a = np.diag(h_0)[:n_a]
    vecs_a = np.eye(n_dim)[:, :n_a]

    h_p = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
    h_p += h_p.conjugate().transpose()

    ham = {(0,): h_0, (1,): h_p}

    h_t, u, u_adj = numerical(
        ham, vecs_a, eigs_a, kpm_params={"num_moments": 10000}
    )

    ab_s = h_t.evaluated[0, 1, :max_ord]
    ab_s = list(ab_s[~ab_s.mask].data)
    ab_s = {i: ab_s[i] for i in range(len(ab_s))}

    assert_almost_zero(ab_s, decimal=1, extra_msg="")
