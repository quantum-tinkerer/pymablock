from itertools import count

import numpy as np
import tinyarray as ta
import pytest

from codes.polynomial_orders_U import compute_next_orders, H_tilde


@pytest.fixture(scope="module", params=range(2, 6))
def wanted_orders(request):
    N_p = 4
    return [
        np.random.randint(0, high=5, size=N_p),
        ta.array([4] + [0 for i in range(N_p - 1)]),
    ]


@pytest.fixture(scope="module")
def Ns():
    return np.random.randint(1, high=5, size=2)


@pytest.fixture(scope="module")
def hamiltonians(Ns, wanted_orders):
    N_p = len(wanted_orders[0])
    orders = ta.array(np.eye(N_p))
    hams = []
    for i in range(2):
        hams.append(np.diag(np.sort(np.random.rand(Ns[i])) - i))

    def matrices_it(N_i, N_j, hermitian):
        """
        Generate random matrices of size N_i x N_j.
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


def test_check_AB(hamiltonians, wanted_orders):
    """
    Test that H_AB is zero for a random Hamiltonian.

    hamiltonians:
    wanted_orders: list of orders to compute
    """
    exp_S = compute_next_orders(*hamiltonians, wanted_orders=wanted_orders)

    H_AB = H_tilde(*hamiltonians, wanted_orders, exp_S, compute_AB=True)[2]

    assert_almost_zero(H_AB)


def test_check_unitary(Ns, hamiltonians, wanted_orders):
    """
    Test that the transformation is unitary.

    Ns
    hamiltonians:
    wanted_orders: list of orders to compute
    """
    decimal = 5
    exp_S = compute_next_orders(*hamiltonians, wanted_orders=wanted_orders)
    transformed = H_tilde(
        np.eye(Ns[0]), np.eye(Ns[1]), {}, {}, {}, wanted_orders, exp_S, compute_AB=True
    )

    for value, block in zip(transformed, "AA BB AB".split()):
        assert_almost_zero(value, decimal, f"{block=}")
