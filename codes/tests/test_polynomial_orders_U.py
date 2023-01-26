from itertools import count

import numpy as np
import tinyarray as ta
import pytest

from codes.polynomial_orders_U import compute_next_orders, H_tilde


def assert_almost_zero(a, decimal, extra_msg=""):
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


@pytest.fixture(scope="module")
def decimal():
    return 5


@pytest.fixture(scope="module")
def N_A():
    return np.random.randint(1, high=10)


@pytest.fixture(scope="module")
def N_B():
    return np.random.randint(1, high=10)


@pytest.fixture(scope="module")
@pytest.mark.repeat(10, scope="session")
def N_p():
    return np.random.randint(0, high=6)


@pytest.fixture(scope="module")
def wanted_orders(N_p):
    return [
        np.random.randint(0, high=3, size=N_p),
        ta.array([4] + [0 for i in range(N_p - 1)]),
    ]


@pytest.fixture(scope="module")
def H_0_AA(N_A):
    return np.diag(np.sort(np.random.rand(N_A)))


@pytest.fixture(scope="module")
def H_0_BB(N_B):
    return np.diag(np.sort(np.random.rand(N_B)))


def matrices_it(N_i, N_j, hermitian):
    """
    Generate random matrices of size N_i x N_j.
    """
    for i in count():
        H = np.random.rand(N_i, N_j) + 1j * np.random.rand(N_i, N_j)
        if hermitian:
            H += H.conj().T
        yield H


@pytest.fixture(scope="module")
def H_p_AA(N_A, N_p):
    orders = ta.array(np.eye(N_p))
    matrices = matrices_it(N_A, N_A, hermitian=True)
    return {order: matrix for order, matrix in zip(orders, matrices)}


@pytest.fixture(scope="module")
def H_p_BB(N_B, N_p):
    orders = ta.array(np.eye(N_p))

    matrices = matrices_it(N_B, N_B, hermitian=True)
    return {order: matrix for order, matrix in zip(orders, matrices)}


@pytest.fixture(scope="module")
def H_p_AB(N_A, N_B, N_p):
    orders = ta.array(np.eye(N_p))
    matrices = matrices_it(N_A, N_B, hermitian=False)
    return {order: matrix for order, matrix in zip(orders, matrices)}


def test_check_AB(decimal, H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders):
    """
    Test that H_AB is zero for a random Hamiltonian.

    H_0_AA: N_A x N_A matrix
    H_0_BB: N_B x N_B matrix
    H_p_AA: N_p x N_A x N_A matrix
    H_p_BB: N_p x N_B x N_B matrix
    H_p_AB: N_p x N_A x N_B matrix
    wanted_orders: list of orders to compute
    """
    exp_S = compute_next_orders(
        H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders=wanted_orders
    )

    H_AB = H_tilde(
        H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, exp_S, compute_AB=True
    )[2]

    assert_almost_zero(H_AB, decimal)
