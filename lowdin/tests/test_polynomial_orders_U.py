from itertools import count, permutations

import numpy as np
import tinyarray as ta
import pytest
from sympy.physics.quantum import Dagger

from lowdin.polynomial_orders_U import compute_next_orders, H_tilde
from lowdin.series import _zero


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

    a: dict to check
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

    hamiltonians: list of Hamiltonians
    wanted_orders: list of orders to compute
    """
    exp_S = compute_next_orders(*hamiltonians)
    H = H_tilde(*hamiltonians, exp_S)
    for order in wanted_orders:
        order = tuple(slice(None, dim_order + 1) for dim_order in order)
        for block in H.evaluated[(0, 1) + order].compressed():
            np.testing.assert_allclose(block, 0, atol=10**-5,  err_msg=f"{block=}, {order=}")


def test_check_unitary(hamiltonians, wanted_orders):
    """
    Test that the transformation is unitary.

    hamiltonians : list of Hamiltonians
    wanted_orders: list of orders to compute
    """
    N_A = hamiltonians[0].shape[0]
    N_B = hamiltonians[1].shape[0]
    exp_S = compute_next_orders(*hamiltonians)
    transformed = H_tilde(np.eye(N_A), np.eye(N_B), {}, {}, {}, exp_S)
    for order in wanted_orders:
        order = tuple(slice(None, dim_order + 1) for dim_order in order)
        for block in ((0, 0), (1, 1), (0, 1)):
            result = transformed.evaluated[tuple(block + order)]
            for index, block in np.ma.ndenumerate(result):
                if not any(index):
                    # Zeroth order is not zero.
                    continue
                np.testing.assert_allclose(
                    block, 0, atol=10**-5, err_msg=f"{block=}, {index=}"
                )

def compute_first_order(H_p_AA, order):
    return H_p_AA[order]

def test_first_order_H_tilde(hamiltonians, wanted_orders):
    """Test that the first order is computed correctly.

    hamiltonians: list of Hamiltonians
    wanted_orders: list of orders to compute
    """
    exp_S = compute_next_orders(*hamiltonians)
    H = H_tilde(*hamiltonians, exp_S)
    Np = len(wanted_orders[0])
    for order in permutations((0,) * (Np - 1) + (1,)):
        result = H.evaluated[(0, 0) + order]
        expected = compute_first_order(hamiltonians[2], order)
        if _zero == result:
            np.testing.assert_allclose(
                0, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
            )
        np.testing.assert_allclose(
            result, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
        )

def compute_second_order(H_0_AA, H_0_BB, H_p_AB, order):
    order = ta.array(order) / 2
    E_A = np.diag(H_0_AA)
    E_B = np.diag(H_0_BB)
    energy_denominators = 1 / (E_A.reshape(-1, 1) - E_B)
    V1 = -H_p_AB[order] * energy_denominators
    return -(V1 @ Dagger(H_p_AB[order]) + H_p_AB[order] @ Dagger(V1)) / 2

def test_second_order_H_tilde(hamiltonians, wanted_orders):
    """Test that the second order is computed correctly.

    hamiltonians: list of Hamiltonians
    wanted_orders: list of orders to compute
    """
    exp_S = compute_next_orders(*hamiltonians)
    H = H_tilde(*hamiltonians, exp_S)
    Np = len(wanted_orders[0])
    for order in permutations((0,) * (Np - 1) + (2,)):
        result = H.evaluated[(0, 0) + order]
        H_0_AA, H_0_BB, H_p_AB = hamiltonians[0], hamiltonians[1], hamiltonians[4]
        expected = compute_second_order(H_0_AA, H_0_BB, H_p_AB, order)
        if _zero == result:
            np.testing.assert_allclose(
                0, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
            )
        np.testing.assert_allclose(
            result, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
        )


def test_check_diagonal():
    """Test that offdiagonal H_0_AA is not allowed if divide_by_energies is not provided."""
    with pytest.raises(ValueError):
        compute_next_orders(
            np.array([[1, 1], [1, 1]]),
            np.eye(2),
            {},
            {},
            {},
        )
