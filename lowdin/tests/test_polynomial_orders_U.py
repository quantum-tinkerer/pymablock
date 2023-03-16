from itertools import count, permutations

import numpy as np
import tinyarray as ta
import pytest
from sympy.physics.quantum import Dagger

from lowdin.polynomial_orders_U import block_diagonalize, to_BlockOperatorSeries
from lowdin.series import BlockOperatorSeries, cauchy_dot_product, _zero


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
def H(Ns, wanted_orders):
    """
    Produce random Hamiltonians to test.

    Ns: dimension of each block (A, B)
    wanted_orders: list of orders to compute

    Returns:
    BlockOperatorSeries of the Hamiltonian
    """
    n_infinite = len(wanted_orders[0])
    orders = ta.array(np.eye(n_infinite))
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

    return to_BlockOperatorSeries(*hams, n_infinite)


def test_check_AB(H, wanted_orders):
    """
    Test that H_AB is zero for a random Hamiltonian.

    H: BlockOperatorSeries of the Hamiltonian
    wanted_orders: list of orders to compute
    """
    H_tilde = block_diagonalize(H)[0]
    for order in wanted_orders:
        order = tuple(slice(None, dim_order + 1) for dim_order in order)
        for block in H_tilde.evaluated[(0, 1) + order].compressed():
            np.testing.assert_allclose(
                block, 0, atol=10**-5, err_msg=f"{block=}, {order=}"
            )


def test_check_unitary(H, wanted_orders):
    """
    Test that the transformation is unitary.

    H: BlockOperatorSeries of the Hamiltonian
    wanted_orders: list of orders to compute
    """
    zero_order = (0,) * len(wanted_orders[0])
    N_A, N_B = H.evaluated[(0, 0) + zero_order].shape[0], H.evaluated[(1, 1) + zero_order].shape[0]
    n_infinite = H.n_infinite
    identity = to_BlockOperatorSeries(np.eye(N_A), np.eye(N_B), {}, {}, {}, n_infinite)
    _, U, U_adjoint = block_diagonalize(H)
    transformed = cauchy_dot_product(U_adjoint, identity, U, hermitian=True)

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


def compute_first_order(H, order):
    return H.evaluated[(0, 0) + order]


def test_first_order_H_tilde(H, wanted_orders):
    """Test that the first order is computed correctly.

    hamiltonians: list of Hamiltonians
    wanted_orders: list of orders to compute
    """
    H_tilde = block_diagonalize(H)[0]
    Np = len(wanted_orders[0])
    for order in permutations((0,) * (Np - 1) + (1,)):
        result = H_tilde.evaluated[(0, 0) + order]
        expected = compute_first_order(H, order)
        if _zero == result:
            np.testing.assert_allclose(
                0, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
            )
        np.testing.assert_allclose(
            result, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
        )


def compute_second_order(H, order):
    n_infinite = H.n_infinite
    order = tuple(value//2 for value in order)
    H_0_AA, H_0_BB, H_p_AB = (
        H.evaluated[(0, 0) + (0,) * n_infinite],
        H.evaluated[(1, 1) + (0,) * n_infinite],
        H.evaluated[(0, 1) + order],
    )

    E_A = np.diag(H_0_AA)
    E_B = np.diag(H_0_BB)
    energy_denominators = 1 / (E_A.reshape(-1, 1) - E_B)
    V1 = -H_p_AB * energy_denominators
    return -(V1 @ Dagger(H_p_AB) + H_p_AB @ Dagger(V1)) / 2


def test_second_order_H_tilde(H, wanted_orders):
    """Test that the second order is computed correctly.

    hamiltonians: list of Hamiltonians
    wanted_orders: list of orders to compute
    """
    H_tilde = block_diagonalize(H)[0]
    n_infinite = H.n_infinite

    for order in permutations((0,) * (n_infinite - 1) + (2,)):
        result = H_tilde.evaluated[(0, 0) + order]
        expected = compute_second_order(H, order)
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
        H = to_BlockOperatorSeries(
            np.array([[1, 1], [1, 1]]),
            np.eye(2),
            {},
            {},
            {},
        )
        block_diagonalize(H)
