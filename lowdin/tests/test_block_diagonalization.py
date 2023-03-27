from itertools import count, permutations

import numpy as np
import tinyarray as ta
import pytest
from sympy.physics.quantum import Dagger

from lowdin.block_diagonalization import general, expanded, to_BlockSeries
from lowdin.series import cauchy_dot_product, zero


@pytest.fixture(
    scope="module",
    params=[
        [(3,)],
        [(2, 2)],
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
    BlockSeries of the Hamiltonian
    """
    n_infinite = len(wanted_orders[0])
    orders = ta.array(np.eye(n_infinite, dtype=int))
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

    return to_BlockSeries(*hams, n_infinite)


def test_check_AB(H, wanted_orders):
    """
    Test that H_AB is zero for a random Hamiltonian.

    H: BlockSeries of the Hamiltonian
    wanted_orders: list of orders to compute
    """
    H_tilde = general(H)[0]
    for order in wanted_orders:
        order = tuple(slice(None, dim_order + 1) for dim_order in order)
        for block in H_tilde.evaluated[(0, 1) + order].compressed():
            np.testing.assert_allclose(
                block, 0, atol=10**-5, err_msg=f"{block=}, {order=}"
            )


def test_check_unitary(H, wanted_orders):
    """
    Test that the transformation is unitary.

    H: BlockSeries of the Hamiltonian
    wanted_orders: list of orders to compute
    """
    zero_order = (0,) * len(wanted_orders[0])
    N_A, N_B = H.evaluated[(0, 0) + zero_order].shape[0], H.evaluated[(1, 1) + zero_order].shape[0]
    n_infinite = H.n_infinite
    identity = to_BlockSeries(np.eye(N_A), np.eye(N_B), {}, {}, {}, n_infinite)
    _, U, U_adjoint = general(H)
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
    """
    Compute the first order correction to the Hamiltonian.
    
    H: BlockSeries of the Hamiltonian
    order: tuple of orders to compute

    Returns:
    BlockSeries of the first order correction obtained explicitly
    """
    return H.evaluated[(0, 0) + order]


def test_first_order_H_tilde(H, wanted_orders):
    """
    Test that the first order is computed correctly.

    hamiltonians: list of Hamiltonians
    wanted_orders: list of orders to compute
    """
    H_tilde = general(H)[0]
    Np = len(wanted_orders[0])
    for order in permutations((0,) * (Np - 1) + (1,)):
        result = H_tilde.evaluated[(0, 0) + order]
        expected = compute_first_order(H, order)
        if zero == result:
            np.testing.assert_allclose(
                0, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
            )
        np.testing.assert_allclose(
            result, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
        )


def compute_second_order(H, order):
    """
    Compute the second order correction to the Hamiltonian.
    
    H: BlockSeries of the Hamiltonian
    order: tuple of orders to compute

    Returns:
    BlockSeries of the second order correction obtained explicitly
    """
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
    H_tilde = general(H)[0]
    n_infinite = H.n_infinite

    for order in permutations((0,) * (n_infinite - 1) + (2,)):
        result = H_tilde.evaluated[(0, 0) + order]
        expected = compute_second_order(H, order)
        if zero == result:
            np.testing.assert_allclose(
                0, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
            )
        np.testing.assert_allclose(
            result, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
        )


def test_check_diagonal():
    """Test that offdiagonal H_0_AA is not allowed if solve_sylvester is not provided."""
    with pytest.raises(ValueError):
        H = to_BlockSeries(
            np.array([[1, 1], [1, 1]]),
            np.eye(2),
            {},
            {},
            {},
        )
        general(H)

@pytest.mark.skip(reason="Not working yet")
def test_equivalence_general_expanded(H, wanted_orders):
    """Test that the general and expanded methods give the same results."""
    H_tilde_general, _, _ = general(H)
    H_tilde_expanded, _, _ = expanded(H)
    for order in wanted_orders:
        order = tuple(slice(None, dim_order + 1) for dim_order in order)
        for block in ((0, 0), (1, 1), (0, 1)):
            result_general = H_tilde_general.evaluated[tuple(block + order)]
            result_expanded = H_tilde_expanded.evaluated[tuple(block + order)]
            if zero == result_general:
                assert zero == result_expanded
            else:
                np.testing.assert_allclose(
                    np.array(result_general).real, np.array(result_expanded).real, atol=10**-2, err_msg=f"{order=}"
                )