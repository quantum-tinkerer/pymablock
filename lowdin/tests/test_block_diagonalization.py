from itertools import count, permutations
from typing import Any, Callable

import numpy as np
import pytest
from sympy.physics.quantum import Dagger

from lowdin.block_diagonalization import general, expanded, to_BlockSeries
from lowdin.series import BlockSeries, cauchy_dot_product, zero, one


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
def H(Ns: np.array, wanted_orders: list[tuple[int, ...]]) -> BlockSeries:
    """
    Produce random Hamiltonians to test.

    Parameters
    ----------
    Ns: dimension of each block (A, B)
    wanted_orders: list of orders to compute

    Returns
    -------
    BlockSeries of the Hamiltonian
    """
    n_infinite = len(wanted_orders[0])
    orders = np.array(np.eye(n_infinite, dtype=int))
    hams = []
    for i in range(2):
        hams.append(np.diag(np.sort(np.random.rand(Ns[i])) - i))

    def matrices_it(N_i, N_j, hermitian):
        """
        Generate random matrices of size N_i x N_j.

        Parameters
        ----------
        N_i: number of rows
        N_j: number of columns
        hermitian: if True, the matrix is hermitian

        Returns
        -------
        generator of random matrices
        """
        for i in count():
            H = np.random.rand(N_i, N_j) + 1j * np.random.rand(N_i, N_j)
            if hermitian:
                H += H.conj().T
            yield H

    for i, j, hermitian in zip([0, 1, 0], [0, 1, 1], [True, True, False]):
        matrices = matrices_it(Ns[i], Ns[j], hermitian)
        hams.append({tuple(order): matrix for order, matrix in zip(orders, matrices)})

    return to_BlockSeries(*hams, n_infinite)


def test_check_AB(H: BlockSeries, wanted_orders: list[tuple[int, ...]]) -> None:
    """
    Test that H_AB is zero for a random Hamiltonian.

    Parameters
    ----------
    H: Hamiltonian
    wanted_orders: list of orders to compute
    """
    H_tilde = general(H)[0]
    for order in wanted_orders:
        order = tuple(slice(None, dim_order + 1) for dim_order in order)
        for block in H_tilde.evaluated[(0, 1) + order].compressed():
            np.testing.assert_allclose(
                block, 0, atol=10**-5, err_msg=f"{block=}, {order=}"
            )


def test_check_unitary(H: BlockSeries, wanted_orders: list[tuple[int, ...]]) -> None:
    """
    Test that the transformation is unitary.

    Parameters
    ----------
    H: Hamiltonian
    wanted_orders: list of orders to compute
    """
    zero_order = (0,) * len(wanted_orders[0])
    N_A = H.evaluated[(0, 0) + zero_order].shape[0]
    N_B = H.evaluated[(1, 1) + zero_order].shape[0]
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


def compute_first_order(H: BlockSeries, order: tuple[int, ...]) -> Any:
    """
    Compute the first order correction to the Hamiltonian.

    Parameters
    ----------
    H: Hamiltonian
    order: tuple of orders to compute

    Returns
    -------
    First order correction obtained explicitly
    """
    return H.evaluated[(0, 0) + order]


def test_first_order_H_tilde(
    H: BlockSeries, wanted_orders: list[tuple[int, ...]]
) -> None:
    """
    Test that the first order is computed correctly.

    Parameters
    ----------
    H : Hamiltonian
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


def compute_second_order(H: BlockSeries, order: tuple[int, ...]) -> Any:
    """
    Compute the second order correction to the Hamiltonian.

    Parameters
    ----------
    H: Hamiltonian
    order: tuple of orders to compute

    Returns
    -------
    BlockSeries of the second order correction obtained explicitly
    """
    n_infinite = H.n_infinite
    order = tuple(value // 2 for value in order)
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


def test_second_order_H_tilde(
    H: BlockSeries, wanted_orders: list[tuple[int, ...]]
) -> None:
    """Test that the second order is computed correctly.

    Parameters
    ----------
    H : Hamiltonian
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


def test_check_diagonal_H_0_AA() -> None:
    """Test that offdiagonal H_0_AA requires solve_sylvester."""
    with pytest.raises(ValueError):
        H = to_BlockSeries(
            np.array([[1, 1], [1, 1]]),
            np.eye(2),
            {},
            {},
            {},
        )
        general(H)


def test_check_diagonal_H_0_BB() -> None:
    """Test that offdiagonal H_0_BB requires solve_sylvester."""
    with pytest.raises(ValueError):
        H = to_BlockSeries(
            np.eye(2),
            np.array([[1, 1], [1, 1]]),
            {},
            {},
            {},
        )
        general(H)


def test_equivalence_general_expanded(
    H: BlockSeries, wanted_orders: list[tuple[int, ...]]
) -> None:
    """
    Test that the general and expanded methods give the same results.

    Parameters
    ----------
    H: BlockSeries of the Hamiltonian
    wanted_orders: list of orders to compute
    """
    H_tilde_general, U_general, _ = general(H)
    H_tilde_expanded, U_expanded, _ = expanded(H)
    for order in wanted_orders:
        for block in ((0, 0), (1, 1), (0, 1)):
            for op_general, op_expanded in zip(
                (H_tilde_general, U_general), (H_tilde_expanded, U_expanded)
            ):
                result_general = op_general.evaluated[block + order]
                result_expanded = op_expanded.evaluated[block + order]
                if zero == result_general:
                    assert zero == result_expanded
                elif zero == result_expanded:
                    np.testing.assert_allclose(
                        0, result_general, atol=10**-5, err_msg=f"{order=}"
                    )
                else:
                    np.testing.assert_allclose(
                        result_general,
                        result_expanded,
                        atol=10**-5,
                        err_msg=f"{order=}",
                    )


def double_orders(data: dict[tuple[int, ...], Any]) -> dict[tuple[int, ...], Any]:
    """
    Double the orders of the keys in a dictionary.

    Parameters
    ----------
    data: dictionary of the form {(block, order): value}

    Returns
    -------
    dictionary of the form {(block, 2*order): value}
    """
    new_data = {}
    for index, value in data.items():
        if zero == value:
            continue
        block = index[:2]
        order = tuple(2 * np.array(index[2:]))
        new_data[block + order] = value
    return new_data


@pytest.mark.parametrize("algorithm", [general, expanded])
def test_doubled_orders(
    algorithm: Callable, H: BlockSeries, wanted_orders: list[tuple[int, ...]]
) -> None:
    """
    Test that doubling the order of the inputs produces the same results on
    the corresponding doubled orders of the outputs.
    This is a consistency check for the algorithm.

    Parameters
    ----------
    H: BlockSeries of the Hamiltonian
    wanted_orders: list of orders to compute
    """

    data = H.data
    H_doubled = BlockSeries(
        data=double_orders(data), shape=H.shape, n_infinite=H.n_infinite
    )

    H_tilde, U, _ = algorithm(H)
    H_tilde_doubled, U_doubled, _ = algorithm(H_doubled)

    for wanted_order in wanted_orders:
        blocks = np.index_exp[:2, :2]
        orders = tuple(slice(None, order + 1, None) for order in wanted_order)
        doubled_orders = tuple(
            slice(None, 2 * (order + 1), None) for order in wanted_order
        )

        for op, op_doubled in zip((H_tilde, U), (H_tilde_doubled, U_doubled)):
            result = op.evaluated[blocks + orders].compressed()
            result_doubled = op_doubled.evaluated[blocks + doubled_orders].compressed()
            assert len(result) == len(result_doubled)
            for result, result_doubled in zip(result, result_doubled):
                if isinstance(result, object):
                    assert isinstance(result_doubled, object)
                    continue
                np.testing.assert_allclose(result, result_doubled, atol=10**-5)


def _test_block(block: Any) -> bool:
    if isinstance(block, np.ndarray):
        check_bool = False
        try:
            np.testing.assert_allclose(
                block, np.eye(block.shape), atol=10**-5, err_msg=f"{block=}"
            )
            check_bool = True
        except:
            np.testing.assert_allclose(
                block,
                np.zeros(block.shape),
                atol=10**-5,
                err_msg=f"{block=}",
            )
            check_bool = True
    elif block == one:
        check_bool = True
    elif block == zero:
        check_bool == zero
    return check_bool


def test_repeated_application(
    H: BlockSeries, wanted_orders: list[tuple[int, ...]]
) -> None:
    """
    Test ensuring invariance of the result upon repeated application
    """
    H_tilde_1, U_1, U_adjoint_1 = general(H)
    H_tilde_2, U_2, U_adjoint_2 = general(H_tilde_1)

    for order in wanted_orders:
        order = tuple(slice(None, dim_order + 1) for dim_order in order)
        for block in U_2.evaluated[(0, 0) + order].compressed():
            check_boolean = _test_block(block)
            assert check_boolean == True

        for block in U_2.evaluated[(1, 1) + order].compressed():
            check_boolean = _test_block(block)
            assert check_boolean == True

    H_tilde_1, U_1, U_adjoint_1 = expanded(H)
    H_tilde_2, U_2, U_adjoint_2 = expanded(H_tilde_1)

    for order in wanted_orders:
        order = tuple(slice(None, dim_order + 1) for dim_order in order)
        for block in U_2.evaluated[(0, 0) + order].compressed():
            check_boolean = _test_block(block)
            assert check_boolean == True

        for block in U_2.evaluated[(1, 1) + order].compressed():
            check_boolean = _test_block(block)
            assert check_boolean == True
