from itertools import count, permutations
from typing import Any, Callable, Optional

import pytest
import numpy as np
import sympy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from sympy.physics.quantum import Dagger

from lowdin.block_diagonalization import (
    block_diagonalize,
    general,
    expanded,
    numerical,
    solve_sylvester_KPM,
    _default_solve_sylvester,
    hamiltonian_to_BlockSeries,
)
from lowdin.series import BlockSeries, cauchy_dot_product, zero, one
from lowdin.linalg import ComplementProjector


@pytest.fixture(
    scope="module",
    params=[
        (3,),
        (2, 2),
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
    return np.random.randint(1, high=4, size=2)


@pytest.fixture(scope="module")
def H(Ns: np.array, wanted_orders: list[tuple[int, ...]]) -> BlockSeries:
    """
    Produce random Hamiltonians to test.

    Parameters
    ----------
    Ns: dimension of each block (A, B)
    wanted_orders: orders to compute

    Returns
    -------
    BlockSeries of the Hamiltonian
    """
    n_infinite = len(wanted_orders)
    orders = np.eye(n_infinite, dtype=int)
    h_0_AA = np.diag(np.sort(np.random.rand(Ns[0])) - 1)
    h_0_BB = np.diag(np.sort(np.random.rand(Ns[1])))


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

    hams = []
    for i, j, hermitian in zip([0, 1, 0], [0, 1, 1], [True, True, False]):
        matrices = matrices_it(Ns[i], Ns[j], hermitian)
        hams.append({tuple(order): matrix for order, matrix in zip(orders, matrices)})

    h_p_AA, h_p_BB, h_p_AB = hams
    zeroth_order = (0,) * n_infinite
    H = BlockSeries(
        data={
            **{(0, 0) + zeroth_order: h_0_AA},
            **{(1, 1) + zeroth_order: h_0_BB},
            **{(0, 0) + tuple(key): value for key, value in h_p_AA.items()},
            **{(0, 1) + tuple(key): value for key, value in h_p_AB.items()},
            **{(1, 0) + tuple(key): Dagger(value) for key, value in h_p_AB.items()},
            **{(1, 1) + tuple(key): value for key, value in h_p_BB.items()},
        },
        shape=(2, 2),
        n_infinite=n_infinite,
    )
    return H


def compare_series(
    series1: BlockSeries,
    series2: BlockSeries,
    wanted_orders: tuple[int, ...],
    atol: Optional[float] = 1e-15,
    rtol: Optional[float] = 0,
) -> None:
    """
    Function that compares two BlockSeries with each other

    Two series are compared for a given list of wanted orders in all orders.
    The test first checks for `~lowdin.series.one` objects since these are
    not masked by the resulting masked arrays. For numeric types, numpy
    arrays, `scipy.sparse.linalg.LinearOperator` types, and scipy.sparse.sp_Matrix,
    the evaluated object is converted to a dense array by multiplying with dense
    identity and numrically compared up to the desired tolerance.

    Parameters:
    --------------
    series1:
        First `~lowdin.series.BlockSeries` to compare
    series2:
        Second `~lowdin.series.BlockSeries` to compare
    wanted_orders:
        Tuple of wanted_orders to check the series for
    atol:
        Optional absolute tolerance for numeric comparison
    rtol:
        Optional relative tolerance for numeric comparison
    """
    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    all_elements = (slice(None),) * len(series1.shape)
    results = [
        np.ma.ndenumerate(series.evaluated[all_elements + order])
        for series in (series1, series2)
    ]
    for (order1, value1), (order2, value2) in zip(*results):
        assert order1 == order2

        if isinstance(value1, type(one)) or isinstance(value2, type(one)):
            assert value1 == value2
            continue
        # Convert all numeric types to dense arrays
        np.testing.assert_allclose(
            value1 @ np.identity(value1.shape[1]),
            value2 @ np.identity(value2.shape[1]),
            atol=atol,
            rtol=rtol,
            err_msg=f"{order1=} {order2=}",
        )


@pytest.fixture(scope="module")
def general_output(H: BlockSeries) -> BlockSeries:
    """
    Return the transformed Hamiltonian.

    Parameters
    ----------
    H: Hamiltonian

    Returns
    -------
    transformed Hamiltonian
    """
    return general(H)


def test_check_AB(
        general_output: BlockSeries,
        wanted_orders: tuple[int, ...]
    ) -> None:
    """
    Test that H_AB is zero for a random Hamiltonian.

    Parameters
    ----------
    H: Hamiltonian
    wanted_orders: orders to compute
    """
    H_tilde = general_output[0]
    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    for block in H_tilde.evaluated[(0, 1) + order].compressed():
        if isinstance(block, np.ndarray):
            np.testing.assert_allclose(
                block, 0, atol=10**-5, err_msg=f"{block=}, {order=}"
            )
        else:
            np.testing.assert_allclose(
                block.toarray(), 0, atol=10**-5, err_msg=f"{block=}, {order=}"
            )


def test_check_unitary(
        general_output: BlockSeries,
        wanted_orders: tuple[int, ...]
    ) -> None:
    """
    Test that the transformation is unitary.

    Parameters
    ----------
    H: Hamiltonian
    wanted_orders: orders to compute
    """
    zero_order = (0,) * len(wanted_orders)
    H_tilde, U, U_adjoint = general_output

    N_A = H_tilde.evaluated[(0, 0) + zero_order].shape[0]
    N_B = H_tilde.evaluated[(1, 1) + zero_order].shape[0]
    n_infinite = H_tilde.n_infinite

    identity = BlockSeries(
        data = {(0, 0) + zero_order: np.eye(N_A), (1, 1) + zero_order: np.eye(N_B)},
        shape=(2, 2),
        n_infinite=n_infinite,
    )
    transformed = cauchy_dot_product(U_adjoint, identity, U, hermitian=True)

    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    for block in ((0, 0), (1, 1), (0, 1)):
        result = transformed.evaluated[tuple(block + order)]
        for index, block in np.ma.ndenumerate(result):
            if not any(index):
                # Zeroth order is not zero.
                continue
        if isinstance(block, np.ndarray):
            np.testing.assert_allclose(
                block, 0, atol=10**-5, err_msg=f"{block=}, {order=}"
            )
        else:
            np.testing.assert_allclose(
                block.toarray(), 0, atol=10**-5, err_msg=f"{block=}, {order=}"
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
    H: BlockSeries, wanted_orders: tuple[int, ...]
) -> None:
    """
    Test that the first order is computed correctly.

    Parameters
    ----------
    H : Hamiltonian
    wanted_orders: orders to compute
    """
    H_tilde = general(H)[0]
    Np = len(wanted_orders)
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
    H: BlockSeries, wanted_orders: tuple[int, ...]
) -> None:
    """Test that the second order is computed correctly.

    Parameters
    ----------
    H : Hamiltonian
    wanted_orders: orders to compute
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
        H = BlockSeries(
                data={(0, 0, 0): np.array([[1, 1], [1, 1]]), (1, 1, 0): np.eye(2)},
                shape=(2, 2),
                n_infinite=1,
        )
        general(H)


def test_check_diagonal_H_0_BB() -> None:
    """Test that offdiagonal H_0_BB requires solve_sylvester."""
    with pytest.raises(ValueError):
        H = BlockSeries(
                data={(0, 0, 0): np.eye(2), (1, 1, 0): np.array([[1, 1], [1, 1]])},
                shape=(2, 2),
                n_infinite=1,
        )
        general(H)


def test_equivalence_general_expanded(
    H: BlockSeries, wanted_orders: tuple[int, ...]
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
    for block in ((0, 0), (1, 1), (0, 1)):
        for op_general, op_expanded in zip(
            (H_tilde_general, U_general), (H_tilde_expanded, U_expanded)
        ):
            result_general = op_general.evaluated[block + wanted_orders]
            result_expanded = op_expanded.evaluated[block + wanted_orders]
            if zero == result_general:
                assert zero == result_expanded
            elif zero == result_expanded:
                np.testing.assert_allclose(
                    0, result_general, atol=10**-5, err_msg=f"{wanted_orders=}"
                )
            else:
                np.testing.assert_allclose(
                    result_general,
                    result_expanded,
                    atol=10**-5,
                    err_msg=f"{wanted_orders=}",
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
    algorithm: Callable, H: BlockSeries, wanted_orders: tuple[int, ...]
) -> None:
    """
    Test that doubling the order of the inputs produces the same results on
    the corresponding doubled orders of the outputs.
    This is a consistency check for the algorithm.

    Parameters
    ----------
    H: BlockSeries of the Hamiltonian
    wanted_orders: orders to compute
    """
    # Get the data directly to avoid defining an extra eval
    data = H._data
    H_doubled = BlockSeries(
        data=double_orders(data), shape=H.shape, n_infinite=H.n_infinite
    )

    H_tilde, U, _ = algorithm(H)
    H_tilde_doubled, U_doubled, _ = algorithm(H_doubled)

    blocks = np.index_exp[:2, :2]
    orders = tuple(slice(None, order + 1, None) for order in wanted_orders)
    doubled_orders = tuple(
        slice(None, 2 * (order + 1), None) for order in wanted_orders
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


@pytest.fixture(scope="module")
def n_dim() -> int:
    """
    Randomly generate integers for size of Hamiltonians

    Returns:
    --------
    n_dim: int
        Dimension of the full Hamiltonian
    """
    n_dim = np.random.randint(low=6, high=12, dtype=int)
    return n_dim


@pytest.fixture(scope="module")
def a_dim(n_dim) -> int:
    """
    Randomly generate size of a subspace

    Parameters:
    --------
    n_dim:
        Dimension of the total system

    Returns:
    -------
    a_dim: int
        Dimension of the A subspace
    """
    a_dim = np.random.randint(low=2, high=n_dim - 3, dtype=int)
    return a_dim


@pytest.fixture(scope="module")
def generate_kpm_hamiltonian(
    n_dim: int, wanted_orders: tuple[int, ...], a_dim: int
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Generate random BlockSeries Hamiltonian in the format required by the numerical
    algorithm (full Hamitonians in the (0,0) block).

    Parameters:
    ----------
    n_dim:
        integer denoting the dimension of the full Hamiltonian.
    n_infinite:
        Number of perturbation terms that are generated.
    a_dim:
        Dimension of the A subspace. This number must be smaller than n_dim.

    Returns:
    --------
    hamiltonian: list
        Unperturbed Hamiltonian and perturbation terms.
    subspaces_vectors: tuple
        Subspaces of the Hamiltonian.
    eigenvalues: tuple
        Eigenvalues of the Hamiltonian.
    """
    n_infinite = len(wanted_orders)

    hamiltonian = []
    h_0 = np.random.randn(n_dim, n_dim) + 1j * np.random.randn(n_dim, n_dim)
    h_0 += Dagger(h_0)

    eigs, vecs = np.linalg.eigh(h_0)
    eigs[:a_dim] -= 10
    h_0 = vecs @ np.diag(eigs) @ Dagger(vecs)
    hamiltonian.append(h_0)
    subspaces_vectors = (vecs[:, :a_dim], vecs[:, a_dim:])
    eigenvalues = (eigs[:a_dim], eigs[a_dim:])

    for _ in range(n_infinite):
        h_p = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
        h_p += Dagger(h_p)
        hamiltonian.append(h_p)

    return hamiltonian, subspaces_vectors, eigenvalues


def test_check_AB_KPM(
    generate_kpm_hamiltonian: tuple[
        list[np.ndarray], np.ndarray, np.ndarray
    ],
    wanted_orders: tuple[int, ...],
) -> None:
    """
    Test that H_AB is zero for a random Hamiltonian using the numerical algorithm.

    Parameters
    ----------
    generate_kpm_hamiltonian:
        Randomly generated Hamiltonian and its eigendecomposition.
    wanted_orders:
        List of orders to compute.
    """
    hamiltonian, subspaces_vectors, eigenvalues = generate_kpm_hamiltonian
    n_dim = hamiltonian[0].shape[0]
    a_dim = subspaces_vectors[0].shape[1]
    b_dim = n_dim - a_dim

    H_tilde_full_b, _, _ = block_diagonalize(
        hamiltonian,
        subspaces_vectors=subspaces_vectors,
        eigenvalues=eigenvalues
    )

    half_subspaces = (subspaces_vectors[0], subspaces_vectors[1][:, : b_dim // 2])
    half_eigenvalues = (eigenvalues[0], eigenvalues[1][: b_dim // 2])
    H_tilde_half_b, _, _ = block_diagonalize(
        hamiltonian,
        subspaces_vectors=half_subspaces,
        eigenvalues=half_eigenvalues,
        kpm_params={"num_moments": 5000}
    )

    kpm_subspaces = (subspaces_vectors[0],)
    kpm_eigenvalues = (eigenvalues[0],)
    H_tilde_kpm, _, _ = block_diagonalize(
        hamiltonian,
        subspaces_vectors=kpm_subspaces,
        eigenvalues=kpm_eigenvalues,
        kpm_params={"num_moments": 10000}
    )

    # full b
    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    for block in H_tilde_full_b.evaluated[(0, 1) + order].compressed():
        np.testing.assert_allclose(
            block, 0, atol=1e-5, err_msg=f"{block=}, {order=}"
        )

    # half b
    for block in H_tilde_half_b.evaluated[(0, 1) + order].compressed():
        np.testing.assert_allclose(
            block, 0, atol=1e-1, err_msg=f"{block=}, {order=}"
        )

    # KPM
    for block in H_tilde_kpm.evaluated[(0, 1) + order].compressed():
        np.testing.assert_allclose(
            block, 0, atol=1e-1, err_msg=f"{block=}, {order=}"
        )


def test_solve_sylvester(
        generate_kpm_hamiltonian: tuple[list[np.ndarray], np.ndarray, np.ndarray],
    ) -> None:
    """
    Test whether the KPM version of solve_sylvester provides approximately
    equivalent results depending on how much of the B subspace is known
    explicitly.

    Parameters:
    ---------
    generate_kpm_hamiltonian:
        Randomly generated Hamiltonian and its eigendecomposition.
    """

    hamiltonian, subspaces_vectors, eigenvalues = generate_kpm_hamiltonian
    h_0 = hamiltonian[0]
    n_dim = h_0.shape[0]
    a_dim = subspaces_vectors[0].shape[1]
    b_dim = n_dim - a_dim

    divide_energies_full_b = solve_sylvester_KPM(h_0, subspaces_vectors, eigenvalues)

    half_subspaces = (subspaces_vectors[0], subspaces_vectors[1][:, : b_dim // 2])
    half_eigenvalues = (eigenvalues[0], eigenvalues[1][: b_dim // 2])
    divide_energies_half_b = solve_sylvester_KPM(
        h_0, half_subspaces, half_eigenvalues, kpm_params={"num_moments": 10000},
    )

    kpm_subspaces = (subspaces_vectors[0],)
    kpm_eigenvalues = (eigenvalues[0],)
    divide_energies_kpm = solve_sylvester_KPM(
        h_0, kpm_subspaces, kpm_eigenvalues, kpm_params={"num_moments": 20000}
    )

    y_trial = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
    y_trial += Dagger(y_trial)
    y_trial = (
        Dagger(subspaces_vectors[0]) @ y_trial @ ComplementProjector(subspaces_vectors[0])
    )

    y_full_b = np.abs(divide_energies_full_b(y_trial))
    y_half_b = np.abs(divide_energies_half_b(y_trial))
    y_kpm = np.abs(divide_energies_kpm(y_trial))

    np.testing.assert_allclose(
        y_full_b,
        y_half_b,
        atol=1e-2,
        err_msg="fail in full/half at max val {}".format(np.max(y_full_b - y_half_b)),
    )
    np.testing.assert_allclose(
        y_full_b,
        y_kpm,
        atol=1e-2,
        err_msg="fail in full/kpm at max val {}".format(np.max(y_full_b - y_kpm)),
    )


@pytest.mark.skip(reason="Sometimes it fails due to precision.")
def test_check_AA_numerical(
    generate_kpm_hamiltonian: tuple[
        BlockSeries, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    wanted_orders: tuple[int, ...],
    a_dim: int,
) -> None:
    """
    TODO: Update test to new UI.
    Test that the numerical and general algorithms coincide.

    Parameters
    ----------
    generate_kpm_hamiltonian:
        Randomly generated Hamiltnonian and its eigendecomposition.
    wanted_orders:
        list of orders to compute.
    a_dim:
        Dimension of the A subspace.
    """
    H_input, vecs_a, eigs_a, vecs_b, eigs_b = generate_kpm_hamiltonian
    n_infinite = H_input.n_infinite

    # construct Hamiltonian for general
    index_rows = np.eye(n_infinite, dtype=int)
    vecs = np.concatenate((vecs_a, vecs_b), axis=-1)
    h_0_aa = np.diag(eigs_a)
    h_0_bb = np.diag(eigs_b)
    h_p_aa = {
        tuple(index_rows[index, :]): (
            Dagger(vecs) @ H_input.evaluated[tuple(index_rows[index, :])] @ vecs
        )[:a_dim, :a_dim]
        for index in range(n_infinite)
    }
    h_p_bb = {
        tuple(index_rows[index, :]): (
            Dagger(vecs) @ H_input.evaluated[tuple(index_rows[index, :])] @ vecs
        )[a_dim:, a_dim:]
        for index in range(n_infinite)
    }
    h_p_ab = {
        tuple(index_rows[index, :]): (
            Dagger(vecs) @ H_input.evaluated[tuple(index_rows[index, :])] @ vecs
        )[:a_dim, a_dim:]
        for index in range(n_infinite)
    }

    H_general = BlockSeries(
        data = {
            (0, 0) + (0,) * n_infinite: h_0_aa,
            (1, 1) + (0,) * n_infinite: h_0_bb,
            **{(0, 0) + tuple(key): value for key, value in h_p_aa.items()},
            **{(1, 1) + tuple(key): value for key, value in h_p_bb.items()},
            **{(0, 1) + tuple(key): value for key, value in h_p_ab.items()},
            **{(1, 0) + tuple(key): Dagger(value) for key, value in h_p_ab.items()},
        },
        shape = (2, 2),
        n_infinite = n_infinite,
    )

    H_tilde_general = general(H_general)[0]
    H_tilde_full_b = numerical(H_input, vecs_a, eigs_a, vecs_b, eigs_b)[0]
    H_tilde_KPM = numerical(H_input, vecs_a, eigs_a, kpm_params={"num_moments": 5000})[
        0
    ]
    order = (0, 0) + tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    for block_full_b, block_general, block_KPM in zip(
        H_tilde_full_b.evaluated[order].compressed(),
        H_tilde_general.evaluated[order].compressed(),
        H_tilde_KPM.evaluated[order].compressed(),
    ):
        np.testing.assert_allclose(
            block_full_b, block_general, atol=1e-4, err_msg=f"{order=}"
        )

        np.testing.assert_allclose(
            block_full_b, block_KPM, atol=1e-4, err_msg=f"{order=}"
        )


def test_solve_sylvester_kpm_vs_default(n_dim: int, a_dim: int)-> None:
    """
    Test whether the KPM ready solve_sylvester gives the same result
    as _default_solve_sylvester when prompted with a diagonal input.

    Paramaters:
    ---------
    n_dim:
        Total size of the Hamiltonian.
    a_dim:
        Size of the A subspace.
    """
    h_0 = np.diag(np.sort(50 * np.random.random(n_dim)))
    eigs, vecs = np.linalg.eigh(h_0)

    subspaces_vectors = [vecs[:, :a_dim], vecs[:, a_dim:]]
    eigenvalues = [eigs[:a_dim], eigs[a_dim:]]

    solve_sylvester_default = _default_solve_sylvester(eigenvalues)
    solve_sylvester_kpm = solve_sylvester_KPM(h_0, subspaces_vectors, eigenvalues)

    y_trial = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
    y_kpm = (
        Dagger(subspaces_vectors[0]) @ y_trial @ ComplementProjector(subspaces_vectors[0])
    )

    y_default = solve_sylvester_default(y_trial[:a_dim, a_dim:])
    y_kpm = solve_sylvester_kpm(y_kpm)

    np.testing.assert_allclose(
        y_default,
        y_kpm[:a_dim, a_dim:],
        atol=1e-2,
        err_msg="fail in full/half at max val {}".format(
            np.max(y_default - y_kpm[:a_dim, a_dim:])
        ),
    )


def test_correct_implicit_subspace(
    generate_kpm_hamiltonian: tuple[list[np.ndarray], np.ndarray, np.ndarray],
    wanted_orders: tuple[int, ...],
) -> None:
    """
    Testing agreement of explicit and implicit subspaces_vectors

    Test that the BB block of H_tilde is a) a LinearOperator type and
    b) the same as the AA block on exchanging veca_a and vecs_b

    Parameters:
    ----------
    generate_kpm_hamiltonian:
        Randomly generated Hamiltonian and its eigendeomposition.
    wanted_orders:
        list of orders to compute.
    """
    hamiltonian, subspaces_vectors, eigenvalues = generate_kpm_hamiltonian

    H_tilde, _, _ = block_diagonalize(
        hamiltonian,
        subspaces_vectors=subspaces_vectors,
        eigenvalues=eigenvalues
    )
    H_tilde_swapped, _, _ = block_diagonalize(
        hamiltonian,
        subspaces_vectors=subspaces_vectors[::-1],
        eigenvalues=eigenvalues[::-1]
    )

    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    h = H_tilde.evaluated[(0, 0) + order].compressed()
    h_swapped = H_tilde_swapped.evaluated[(1, 1) + order].compressed()
    for block_aa, block_bb in zip(h, h_swapped):
        assert isinstance(block_bb, LinearOperator)
        np.testing.assert_allclose(
            block_aa,
            Dagger(subspaces_vectors[0]) @ block_bb @ subspaces_vectors[0],
            atol=1e-14
        )


def test_repeated_application(
    H: BlockSeries, wanted_orders: tuple[int, ...]
) -> None:
    """
    Test ensuring invariance of the result upon repeated application

    Tests if the unitary transform returns identity when the algorithm is applied twice

    Parameters:
    -----------
    H:
        Hamiltonian
    wanted_orders:
        list of wanted orders
    """
    H_tilde_1, U_1, U_adjoint_1 = expanded(H)
    H_tilde_2, U_2, U_adjoint_2 = expanded(H_tilde_1)

    zero_index = (0,) * H_tilde_1.n_infinite
    U_target = BlockSeries(
        data={(i, i, *zero_index): one for i in range(H_tilde_1.shape[0])},
        shape=H_tilde_1.shape,
        n_infinite=H_tilde_1.n_infinite,
    )
    compare_series(H_tilde_2, H_tilde_1, wanted_orders, atol=1e-10)
    compare_series(U_2, U_target, wanted_orders)


def test_input_hamiltonian_KPM(generate_kpm_hamiltonian):
    """
    Test that KPM Hamiltonians are interpreted correctly.

    TODO: Improve test
    """
    hamiltonian, subspaces_vectors, _ = generate_kpm_hamiltonian
    H = hamiltonian_to_BlockSeries(hamiltonian, subspaces_vectors=subspaces_vectors)
    assert H.shape == (2, 2)
    for block in ((0, 1), (1, 0)):
        assert zero == H.evaluated[block + (0,) * H.n_infinite]


def test_input_hamiltonian_BlockSeries(H):
    """
    Test that BlockSeries Hamiltonians are interpreted correctly.
    """
    # List input for diagonal H_0
    hamiltonian = hamiltonian_to_BlockSeries(H)
    assert hamiltonian.shape == H.shape
    assert hamiltonian.n_infinite == H.n_infinite
    for block in ((0, 0), (1, 1), (0, 1), (1, 0)):
        index = block + (0,) * H.n_infinite
        if zero == H.evaluated[index]:
            assert zero == hamiltonian.evaluated[index]
            continue
        np.allclose(H.evaluated[index], hamiltonian.evaluated[index])


@pytest.fixture(scope="module", params=[0, 1, 2, 3])
def diagonal_hamiltonians(request):
    subspaces_indices = [0, 1, 1, 0]
    eigenvalues = [-1, 1, 1, -1]
    perturbation = np.random.random(4)
    perturbation += Dagger(perturbation)
    hamiltonians = [
        [np.diag(eigenvalues), perturbation],
        {(0, 0): np.diag(eigenvalues), (1, 0): perturbation, (0, 1): perturbation},
        [sparse.diags(eigenvalues), perturbation, perturbation],
        {(0,): sparse.diags(eigenvalues), (1,): perturbation}
    ]
    diagonals = [np.array([-1, -1]), np.array([1, 1])]
    return hamiltonians[request.param], subspaces_indices, diagonals


def test_input_hamiltonian_diagonal_indices(diagonal_hamiltonians):
    """
    Test inputs where the unperturbed Hamiltonian is diagonal.

    We test the following inputs:
    - list of numpy arrays
    - dictionary of numpy arrays
    - list of sparse matrices
    - dictionary of sparse matrices
    """
    hamiltonian, subspaces_indices, diagonals = diagonal_hamiltonians
    H = hamiltonian_to_BlockSeries(hamiltonian, subspaces_indices=subspaces_indices)
    assert H.shape == (2, 2)
    assert H.n_infinite == len(hamiltonian) - 1
    for block, eigvals in zip(((0, 0), (1, 1)), diagonals):
        index = block + (0,) * H.n_infinite
        np.testing.assert_allclose(H.evaluated[index].diagonal(), eigvals)
    for block in ((0, 1), (1, 0)):
        zero == H.evaluated[block + (0,) * H.n_infinite]
    with pytest.raises(ValueError):
        H = hamiltonian_to_BlockSeries(hamiltonian)
        H.evaluated[(0, 0) + (0,) * H.n_infinite]


def test_input_hamiltonian_from_subspaces():
    """
    Test that the algorithm works with a Hamiltonian defined on subspaces_vectors.
    The test now does not test the perturbation.
    """
    h_0 = np.random.random((4, 4))
    h_0 += Dagger(h_0)
    perturbation = 0.1 * np.random.random((4, 4))
    perturbation += Dagger(perturbation)

    eigenvalues, eigvecs = np.linalg.eigh(h_0)
    diagonals = [eigenvalues[:2], eigenvalues[2:]]
    subspaces_vectors = [eigvecs[:, :2], eigvecs[:, 2:]]

    hamiltonians = [
        [h_0, perturbation],
        {(0, 0): h_0, (1, 0): perturbation, (0, 1): perturbation},
    ]

    for hamiltonian in hamiltonians:
        H = hamiltonian_to_BlockSeries(hamiltonian, subspaces_vectors=subspaces_vectors)
        assert H.shape == (2, 2)
        assert H.n_infinite == len(hamiltonian) - 1
        for block, eigvals in zip(((0, 0), (1, 1)), diagonals):
            index = block + (0,) * H.n_infinite
            np.testing.assert_allclose(H.evaluated[index].diagonal(), eigvals)
        for block in ((0, 1), (1, 0)):
            index = block + (0,) * H.n_infinite
            zero == H.evaluated[index]


def test_input_hamiltonian_blocks():
    """
    Test inputs that come separated by subspace.

    We test the following inputs:
    - list of Hamiltonians where each Hamiltonian is a list of blocks.
    - dictionary of Hamiltonians where each Hamiltonian is a list of blocks.
    """
    hermitian_block = np.random.random((2, 2))
    block = np.random.random((2, 2))
    block += Dagger(block)
    h_0 = [
        [np.diag([-1, -1]), np.zeros((2, 2))],
        [np.zeros((2, 2)), np.diag([1, 1])]
    ]
    perturbation = [
        [-block, hermitian_block],
        [Dagger(hermitian_block), block]
    ]

    hamiltonians = [
        [h_0, perturbation],
        {(0, 0): h_0, (1, 0): perturbation, (0, 1): perturbation},
    ]

    for hamiltonian in hamiltonians:
        H = hamiltonian_to_BlockSeries(hamiltonian)
        assert H.shape == (2, 2)
        assert H.n_infinite == len(hamiltonian) - 1
        np.testing.assert_allclose(
            H.evaluated[(0, 0) + (0,) * H.n_infinite].diagonal(),
            np.array([-1, -1])
        )
        np.testing.assert_allclose(
            H.evaluated[(1, 1) + (0,) * H.n_infinite].diagonal(),
            np.array([1, 1])
        )
        assert zero == H.evaluated[(0, 1) + (0,) * H.n_infinite]
        assert zero == H.evaluated[(1, 0) + (0,) * H.n_infinite]


def test_input_hamiltonian_symbolic():
    """
    Test that the algorithm works with a symbolic Hamiltonian.
    """
    k_x, k_y, k_z, alpha, beta, h, m = sympy.symbols(
        "k_x k_y k_z alpha beta h m", real=True, positive=True, constant=True
    )
    kinetic_term = sum(h**2 * k**2 / (2*m) for k in (k_x, k_y, k_z))
    h_00 = -beta + alpha * k_z + kinetic_term
    h_11 = beta - alpha * k_z + kinetic_term
    h_01 = alpha * k_x - sympy.I * alpha * k_y
    h_10 = alpha * k_x + sympy.I * alpha * k_y
    hamiltonian = sympy.Matrix(
        [
            [h_00, h_01],
            [h_10, h_11]
        ]
    )

    subspaces_vectors = [sympy.Matrix([1, 0]), sympy.Matrix([0, 1])]
    subspaces_indices = [0, 1]

    symbols = [k_x, k_y, k_z]
    # Test if subspaces_vectors are provided
    H_1 = hamiltonian_to_BlockSeries(
        hamiltonian,
        subspaces_vectors=subspaces_vectors,
        symbols=symbols
    )
    # Test if subspaces_indices are provided
    H_2 = hamiltonian_to_BlockSeries(
        hamiltonian,
        subspaces_indices=subspaces_indices,
        symbols=symbols
    )
    assert H_1.shape == H_2.shape == (2, 2)
    assert H_1.n_infinite == H_2.n_infinite == len(symbols)

    for H in (H_1, H_2):
        for block in ((0, 1), (1, 0)):
            assert zero == H.evaluated[block + (0,) * H.n_infinite]


@pytest.mark.xfail(reason="solve_sylvester won't be determined with current code")
def test_block_diagonalize(
    diagonal_hamiltonians: tuple | list,
    wanted_orders : tuple[int, ...]
):
    """
    Test that `block_diagonalize` chooses the right algorithm and the
    solve_sylvester function.

    Parameters
    ----------
    diagonal_hamiltonians :
        Hamiltonians to test.
    wanted_orders : list
        List of orders to test.
    """

    hamiltonian, subspaces_indices, diagonals = diagonal_hamiltonians
    H_tilde, _, _ = block_diagonalize(hamiltonian, subspaces_indices=subspaces_indices)
    test_check_AB(H_tilde, wanted_orders)
    test_check_unitary(H_tilde, wanted_orders)