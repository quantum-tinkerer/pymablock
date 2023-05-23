from itertools import count, permutations
from typing import Any, Callable, Optional

import pytest
import numpy as np
import sympy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from sympy.physics.quantum import Dagger

from pymablock.block_diagonalization import (
    block_diagonalize,
    general,
    expanded,
    implicit,
    solve_sylvester_KPM,
    solve_sylvester_direct,
    solve_sylvester_diagonal,
    hamiltonian_to_BlockSeries,
)
from pymablock.series import BlockSeries, cauchy_dot_product, zero, one
from pymablock.linalg import ComplementProjector


@pytest.fixture(scope="module", params=[(3,), (2, 2)])
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
    n_a = np.random.randint(1, 3)
    return n_a, n_a + 1


@pytest.fixture(scope="module")
def H(Ns: np.array, wanted_orders: list[tuple[int, ...]]) -> BlockSeries:
    """
    Produce random Hamiltonians to test.

    Parameters
    ----------
    Ns:
        Dimension of each block (A, B)
    wanted_orders:
        Orders to compute

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


@pytest.fixture(scope="module", params=[0, 1])
def generate_kpm_hamiltonian(
    Ns: tuple[int, int], wanted_orders: tuple[int, ...], request: Any
) -> tuple[list | dict, np.ndarray, np.ndarray]:
    """
    Generate random BlockSeries Hamiltonian in the format required by the implicit
    algorithm (full Hamitonians in the (0,0) block).

    Parameters:
    ----------
    Ns:
        Dimensions of the subspaces.
    wanted_orders:
        Orders to compute.
    request:
        Pytest request object.

    Returns:
    --------
    hamiltonian: list | dict
        Unperturbed Hamiltonian and perturbation terms.
    subspace_eigenvectors: tuple
        Subspaces of the Hamiltonian.
    """
    a_dim, b_dim = Ns
    n_dim = a_dim + b_dim
    n_infinite = len(wanted_orders)

    hamiltonian_list = []
    hamiltonian_dict = {}
    h_0 = np.random.randn(n_dim, n_dim) + 1j * np.random.randn(n_dim, n_dim)
    h_0 += Dagger(h_0)

    eigs, vecs = np.linalg.eigh(h_0)
    eigs[:a_dim] -= 10
    h_0 = vecs @ np.diag(eigs) @ Dagger(vecs)
    hamiltonian_list.append(h_0)
    hamiltonian_dict[(0,) * n_infinite] = h_0
    subspace_eigenvectors = (vecs[:, :a_dim], vecs[:, a_dim:])

    for i in range(n_infinite):
        h_p = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
        h_p += Dagger(h_p)
        hamiltonian_list.append(h_p)
        order = tuple(np.eye(n_infinite, dtype=int)[i])
        hamiltonian_dict[order] = h_p

    hamiltonians = [hamiltonian_list, hamiltonian_dict]
    return hamiltonians[request.param], subspace_eigenvectors


@pytest.fixture(scope="module", params=[0, 1, 2, 3])
def diagonal_hamiltonian(wanted_orders, request):
    """

    Parameters:
    -----------
    wanted_orders:
        Orders to compute
    request:
        pytest request object
    """
    subspace_indices = [0, 1, 1, 0]
    eigenvalues = [-1, 1, 1, -1]

    n_infinite = len(wanted_orders)

    h_list = [np.diag(eigenvalues)]
    h_list_all_sparse = [sparse.diags(eigenvalues)]
    h_dict = {(0,) * n_infinite: np.diag(eigenvalues)}
    h_dict_all_sparse = {(0,) * n_infinite: sparse.diags(eigenvalues)}
    for i in range(n_infinite):
        sparse_perturbation = 0.1 * sparse.random(4, 4, density=0.2)
        sparse_perturbation += Dagger(sparse_perturbation)
        perturbation = sparse_perturbation.toarray()

        h_list.append(perturbation)
        h_list_all_sparse.append(sparse_perturbation)
        order = tuple(np.eye(n_infinite, dtype=int)[i])
        h_dict[order] = perturbation
        h_dict_all_sparse[order] = sparse.csr_array(sparse_perturbation)

    hamiltonians = [
        h_list,
        h_dict,
        h_list_all_sparse,
        h_dict_all_sparse,
    ]
    diagonals = [np.array([-1, -1]), np.array([1, 1])]
    return hamiltonians[request.param], subspace_indices, diagonals


@pytest.fixture(params=[0, 1, 2])
def symbolic_hamiltonian(request):
    """
    Return a symbolic Hamiltonian in the form of a sympy.Matrix.
    """
    # Symbolic Hamiltonian in sympy.Matrix
    k_x, k_y, k_z, alpha, beta, h, m = sympy.symbols(
        "k_x k_y k_z alpha beta h m", real=True, positive=True, constant=True
    )
    kinetic_term = sum(h**2 * k**2 / (2 * m) for k in (k_x, k_y, k_z))
    h_00 = -beta + alpha * k_z + kinetic_term
    h_11 = beta - alpha * k_z + kinetic_term
    h_01 = alpha * k_x - sympy.I * alpha * k_y
    h_10 = alpha * k_x + sympy.I * alpha * k_y
    hamiltonian_1 = sympy.Matrix([[h_00, h_01], [h_10, h_11]])

    # Symbolic Hamiltonian in dictionary of sympy.Matrix
    sigma_x = sympy.Matrix([[0, 1], [1, 0]])
    sigma_y = sympy.Matrix([[0, -1j], [1j, 0]])
    sigma_z = sympy.Matrix([[1, 0], [0, -1]])
    # Also test that having zero terms in the Hamiltonian is fine by having a
    # zero block of the 0th order term.
    hamiltonian_2 = {
        sympy.Rational(1): beta * sympy.Matrix([[0, 0], [0, 1]]),
        k_x**2: h**2 * sympy.Identity(2) / (2 * m),
        k_y**2: h**2 * sympy.Identity(2) / (2 * m),
        k_z**2: h**2 * sympy.Identity(2) / (2 * m),
        k_z: alpha * sigma_z,
        k_x: alpha * sigma_x,
        k_y: alpha * sigma_y,
    }

    # Symbolic Hamiltonian in dictionary of numpy.ndarray
    m = h = alpha = beta = 1  # values must be numeric, TODO: test symbolic
    hamiltonian_3 = {
        k_x**2: h**2 * np.eye(2) / (2 * m),
        k_y**2: h**2 * np.eye(2) / (2 * m),
        k_z**2: h**2 * np.eye(2) / (2 * m),
        sympy.Rational(1): beta * np.diag([-1, 1]),
        k_z: alpha * np.diag([1, -1]),
        k_x: alpha * np.array([[0, 1], [1, 0]]),
        k_y: alpha * np.array([[0, -1j], [1j, 0]]),
    }

    hamiltonians = [hamiltonian_1, hamiltonian_2, hamiltonian_3]
    symbols = tuple([k_x, k_y, k_z])
    subspace_eigenvectors = [sympy.Matrix([1, 0]), sympy.Matrix([0, 1])]
    subspace_indices = [0, 1]
    return hamiltonians[request.param], symbols, subspace_eigenvectors, subspace_indices


def test_input_hamiltonian_diagonal_indices(diagonal_hamiltonian):
    """
    Test inputs where the unperturbed Hamiltonian is diagonal.

    Parameters:
    -----------
    diagonal_hamiltonian:
        Hamiltonian with diagonal unperturbed Hamiltonian
    """
    hamiltonian, subspace_indices, diagonals = diagonal_hamiltonian
    H = hamiltonian_to_BlockSeries(hamiltonian, subspace_indices=subspace_indices)
    assert H.shape == (2, 2)
    assert H.n_infinite == len(hamiltonian) - 1
    assert H.dimension_names == ()
    for block, eigvals in zip(((0, 0), (1, 1)), diagonals):
        index = block + (0,) * H.n_infinite
        np.testing.assert_allclose(H[index].diagonal(), eigvals)
    for block in ((0, 1), (1, 0)):
        zero == H[block + (0,) * H.n_infinite]
    with pytest.raises(ValueError):
        H = hamiltonian_to_BlockSeries(hamiltonian)
        H[(0, 0) + (0,) * H.n_infinite]


def test_input_hamiltonian_from_subspaces():
    """
    Test that the algorithm works with a Hamiltonian defined on `subspace_eigenvectors`.
    The test now does not test the perturbation.
    """
    h_0 = np.random.random((4, 4))
    h_0 += Dagger(h_0)
    perturbation = 0.1 * np.random.random((4, 4))
    perturbation += Dagger(perturbation)

    eigenvalues, eigvecs = np.linalg.eigh(h_0)
    diagonals = [eigenvalues[:2], eigenvalues[2:]]
    subspace_eigenvectors = [eigvecs[:, :2], eigvecs[:, 2:]]

    hamiltonians = [
        [h_0, perturbation],
        {(0, 0): h_0, (1, 0): perturbation, (0, 1): perturbation},
    ]

    for hamiltonian in hamiltonians:
        H = hamiltonian_to_BlockSeries(
            hamiltonian, subspace_eigenvectors=subspace_eigenvectors
        )
        assert H.shape == (2, 2)
        assert H.n_infinite == len(hamiltonian) - 1
        assert H.dimension_names == ()
        for block, eigvals in zip(((0, 0), (1, 1)), diagonals):
            index = block + (0,) * H.n_infinite
            np.testing.assert_allclose(H[index].diagonal(), eigvals)
        for block in ((0, 1), (1, 0)):
            index = block + (0,) * H.n_infinite
            zero == H[index]


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
    h_0 = [[np.diag([-1, -1]), np.zeros((2, 2))], [np.zeros((2, 2)), np.diag([1, 1])]]
    perturbation = [[-block, hermitian_block], [Dagger(hermitian_block), block]]

    hamiltonians = [
        [h_0, perturbation],
        {(0, 0): h_0, (1, 0): perturbation, (0, 1): perturbation},
    ]

    for hamiltonian in hamiltonians:
        H = hamiltonian_to_BlockSeries(hamiltonian)
        assert H.shape == (2, 2)
        assert H.n_infinite == len(hamiltonian) - 1
        assert H.dimension_names == ()
        np.testing.assert_allclose(
            H[(0, 0) + (0,) * H.n_infinite].diagonal(), np.array([-1, -1])
        )
        np.testing.assert_allclose(
            H[(1, 1) + (0,) * H.n_infinite].diagonal(), np.array([1, 1])
        )
        assert zero == H[(0, 1) + (0,) * H.n_infinite]
        assert zero == H[(1, 0) + (0,) * H.n_infinite]


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
    The test first checks for `~pymablock.series.one` objects since these are
    not masked by the resulting masked arrays. For numeric types, numpy
    arrays, `scipy.sparse.linalg.LinearOperator` types, and scipy.sparse.sp_Matrix,
    the evaluated object is converted to a dense array by multiplying with dense
    identity and numrically compared up to the desired tolerance.

    Parameters:
    --------------
    series1:
        First `~pymablock.series.BlockSeries` to compare
    series2:
        Second `~pymablock.series.BlockSeries` to compare
    wanted_orders:
        Order until which to compare the series
    atol:
        Optional absolute tolerance for numeric comparison
    rtol:
        Optional relative tolerance for numeric comparison
    """
    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    all_elements = (slice(None),) * len(series1.shape)
    results = [
        np.ma.ndenumerate(series[all_elements + order]) for series in (series1, series2)
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


def test_check_AB(general_output: BlockSeries, wanted_orders: tuple[int, ...]) -> None:
    """
    Test that H_AB is zero for a random Hamiltonian.

    Parameters
    ----------
    general_output:
        transformed Hamiltonian to test
    wanted_orders:
        orders to compute
    """
    H_tilde = general_output[0]
    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    for matrix in H_tilde[(0, 1) + order].compressed():
        if isinstance(matrix, np.ndarray):
            np.testing.assert_allclose(
                matrix, 0, atol=10**-5, err_msg=f"{matrix=}, {order=}"
            )
        elif sparse.issparse(matrix):
            np.testing.assert_allclose(
                matrix.toarray(), 0, atol=10**-5, err_msg=f"{matrix=}, {order=}"
            )
        elif isinstance(matrix, sympy.MatrixBase):
            assert matrix.is_zero_matrix
        else:
            raise TypeError(f"Unknown type {type(matrix)}")


def test_check_unitary(
    general_output: tuple[BlockSeries, BlockSeries, BlockSeries],
    wanted_orders: tuple[int, ...],
    N_A: Optional[int] = None,
    N_B: Optional[int] = None,
) -> None:
    """
    Test that the transformation is unitary.

    Parameters
    ----------
    H:
        Hamiltonian
    wanted_orders:
        orders to compute
    """
    zero_order = (0,) * len(wanted_orders)
    H_tilde, U, U_adjoint = general_output

    N_A = N_A or H_tilde[(0, 0) + zero_order].shape[0]
    N_B = N_B or H_tilde[(1, 1) + zero_order].shape[0]
    n_infinite = H_tilde.n_infinite

    identity = BlockSeries(
        data={(0, 0) + zero_order: np.eye(N_A), (1, 1) + zero_order: np.eye(N_B)},
        shape=(2, 2),
        n_infinite=n_infinite,
    )
    transformed = cauchy_dot_product(U_adjoint, identity, U, hermitian=True)

    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    for block in ((0, 0), (1, 1), (0, 1)):
        result = transformed[tuple(block + order)]
        for index, matrix in np.ma.ndenumerate(result):
            if not any(index):
                # Zeroth order is not zero.
                continue
            if isinstance(matrix, np.ndarray):
                np.testing.assert_allclose(
                    matrix, 0, atol=10**-5, err_msg=f"{matrix=}, {order=}"
                )
            elif sparse.issparse(matrix):
                np.testing.assert_allclose(
                    matrix.toarray(), 0, atol=10**-5, err_msg=f"{matrix=}, {order=}"
                )
            elif isinstance(matrix, sympy.MatrixBase):
                assert matrix.is_zero_matrix
            else:
                raise TypeError(f"Unknown type {type(matrix)}")


def compute_first_order(H: BlockSeries, order: tuple[int, ...]) -> Any:
    """
    Compute the first order correction to the Hamiltonian.

    Parameters
    ----------
    H:
        Hamiltonian
    order:
        tuple of orders to compute

    Returns
    -------
    First order correction obtained explicitly
    """
    return H[(0, 0) + order]


def test_first_order_H_tilde(H: BlockSeries, wanted_orders: tuple[int, ...]) -> None:
    """
    Test that the first order is computed correctly.

    Parameters
    ----------
    H :
        Hamiltonian
    wanted_orders:
        orders to compute
    """
    H_tilde = general(H)[0]
    Np = len(wanted_orders)
    for order in permutations((0,) * (Np - 1) + (1,)):
        result = H_tilde[(0, 0) + order]
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
    H:
        Hamiltonian
    order:
        Orders to compute

    Returns
    -------
    BlockSeries of the second order correction obtained explicitly
    """
    n_infinite = H.n_infinite
    order = tuple(value // 2 for value in order)
    h_0_AA, h_0_BB, h_p_AB = (
        H[(0, 0) + (0,) * n_infinite],
        H[(1, 1) + (0,) * n_infinite],
        H[(0, 1) + order],
    )

    eigs_A = np.diag(h_0_AA)
    eigs_B = np.diag(h_0_BB)
    energy_denominators = 1 / (eigs_A.reshape(-1, 1) - eigs_B)
    V1 = -h_p_AB * energy_denominators
    return -(V1 @ Dagger(h_p_AB) + h_p_AB @ Dagger(V1)) / 2


def test_second_order_H_tilde(H: BlockSeries, wanted_orders: tuple[int, ...]) -> None:
    """Test that the second order is computed correctly.

    Parameters
    ----------
    H :
        Hamiltonian
    wanted_orders:
        Orders to compute
    """
    H_tilde = general(H)[0]
    n_infinite = H.n_infinite

    for order in permutations((0,) * (n_infinite - 1) + (2,)):
        result = H_tilde[(0, 0) + order]
        expected = compute_second_order(H, order)
        if zero == result:
            np.testing.assert_allclose(
                0, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
            )
        np.testing.assert_allclose(
            result, expected, atol=10**-5, err_msg=f"{result=}, {expected=}"
        )


def test_check_diagonal_h_0_A() -> None:
    """Test that offdiagonal h_0_AA requires solve_sylvester."""
    with pytest.raises(ValueError):
        H = BlockSeries(
            data={(0, 0, 0): np.array([[1, 1], [1, 1]]), (1, 1, 0): np.eye(2)},
            shape=(2, 2),
            n_infinite=1,
        )
        general(H)


def test_check_diagonal_h_0_B() -> None:
    """Test that offdiagonal h_0_BB requires solve_sylvester."""
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
    H:
        Hamiltonian
    wanted_orders:
        Orders to compute
    """
    H_tilde_general, U_general, _ = general(H)
    H_tilde_expanded, U_expanded, _ = expanded(H)
    for block in ((0, 0), (1, 1), (0, 1)):
        for op_general, op_expanded in zip(
            (H_tilde_general, U_general), (H_tilde_expanded, U_expanded)
        ):
            result_general = op_general[block + wanted_orders]
            result_expanded = op_expanded[block + wanted_orders]
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
    data:
        dictionary of the form {(block, order): value} with BlockSeries data

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
    H:
        Hamiltonian
    wanted_orders:
        Orders to compute
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
        result = op[blocks + orders].compressed()
        result_doubled = op_doubled[blocks + doubled_orders].compressed()
        assert len(result) == len(result_doubled)
        for result, result_doubled in zip(result, result_doubled):
            if isinstance(result, object):
                assert isinstance(result_doubled, object)
                continue
            np.testing.assert_allclose(result, result_doubled, atol=10**-5)


def test_check_AB_KPM(
    Ns: tuple[int, int],
    generate_kpm_hamiltonian: tuple[list[np.ndarray], np.ndarray, np.ndarray],
    wanted_orders: tuple[int, ...],
) -> None:
    """
    Test that H_AB is zero for a random Hamiltonian using the implicit algorithm.

    Parameters
    ----------
    Ns:
        Dimensions of the Hamiltonian.
    generate_kpm_hamiltonian:
        Randomly generated Hamiltonian and its eigendecomposition.
    wanted_orders:
        List of orders to compute.
    """
    _, b_dim = Ns
    hamiltonian, subspace_eigenvectors = generate_kpm_hamiltonian

    almost_eigenvectors = (subspace_eigenvectors[0], subspace_eigenvectors[1][:, :-1])
    H_tilde_full_b, _, _ = block_diagonalize(
        hamiltonian,
        subspace_eigenvectors=almost_eigenvectors,
        direct_solver=False,
        solver_options={"num_moments": 5000},
        atol=1e-8,
    )

    half_eigenvectors = (
        subspace_eigenvectors[0],
        subspace_eigenvectors[1][:, : b_dim // 2],
    )
    H_tilde_half_b, _, _ = block_diagonalize(
        hamiltonian,
        subspace_eigenvectors=half_eigenvectors,
        direct_solver=False,
        solver_options={"num_moments": 5000},
        atol=1e-8,
    )

    kpm_subspaces = (subspace_eigenvectors[0],)
    H_tilde_kpm, _, _ = block_diagonalize(
        hamiltonian,
        subspace_eigenvectors=kpm_subspaces,
        direct_solver=False,
        solver_options={"num_moments": 10000},
        atol=1e-8,
    )

    # # full b
    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    for block in H_tilde_full_b[(0, 1) + order].compressed():
        np.testing.assert_allclose(block, 0, atol=1e-6, err_msg=f"{block=}, {order=}")

    # half b
    for block in H_tilde_half_b[(0, 1) + order].compressed():
        np.testing.assert_allclose(block, 0, atol=1e-6, err_msg=f"{block=}, {order=}")

    # KPM
    for block in H_tilde_kpm[(0, 1) + order].compressed():
        np.testing.assert_allclose(block, 0, atol=1e-6, err_msg=f"{block=}, {order=}")


def test_solve_sylvester(
    Ns: tuple[int, int],
    wanted_orders: tuple[int, ...],
    generate_kpm_hamiltonian: tuple[list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """
    Test whether the KPM version of solve_sylvester provides approximately
    equivalent results depending on how much of the B subspace is known
    explicitly.

    Parameters:
    ---------
    Ns:
        Dimensions of the Hamiltonian.
    wanted_orders:
        Orders to compute.
    generate_kpm_hamiltonian:
        Randomly generated Hamiltonian and its eigendecomposition.
    """
    a_dim, b_dim = Ns
    n_dim = a_dim + b_dim
    hamiltonian, subspace_eigenvectors = generate_kpm_hamiltonian

    if isinstance(hamiltonian, list):
        h_0 = hamiltonian[0]
    elif isinstance(hamiltonian, dict):
        h_0 = hamiltonian[(0,) * len(wanted_orders)]

    divide_energies_full_b = solve_sylvester_KPM(h_0, subspace_eigenvectors)

    half_subspaces = (
        subspace_eigenvectors[0],
        subspace_eigenvectors[1][:, : b_dim // 2],
    )
    divide_energies_half_b = solve_sylvester_KPM(
        h_0,
        half_subspaces,
        solver_options={"num_moments": 10000},
    )

    kpm_subspaces = (subspace_eigenvectors[0],)
    divide_energies_kpm = solve_sylvester_KPM(
        h_0, kpm_subspaces, solver_options={"num_moments": 20000}
    )

    y_trial = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
    y_trial += Dagger(y_trial)
    y_trial = (
        Dagger(subspace_eigenvectors[0])
        @ y_trial
        @ ComplementProjector(subspace_eigenvectors[0])
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


@pytest.mark.xfail(reason="Sometimes it fails due to precision.")
def test_implicit_consistent_on_A(
    generate_kpm_hamiltonian: tuple[
        BlockSeries, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    wanted_orders: tuple[int, ...],
    a_dim: int,
) -> None:
    """
    TODO: Update test to new UI.
    Test that the implicit and general algorithms coincide.

    Parameters
    ----------
    generate_kpm_hamiltonian:
        Randomly generated Hamiltnonian and its eigendecomposition.
    wanted_orders:
        list of orders to compute.
    a_dim:
        Dimension of the A subspace.
    """
    H_input, vecs_A, eigs_A, vecs_B, eigs_B = generate_kpm_hamiltonian
    n_infinite = H_input.n_infinite

    # construct Hamiltonian for general
    index_rows = np.eye(n_infinite, dtype=int)
    vecs = np.concatenate((vecs_A, vecs_B), axis=-1)
    h_0_AA = np.diag(eigs_A)
    h_0_BB = np.diag(eigs_B)
    h_p_AA = {
        tuple(index_rows[index, :]): (
            Dagger(vecs) @ H_input[tuple(index_rows[index, :])] @ vecs
        )[:a_dim, :a_dim]
        for index in range(n_infinite)
    }
    h_p_BB = {
        tuple(index_rows[index, :]): (
            Dagger(vecs) @ H_input[tuple(index_rows[index, :])] @ vecs
        )[a_dim:, a_dim:]
        for index in range(n_infinite)
    }
    h_p_AB = {
        tuple(index_rows[index, :]): (
            Dagger(vecs) @ H_input[tuple(index_rows[index, :])] @ vecs
        )[:a_dim, a_dim:]
        for index in range(n_infinite)
    }

    H_general = BlockSeries(
        data={
            (0, 0) + (0,) * n_infinite: h_0_AA,
            (1, 1) + (0,) * n_infinite: h_0_BB,
            **{(0, 0) + tuple(key): value for key, value in h_p_AA.items()},
            **{(1, 1) + tuple(key): value for key, value in h_p_BB.items()},
            **{(0, 1) + tuple(key): value for key, value in h_p_AB.items()},
            **{(1, 0) + tuple(key): Dagger(value) for key, value in h_p_AB.items()},
        },
        shape=(2, 2),
        n_infinite=n_infinite,
    )

    H_tilde_general = general(H_general)[0]
    H_tilde_full_b = implicit(H_input, vecs_A, eigs_A, vecs_B, eigs_B)[0]
    H_tilde_KPM = implicit(
        H_input, vecs_A, eigs_A, solver_options={"num_moments": 5000}
    )[0]
    order = (0, 0) + tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    for block_full_b, block_general, block_KPM in zip(
        H_tilde_full_b[order].compressed(),
        H_tilde_general[order].compressed(),
        H_tilde_KPM[order].compressed(),
    ):
        np.testing.assert_allclose(
            block_full_b, block_general, atol=1e-4, err_msg=f"{order=}"
        )

        np.testing.assert_allclose(
            block_full_b, block_KPM, atol=1e-4, err_msg=f"{order=}"
        )


def test_solve_sylvester_kpm_vs_diagonal(Ns: tuple[int, int]) -> None:
    """
    Test whether the KPM ready solve_sylvester gives the same result
    as solve_sylvester_diagonal when prompted with a diagonal input.

    Paramaters:
    ---------
    Ns:
        tuple of dimensions of the two subspaces.
    """
    a_dim, b_dim = Ns
    n_dim = a_dim + b_dim
    h_0 = np.diag(np.sort(50 * np.random.random(n_dim)))
    eigs, vecs = np.linalg.eigh(h_0)

    subspace_eigenvectors = [vecs[:, :a_dim], vecs[:, a_dim:]]
    eigenvalues = [eigs[:a_dim], eigs[a_dim:]]

    solve_sylvester_default = solve_sylvester_diagonal(*eigenvalues)
    solve_sylvester_kpm = solve_sylvester_KPM(h_0, subspace_eigenvectors)

    y_trial = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
    y_kpm = (
        Dagger(subspace_eigenvectors[0])
        @ y_trial
        @ ComplementProjector(subspace_eigenvectors[0])
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


def test_solve_sylvester_direct_vs_diagonal() -> None:
    n = 300
    a_dim = 5
    E = np.random.randn(n)
    t = np.random.rand(n - 1) * np.exp(2j * np.pi * np.random.rand(n - 1))
    h = sparse.diags([t, E, t.conj()], [-1, 0, 1])
    eigvals, eigvecs = np.linalg.eigh(h.toarray())
    eigvecs, eigvecs_rest = eigvecs[:, :a_dim], eigvecs[:, a_dim:]

    diagonal = solve_sylvester_diagonal(eigvals[:a_dim], eigvals[a_dim:], eigvecs_rest)
    direct = solve_sylvester_direct(h, eigvecs)

    y = np.random.randn(a_dim, n - a_dim) + 1j * np.random.randn(a_dim, n - a_dim)
    y = y @ Dagger(eigvecs_rest)

    y_default = diagonal(y)
    y_direct = direct(y)

    np.testing.assert_allclose(y_default, y_direct)


def test_consistent_implicit_subspace(
    generate_kpm_hamiltonian: tuple[list[np.ndarray], np.ndarray, np.ndarray],
    wanted_orders: tuple[int, ...],
) -> None:
    """
    Testing agreement of explicit and implicit subspace_eigenvectors

    Test that the BB block of H_tilde is a) a LinearOperator type and
    b) the same as the A block on exchanging vecs_A and vecs_B

    Parameters:
    ----------
    generate_kpm_hamiltonian:
        Randomly generated Hamiltonian and its eigendeomposition.
    wanted_orders:
        Orders to compute.
    """
    hamiltonian, subspace_eigenvectors = generate_kpm_hamiltonian

    # Catch error if the dimension of the eigenvectors do not match the Hamiltonian
    with pytest.raises(ValueError):
        faulted_eigenvectors = (
            subspace_eigenvectors[0][:1, :],
            subspace_eigenvectors[1],
        )
        H_tilde, _, _ = block_diagonalize(
            hamiltonian, subspace_eigenvectors=faulted_eigenvectors, direct_solver=False
        )

    almost_eigenvectors = (subspace_eigenvectors[0], subspace_eigenvectors[1][:, :-1])
    H_tilde, _, _ = block_diagonalize(
        hamiltonian,
        subspace_eigenvectors=almost_eigenvectors,
        direct_solver=False,
        solver_options={"num_moments": 10000},
        atol=1e-8,
    )
    reversed_eigenvectors = (subspace_eigenvectors[1], subspace_eigenvectors[0][:, :-1])
    H_tilde_swapped, _, _ = block_diagonalize(
        hamiltonian,
        subspace_eigenvectors=reversed_eigenvectors,
        direct_solver=False,
        solver_options={"num_moments": 10000},
        atol=1e-8,
    )

    assert H_tilde.shape == H_tilde_swapped.shape
    assert H_tilde.n_infinite == H_tilde_swapped.n_infinite
    assert H_tilde.dimension_names == H_tilde_swapped.dimension_names

    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    h = H_tilde[(0, 0) + order].compressed()
    h_swapped = H_tilde_swapped[(1, 1) + order].compressed()
    for block_A, block_B in zip(h, h_swapped):
        assert isinstance(block_B, LinearOperator)
        np.testing.assert_allclose(
            block_A,
            Dagger(subspace_eigenvectors[0]) @ block_B @ subspace_eigenvectors[0],
            atol=1e-8,
        )


def test_repeated_application(H: BlockSeries, wanted_orders: tuple[int, ...]) -> None:
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


def test_input_hamiltonian_kpm(generate_kpm_hamiltonian):
    """
    Test that KPM Hamiltonians are interpreted correctly.

    Parameters:
    -----------
    generate_kpm_hamiltonian:
        Randomly generated Hamiltonian and its eigendeomposition.
    """
    hamiltonian, subspace_eigenvectors = generate_kpm_hamiltonian
    H = hamiltonian_to_BlockSeries(
        hamiltonian, subspace_eigenvectors=subspace_eigenvectors, implicit=True
    )
    assert H.shape == (2, 2)
    assert H.n_infinite == len(hamiltonian) - 1
    assert H.dimension_names == ()
    for block in ((0, 1), (1, 0)):
        index = block + (0,) * H.n_infinite
        if zero == H[index]:
            continue
        np.testing.assert_allclose(H[index], 0, atol=1e-12)
    assert isinstance(H[(1, 1) + (0,) * H.n_infinite], LinearOperator)


def test_input_hamiltonian_BlockSeries(H):
    """
    Test that BlockSeries Hamiltonians are interpreted correctly.

    Parameters:
    -----------
    H:
        Hamiltonian
    """
    # List input for diagonal H_0
    hamiltonian = hamiltonian_to_BlockSeries(H)
    assert hamiltonian.shape == H.shape
    assert hamiltonian.n_infinite == H.n_infinite
    assert hamiltonian.dimension_names == H.dimension_names
    for block in ((0, 0), (1, 1), (0, 1), (1, 0)):
        index = block + (0,) * H.n_infinite
        if zero == H[index]:
            assert zero == hamiltonian[index]
            continue
        np.allclose(H[index], hamiltonian[index])


def test_input_hamiltonian_symbolic(symbolic_hamiltonian):
    """
    Test that the algorithm works with a symbolic Hamiltonian.

    Parameters
    ----------
    symbolic_hamiltonian :
        Sympy Matrix or dictionary of sympy expressions.
    """
    hamiltonian, symbols, subspace_eigenvectors, subspace_indices = symbolic_hamiltonian

    # Test if subspace_eigenvectors are provided
    H_1 = hamiltonian_to_BlockSeries(
        hamiltonian, subspace_eigenvectors=subspace_eigenvectors, symbols=symbols
    )
    # Test if subspace_indices are provided
    H_2 = hamiltonian_to_BlockSeries(
        hamiltonian, subspace_indices=subspace_indices, symbols=symbols
    )
    assert H_1.shape == H_2.shape == (2, 2)
    assert H_1.n_infinite == H_2.n_infinite == len(symbols)
    assert H_1.dimension_names == H_2.dimension_names == symbols

    for H in (H_1, H_2):
        for block in ((0, 1), (1, 0)):
            assert zero == H[block + (0,) * H.n_infinite]


def test_block_diagonalize_hamiltonian_diagonal(
    diagonal_hamiltonian: tuple | list, wanted_orders: tuple[int, ...]
):
    """
    Test that `block_diagonalize` chooses the right algorithm and the
    `solve_sylvester` function.

    Parameters
    ----------
    diagonal_hamiltonian :
        Hamiltonian
    wanted_orders :
        orders to compute
    """
    hamiltonian, subspace_indices, diagonals = diagonal_hamiltonian
    H_tilde, U, U_adjoint = block_diagonalize(
        hamiltonian, subspace_indices=subspace_indices
    )

    assert H_tilde.shape == (2, 2)
    assert H_tilde.n_infinite == len(hamiltonian) - 1 == len(wanted_orders)
    assert H_tilde.dimension_names == ()
    test_check_AB([H_tilde, U, U_adjoint], wanted_orders)
    test_check_unitary([H_tilde, U, U_adjoint], wanted_orders)


def test_block_diagonalize_hamiltonian_symbolic(
    symbolic_hamiltonian: tuple[sympy.Matrix | dict, list[sympy.Symbol]],
):
    """
    Test that `block_diagonalize` chooses the right algorithm and the
    `solve_sylvester` function.

    Parameters
    ----------
    symbolic_hamiltonian :
        Hamiltonian
    wanted_orders :
        orders to compute
    """
    # Test if subspace_eigenvectors are provided
    hamiltonian, symbols, subspace_eigenvectors, subspace_indices = symbolic_hamiltonian
    H_tilde, U, U_adjoint = block_diagonalize(
        hamiltonian, symbols=symbols, subspace_eigenvectors=subspace_eigenvectors
    )
    assert H_tilde.shape == (2, 2)
    assert H_tilde.n_infinite == len(symbols)
    assert H_tilde.dimension_names == symbols
    test_check_AB([H_tilde, U, U_adjoint], (1,) * len(symbols))
    test_check_unitary([H_tilde, U, U_adjoint], (1,) * len(symbols), 1, 1)

    # Test if subspace_indices are provided
    hamiltonian, symbols, subspace_eigenvectors, subspace_indices = symbolic_hamiltonian
    H_tilde, U, U_adjoint = block_diagonalize(
        hamiltonian, symbols=symbols, subspace_indices=subspace_indices
    )
    assert H_tilde.shape == (2, 2)
    assert H_tilde.n_infinite == len(symbols)
    assert H_tilde.dimension_names == symbols
    test_check_AB([H_tilde, U, U_adjoint], (1,) * len(symbols))
    test_check_unitary([H_tilde, U, U_adjoint], (1,) * len(symbols), 1, 1)

    # Raise if eigenvectors are not orthonormal
    with pytest.raises(ValueError):
        faulted_eigenvectors = [2 * evecs for evecs in subspace_eigenvectors]
        block_diagonalize(hamiltonian, subspace_eigenvectors=faulted_eigenvectors)


def test_unknown_data_type():
    """
    Test that the algorithm raises an error if the data type is unknown.
    """

    class Unknown:
        """Black box class with a minimal algebraic interface."""

        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return self.name

        __str__ = __repr__

        def __mul__(self, other):
            return Unknown(f"{self.name} * {other}")

        def __rmul__(self, other):
            return Unknown(f"{other} * {self.name}")

        def __add__(self, other):
            return Unknown(f"{self.name} + {other}")

        def adjoint(self):
            return Unknown(f"{self.name}^*")

        def __neg__(self):
            return Unknown(f"-{self.name}")

    H = [
        [[Unknown("a"), zero], [zero, Unknown("b")]],
        [[Unknown("c"), Unknown("d")], [Unknown("e"), Unknown("f")]],
    ]
    H_tilde, *_ = block_diagonalize(H, solve_sylvester=lambda x: x)
    H_tilde[:, :, :3]
    # Shouldn't work without the solve_sylvester function
    with pytest.raises(NotImplementedError):
        H_tilde, *_ = block_diagonalize(H)


def test_zero_h_0():
    """
    Test that the algorithm works if the first block is zero.
    """
    h_0 = np.diag([0.0, 0.0, 1.0, 1.0])
    h_p = np.random.randn(4, 4)
    h_p += h_p.T
    h = [h_0, h_p]
    block_diagonalize(h, subspace_indices=[0, 0, 1, 1])[0][:, :, 3]
    h_sparse = [sparse.csr_array(term) for term in h]
    block_diagonalize(h_sparse, subspace_indices=[0, 0, 1, 1])[0][:, :, 3]
    h_sympy = [sympy.Matrix(term) for term in h]
    block_diagonalize(h_sympy, subspace_indices=[0, 0, 1, 1])[0][:, :, 3]
    h_blocked = [
        [[term[:2, :2], term[2:, :2]], [term[:2, :2], term[2:, 2:]]] for term in h
    ]
    block_diagonalize(h_blocked)[0][:, :, 3]


def test_single_symbol_input():
    a = sympy.Symbol("a")
    H = sympy.Matrix([[a, 0], [0, sympy.Symbol("b")]])
    H_tilde = block_diagonalize(H, symbols=a, subspace_indices=[0, 1])[0]
    assert H_tilde.shape == (2, 2)
    assert H_tilde.n_infinite == 1
    assert H_tilde.dimension_names == [a]
    # Check that execution doesn't raise
    H_tilde[:, :, 3]
