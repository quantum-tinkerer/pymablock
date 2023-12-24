from itertools import permutations, chain
from typing import Any, Union, Callable

import pytest
import numpy as np
import sympy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from sympy.physics.quantum import Dagger

from pymablock.block_diagonalization import (
    block_diagonalize,
    general,
    solve_sylvester_KPM,
    solve_sylvester_direct,
    solve_sylvester_diagonal,
    hamiltonian_to_BlockSeries,
)
from pymablock.series import BlockSeries, cauchy_dot_product, zero, one


# Auxiliary comparison functions
def compare_series(
    series1: BlockSeries,
    series2: BlockSeries,
    wanted_orders: tuple[int, ...],
    atol: float = 1e-15,
    rtol: float = 0,
) -> None:
    """
    Function that compares two BlockSeries with each other

    Two series are compared for a given list of wanted orders in all orders.
    The test first checks for `~pymablock.series.one` objects since these are
    not masked by the resulting masked arrays. For numeric types, numpy
    arrays, `scipy.sparse.linalg.LinearOperator` types, and scipy.sparse.sp_Matrix,
    the evaluated object is converted to a dense array by multiplying with dense
    identity and numrically compared up to the desired tolerance.

    Missing values are converted to 0 before comparison.

    Parameters:
    --------------
    series1:
        First `~pymablock.series.BlockSeries` to compare
    series2:
        Second `~pymablock.series.BlockSeries` to compare
    wanted_orders:
        Order until which to compare the series
    atol:
        Absolute tolerance for numeric comparison
    rtol:
        Relative tolerance for numeric comparison
    """
    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    all_elements = (slice(None),) * len(series1.shape)
    results = [
        np.ndenumerate(np.ma.filled(series[all_elements + order], zero))
        for series in (series1, series2)
    ]
    for (order, value1), (_, value2) in zip(*results):
        values = [value1, value2]
        if any(isinstance(value, type(one)) for value in values):
            assert value1 == value2
        elif all(isinstance(value, sympy.MatrixBase) for value in values):
            assert value1 == value2
        elif any(isinstance(value, sympy.MatrixBase) for value in values):
            # The only non-symbolic option is zero
            values.remove(zero)
            assert values[0].is_zero_matrix
        else:
            # Convert all numeric types to dense arrays
            values = [
                value @ np.identity(value.shape[1]) if zero != value else 0
                for value in values
            ]
            np.testing.assert_allclose(
                *values, atol=atol, rtol=rtol, err_msg=f"Series unequal at {order=}"
            )


def is_diagonal(
    series: BlockSeries, wanted_orders: tuple[int, ...], atol=1e-10
) -> None:
    """
    Test that the offdiagonal blocks of a series are zero.

    Parameters
    ----------
    H:
        Hamiltonian to test
    wanted_orders:
        orders to compute
    """
    order = tuple(slice(None, dim_order + 1) for dim_order in wanted_orders)
    for matrix in chain(
        series[(0, 1) + order].compressed(), series[(1, 0) + order].compressed()
    ):
        if isinstance(matrix, np.ndarray):
            np.testing.assert_allclose(
                matrix, 0, atol=atol, err_msg=f"{matrix=}, {order=}"
            )
        elif sparse.issparse(matrix):
            np.testing.assert_allclose(
                matrix.toarray(), 0, atol=atol, err_msg=f"{matrix=}, {order=}"
            )
        elif isinstance(matrix, sympy.MatrixBase):
            assert matrix.is_zero_matrix
        else:
            raise TypeError(f"Unknown type {type(matrix)}")


def identity_like(U: BlockSeries):
    """An identity-like series with the same dimensions as U"""
    return BlockSeries(
        data={(block + U.n_infinite * (0,)): one for block in ((0, 0), (1, 1))},
        shape=U.shape,
        n_infinite=U.n_infinite,
        dimension_names=U.dimension_names,
    )


def is_unitary(
    U: BlockSeries,
    U_dagger: BlockSeries,
    wanted_orders: tuple[int, ...],
    atol: float = 1e-15,
) -> None:
    """
    Test that the transformation is unitary.

    Parameters
    ----------
    U:
        Transformation
    U_dagger:
        Adjoint transformation
    wanted_orders:
        Order until which to compare the series
    atol:
        Absolute tolerance for numeric comparison
    rtol:
        Relative tolerance for numeric comparison
    """
    identity = identity_like(U)
    transformed = cauchy_dot_product(U_dagger, U)
    compare_series(transformed, identity, wanted_orders, atol=atol, rtol=0)


# Fixtures
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
        while True:
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


@pytest.fixture(scope="module", params=[0, 1])
def implicit_problem(
    Ns: tuple[int, int], wanted_orders: tuple[int, ...], request: Any
) -> tuple[Union[list, dict], np.ndarray, np.ndarray]:
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


# Tests
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
    # Default dimension names
    assert H.dimension_names == tuple(f"n_{i}" for i in range(H.n_infinite))
    for block, eigvals in zip(((0, 0), (1, 1)), diagonals):
        index = block + (0,) * H.n_infinite
        np.testing.assert_allclose(H[index].diagonal(), eigvals)
    for block in ((0, 1), (1, 0)):
        assert zero == H[block + (0,) * H.n_infinite]
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
        # Default dimension names
        assert H.dimension_names == tuple(f"n_{i}" for i in range(H.n_infinite))
        for block, eigvals in zip(((0, 0), (1, 1)), diagonals):
            index = block + (0,) * H.n_infinite
            np.testing.assert_allclose(H[index].diagonal(), eigvals)
        for block in ((0, 1), (1, 0)):
            index = block + (0,) * H.n_infinite
            assert zero == H[index]


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
        # Default dimension names
        assert H.dimension_names == tuple(f"n_{i}" for i in range(H.n_infinite))
        np.testing.assert_allclose(
            H[(0, 0) + (0,) * H.n_infinite].diagonal(), np.array([-1, -1])
        )
        np.testing.assert_allclose(
            H[(1, 1) + (0,) * H.n_infinite].diagonal(), np.array([1, 1])
        )
        assert zero == H[(0, 1) + (0,) * H.n_infinite]
        assert zero == H[(1, 0) + (0,) * H.n_infinite]


def test_H_tilde_diagonal(H: BlockSeries, wanted_orders: tuple[int, ...]) -> None:
    """
    Test that H_tilde_AB is zero for a random Hamiltonian.

    Parameters
    ----------
    H:
        Hamiltonian to test
    wanted_orders:
        orders to compute
    """
    is_diagonal(general(H)[0], wanted_orders)


def test_check_unitary(
    H: BlockSeries,
    wanted_orders: tuple[int, ...],
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
    is_unitary(*general(H)[1:], wanted_orders, atol=1e-10)


def test_check_invertible(
    H: BlockSeries,
    wanted_orders: tuple[int, ...],
) -> None:
    """
    Test that the transformation is invertible, so H = U @ H_tilde @ U_dagger.

    Parameters
    ----------
    H:
        Hamiltonian
    wanted_orders:
        orders to compute
    """
    H_tilde, U, U_dagger = general(H)
    H_reconstructed = cauchy_dot_product(U, cauchy_dot_product(H_tilde, U_dagger))
    compare_series(H, H_reconstructed, wanted_orders, atol=1e-10)


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
    H_tilde_1, U_1, U_adjoint_1 = general(H)
    H_tilde_2, U_2, U_adjoint_2 = general(H_tilde_1)

    compare_series(H_tilde_2, H_tilde_1, wanted_orders, atol=1e-10)
    compare_series(U_2, identity_like(U_2), wanted_orders, atol=1e-10)


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
                0, expected, atol=1e-10, err_msg=f"{result=}, {expected=}"
            )
        np.testing.assert_allclose(
            result, expected, atol=1e-10, err_msg=f"{result=}, {expected=}"
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
                0, expected, atol=1e-10, err_msg=f"{result=}, {expected=}"
            )
        np.testing.assert_allclose(
            result, expected, atol=1e-10, err_msg=f"{result=}, {expected=}"
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


def test_doubled_orders(H: BlockSeries, wanted_orders: tuple[int, ...]) -> None:
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

    def doubled_eval(H: BlockSeries) -> Callable:
        def eval(*index):
            element, order = index[:2], index[2:]
            if any(i % 2 for i in order):
                return zero
            return H[element + tuple(i // 2 for i in order)]

        return eval

    H_doubled = BlockSeries(
        eval=doubled_eval(H), shape=H.shape, n_infinite=H.n_infinite
    )

    H_tilde, U, _ = general(H)
    H_tilde_doubled_directly = BlockSeries(
        eval=doubled_eval(H_tilde), shape=H_tilde.shape, n_infinite=H_tilde.n_infinite
    )
    U_doubled_directly = BlockSeries(
        eval=doubled_eval(U), shape=U.shape, n_infinite=U.n_infinite
    )
    H_tilde_doubled, U_doubled, _ = general(H_doubled)

    compare_series(H_tilde_doubled_directly, H_tilde_doubled, wanted_orders)
    compare_series(U_doubled_directly, U_doubled, wanted_orders)


def test_equivalence_explicit_implicit() -> None:
    """
    Test that the explicit and implicit algorithms give the same results.

    Specifically, we check that:
    - The implicit algorithm has BB blocks as linear operators.
    - Using the implicit solve_sylvester combined with an explicit Hamiltonian
      (which is also wrapped in projectors) gives the same result as implicit
      one.
    - That providing full eigenvectors (and therefore using the explicit
      algorithm with standard solve_sylvester_diagonal) gives the same ``H^AA``

    The correctness of the solve_sylvester functions for the implicit problem is
    checked in a different test.
    """
    # KPM is slower and does not guarantee convergence. Therefore we only run
    # this test for the direct solver.
    pytest.importorskip("kwant.linalg.mumps", reason="mumps not installed")
    n = 30
    a_dim = 2

    def random_H(*index):
        h = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        return h + Dagger(h)

    H = BlockSeries(
        eval=random_H,
        shape=(),
        n_infinite=1,
    )
    H_0 = H[0]
    eigvals, eigvecs = np.linalg.eigh(H_0)
    solve_sylvester = solve_sylvester_direct(sparse.coo_matrix(H_0), eigvecs[:, :a_dim])

    implicit_H = hamiltonian_to_BlockSeries(
        H, subspace_eigenvectors=(eigvecs[:, :a_dim],), implicit=True
    )

    def explicit_wrapped_H_eval(*index):
        result = implicit_H[index]
        if index[:2] == (1, 1):
            return result @ np.eye(n)
        return result

    explicit_wrapped_H = BlockSeries(
        eval=explicit_wrapped_H_eval,
        shape=(2, 2),
        n_infinite=H.n_infinite,
    )

    fully_explicit_H = hamiltonian_to_BlockSeries(
        H, subspace_eigenvectors=(eigvecs[:, :a_dim], eigvecs[:, a_dim:])
    )

    implicit_H_tilde, *_ = general(implicit_H, solve_sylvester=solve_sylvester)
    explicit_wrapped_H_tilde, *_ = general(
        explicit_wrapped_H, solve_sylvester=solve_sylvester
    )
    fully_explicit_H_tilde, *_ = general(fully_explicit_H)

    assert all(isinstance(implicit_H_tilde[1, 1, i], LinearOperator) for i in range(2))

    compare_series(implicit_H_tilde, explicit_wrapped_H_tilde, (2,))
    compare_series(
        implicit_H_tilde[0, 0], fully_explicit_H_tilde[0, 0], (2,), atol=1e-10
    )


def test_solve_sylvester_direct_vs_diagonal() -> None:
    """
    Test whether the solve_sylvester_direct gives the result consistent with
    solve_sylvester_diagonal.
    """
    pytest.importorskip("kwant.linalg.mumps", reason="mumps not installed")
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


def test_solve_sylvester_kpm_vs_diagonal() -> None:
    """
    Test whether the solve_sylvester_direct gives the result consistent with
    solve_sylvester_diagonal.

    Same as test_solve_sylvester_direct_vs_diagonal, but with a higher error
    tolerance because the KPM solver is not as accurate. See also
    https://gitlab.kwant-project.org/qt/pymablock/-/issues/38

    We also introduce an energy gap between subspaces to make the convergence of
    the KPM solver faster.
    """
    n = 30
    a_dim = 5
    # Introduce a gap between the two subspaces
    E = np.random.randn(n) - 5 * (np.arange(n) < a_dim)
    t = np.random.rand(n - 1) * np.exp(2j * np.pi * np.random.rand(n - 1))
    h = sparse.diags([t, E, t.conj()], [-1, 0, 1])
    eigvals, eigvecs = np.linalg.eigh(h.toarray())
    eigvecs, eigvecs_partial, eigvecs_rest = (
        eigvecs[:, :a_dim],
        eigvecs[:, a_dim : 3 * a_dim],
        eigvecs[:, a_dim:],
    )

    diagonal = solve_sylvester_diagonal(eigvals[:a_dim], eigvals[a_dim:], eigvecs_rest)
    kpm = solve_sylvester_KPM(h, [eigvecs], {"num_moments": 1000})
    hybrid = solve_sylvester_KPM(h, [eigvecs, eigvecs_partial], {"num_moments": 1000})

    y = np.random.randn(a_dim, n - a_dim) + 1j * np.random.randn(a_dim, n - a_dim)
    y = y @ Dagger(eigvecs_rest)

    y_default = diagonal(y)
    y_kpm = kpm(y)
    y_hybrid = hybrid(y)

    # Use a lower tolerance until KPM estimates error bounds.
    np.testing.assert_allclose(y_default, y_kpm, atol=1e-2)
    np.testing.assert_allclose(y_default, y_hybrid, atol=1e-2)


def test_input_hamiltonian_implicit(implicit_problem):
    """
    Test that KPM Hamiltonians are interpreted correctly.

    Parameters:
    -----------
    implicit_problem:
        Randomly generated Hamiltonian and its eigendecomposition.
    """
    hamiltonian, subspace_eigenvectors = implicit_problem
    H = hamiltonian_to_BlockSeries(
        hamiltonian, subspace_eigenvectors=subspace_eigenvectors, implicit=True
    )
    assert H.shape == (2, 2)
    assert H.n_infinite == len(hamiltonian) - 1
    # Default dimension names
    assert H.dimension_names == tuple(f"n_{i}" for i in range(H.n_infinite))
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
    diagonal_hamiltonian: Union[tuple, list], wanted_orders: tuple[int, ...]
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
    # Default dimension names
    assert H_tilde.dimension_names == tuple(f"n_{i}" for i in range(H_tilde.n_infinite))
    is_diagonal(H_tilde, wanted_orders)
    is_unitary(U, U_adjoint, wanted_orders)


def test_block_diagonalize_hamiltonian_symbolic(
    symbolic_hamiltonian: tuple[Union[sympy.Matrix, dict], list[sympy.Symbol]],
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
    is_diagonal(H_tilde, (1,) * len(symbols))
    is_unitary(U, U_adjoint, (1,) * len(symbols))

    # Test if subspace_indices are provided
    hamiltonian, symbols, subspace_eigenvectors, subspace_indices = symbolic_hamiltonian
    H_tilde, U, U_adjoint = block_diagonalize(
        hamiltonian, symbols=symbols, subspace_indices=subspace_indices
    )
    assert H_tilde.shape == (2, 2)
    assert H_tilde.n_infinite == len(symbols)
    assert H_tilde.dimension_names == symbols
    is_diagonal(H_tilde, (1,) * len(symbols))
    is_unitary(U, U_adjoint, (1,) * len(symbols))

    # Raise if eigenvectors are not orthonormal
    with pytest.raises(ValueError):
        faulted_eigenvectors = [2 * evecs for evecs in subspace_eigenvectors]
        block_diagonalize(hamiltonian, subspace_eigenvectors=faulted_eigenvectors)


def test_unknown_data_type():
    """
    Test that the algorithm works with a class implementing algebra.

    This requires a solve_sylvester function to be provided, and otherwise
    should raise an error.
    """

    class Unknown:
        """Black box class with a minimal algebraic interface."""

        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return self.name

        __str__ = __repr__

        def __mul__(self, other):
            return Unknown(f"({self} * {other})")

        def __rmul__(self, other):
            return Unknown(f"({other} * {self})")

        def __add__(self, other):
            return Unknown(f"({self} + {other})")

        def adjoint(self):
            return Unknown(f"({self}^*)")

        def __neg__(self):
            return Unknown(f"(-{self})")

        def __sub__(self, other):
            return Unknown(f"({self} - {other})")

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

    This is a regression test for a bug that was present in the algorithm.
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
    """
    Test that the algorithm works with a single symbol as input.

    This is a regression test for a bug that was present in the algorithm.
    """
    a = sympy.Symbol("a")
    H = sympy.Matrix([[a, 0], [0, sympy.Symbol("b")]])
    H_tilde = block_diagonalize(H, symbols=a, subspace_indices=[0, 1])[0]
    assert H_tilde.shape == (2, 2)
    assert H_tilde.n_infinite == 1
    assert H_tilde.dimension_names == [a]
    # Check that execution doesn't raise
    H_tilde[:, :, 3]
