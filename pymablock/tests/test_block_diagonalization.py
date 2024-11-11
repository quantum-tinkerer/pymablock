import operator
import tracemalloc
from collections import Counter
from collections.abc import Callable
from itertools import chain, permutations, product
from typing import Any

import numpy as np
import pytest
import sympy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from sympy.physics.quantum import Dagger

from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main
from pymablock.block_diagonalization import (
    _dict_to_BlockSeries,
    block_diagonalize,
    hamiltonian_to_BlockSeries,
    solve_sylvester_diagonal,
    solve_sylvester_direct,
    solve_sylvester_KPM,
)
from pymablock.series import AlgebraElement, BlockSeries, cauchy_dot_product, one, zero


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
    arrays, `scipy.sparse.linalg.LinearOperator` types, and scipy.sparse.sparray,
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
            assert sympy.simplify(value1 - value2).is_zero_matrix
        elif any(isinstance(value, sympy.MatrixBase) for value in values):
            # The only non-symbolic option is zero
            values.remove(zero)
            values = values[0]
            values.simplify()
            assert values.is_zero_matrix
        else:
            # Convert all numeric types to dense arrays
            values = [
                value @ np.identity(value.shape[1]) if value is not zero else 0
                for value in values
            ]
            np.testing.assert_allclose(
                *values, atol=atol, rtol=rtol, err_msg=f"Series unequal at {order=}"
            )


def is_diagonal_series(
    series: BlockSeries, wanted_orders: tuple[int, ...], atol=1e-7
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
        series[(0, 1, *order)].compressed(), series[(1, 0, *order)].compressed()
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
        data={((i, i) + U.n_infinite * (0,)): one for i in range(U.shape[0])},
        shape=U.shape,
        n_infinite=U.n_infinite,
        dimension_names=U.dimension_names,
        name="I",
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
@pytest.fixture(scope="module", params=[(5,), (4, 2)])
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
    N_a = np.random.randint(1, 3)
    return N_a, N_a + 1


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
            **{(0, 0, *zeroth_order): h_0_AA},
            **{(1, 1, *zeroth_order): h_0_BB},
            **{(0, 0, *key): value for key, value in h_p_AA.items()},
            **{(0, 1, *key): value for key, value in h_p_AB.items()},
            **{(1, 0, *key): Dagger(value) for key, value in h_p_AB.items()},
            **{(1, 1, *key): value for key, value in h_p_BB.items()},
        },
        shape=(2, 2),
        n_infinite=n_infinite,
    )
    return H


@pytest.fixture(scope="module", params=[0, 1])
def implicit_problem(
    Ns: tuple[int, int], wanted_orders: tuple[int, ...], request: Any
) -> tuple[list, np.ndarray, np.ndarray]:
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
    hamiltonian: list
        Unperturbed Hamiltonian and perturbation terms.
    subspace_eigenvectors: tuple
        Subspaces of the Hamiltonian.
    """
    a_dim, b_dim = Ns
    n_dim = a_dim + b_dim
    n_infinite = len(wanted_orders)

    hamiltonian_list = []
    hamiltonian_dict = {}
    rng = np.random.default_rng()
    h_0 = rng.standard_normal(size=(n_dim, n_dim)) + 1j * rng.standard_normal(
        size=(n_dim, n_dim)
    )
    h_0 += Dagger(h_0)

    eigs, vecs = np.linalg.eigh(h_0)
    eigs[:a_dim] -= 10
    h_0 = sparse.coo_array(vecs @ np.diag(eigs) @ Dagger(vecs))
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

    hamiltonian = [hamiltonian_1, hamiltonian_2, hamiltonian_3][request.param]
    symbols = tuple([k_x, k_y, k_z])
    subspace_eigenvectors = [sympy.Matrix([1, 0]), sympy.Matrix([0, 1])]
    if isinstance(hamiltonian, dict):
        if not isinstance(hamiltonian[k_x], sympy.Matrix):
            subspace_eigenvectors = [
                np.array(vecs).astype(complex) for vecs in subspace_eigenvectors
            ]
    subspace_indices = [0, 1]
    return hamiltonian, symbols, subspace_eigenvectors, subspace_indices


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
    assert H.name == "H"
    for block, eigvals in zip(((0, 0), (1, 1)), diagonals):
        index = block + (0,) * H.n_infinite
        np.testing.assert_allclose(H[index].diagonal(), eigvals)
    for block in ((0, 1), (1, 0)):
        assert H[block + (0,) * H.n_infinite] is zero

    # Check that without subspace indices we get a single block of the full size.
    H_one_block = hamiltonian_to_BlockSeries(hamiltonian)
    H_0 = H_one_block[(0, 0, *(0,) * H.n_infinite)]
    H_one_block_explicit = hamiltonian_to_BlockSeries(
        hamiltonian, subspace_indices=np.zeros_like(subspace_indices)
    )
    H_0_explicit = H_one_block_explicit[(0, 0, *(0,) * H_one_block_explicit.n_infinite)]
    # Hamiltonians are converted to sparse matrices
    np.testing.assert_equal((H_0 - H_0_explicit).data, 0)


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
        assert H.name == "H"
        for block, eigvals in zip(((0, 0), (1, 1)), diagonals):
            index = block + (0,) * H.n_infinite
            np.testing.assert_allclose(H[index].diagonal(), eigvals)
        for block in ((0, 1), (1, 0)):
            index = block + (0,) * H.n_infinite
            assert H[index] is zero


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
        assert H.name == "H"
        np.testing.assert_allclose(
            H[(0, 0, *(0,) * H.n_infinite)].diagonal(), np.array([-1, -1])
        )
        np.testing.assert_allclose(
            H[(1, 1, *(0,) * H.n_infinite)].diagonal(), np.array([1, 1])
        )
        assert H[(0, 1, *(0,) * H.n_infinite)] is zero
        assert H[(1, 0, *(0,) * H.n_infinite)] is zero


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
    is_diagonal_series(block_diagonalize(H)[0], wanted_orders)


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
    is_unitary(*block_diagonalize(H)[1:], wanted_orders, atol=1e-6)


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
    H_tilde, U, U_dagger = block_diagonalize(H)
    H_reconstructed = cauchy_dot_product(U, cauchy_dot_product(H_tilde, U_dagger))
    compare_series(H, H_reconstructed, wanted_orders, atol=1e-6)


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
    H_tilde_1, *_ = block_diagonalize(H)
    H_tilde_2, U_2, _ = block_diagonalize(H_tilde_1)

    compare_series(H_tilde_2, H_tilde_1, wanted_orders, atol=1e-7)
    compare_series(U_2, identity_like(U_2), wanted_orders, atol=1e-7)


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
    H_tilde = block_diagonalize(H)[0]
    Np = len(wanted_orders)
    for order in permutations((0,) * (Np - 1) + (1,)):
        np.testing.assert_allclose(H_tilde[(0, 0, *order)], H[(0, 0, *order)], atol=1e-8)


def second_order(H: BlockSeries, order: tuple[int, ...]) -> Any:
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
        H[(0, 0, *(0,) * n_infinite)],
        H[(1, 1, *(0,) * n_infinite)],
        H[(0, 1, *order)],
    )

    eigs_A = np.diag(h_0_AA)
    eigs_B = np.diag(h_0_BB)
    energy_denominators = 1 / (eigs_A.reshape(-1, 1) - eigs_B)
    V1 = -h_p_AB * energy_denominators
    return -(V1 @ Dagger(h_p_AB) + h_p_AB @ Dagger(V1)) / 2


def test_second_order_H_tilde(H: BlockSeries) -> None:
    """Test that the second order is computed correctly.

    Parameters
    ----------
    H :
        Hamiltonian
    wanted_orders:
        Orders to compute
    """
    H_tilde = block_diagonalize(H)[0]
    n_infinite = H.n_infinite

    for order in permutations((0,) * (n_infinite - 1) + (2,)):
        np.testing.assert_allclose(
            H_tilde[(0, 0, *order)], second_order(H, order), atol=1e-8
        )


def test_check_diagonal_h_0_A() -> None:
    """Test that offdiagonal h_0_AA requires solve_sylvester."""
    with pytest.warns(
        UserWarning, match="Cannot confirm that the unperturbed Hamiltonian is diagonal"
    ):
        H = BlockSeries(
            data={(0, 0, 0): np.array([[1, 1], [1, 1]]), (1, 1, 0): 3 * np.eye(2)},
            shape=(2, 2),
            n_infinite=1,
        )
        block_diagonalize(H)


def test_check_diagonal_h_0_B() -> None:
    """Test that offdiagonal h_0_BB requires solve_sylvester."""
    with pytest.warns(
        UserWarning, match="Cannot confirm that the unperturbed Hamiltonian is diagonal"
    ):
        H = BlockSeries(
            data={(0, 0, 0): 3 * np.eye(2), (1, 1, 0): np.array([[1, 1], [1, 1]])},
            shape=(2, 2),
            n_infinite=1,
        )
        block_diagonalize(H)


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

    H_doubled = BlockSeries(eval=doubled_eval(H), shape=H.shape, n_infinite=H.n_infinite)

    H_tilde, U, _ = block_diagonalize(H)
    H_tilde_doubled_directly = BlockSeries(
        eval=doubled_eval(H_tilde), shape=H_tilde.shape, n_infinite=H_tilde.n_infinite
    )
    U_doubled_directly = BlockSeries(
        eval=doubled_eval(U), shape=U.shape, n_infinite=U.n_infinite
    )
    H_tilde_doubled, U_doubled, _ = block_diagonalize(H_doubled)

    compare_series(H_tilde_doubled_directly, H_tilde_doubled, wanted_orders)
    compare_series(U_doubled_directly, U_doubled, wanted_orders)


def test_one_sized_subspace():
    """
    Tests that BlockSeries have correct shapes when one of the subspaces has
    size 1, see issue #127.
    """
    N = 3
    H_0 = np.diag(np.arange(N))
    H_1 = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    H_1 = H_1 + H_1.T.conj()

    for N_A in (1, 2):
        subspace_indices = N_A * [0] + (N - N_A) * [1]
        H_tilde, U, U_adjoint = block_diagonalize(
            [H_0, H_1], subspace_indices=subspace_indices
        )
        for output in (H_tilde, U, U_adjoint):
            blocks = [(0, 0), (1, 1), (0, 1), (1, 0)]
            shapes = [(N_A, N_A), (N - N_A, N - N_A), (N_A, N - N_A), (N - N_A, N_A)]
            for block, shape in zip(blocks, shapes):
                if output[(*block, 3)] is not zero:
                    assert output[(*block, 3)].shape == shape


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
    # KPM is slower. Therefore we only run this test for the direct solver.
    n = 30
    a_dim = 2

    def random_H(*index):  # noqa: ARG001
        rng = np.random.default_rng()
        h = rng.standard_normal(size=(n, n)) + 1j * rng.standard_normal(size=(n, n))
        return h + Dagger(h)

    H = BlockSeries(
        eval=random_H,
        shape=(),
        n_infinite=1,
    )
    H_0 = H[0]
    _, eigvecs = np.linalg.eigh(H_0)
    solve_sylvester = solve_sylvester_direct(sparse.coo_array(H_0), [eigvecs[:, :a_dim]])

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

    implicit_H_tilde, *_ = block_diagonalize(implicit_H, solve_sylvester=solve_sylvester)
    explicit_wrapped_H_tilde, *_ = block_diagonalize(
        explicit_wrapped_H, solve_sylvester=solve_sylvester
    )
    fully_explicit_H_tilde, *_ = block_diagonalize(fully_explicit_H)

    assert all(isinstance(implicit_H_tilde[1, 1, i], LinearOperator) for i in range(2))

    compare_series(implicit_H_tilde, explicit_wrapped_H_tilde, (2,), atol=1e-12)
    compare_series(implicit_H_tilde[0, 0], fully_explicit_H_tilde[0, 0], (2,), atol=1e-8)


def test_dtype_mismatch_error_implicit():
    """Test that the implicit algorithm raises an error when the dtype of the
    Hamiltonian and perturbation do not match."""
    rng = np.random.default_rng()
    h_0 = rng.standard_normal(size=(4, 4))
    h_0 += Dagger(h_0)
    vecs = np.linalg.eigh(h_0)[1]
    perturbation = rng.standard_normal(size=(4, 4)) + 1j * rng.standard_normal(
        size=(4, 4)
    )
    perturbation += Dagger(perturbation)

    H_tilde, *_ = block_diagonalize(
        [h_0, perturbation], subspace_eigenvectors=[vecs[:, :1]]
    )
    # This should evaluate normally.
    H_tilde[0, 0, 2]


def test_solve_sylvester_direct_vs_diagonal() -> None:
    """
    Test whether the solve_sylvester_direct gives the result consistent with
    solve_sylvester_diagonal.
    """
    n = 300
    a_dim = 5
    rng = np.random.default_rng()
    E = rng.standard_normal(n)
    t = np.random.rand(n - 1) * np.exp(2j * np.pi * np.random.rand(n - 1))
    h = sparse.diags([t, E, t.conj()], [-1, 0, 1])
    eigvals, eigvecs = np.linalg.eigh(h.toarray())
    eigvecs, eigvecs_rest = eigvecs[:, :a_dim], eigvecs[:, a_dim:]

    diagonal = solve_sylvester_diagonal((eigvals[:a_dim], eigvals[a_dim:]), eigvecs_rest)
    direct = solve_sylvester_direct(h, [eigvecs])

    y = rng.standard_normal(size=(a_dim, n - a_dim)) + 1j * rng.standard_normal(
        size=(a_dim, n - a_dim)
    )
    y = y @ Dagger(eigvecs_rest)

    y_default = diagonal(y, (0, 1))
    y_direct = direct(y, (0, 1))

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
    rng = np.random.default_rng()
    # Introduce a gap between the two subspaces
    E = rng.standard_normal(n) - 10 * (np.arange(n) < a_dim)
    t = np.random.rand(n - 1) * np.exp(2j * np.pi * np.random.rand(n - 1))
    h = sparse.diags([t, E, t.conj()], [-1, 0, 1])
    eigvals, eigvecs = np.linalg.eigh(h.toarray())
    eigvecs, eigvecs_partial, eigvecs_rest = (
        eigvecs[:, :a_dim],
        eigvecs[:, a_dim : 3 * a_dim],
        eigvecs[:, a_dim:],
    )

    diagonal = solve_sylvester_diagonal((eigvals[:a_dim], eigvals[a_dim:]), eigvecs_rest)
    kpm = solve_sylvester_KPM(h, [eigvecs], solver_options={"atol": 1e-3})
    hybrid = solve_sylvester_KPM(
        h, [eigvecs], solver_options={"atol": 1e-3, "aux_vectors": eigvecs_partial}
    )

    y = rng.standard_normal(size=(a_dim, n - a_dim)) + 1j * rng.standard_normal(
        size=(a_dim, n - a_dim)
    )
    y = y @ Dagger(eigvecs_rest)

    y_default = diagonal(y, (0, 1))
    y_kpm = kpm(y, (0, 1))
    y_hybrid = hybrid(y, (0, 1))

    # Use a lower tolerance until KPM estimates error bounds.
    np.testing.assert_allclose(y_default, y_kpm, atol=1e-3)
    np.testing.assert_allclose(y_default, y_hybrid, atol=1e-3)


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
        hamiltonian, subspace_eigenvectors=subspace_eigenvectors[:-1], implicit=True
    )
    assert H.shape == (2, 2)
    assert H.n_infinite == len(hamiltonian) - 1
    # Default dimension names
    assert H.dimension_names == tuple(f"n_{i}" for i in range(H.n_infinite))
    assert H.name == "H"
    for block in ((0, 1), (1, 0)):
        index = block + (0,) * H.n_infinite
        assert H[index] is zero
    assert isinstance(H[(1, 1) + (0,) * H.n_infinite], LinearOperator)

    # Test that block_diagonalize does the same processing.
    H_tilde, *_ = block_diagonalize(
        hamiltonian, subspace_eigenvectors=[subspace_eigenvectors[0]]
    )
    # Also try that BlockSeries input works
    if isinstance(hamiltonian, dict):
        block_diagonalize(
            _dict_to_BlockSeries(hamiltonian)[0],
            subspace_eigenvectors=[subspace_eigenvectors[0]],
        )
    # hamiltonian is either a list or a dictionary.
    try:
        hamiltonian = hamiltonian[0]
    except KeyError:
        hamiltonian = hamiltonian[(0,) * H.n_infinite]
    solve_sylvester = solve_sylvester_direct(hamiltonian, [subspace_eigenvectors[0]])

    compare_series(
        block_diagonalize(H, solve_sylvester=solve_sylvester)[0][0, 0],
        H_tilde[0, 0],
        (2,) * H.n_infinite,
    )


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
    assert hamiltonian.name == H.name
    for block in ((0, 0), (1, 1), (0, 1), (1, 0)):
        index = block + (0,) * H.n_infinite
        if H[index] is zero:
            assert hamiltonian[index] is zero
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
    assert H_1.name == H_2.name == "H"

    for H in (H_1, H_2):
        for block in ((0, 1), (1, 0)):
            assert H[block + (0,) * H.n_infinite] is zero


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
    # Default dimension names
    assert H_tilde.dimension_names == tuple(f"n_{i}" for i in range(H_tilde.n_infinite))
    assert H_tilde.name == "H_tilde"
    is_diagonal_series(H_tilde, wanted_orders)
    is_unitary(U, U_adjoint, wanted_orders)


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
    assert H_tilde.name == "H_tilde"
    is_diagonal_series(H_tilde, (1,) * len(symbols))
    is_unitary(U, U_adjoint, (1,) * len(symbols))

    # Test if subspace_indices are provided
    hamiltonian, symbols, subspace_eigenvectors, subspace_indices = symbolic_hamiltonian
    H_tilde, U, U_adjoint = block_diagonalize(
        hamiltonian, symbols=symbols, subspace_indices=subspace_indices
    )
    assert H_tilde.shape == (2, 2)
    assert H_tilde.n_infinite == len(symbols)
    assert H_tilde.dimension_names == symbols
    assert H_tilde.name == "H_tilde"
    is_diagonal_series(H_tilde, (1,) * len(symbols))
    is_unitary(U, U_adjoint, (1,) * len(symbols))

    # Raise if eigenvectors are not orthonormal
    with pytest.raises(ValueError):
        faulted_eigenvectors = [2 * evecs for evecs in subspace_eigenvectors]
        block_diagonalize(hamiltonian, subspace_eigenvectors=faulted_eigenvectors)


def test_algebra_element_data_type():
    """
    Test that the algorithm works with a class implementing algebra.

    This requires a solve_sylvester function to be provided, and otherwise
    should raise an error.
    """
    H = [
        [[AlgebraElement("a"), zero], [zero, AlgebraElement("b")]],
        [
            [AlgebraElement("c"), AlgebraElement("d")],
            [AlgebraElement("e"), AlgebraElement("f")],
        ],
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
    rng = np.random.default_rng()
    h_0 = np.diag([0.0, 0.0, 1.0, 1.0])
    h_p = rng.standard_normal(size=(4, 4))
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
    assert H_tilde.name == "H_tilde"
    # Check that execution doesn't raise
    H_tilde[:, :, 3]


def test_warning_non_diagonal_input():
    muL, muR, B, E, t, tso, Delta = sympy.symbols("mu_L mu_R B E t t_s Delta", real=True)
    h_0 = sympy.Matrix(
        [
            [muL, 0, 0, 0, 0, 0, 0, 0],
            [0, muR, 0, 0, 0, 0, 0, 0],
            [0, 0, -muL, 0, 0, 0, 0, 0],
            [0, 0, 0, -muR, 0, 0, 0, 0],
            [0, 0, 0, 0, B + E, 0, 0, Delta],
            [0, 0, 0, 0, 0, -B + E, Delta, 0],
            [0, 0, 0, 0, 0, Delta, -B - E, 0],
            [0, 0, 0, 0, Delta, 0, 0, B - E],
        ]
    )
    h_p = sympy.Matrix(
        [
            [0, 0, 0, 0, sympy.I * tso, t, 0, 0],
            [0, 0, 0, 0, -sympy.I * tso, t, 0, 0],
            [0, 0, 0, 0, 0, 0, -sympy.I * tso, -t],
            [0, 0, 0, 0, 0, 0, sympy.I * tso, -t],
            [-sympy.I * tso, sympy.I * tso, 0, 0, 0, 0, 0, 0],
            [t, t, 0, 0, 0, 0, 0, 0],
            [0, 0, sympy.I * tso, -sympy.I * tso, 0, 0, 0, 0],
            [0, 0, -t, -t, 0, 0, 0, 0],
        ]
    )
    P, _ = h_0.diagonalize()

    with pytest.warns(UserWarning):
        block_diagonalize([h_0, h_p], subspace_eigenvectors=[P[:, :4], P[:, 4:]])[0]


def test_memory_usage_implicit():
    """
    Test that the implicit algorithm does not use more memory than expected.
    A failure of this test would indicate that the implicit algorithm is
    broken.

    Observed memory usage running this test with explicit and implicit algorithms:
    - explicit -> total_KiB = [411, 0, 312, 945, 2512, 4400, 6600]
    - implicit -> total_KiB = [411, 0, 0, 8, 103, 115, 128]

    """
    a_dim, b_dim = 1, 200
    n_dim = a_dim + b_dim

    tracemalloc.start()
    h_0 = np.diag(np.linspace(0.1, 1, n_dim))
    h_p = sparse.random(n_dim, n_dim, density=0.1)
    h_p += Dagger(h_p)
    snapshots = [tracemalloc.take_snapshot()]

    eigenvectors = np.eye(n_dim)
    subspace_eigenvectors = eigenvectors[:, :a_dim], eigenvectors[:, a_dim:]
    hamiltonian = [sparse.csr_array(h_0), sparse.csr_array(h_p)]

    H_tilde, *_ = block_diagonalize(
        hamiltonian, subspace_eigenvectors=[subspace_eigenvectors[0]]
    )
    tracemalloc.clear_traces()

    for order in range(6):
        H_tilde[:, :, order]
        snapshots.append(tracemalloc.take_snapshot())
    tracemalloc.stop()

    total_KiB = []
    for snapshot in snapshots:
        dom_filter = tracemalloc.DomainFilter(
            inclusive=True, domain=np.lib.tracemalloc_domain
        )
        snapshot = snapshot.filter_traces([dom_filter])
        top_stats = snapshot.statistics("lineno")
        # We sum the memory usage of the top 10 traces in the snapshot
        total_KiB.append(round(sum(stat.size for stat in top_stats) / (1024)))

    total_KiB = np.array(total_KiB)
    if np.any(total_KiB[1:] > 200):
        raise ValueError("Memory usage unexpectedly high for implicit algorithm.")


def test_number_products(data_regression):
    """
    Test that the number of products per order of the transformed Hamiltonian
    is as expected.
    This is a regression test so that we don't accidentally change the algorithm
    in a way that increases the number of products.

    If the number of products needs to be updated, check the output of this test
    and update the yaml file accordingly by running `pytest --force-regen`.
    """

    def solve_sylvester(A):
        return AlgebraElement(f"S({A})")

    def eval_dense_first_order(*index):
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        if index[2] > 1 or any(index[3:]):
            return zero
        return AlgebraElement(f"H{index}")

    def eval_dense_every_order(*index):
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        return AlgebraElement(f"H{index}")

    def eval_offdiagonal_every_order(*index):
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        if index[0] == index[1] and sum(index[2:]) != 0:
            return zero
        return AlgebraElement(f"H{index}")

    def eval_randomly_sparse(*index):
        np.random.seed(index[2])
        p = np.random.random(3)
        if index[0] != index[1] and sum(index[2:]) == 0:  # H_0 is diagonal
            return zero
        if index[0] == index[1] == 0 and sum(index[2:]) == 0 and p[0] > 0.4:
            return zero
        if index[0] == index[1] == 1 and sum(index[2:]) == 0:
            return AlgebraElement(f"H{index}")  # Not both diagonal blocks are zero
        if index[0] == index[1] and p[1] > 0.4:
            return zero
        if index[0] != index[1] and p[2] > 0.4:
            return zero
        return AlgebraElement(f"H{index}")

    evals = {
        "dense_first_order": eval_dense_first_order,
        "dense_every_order": eval_dense_every_order,
        "offdiagonal_every_order": eval_offdiagonal_every_order,
        "random_every_order": eval_randomly_sparse,
    }

    blocks = {
        "aa": [(0, 0)],
        "bb": [(1, 1)],
        "both": [(0, 0), (1, 1)],
    }

    orders = {
        "all": lambda order: tuple(range(order + 1)),
        "highest": lambda order: (order,),
    }

    multiplication_counts = {}
    for structure in evals.keys():
        multiplication_counts[structure] = {}
        for order in orders.keys():
            multiplication_counts[structure][order] = {}
            for block in blocks.keys():
                multiplication_counts[structure][order][block] = {}
                for highest_order in range(10):
                    AlgebraElement.log = []
                    H = BlockSeries(
                        eval=evals[structure],
                        shape=(2, 2),
                        n_infinite=1,
                    )

                    H_tilde, *_ = block_diagonalize(
                        H,
                        solve_sylvester=solve_sylvester,
                    )
                    for index in blocks[block]:
                        for _order in orders[order](highest_order):
                            H_tilde[(*index, _order)]

                    multiplication_counts[structure][order][block][highest_order] = (
                        Counter(call[1] for call in AlgebraElement.log)["__mul__"]
                    )

    data_regression.check(multiplication_counts)


def test_delete_intermediate_terms():
    """Test that the algorithm deletes all intermediate terms that are accessed once.

    The terms to delete are calculated automatically by the algorithm,
    so we check against the previous manual result.

    Note that this test does not check whether we delete too many terms,
    that is ensured by the test `test_number_products` instead.
    """

    def dense_eval(*index):
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        return AlgebraElement(f"H{index}")

    H = BlockSeries(
        eval=dense_eval,
        shape=(2, 2),
        n_infinite=1,
    )

    max_order = 10
    series, linear_operator_series = series_computation(
        {"H": H},
        algorithm=main,
        scope={
            "solve_sylvester": (lambda x, _: x),
            "two_block_optimized": True,
            "commuting_blocks": [True, True],
            "offdiag": None,
            "diag": (lambda x, index: x[index] if isinstance(x, BlockSeries) else x),
        },
        operator=operator.mul,
    )
    series["H_tilde"][:, :, :max_order]

    # Manual result of terms to delete
    to_delete = {
        "U'† @ U'": [(0, 0), (1, 1)],
        "X": [(0, 1)],
        "H'_diag @ U'": [(0, 1), (1, 0)],
        "H'_offdiag @ U'": [(0, 1)],
        "U'† @ B": [(0, 1), (1, 0)],
    }

    for term, indices in to_delete.items():
        for which in series, linear_operator_series:
            # We start from 1 because the 0th order is precomputed and thus not deleted.
            for order in range(1, max_order):
                for index in indices:
                    assert (*index, order) not in which[term]._data


def H_list(wanted_orders, N):
    """Random Hamiltonian of a given size."""

    def random_hermitian():
        H_p = np.random.randn(N, N) + 1.0j * np.random.randn(N, N)
        H_p += H_p.T.conj()
        return H_p

    H_0 = np.diag(np.random.randn(N) + 8 * np.arange(N))
    H_ps = [random_hermitian() for _ in wanted_orders]

    return H_0, H_ps


def test_three_blocks(wanted_orders):
    N = 6
    H_0, H_ps = H_list(wanted_orders, N)
    H = hamiltonian_to_BlockSeries([H_0, *H_ps], subspace_indices=np.arange(N) // 2)
    H_tilde, U, U_adjoint = block_diagonalize(H)
    is_unitary(U, U_adjoint, wanted_orders, atol=1e-6)
    H_prime = cauchy_dot_product(U, H_tilde, U_adjoint)
    compare_series(H, H_prime, wanted_orders, atol=1e-6)


def test_hamiltonian_shared_decoupled_eigenvalues(wanted_orders):
    """
    Test that blocks may overlap in the eigenvalues if their coupling is zero.
    """
    N = 4
    H_0, H_ps = H_list(wanted_orders, N)
    H_0 = np.kron(np.eye(2), H_0)
    H_ps = [np.kron(np.eye(2), H_p) for H_p in H_ps]
    H = hamiltonian_to_BlockSeries([H_0, *H_ps], subspace_indices=np.arange(0, 2 * N))
    H_tilde, *_ = block_diagonalize(H)
    compare_series(H_tilde[:N, :N], H_tilde[N:, N:], wanted_orders, atol=1e-6)


def test_analytic_full_and_selective():
    H_0 = sympy.diag(*[sympy.Symbol(f"H_{i}", real=True) for i in range(3)])
    H_1 = sympy.Matrix(
        [
            [sympy.Symbol(f"H_{sorted([i,j])}", real=True) for i in range(3)]
            for j in range(3)
        ]
    )
    H = hamiltonian_to_BlockSeries([H_0, H_1])
    H_tilde, U, U_adjoint = block_diagonalize(H)
    is_unitary(U, U_adjoint, (3,), atol=1e-6)

    # Only check up to 2nd order for performance reasons
    compare_series(
        cauchy_dot_product(U, H_tilde), cauchy_dot_product(H, U), (2,), atol=1e-6
    )
    # Now the same but only eliminate the (0, 2) matrix element
    H_tilde, U, U_adjoint = block_diagonalize(
        H, fully_diagonalize={0: np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])}
    )
    is_unitary(U, U_adjoint, (3,), atol=1e-6)
    compare_series(
        cauchy_dot_product(U, H_tilde), cauchy_dot_product(H, U), (2,), atol=1e-6
    )
    assert H_tilde[0, 0, 3][0, 2].simplify() == 0


def test_three_blocks_repeated(wanted_orders):
    """
    Test block-diagonalization of a 3x3 Hamiltonian.

    Verifies that block-diagonalizing a 3x3 Hamiltonian into 1x1 + 2x2, then
    the second block further into 1x1 + 1x1, matches direct
    block-diagonalization into 1x1 + 1x1 + 1x1.

    Parameters:
    wanted_orders (list): Desired orders for the Hamiltonian.
    """
    N = 3
    H_0, H_ps = H_list(wanted_orders, N=N)

    H_tilde, *_ = block_diagonalize([H_0, *H_ps], subspace_indices=np.arange(N))

    H = hamiltonian_to_BlockSeries(
        [H_0, *H_ps], subspace_indices=np.clip(np.arange(N), 0, 1)
    )
    H_tilde_A_BC, *_ = block_diagonalize(H)

    H = hamiltonian_to_BlockSeries(
        {
            orders: H_tilde_A_BC[(1, 1, *orders)]
            for orders in product(*(range(order + 1) for order in wanted_orders))
        },
        subspace_indices=np.arange(N - 1),
    )
    H_tilde_B_C, *_ = block_diagonalize(H)

    def H_tilde_repeated_eval(*index):
        if index[0] == index[1] == 0:
            return H_tilde[index]
        if index[0] == 0 or index[1] == 0:
            return zero
        index = (index[0] - 1, index[1] - 1, *index[2:])
        return H_tilde_B_C[index]

    H_tilde_repeated = BlockSeries(
        eval=H_tilde_repeated_eval,
        shape=H_tilde.shape,
        n_infinite=H_tilde.n_infinite,
    )

    compare_series(
        H_tilde,
        H_tilde_repeated,
        wanted_orders=wanted_orders,
        atol=1e-6,
    )


def test_one_block_vs_multiblock(wanted_orders):
    N = 6
    H_0, H_ps = H_list(wanted_orders, N)
    # First, multiblock
    H_tilde, *_ = block_diagonalize([H_0, *H_ps], subspace_indices=np.arange(N))

    H_tilde_single, *_ = block_diagonalize([H_0, *H_ps])

    H_tilde_split = BlockSeries(
        eval=lambda *index: np.array(
            H_tilde_single[(0, 0, *index[2:])][index[:2]].reshape((1, 1))
        ),
        shape=(N, N),
        n_infinite=len(wanted_orders),
    )
    compare_series(H_tilde, H_tilde_split, wanted_orders, atol=1e-10)


def test_mixed_full_partial(wanted_orders):
    N = 6
    H_0, H_ps = H_list(wanted_orders, N)
    with pytest.raises(ValueError):
        block_diagonalize(
            [H_0, *H_ps], subspace_eigenvectors=[np.eye(N)[:, :2]], fully_diagonalize=[1]
        )

    H_tilde_mixed_implicit, *_ = block_diagonalize(
        [H_0, *H_ps], subspace_eigenvectors=[np.eye(N)[:, :2]], fully_diagonalize=[0]
    )
    H_tilde_mixed, *_ = block_diagonalize(
        [H_0, *H_ps], subspace_indices=[0] * 2 + [1] * 4, fully_diagonalize=[0]
    )
    H_tilde_full, *_ = block_diagonalize([H_0, *H_ps])
    H_tilde_full_subblock = BlockSeries(
        eval=lambda *index: H_tilde_full[(0, 0, *index[2:])][:2, :2],
        shape=(1, 1),
        n_infinite=len(wanted_orders),
    )
    # The slicing syntax is a workaround of #154
    compare_series(
        H_tilde_mixed_implicit[:1, :1], H_tilde_mixed[:1, :1], wanted_orders, atol=1e-10
    )
    compare_series(
        H_tilde_mixed[:1, :1], H_tilde_full_subblock, wanted_orders, atol=1e-10
    )


def test_multiblock_kpm_auxiliary(wanted_orders):
    """Test that the multiblock KPM solver correctly works with auxiliary vectors."""
    N = 6
    H_0, H_ps = H_list(wanted_orders, N)
    # 3 blocks, last one missing
    H_tilde_implicit, *_ = block_diagonalize(
        [H_0, *H_ps],
        subspace_eigenvectors=[np.eye(N)[:, :2], np.eye(N)[:, 2:4]],
        direct_solver=False,
        # We have all vectors as auxiliary vectors to avoid the problem with
        # the KPM convergence
        solver_options={"atol": 1e-6, "aux_vectors": np.eye(N)[:, 4:]},
    )
    H_tilde_full, *_ = block_diagonalize([H_0, *H_ps], subspace_indices=np.arange(6) // 2)
    # The slicing is a workaround of #154
    compare_series(
        H_tilde_implicit[:1, :1], H_tilde_full[:1, :1], wanted_orders, atol=1e-3
    )
    compare_series(
        H_tilde_implicit[1:2, 1:2], H_tilde_full[1:2, 1:2], wanted_orders, atol=1e-3
    )


def test_selective_diagonalization(wanted_orders):
    N = 20
    H_0, H_ps = H_list(wanted_orders, N)
    H = hamiltonian_to_BlockSeries([H_0, *H_ps])
    to_eliminate = np.random.rand(N, N) > 0.8
    to_eliminate = np.logical_or(to_eliminate, to_eliminate.T)
    np.fill_diagonal(to_eliminate, False)
    H_tilde, U, U_adjoint = block_diagonalize(H, fully_diagonalize={0: to_eliminate})
    is_unitary(U, U_adjoint, wanted_orders, atol=1e-6)
    compare_series(H, cauchy_dot_product(U, H_tilde, U_adjoint), wanted_orders, atol=1e-6)
    # Check that the eliminated elements are zero
    np.testing.assert_equal(H_tilde[(0, 0, *wanted_orders)][to_eliminate], 0)
