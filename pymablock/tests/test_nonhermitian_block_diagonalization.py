import operator
from collections import Counter
from collections.abc import Callable
from itertools import pairwise, product

import numpy as np
import pytest
import sympy
from scipy import sparse
from sympy.physics.quantum import Dagger

from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main, nonhermitian
from pymablock.block_diagonalization import (
    block_diagonalize,
    solve_sylvester_direct,
)
from pymablock.series import AlgebraElement, BlockSeries, cauchy_dot_product, one, zero

from .test_block_diagonalization import (
    compare_series,
    is_unitary,
    random_hermitian_matrix,
)


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _block_slices(block_dims: tuple[int, ...]) -> tuple[slice, ...]:
    offsets = np.cumsum((0, *block_dims))
    return tuple(slice(start, stop) for start, stop in pairwise(offsets))


def _make_block_h(
    block_dims: tuple[int, ...],
    wanted_orders: int | tuple[int, ...],
    matrix_factory: Callable[[int], np.ndarray],
) -> tuple[BlockSeries, list[float]]:
    if isinstance(wanted_orders, int):
        wanted_orders = (wanted_orders,)
    n_infinite = len(wanted_orders)
    zero_order = (0,) * n_infinite
    block_centers = np.linspace(-2.0, 2.0, len(block_dims))
    block_slices = _block_slices(block_dims)
    total_dim = sum(block_dims)

    data = {
        (i, i, *zero_order): np.eye(dim, dtype=complex) * block_centers[i]
        for i, dim in enumerate(block_dims)
    }
    for order in np.ndindex(tuple(order + 1 for order in wanted_orders)):
        if order == zero_order:
            continue
        matrix = matrix_factory(total_dim)
        for i, row_slice in enumerate(block_slices):
            for j, column_slice in enumerate(block_slices):
                data[(i, j, *order)] = matrix[row_slice, column_slice]

    return BlockSeries(
        data=data,
        shape=(len(block_dims), len(block_dims)),
        n_infinite=n_infinite,
        name="H",
    ), list(block_centers)


def _solve_sylvester(energies: list[float]):
    def solve(rhs, index):
        if rhs is zero:
            return zero
        left = np.atleast_1d(energies[index[0]]).reshape(-1, 1)
        right = np.atleast_1d(energies[index[1]]).reshape(1, -1)
        return rhs / (left - right)

    return solve


def _run_nonhermitian(H: BlockSeries, energies: list[float]) -> dict[str, BlockSeries]:
    series, _ = series_computation(
        {"H": H},
        algorithm=nonhermitian,
        scope={
            "solve_sylvester": _solve_sylvester(energies),
            "two_block_optimized": False,
            "commuting_blocks": [False] * H.shape[0],
        },
    )
    return series


def _assert_zero(value, *, atol: float) -> None:
    if value is zero:
        return
    if sparse.issparse(value):
        np.testing.assert_allclose(value.toarray(), 0, atol=atol, rtol=0)
        return
    if isinstance(value, sympy.MatrixBase):
        assert value.is_zero_matrix
        return
    np.testing.assert_allclose(value, 0, atol=atol, rtol=0)


def _assert_inverse_relations(
    U: BlockSeries, U_inv: BlockSeries, wanted_orders: tuple[int, ...]
) -> None:
    is_unitary(U, U_inv, wanted_orders, atol=1e-12)
    is_unitary(U_inv, U, wanted_orders, atol=1e-12)


def _assert_roundtrip(
    H: BlockSeries, series: dict[str, BlockSeries], wanted_orders: tuple[int, ...]
) -> BlockSeries:
    H_tilde = series["H_tilde"]
    U = series["U"]
    U_inv = series["U†"]

    _assert_inverse_relations(U, U_inv, wanted_orders)
    compare_series(cauchy_dot_product(U_inv, H, U), H_tilde, wanted_orders, atol=1e-14)
    compare_series(cauchy_dot_product(U, H_tilde, U_inv), H, wanted_orders, atol=1e-14)
    return H_tilde


def _assert_block_offdiag_zero(
    H_tilde: BlockSeries,
    wanted_orders: tuple[int, ...],
    *,
    atol: float = 1e-10,
) -> None:
    zero_order = (0,) * len(wanted_orders)
    for order in np.ndindex(tuple(order + 1 for order in wanted_orders)):
        if order == zero_order:
            continue
        for i in range(H_tilde.shape[0]):
            for j in range(H_tilde.shape[1]):
                if i == j:
                    continue
                _assert_zero(H_tilde[(i, j, *order)], atol=atol)


def _assert_nonhermitian_gauge(
    U: BlockSeries, U_inv: BlockSeries, wanted_orders: tuple[int, ...]
) -> None:
    for order in np.ndindex(tuple(order + 1 for order in wanted_orders)):
        if sum(order) != 1:
            continue
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                if i == j:
                    continue
                _assert_zero(U_inv[(i, j, *order)] + U[(i, j, *order)], atol=1e-12)


def _make_asymmetric_mask(
    rng: np.random.Generator, size: int, *, threshold: float = 0.8
) -> np.ndarray:
    mask = rng.random((size, size)) > threshold
    np.fill_diagonal(mask, False)
    mask[0, 1] = True
    mask[1, 0] = False
    if size > 4:
        mask[2, 4] = False
        mask[4, 2] = True
    assert not np.array_equal(mask, mask.T)
    return mask


def _assert_mask_eliminated(
    H_tilde: BlockSeries, masks: dict[int, np.ndarray], *, max_order: int
) -> None:
    for order in range(1, max_order + 1):
        for block, mask in masks.items():
            np.testing.assert_allclose(
                H_tilde[(block, block, order)][mask], 0, atol=1e-10, rtol=0
            )


def _make_direct_solver_case(
    n: int, a_dim: int
) -> tuple[np.random.Generator, sparse.dia_matrix, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    energies = rng.standard_normal(n)
    hoppings = rng.random(n - 1) * np.exp(2j * np.pi * rng.random(n - 1))
    h_0 = sparse.diags([hoppings, energies, hoppings.conj()], [-1, 0, 1])
    eigvals, eigvecs = np.linalg.eigh(h_0.toarray())
    return rng, h_0, eigvals, eigvecs[:, :a_dim], eigvecs[:, a_dim:]


def _make_biorthogonal_direct_solver_case(
    n: int, a_dim: int
) -> tuple[
    np.random.Generator,
    sparse.csr_array,
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    rng = np.random.default_rng()
    eigvals = np.linspace(-3.0, 3.0, n) + 0.2j * rng.standard_normal(n)
    transform = np.diag(5.0 + rng.random(n)).astype(complex)
    transform += 0.2 * _complex_normal(rng, (n, n))
    inverse_transform = np.linalg.inv(transform)
    left = inverse_transform.conj().T
    h_0 = sparse.csr_array(transform @ np.diag(eigvals) @ inverse_transform)
    return (
        rng,
        h_0,
        eigvals,
        (transform[:, :a_dim], left[:, :a_dim]),
        (transform[:, a_dim:], left[:, a_dim:]),
        (transform, left),
    )


def test_number_products_nonhermitian_two_block(data_regression):
    """Regression test for the number of products in the two-block NH algorithm."""
    op = AlgebraElement("A")

    def solve_sylvester(Y, index=None):  # noqa: ARG001
        return zero if Y is zero else op

    def eval_dense_first_order(*index):
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        if index[2] > 1 or any(index[3:]):
            return zero
        return op

    def eval_dense_every_order(*index):
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        return op

    def eval_offdiagonal_every_order(*index):
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        if index[0] == index[1] and sum(index[2:]) != 0:
            return zero
        return op

    def eval_randomly_sparse(*index):
        np.random.seed(index[2])
        p = np.random.random(3)
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        if index[0] == index[1] == 0 and sum(index[2:]) == 0 and p[0] > 0.4:
            return zero
        if index[0] == index[1] == 1 and sum(index[2:]) == 0:
            return op
        if index[0] == index[1] and p[1] > 0.4:
            return zero
        if index[0] != index[1] and p[2] > 0.4:
            return zero
        return op

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

    mul_counts = {}
    for (structure, eval_), (order, query), (block, indices), highest_order in product(
        evals.items(), orders.items(), blocks.items(), range(10)
    ):
        key = f"{structure=}, {order=}, {block=}, {highest_order=}"
        AlgebraElement.log = []
        H = BlockSeries(eval=eval_, shape=(2, 2), n_infinite=1)

        H_tilde, *_ = block_diagonalize(
            H,
            solve_sylvester=solve_sylvester,
            hermitian=False,
        )
        for index in indices:
            for _order in query(highest_order):
                H_tilde[(*index, _order)]

        mul_counts[key] = Counter(call[1] for call in AlgebraElement.log)["__mul__"]

    data_regression.check(mul_counts)


def test_number_products_nonhermitian_one_block(data_regression):
    """Regression test for the number of products in one-block NH mode."""
    op = AlgebraElement("A")

    def func(A, index=None):  # noqa: ARG001
        return zero if A is zero else op

    def eval_all(*index):  # noqa: ARG001
        return op

    def eval_first(*index):
        return op if index[2] < 2 else zero

    evals = {
        "all": eval_all,
        "first": eval_first,
    }

    mul_counts = {}
    for structure, order in product(evals.keys(), range(2, 6)):
        key = f"{structure=}, {order=}"
        AlgebraElement.log = []
        series_computation(
            {"H": BlockSeries(eval=evals[structure], shape=(1, 1), n_infinite=1)},
            algorithm=nonhermitian,
            scope={
                "solve_sylvester": func,
                "offdiag": func,
                "diag": func,
            },
            operator=operator.mul,
        )[0]["H_tilde"][0, 0, order]

        mul_counts[key] = sum(call[1] == "__mul__" for call in AlgebraElement.log)

    data_regression.check(mul_counts)


def test_number_products_nonhermitian_three_block(data_regression):
    """Regression test for the number of products in the three-block NH algorithm."""
    op = AlgebraElement("A")

    def solve_sylvester(Y, index=None):  # noqa: ARG001
        return zero if Y is zero else op

    def eval_dense_first_order(*index):
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        if index[2] > 1 or any(index[3:]):
            return zero
        return op

    def eval_dense_every_order(*index):
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        return op

    def eval_offdiagonal_every_order(*index):
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        if index[0] == index[1] and sum(index[2:]) != 0:
            return zero
        return op

    def eval_randomly_sparse(*index):
        np.random.seed(index[2])
        p = np.random.random(3)
        if index[0] != index[1] and sum(index[2:]) == 0:
            return zero
        if index[0] == index[1] == 0 and sum(index[2:]) == 0 and p[0] > 0.4:
            return zero
        if index[0] == index[1] == 1 and sum(index[2:]) == 0:
            return op
        if index[0] == index[1] and p[1] > 0.4:
            return zero
        if index[0] != index[1] and p[2] > 0.4:
            return zero
        return op

    evals = {
        "dense_first_order": eval_dense_first_order,
        "dense_every_order": eval_dense_every_order,
        "offdiagonal_every_order": eval_offdiagonal_every_order,
        "random_every_order": eval_randomly_sparse,
    }

    blocks = {
        "aa": [(0, 0)],
        "bb": [(1, 1)],
        "cc": [(2, 2)],
        "all": [(0, 0), (1, 1), (2, 2)],
    }

    orders = {
        "all": lambda order: tuple(range(2, order + 1)),
        "highest": lambda order: (order,),
    }

    mul_counts = {}
    for (structure, eval_), (order, query), (block, indices), highest_order in product(
        evals.items(), orders.items(), blocks.items(), range(2, 5)
    ):
        key = f"{structure=}, {order=}, {block=}, {highest_order=}"
        AlgebraElement.log = []
        H = BlockSeries(eval=eval_, shape=(3, 3), n_infinite=1)

        H_tilde, *_ = block_diagonalize(
            H,
            solve_sylvester=solve_sylvester,
            hermitian=False,
        )
        for index in indices:
            for _order in query(highest_order):
                H_tilde[(*index, _order)]

        mul_counts[key] = Counter(call[1] for call in AlgebraElement.log)["__mul__"]

    data_regression.check(mul_counts)


@pytest.mark.parametrize(
    ("block_dims", "wanted_orders"),
    [((1, 1), (4,)), ((2, 1), (3,)), ((1, 2), (2, 1)), ((2, 1, 1), (1, 1))],
    ids=[
        "scalar-2-blocks-order-4",
        "matrix-2-blocks-order-3",
        "matrix-2-blocks-bivariate",
        "matrix-3-blocks-bivariate",
    ],
)
def test_nonhermitian_roundtrip(block_dims, wanted_orders):
    rng = np.random.default_rng()
    H, energies = _make_block_h(
        block_dims=block_dims,
        wanted_orders=wanted_orders,
        matrix_factory=lambda n: _complex_normal(rng, (n, n)),
    )
    series = _run_nonhermitian(H, energies)
    H_tilde = _assert_roundtrip(H, series, wanted_orders)
    _assert_block_offdiag_zero(H_tilde, wanted_orders)
    _assert_nonhermitian_gauge(series["U"], series["U†"], wanted_orders)


@pytest.mark.parametrize(("block_dims", "max_order"), [((2, 2), 4), ((2, 1, 2), 3)])
def test_nonhermitian_matches_hermitian_on_hermitian_input(block_dims, max_order):
    H, energies = _make_block_h(
        block_dims=block_dims,
        wanted_orders=max_order,
        matrix_factory=random_hermitian_matrix,
    )
    scope = {
        "solve_sylvester": _solve_sylvester(energies),
        "two_block_optimized": False,
        "commuting_blocks": [False] * len(block_dims),
    }

    series_h, _ = series_computation({"H": H}, algorithm=main, scope=scope)
    series_nh, _ = series_computation({"H": H}, algorithm=nonhermitian, scope=scope)

    compare_series(series_h["H_tilde"], series_nh["H_tilde"], (max_order,), atol=1e-14)
    compare_series(series_h["U"], series_nh["U"], (max_order,), atol=1e-14)
    compare_series(series_h["U†"], series_nh["U†"], (max_order,), atol=1e-14)


def test_nonhermitian_arbitrary_asymmetric_mask():
    n = 6
    max_order = 3
    rng = np.random.default_rng()
    energies = np.linspace(-3.0, 3.0, n)

    data = {(0, 0, 0): np.diag(energies).astype(complex)}
    for order in range(1, max_order + 1):
        data[(0, 0, order)] = _complex_normal(rng, (n, n))
    H = BlockSeries(data=data, shape=(1, 1), n_infinite=1, name="H")

    to_eliminate = _make_asymmetric_mask(rng, n)
    to_keep = np.logical_not(to_eliminate)

    denom = energies[:, None] - energies[None, :]

    def solve_sylvester(rhs, index):  # noqa: ARG001
        if rhs is zero:
            return zero
        out = np.zeros_like(rhs, dtype=complex)
        np.divide(rhs, denom, out=out, where=denom != 0)
        return out

    def diag(x, index):
        x = x[index] if isinstance(x, BlockSeries) else x
        if x is zero:
            return zero
        return x * to_keep

    def offdiag(x, index):
        x = x[index] if isinstance(x, BlockSeries) else x
        if x is zero:
            return zero
        return x * to_eliminate

    series, _ = series_computation(
        {"H": H},
        algorithm=nonhermitian,
        scope={
            "solve_sylvester": solve_sylvester,
            "two_block_optimized": False,
            "commuting_blocks": [False],
            "diag": diag,
            "offdiag": offdiag,
        },
    )

    H_tilde = series["H_tilde"]

    _assert_mask_eliminated(H_tilde, {0: to_eliminate}, max_order=max_order)
    opposite_direction_nonzero = False
    for order in range(1, max_order + 1):
        block = H_tilde[(0, 0, order)]
        opposite_direction_nonzero = opposite_direction_nonzero or (
            not np.allclose(block[1, 0], 0, atol=1e-10, rtol=0)
        )

    assert opposite_direction_nonzero


@pytest.mark.parametrize("block_sizes", [(6,), (3, 2)], ids=["single-block", "two-block"])
def test_block_diagonalize_nonhermitian_accepts_asymmetric_mask(block_sizes):
    max_order = 3
    rng = np.random.default_rng()
    n = sum(block_sizes)

    h_0 = np.diag(np.linspace(-3.0, 3.0, n)).astype(complex)
    h_1 = _complex_normal(rng, (n, n))
    subspace_indices = np.repeat(np.arange(len(block_sizes)), block_sizes)
    masks = {
        block: _make_asymmetric_mask(rng, size, threshold=0.75 if size == 2 else 0.8)
        for block, size in enumerate(block_sizes)
    }
    fully_diagonalize = masks[0] if len(masks) == 1 else masks

    H_tilde, *_ = block_diagonalize(
        [h_0, h_1],
        subspace_indices=subspace_indices,
        hermitian=False,
        fully_diagonalize=fully_diagonalize,
    )

    _assert_mask_eliminated(H_tilde, masks, max_order=max_order)


def test_nonhermitian_selective_mask_preserves_trace_and_hermiticity():
    # Column-major vectorization convention: vec(A X B) = (B^T \kron A) vec(X).
    def coherent_superoperator(H):
        dim = H.shape[0]
        eye = np.eye(dim, dtype=complex)
        return -1j * (np.kron(eye, H) - np.kron(H.T, eye))

    def dissipator_superoperator(J):
        dim = J.shape[0]
        eye = np.eye(dim, dtype=complex)
        JdagJ = J.conj().T @ J
        return np.kron(J.conj(), J) - 0.5 * (
            np.kron(eye, JdagJ) + np.kron(JdagJ.T, eye)
        )

    def random_hermitian(rng, dim):
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        return (A + A.conj().T) / 2

    def dagger_index(idx, dim):
        i = idx % dim
        j = idx // dim
        return j + dim * i

    def to_dense(value, dim):
        if value is zero:
            return np.zeros((dim, dim), dtype=complex)
        if value is one:
            return np.eye(dim, dtype=complex)
        if sparse.issparse(value):
            return value.toarray()
        return np.asarray(value)

    dim_hilbert = 3
    dim_liouville = dim_hilbert**2
    trace_idx = np.array([i + dim_hilbert * i for i in range(dim_hilbert)])

    for seed in range(8):
        rng = np.random.default_rng(seed)

        # Keep L0 diagonal so the default Sylvester solver is exact and cheap.
        energies = np.sort(rng.uniform(-2.5, 2.5, size=dim_hilbert))
        h_0 = np.diag(energies)
        h_1 = random_hermitian(rng, dim_hilbert)

        jumps_0 = []
        for _ in range(2):
            vals = 0.2 * (
                rng.normal(size=dim_hilbert) + 1j * rng.normal(size=dim_hilbert)
            )
            jumps_0.append(np.diag(vals))

        jumps_1 = []
        for _ in range(2):
            jumps_1.append(
                0.15
                * (
                    rng.normal(size=(dim_hilbert, dim_hilbert))
                    + 1j * rng.normal(size=(dim_hilbert, dim_hilbert))
                )
            )

        L_0 = coherent_superoperator(h_0) + sum(
            dissipator_superoperator(J) for J in jumps_0
        )
        L_1 = coherent_superoperator(h_1) + sum(
            dissipator_superoperator(J) for J in jumps_1
        )
        np.testing.assert_allclose(L_0, np.diag(np.diag(L_0)), atol=1e-12, rtol=0)

        # Structure-preserving selective mask:
        # - asymmetric random elimination pattern,
        # - never eliminate diagonal matrix elements,
        # - preserve trace equations by keeping all rows carrying trace support,
        # - preserve hermiticity by pairing each eliminated entry with its dagger partner,
        # - avoid eliminating degenerate-denominator pairs.
        to_eliminate = rng.random((dim_liouville, dim_liouville)) < 0.35
        np.fill_diagonal(to_eliminate, False)
        to_eliminate[trace_idx, :] = False
        for i in range(dim_liouville):
            i_dag = dagger_index(i, dim_hilbert)
            for j in range(dim_liouville):
                j_dag = dagger_index(j, dim_hilbert)
                eliminate = to_eliminate[i, j] or to_eliminate[i_dag, j_dag]
                to_eliminate[i, j] = eliminate
                to_eliminate[i_dag, j_dag] = eliminate

        diagonal = np.diag(L_0)
        equal_eigs = np.abs(diagonal.reshape(-1, 1) - diagonal) < 1e-12
        to_eliminate[equal_eigs] = False

        if not to_eliminate.any():
            candidates = np.argwhere(~equal_eigs)
            candidates = np.array(
                [ij for ij in candidates if ij[0] not in trace_idx and ij[0] != ij[1]]
            )
            i, j = candidates[0]
            to_eliminate[i, j] = True
            to_eliminate[dagger_index(i, dim_hilbert), dagger_index(j, dim_hilbert)] = (
                True
            )

        L_tilde, U, U_inv = block_diagonalize(
            [L_0, L_1],
            subspace_indices=np.zeros(dim_liouville, dtype=int),
            hermitian=False,
            fully_diagonalize=to_eliminate,
        )

        trace_row = np.zeros(dim_liouville, dtype=complex)
        trace_row[trace_idx] = 1.0
        trace_series = BlockSeries(
            data={(0, 0, 0): trace_row.reshape(1, -1)},
            shape=(1, 1),
            n_infinite=1,
            name="tr",
        )
        zero_series = BlockSeries(
            data={},
            shape=(1, 1),
            n_infinite=1,
            name="zero_trace",
        )

        max_order = 3
        compare_series(
            cauchy_dot_product(trace_series, U),
            trace_series,
            (max_order,),
            atol=1e-10,
            rtol=0,
        )
        compare_series(
            cauchy_dot_product(trace_series, U_inv),
            trace_series,
            (max_order,),
            atol=1e-10,
            rtol=0,
        )
        compare_series(
            cauchy_dot_product(trace_series, L_tilde),
            zero_series,
            (max_order,),
            atol=1e-10,
            rtol=0,
        )

        hermiticity_conjugation = np.zeros((dim_liouville, dim_liouville), dtype=complex)
        for idx in range(dim_liouville):
            hermiticity_conjugation[dagger_index(idx, dim_hilbert), idx] = 1.0

        # Input perturbative terms must already preserve Hermiticity.
        for L_order in (L_0, L_1):
            np.testing.assert_allclose(
                hermiticity_conjugation @ L_order.conj(),
                L_order @ hermiticity_conjugation,
                atol=1e-10,
                rtol=0,
            )

        for order in range(max_order + 1):
            L_order = to_dense(L_tilde[(0, 0, order)], dim_liouville)
            np.testing.assert_allclose(
                hermiticity_conjugation @ L_order.conj(),
                L_order @ hermiticity_conjugation,
                atol=1e-10,
                rtol=0,
            )


def test_block_diagonalize_nonhermitian_rejects_legacy_two_block_solver():
    rng = np.random.default_rng()
    h_0 = np.diag(np.array([-3.0, 0.7, 1.4, 2.6], dtype=float)).astype(complex)
    h_1 = _complex_normal(rng, (4, 4))
    subspace_indices = np.array([0, 1, 1, 1], dtype=int)
    denominators = {
        (1, 3): h_0[:1, :1].diagonal().reshape(-1, 1) - h_0[1:, 1:].diagonal(),
        (3, 1): h_0[1:, 1:].diagonal().reshape(-1, 1) - h_0[:1, :1].diagonal(),
    }

    def solve_sylvester(rhs):
        if rhs is zero:
            return zero
        return rhs / denominators[rhs.shape]

    with pytest.raises(
        NotImplementedError,
        match="require `solve_sylvester\\(Y, index\\)`",
    ):
        block_diagonalize(
            [h_0, h_1],
            subspace_indices=subspace_indices,
            hermitian=False,
            solve_sylvester=solve_sylvester,
        )


def test_block_diagonalize_nonhermitian_accepts_bare_sympy_matrix():
    x = sympy.Symbol("x")
    H_tilde, U, U_inv = block_diagonalize(
        sympy.Matrix([[1 + x, x], [0, 2]]),
        symbols=(x,),
        subspace_indices=[0, 1],
        hermitian=False,
    )

    assert H_tilde[(0, 1, 1)] is zero
    assert H_tilde[(1, 0, 1)] is zero
    assert H_tilde[(0, 1, 2)] is zero
    assert H_tilde[(1, 0, 2)] is zero

    assert U[(0, 1, 1)] == sympy.Matrix([[x]])
    assert U[(0, 1, 2)] == sympy.Matrix([[x**2]])
    assert U_inv[(0, 1, 1)] == sympy.Matrix([[-x]])
    assert U_inv[(0, 1, 2)] == sympy.Matrix([[-(x**2)]])


@pytest.mark.parametrize("index", [(0, 1), (1, 0)])
def test_nonhermitian_direct_solver_supports_both_offdiagonal_orientations(
    index,
) -> None:
    pytest.importorskip("mumps", reason="python-mumps is not installed")

    n = 300
    a_dim = 5
    rng, h_0, eigvals, eigvecs, eigvecs_rest = _make_direct_solver_case(n, a_dim)
    direct = solve_sylvester_direct(h_0, [eigvecs], nonhermitian=True)

    if index == (0, 1):
        rhs = rng.standard_normal(size=(a_dim, n - a_dim)) + 1j * rng.standard_normal(
            size=(a_dim, n - a_dim)
        )
        rhs = rhs @ Dagger(eigvecs_rest)
        expected = (
            (rhs @ eigvecs_rest) / (eigvals[:a_dim].reshape(-1, 1) - eigvals[a_dim:])
        ) @ Dagger(eigvecs_rest)
    else:
        rhs = rng.standard_normal(size=(n - a_dim, a_dim)) + 1j * rng.standard_normal(
            size=(n - a_dim, a_dim)
        )
        rhs = eigvecs_rest @ rhs
        expected = eigvecs_rest @ (
            (Dagger(eigvecs_rest) @ rhs)
            / (eigvals[a_dim:].reshape(-1, 1) - eigvals[:a_dim])
        )

    np.testing.assert_allclose(expected, direct(rhs, index))


def test_nonhermitian_direct_solver_requires_flag_for_left_implicit_solve() -> None:
    pytest.importorskip("mumps", reason="python-mumps is not installed")

    n = 40
    a_dim = 4
    rng, h_0, _, eigvecs, eigvecs_rest = _make_direct_solver_case(n, a_dim)
    direct = solve_sylvester_direct(h_0, [eigvecs])

    rhs = rng.standard_normal(size=(n - a_dim, a_dim)) + 1j * rng.standard_normal(
        size=(n - a_dim, a_dim)
    )
    rhs = eigvecs_rest @ rhs

    with pytest.raises(NotImplementedError, match="nonhermitian=True"):
        direct(rhs, (1, 0))


@pytest.mark.parametrize("index", [(0, 1), (1, 0)])
def test_nonhermitian_direct_solver_supports_biorthogonal_subspaces(index) -> None:
    pytest.importorskip("mumps", reason="python-mumps is not installed")

    n = 40
    a_dim = 4
    rng, h_0, eigvals, explicit, implicit, _ = _make_biorthogonal_direct_solver_case(
        n, a_dim
    )
    right_a, left_a = explicit
    right_rest, left_rest = implicit
    direct = solve_sylvester_direct(h_0, [(right_a, left_a)], nonhermitian=True)

    if index == (0, 1):
        rhs = _complex_normal(rng, (a_dim, n - a_dim)) @ Dagger(left_rest)
        expected = (
            (rhs @ right_rest) / (eigvals[:a_dim].reshape(-1, 1) - eigvals[a_dim:])
        ) @ Dagger(left_rest)
    else:
        rhs = right_rest @ _complex_normal(rng, (n - a_dim, a_dim))
        expected = right_rest @ (
            (Dagger(left_rest) @ rhs) / (eigvals[a_dim:].reshape(-1, 1) - eigvals[:a_dim])
        )

    np.testing.assert_allclose(expected, direct(rhs, index))


def test_block_diagonalize_nonhermitian_accepts_complete_subspace_eigenvectors() -> None:
    n = 40
    a_dim = 4
    rng, h_0, eigvals, eigvecs_a, eigvecs_rest = _make_direct_solver_case(n, a_dim)
    eigvecs = np.hstack((eigvecs_a, eigvecs_rest))
    subspace_indices = np.array([0] * a_dim + [1] * (n - a_dim))
    h_1 = _complex_normal(rng, (n, n)).astype(complex)

    H_tilde_subspaces, *_ = block_diagonalize(
        [h_0.toarray(), h_1],
        subspace_eigenvectors=[eigvecs_a, eigvecs_rest],
        hermitian=False,
    )
    H_tilde_indices, *_ = block_diagonalize(
        [np.diag(eigvals), Dagger(eigvecs) @ h_1 @ eigvecs],
        subspace_indices=subspace_indices,
        hermitian=False,
    )

    compare_series(H_tilde_subspaces, H_tilde_indices, (2,), atol=1e-10, rtol=1e-11)


def test_block_diagonalize_nonhermitian_accepts_biorthogonal_subspace_eigenvectors():
    n = 10
    a_dim = 3
    rng, h_0, eigvals, explicit, implicit, full_basis = (
        _make_biorthogonal_direct_solver_case(n, a_dim)
    )
    right_a, left_a = explicit
    right_rest, left_rest = implicit
    right_full, left_full = full_basis
    subspace_indices = np.array([0] * a_dim + [1] * (n - a_dim))
    h_1 = _complex_normal(rng, (n, n))

    H_tilde_pairs, *_ = block_diagonalize(
        [h_0.toarray(), h_1],
        subspace_eigenvectors=[(right_a, left_a), (right_rest, left_rest)],
        hermitian=False,
    )
    H_tilde_indices, *_ = block_diagonalize(
        [np.diag(eigvals), Dagger(left_full) @ h_1 @ right_full],
        subspace_indices=subspace_indices,
        hermitian=False,
    )

    compare_series(H_tilde_pairs, H_tilde_indices, (2,), atol=1e-10, rtol=1e-11)


def test_block_diagonalize_nonhermitian_implicit_direct_solver_supports_biorthogonal_pairs() -> (
    None
):
    pytest.importorskip("mumps", reason="python-mumps is not installed")

    n = 24
    a_dim = 4
    rng, h_0, _, explicit, implicit, _ = _make_biorthogonal_direct_solver_case(n, a_dim)
    right_a, left_a = explicit
    right_rest, left_rest = implicit
    h_1 = _complex_normal(rng, (n, n))

    H_tilde_implicit, *_ = block_diagonalize(
        [h_0, h_1],
        subspace_eigenvectors=[(right_a, left_a)],
        hermitian=False,
    )
    H_tilde_explicit, *_ = block_diagonalize(
        [h_0.toarray(), h_1],
        subspace_eigenvectors=[(right_a, left_a), (right_rest, left_rest)],
        hermitian=False,
    )

    compare_series(H_tilde_implicit[0, 0], H_tilde_explicit[0, 0], (2,), atol=1e-6)


def test_block_diagonalize_nonhermitian_rejects_implicit_kpm():
    rng = np.random.default_rng()
    h_0 = sparse.diags(np.linspace(-3.0, 3.0, 8)).astype(complex)
    h_1 = _complex_normal(rng, (8, 8))
    subspace_eigenvectors = [np.eye(8, dtype=complex)[:, :2]]

    with pytest.raises(NotImplementedError, match="does not support the KPM solver"):
        block_diagonalize(
            [h_0, h_1],
            subspace_eigenvectors=subspace_eigenvectors,
            hermitian=False,
            direct_solver=False,
        )
