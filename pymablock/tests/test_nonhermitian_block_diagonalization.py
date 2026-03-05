import numpy as np
import pytest
from scipy import sparse

from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main, nonhermitian
from pymablock.block_diagonalization import block_diagonalize
from pymablock.series import BlockSeries, cauchy_dot_product, one, zero

from .test_block_diagonalization import compare_series, identity_like


def _make_nonhermitian_h(
    n_blocks: int, max_order: int, seed: int = 17
) -> tuple[BlockSeries, list[float]]:
    energies = list(np.linspace(-2.0, 2.0, n_blocks))
    rng = np.random.default_rng(seed)

    data = {(i, i, 0): np.array([[energies[i]]], dtype=complex) for i in range(n_blocks)}
    for order in range(1, max_order + 1):
        for i in range(n_blocks):
            for j in range(n_blocks):
                data[(i, j, order)] = rng.normal(size=(1, 1)) + 1j * rng.normal(
                    size=(1, 1)
                )

    return BlockSeries(
        data=data, shape=(n_blocks, n_blocks), n_infinite=1, name="H"
    ), energies


def _make_nonhermitian_h_multivariate(
    n_blocks: int, wanted_orders: tuple[int, ...], seed: int = 31
) -> tuple[BlockSeries, list[float]]:
    energies = list(np.linspace(-2.0, 2.0, n_blocks))
    rng = np.random.default_rng(seed)

    n_infinite = len(wanted_orders)
    zero_order = (0,) * n_infinite
    data = {
        (i, i, *zero_order): np.array([[energies[i]]], dtype=complex)
        for i in range(n_blocks)
    }

    for order in np.ndindex(tuple(order + 1 for order in wanted_orders)):
        if order == zero_order:
            continue
        for i in range(n_blocks):
            for j in range(n_blocks):
                data[(i, j, *order)] = rng.normal(size=(1, 1)) + 1j * rng.normal(
                    size=(1, 1)
                )

    return BlockSeries(
        data=data, shape=(n_blocks, n_blocks), n_infinite=n_infinite, name="H"
    ), energies


def _solve_sylvester(energies: list[float]):
    def solve(rhs, index):
        if rhs is zero:
            return zero
        return rhs / (energies[index[0]] - energies[index[1]])

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


def _make_hermitian_h(
    n_blocks: int, max_order: int, seed: int = 23
) -> tuple[BlockSeries, list[float]]:
    energies = list(np.linspace(-2.0, 2.0, n_blocks))
    rng = np.random.default_rng(seed)

    data = {(i, i, 0): np.array([[energies[i]]], dtype=complex) for i in range(n_blocks)}
    for order in range(1, max_order + 1):
        for i in range(n_blocks):
            data[(i, i, order)] = np.array([[float(rng.normal())]], dtype=complex)
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                value = rng.normal() + 1j * rng.normal()
                data[(i, j, order)] = np.array([[value]], dtype=complex)
                data[(j, i, order)] = np.array([[np.conjugate(value)]], dtype=complex)

    return BlockSeries(
        data=data, shape=(n_blocks, n_blocks), n_infinite=1, name="H"
    ), energies


def _as_array(value):
    if value is zero:
        return np.zeros((1, 1), dtype=complex)
    return value


@pytest.mark.parametrize("n_blocks", [2, 3])
def test_nonhermitian_structural_first_order(n_blocks):
    H, energies = _make_nonhermitian_h(n_blocks=n_blocks, max_order=1)
    series = _run_nonhermitian(H, energies)

    H_tilde = series["H_tilde"]
    U = series["U"]
    U_inv = series["U†"]

    compare_series(cauchy_dot_product(U_inv, U), identity_like(U), (0,))
    compare_series(cauchy_dot_product(U, U_inv), identity_like(U), (0,))

    for i in range(n_blocks):
        for j in range(n_blocks):
            if i == j:
                continue
            np.testing.assert_allclose(
                _as_array(U_inv[(i, j, 1)] + U[(i, j, 1)]), 0, atol=1e-12, rtol=0
            )
            np.testing.assert_allclose(
                _as_array(H_tilde[(i, j, 1)]), 0, atol=1e-12, rtol=0
            )


@pytest.mark.parametrize(("n_blocks", "max_order"), [(2, 4), (3, 3)])
def test_nonhermitian_inverse_higher_order_and_multiblock(n_blocks, max_order):
    H, energies = _make_nonhermitian_h(n_blocks=n_blocks, max_order=max_order)
    series = _run_nonhermitian(H, energies)
    H_tilde = series["H_tilde"]
    U = series["U"]
    U_inv = series["U†"]

    compare_series(cauchy_dot_product(U_inv, U), identity_like(U), (max_order,))
    compare_series(cauchy_dot_product(U, U_inv), identity_like(U), (max_order,))
    compare_series(cauchy_dot_product(U_inv, H, U), H_tilde, (max_order,))
    compare_series(cauchy_dot_product(U, H_tilde, U_inv), H, (max_order,))

    for order in range(1, max_order + 1):
        for i in range(n_blocks):
            for j in range(n_blocks):
                if i == j:
                    continue
                np.testing.assert_allclose(
                    _as_array(H_tilde[(i, j, order)]), 0, atol=1e-10, rtol=0
                )


@pytest.mark.parametrize(("n_blocks", "wanted_orders"), [(2, (2, 1)), (3, (1, 1))])
def test_nonhermitian_multivariate_backtransform(n_blocks, wanted_orders):
    H, energies = _make_nonhermitian_h_multivariate(
        n_blocks=n_blocks, wanted_orders=wanted_orders
    )
    series = _run_nonhermitian(H, energies)
    H_tilde = series["H_tilde"]
    U = series["U"]
    U_inv = series["U†"]

    compare_series(cauchy_dot_product(U_inv, U), identity_like(U), wanted_orders)
    compare_series(cauchy_dot_product(U, U_inv), identity_like(U), wanted_orders)
    compare_series(cauchy_dot_product(U_inv, H, U), H_tilde, wanted_orders)
    compare_series(cauchy_dot_product(U, H_tilde, U_inv), H, wanted_orders)

    zero_order = (0,) * len(wanted_orders)
    for order in np.ndindex(tuple(order + 1 for order in wanted_orders)):
        if order == zero_order:
            continue
        for i in range(n_blocks):
            for j in range(n_blocks):
                if i == j:
                    continue
                np.testing.assert_allclose(
                    _as_array(H_tilde[(i, j, *order)]), 0, atol=1e-10, rtol=0
                )


@pytest.mark.parametrize(("n_blocks", "max_order"), [(2, 4), (3, 3)])
def test_nonhermitian_matches_hermitian_on_hermitian_input(n_blocks, max_order):
    H, energies = _make_hermitian_h(n_blocks=n_blocks, max_order=max_order)
    scope = {
        "solve_sylvester": _solve_sylvester(energies),
        "two_block_optimized": False,
        "commuting_blocks": [False] * n_blocks,
    }

    series_h, _ = series_computation({"H": H}, algorithm=main, scope=scope)
    series_nh, _ = series_computation({"H": H}, algorithm=nonhermitian, scope=scope)

    compare_series(series_h["H_tilde"], series_nh["H_tilde"], (max_order,), atol=1e-10)
    compare_series(series_h["U"], series_nh["U"], (max_order,), atol=1e-10)
    compare_series(series_h["U†"], series_nh["U†"], (max_order,), atol=1e-10)


def test_nonhermitian_arbitrary_asymmetric_mask():
    n = 6
    max_order = 3
    rng = np.random.default_rng(1234)
    energies = np.linspace(-3.0, 3.0, n)

    data = {(0, 0, 0): np.diag(energies).astype(complex)}
    for order in range(1, max_order + 1):
        data[(0, 0, order)] = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = BlockSeries(data=data, shape=(1, 1), n_infinite=1, name="H")

    to_eliminate = rng.random((n, n)) > 0.8
    np.fill_diagonal(to_eliminate, False)
    to_eliminate[0, 1] = True
    to_eliminate[1, 0] = False
    to_eliminate[2, 4] = False
    to_eliminate[4, 2] = True
    assert not np.array_equal(to_eliminate, to_eliminate.T)
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

    opposite_direction_nonzero = False
    for order in range(1, max_order + 1):
        block = H_tilde[(0, 0, order)]
        np.testing.assert_allclose(block[to_eliminate], 0, atol=1e-10, rtol=0)
        opposite_direction_nonzero = opposite_direction_nonzero or (
            not np.allclose(block[1, 0], 0, atol=1e-10, rtol=0)
        )

    assert opposite_direction_nonzero


def test_block_diagonalize_nonhermitian_accepts_asymmetric_mask():
    n = 6
    max_order = 3
    rng = np.random.default_rng(4321)

    h_0 = np.diag(np.linspace(-3.0, 3.0, n)).astype(complex)
    h_1 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))

    to_eliminate = rng.random((n, n)) > 0.8
    np.fill_diagonal(to_eliminate, False)
    to_eliminate[0, 1] = True
    to_eliminate[1, 0] = False
    assert not np.array_equal(to_eliminate, to_eliminate.T)

    H_tilde, *_ = block_diagonalize(
        [h_0, h_1],
        subspace_indices=np.zeros(n, dtype=int),
        hermitian=False,
        fully_diagonalize=to_eliminate,
    )

    for order in range(1, max_order + 1):
        block = H_tilde[(0, 0, order)]
        np.testing.assert_allclose(block[to_eliminate], 0, atol=1e-10, rtol=0)


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
        return np.kron(J.conj(), J) - 0.5 * (np.kron(eye, JdagJ) + np.kron(JdagJ.T, eye))

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
