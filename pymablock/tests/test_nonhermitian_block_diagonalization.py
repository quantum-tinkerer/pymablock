import numpy as np
import pytest
import sympy

from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main, nonhermitian
from pymablock.block_diagonalization import block_diagonalize
from pymablock.series import BlockSeries, cauchy_dot_product, zero

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


def test_block_diagonalize_nonhermitian_accepts_legacy_two_block_solver():
    rng = np.random.default_rng(97531)
    h_0 = np.diag(np.array([-3.0, 0.7, 1.4, 2.6], dtype=float)).astype(complex)
    h_1 = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    subspace_indices = np.array([0, 1, 1, 1], dtype=int)

    denominators = {
        (1, 3): h_0[:1, :1].diagonal().reshape(-1, 1) - h_0[1:, 1:].diagonal(),
        (3, 1): h_0[1:, 1:].diagonal().reshape(-1, 1) - h_0[:1, :1].diagonal(),
    }

    def solve_sylvester(rhs):
        if rhs is zero:
            return zero
        return rhs / denominators[rhs.shape]

    H_tilde, *_ = block_diagonalize(
        [h_0, h_1],
        subspace_indices=subspace_indices,
        hermitian=False,
        solve_sylvester=solve_sylvester,
    )

    assert H_tilde[(0, 1, 1)] is zero
    assert H_tilde[(1, 0, 1)] is zero


def test_block_diagonalize_nonhermitian_accepts_bare_sympy_matrix():
    x = sympy.Symbol("x")
    H_tilde, *_ = block_diagonalize(
        sympy.Matrix([[1, x], [0, 2]]),
        symbols=(x,),
        subspace_indices=[0, 1],
        hermitian=False,
    )

    assert H_tilde[(0, 1, 1)] is zero
    assert H_tilde[(1, 0, 1)] is zero
