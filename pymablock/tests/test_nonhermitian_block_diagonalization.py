import numpy as np
import pytest

from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main
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


def _solve_sylvester(energies: list[float]):
    def solve(rhs, index):
        if rhs is zero:
            return zero
        return rhs / (energies[index[0]] - energies[index[1]])

    return solve


def _run_nonhermitian(H: BlockSeries, energies: list[float]) -> dict[str, BlockSeries]:
    series, _ = series_computation(
        {"H": H},
        algorithm=main,
        scope={
            "solve_sylvester": _solve_sylvester(energies),
            "two_block_optimized": False,
            "commuting_blocks": [False] * H.shape[0],
            "hermitian_problem": False,
        },
    )
    return series


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

    for order in range(1, max_order + 1):
        for i in range(n_blocks):
            for j in range(n_blocks):
                if i == j:
                    continue
                np.testing.assert_allclose(
                    _as_array(H_tilde[(i, j, order)]), 0, atol=1e-10, rtol=0
                )
