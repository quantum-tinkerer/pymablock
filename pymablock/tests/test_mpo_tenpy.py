from functools import reduce

import numpy as np
import pytest

tenpy = pytest.importorskip("tenpy", minversion="1.1")

from tenpy.networks.mps import MPS  # noqa: E402
from tenpy.networks.site import SpinHalfSite  # noqa: E402

from docs.source.tutorial.tenpy_implicit_backend import (  # noqa: E402
    TenpyImplicitBackend,
)
from pymablock import block_diagonalize  # noqa: E402
from pymablock.backends.tenpy import (  # noqa: E402
    TenpyMPOBackend,
    mpo_to_dense,
    mpo_to_mps,
    mps_to_mpo,
    product_mpo,
)
from pymablock.implicit import block_diagonalize_implicit  # noqa: E402
from pymablock.mpo import (  # noqa: E402
    BackendMPO,
    make_mpo_sylvester_solver,
)


@pytest.fixture
def mpo_problem():
    sites = [SpinHalfSite(conserve=None) for _ in range(2)]
    identity = np.eye(2)
    x = np.array([[0, 1], [1, 0]], dtype=float)
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1.0, -1.0])
    return sites, identity, x, y, z


def test_tenpy_mpo_algebra_and_roundtrip(mpo_problem):
    sites, identity, x, y, z = mpo_problem
    backend = TenpyMPOBackend()
    left = product_mpo(sites, [x, y])
    right = product_mpo(sites, [z, x])

    np.testing.assert_allclose(
        mpo_to_dense(backend.add(left, right)),
        np.kron(x, y) + np.kron(z, x),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mpo_to_dense(backend.matmul(left, right)),
        np.kron(x, y) @ np.kron(z, x),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mpo_to_dense(backend.scale(left, 0.3j)),
        0.3j * np.kron(x, y),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mpo_to_dense(backend.adjoint(backend.scale(left, 0.3j))),
        (0.3j * np.kron(x, y)).conj().T,
        atol=1e-12,
    )

    roundtrip = mps_to_mpo(mpo_to_mps(backend.add(left, right)), sites)
    np.testing.assert_allclose(
        mpo_to_dense(roundtrip),
        np.kron(x, y) + np.kron(z, x),
        atol=1e-12,
    )
    assert max(roundtrip.chi) <= backend.chi_max


def test_tenpy_mpo_exact_cancellation(mpo_problem):
    sites, _, x, y, _ = mpo_problem
    backend = TenpyMPOBackend()
    operator = product_mpo(sites, [x, y])

    result = backend.add(operator, backend.scale(operator, -1))

    np.testing.assert_array_equal(mpo_to_dense(result), 0)
    assert result.chi == [1, 1, 1]
    assert backend.compression_records[-1].truncation_error == 0

    solve_result = backend.solve_sylvester(
        operator,
        operator,
        result,
        (0, 1, 2),
    )
    assert solve_result.converged
    assert solve_result.relative_residual == 0
    np.testing.assert_array_equal(mpo_to_dense(solve_result.operator), 0)


def _minimal_problem(mpo_problem):
    sites, identity, x, _, z = mpo_problem
    backend = TenpyMPOBackend(
        chi_max=64,
        svd_min=1e-12,
        krylov_dimension=8,
        max_restarts=10,
        solver_tolerance=1e-9,
    )
    identity_mpo = product_mpo(sites, [identity, identity])
    z_left = product_mpo(sites, [z, identity])
    z_right = product_mpo(sites, [identity, z])
    x_left = product_mpo(sites, [x, identity])
    x_right = product_mpo(sites, [identity, x])
    block_a = backend.scale(backend.add(z_left, z_right), 0.3)
    block_b = backend.add(block_a, backend.scale(identity_mpo, 3))
    coupling = backend.add(x_left, backend.scale(x_right, 0.2))
    return backend, block_a, block_b, coupling


def test_tenpy_mpo_sylvester(mpo_problem):
    backend, block_a, block_b, coupling = _minimal_problem(mpo_problem)
    sites, identity, _, y, _ = mpo_problem
    block_b = backend.add(
        block_b,
        backend.scale(product_mpo(sites, [y, identity]), 0.1),
    )
    result = backend.solve_sylvester(
        block_a,
        block_b,
        coupling,
        (0, 1, 1),
    )

    dense_a = mpo_to_dense(block_a)
    dense_b = mpo_to_dense(block_b)
    dense_rhs = mpo_to_dense(coupling)
    dense_solution = mpo_to_dense(result.operator)
    measured_residual = np.linalg.norm(
        dense_a @ dense_solution - dense_solution @ dense_b - dense_rhs
    ) / np.linalg.norm(dense_rhs)

    assert result.converged
    assert result.relative_residual < 1e-8
    assert measured_residual < 1e-8
    assert max(result.operator.chi) <= backend.chi_max


def test_tenpy_mpo_block_diagonalization_matches_dense(mpo_problem):
    backend, block_a, block_b, coupling = _minimal_problem(mpo_problem)
    wrapped_a = BackendMPO(block_a, backend)
    wrapped_b = BackendMPO(block_b, backend)
    wrapped_coupling = BackendMPO(coupling, backend)
    solve = make_mpo_sylvester_solver(
        [wrapped_a, wrapped_b],
        backend.solve_sylvester,
        max_relative_residual=1e-8,
    )

    H_tilde, U, _ = block_diagonalize(
        [
            [[wrapped_a, 0], [0, wrapped_b]],
            [[0, wrapped_coupling], [wrapped_coupling.adjoint(), 0]],
        ],
        solve_sylvester=solve,
    )

    dense_a = mpo_to_dense(block_a)
    dense_b = mpo_to_dense(block_b)
    dense_coupling = mpo_to_dense(coupling)
    dense_H_tilde, dense_U, _ = block_diagonalize(
        [
            [[dense_a, 0], [0, dense_b]],
            [[0, dense_coupling], [dense_coupling.conj().T, 0]],
        ]
    )
    np.testing.assert_allclose(
        mpo_to_dense(U[0, 1, 1].operator),
        dense_U[0, 1, 1],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        mpo_to_dense(H_tilde[0, 0, 2].operator),
        dense_H_tilde[0, 0, 2],
        atol=1e-8,
        rtol=0,
    )
    assert (
        max(max(record.output_bond_dimensions) for record in backend.compression_records)
        <= backend.chi_max
    )


def test_tenpy_implicit_ising_perturbation_matches_analytical_and_dense():
    length = 4
    coupling = 0.7
    longitudinal_field = 0.3
    sites = [SpinHalfSite(conserve=None) for _ in range(length)]
    identity = np.eye(2)
    x = np.array([[0, 1], [1, 0]], dtype=float)
    z = np.diag([1.0, -1.0])
    backend = TenpyImplicitBackend(
        chi_max=16,
        svd_min=1e-12,
        max_sweeps=10,
        solver_tolerance=1e-9,
    )

    def term(operators):
        return product_mpo(
            sites,
            [operators.get(site, identity) for site in range(length)],
        )

    def add_all(operators):
        return reduce(backend.mpo_backend.add, operators)

    h_0 = backend.mpo_backend.add(
        backend.mpo_backend.scale(
            add_all([term({site: z, site + 1: z}) for site in range(length - 1)]),
            -coupling,
        ),
        backend.mpo_backend.scale(
            add_all([term({site: z}) for site in range(length)]),
            -longitudinal_field,
        ),
    )
    perturbation = backend.mpo_backend.scale(
        add_all([term({site: x}) for site in range(length)]),
        -1,
    )
    reference = MPS.from_product_state(
        sites,
        ["up"] * length,
        bc="finite",
        unit_cell_width=length,
    )

    implicit_h_tilde, implicit_u, _ = block_diagonalize_implicit(
        [h_0, perturbation],
        [reference],
        backend,
        backend.solve_shifted,
        max_relative_residual=1e-8,
    )
    second_order = implicit_h_tilde[0, 0, 2].dense[0, 0]
    fourth_order = implicit_h_tilde[0, 0, 4].dense[0, 0]

    gaps = [
        2 * longitudinal_field + 2 * coupling * ((site > 0) + (site < length - 1))
        for site in range(length)
    ]
    np.testing.assert_allclose(second_order, -sum(1 / gap for gap in gaps), atol=1e-10)

    dense_h_0 = np.diag(np.diag(mpo_to_dense(h_0)))
    dense_perturbation = mpo_to_dense(perturbation)
    dense_h_tilde, _, _ = block_diagonalize(
        [dense_h_0, dense_perturbation],
        subspace_indices=np.array([0] + [1] * (2**length - 1)),
    )
    np.testing.assert_allclose(second_order, dense_h_tilde[0, 0, 2][0, 0], atol=1e-10)
    np.testing.assert_allclose(fourth_order, dense_h_tilde[0, 0, 4][0, 0], atol=1e-9)

    first_order_state = implicit_u[1, 0, 1].states[0]
    assert max(first_order_state.chi) <= backend.chi_max
    assert backend.solver_records
    assert all(
        record.relative_residuals[-1] < 1e-8
        and record.orthogonality_errors[-1] < 1e-8
        and max(record.maximum_bond_dimensions) <= backend.chi_max
        for record in backend.solver_records
    )


def test_open_boundary_mpo_preserves_model_operator():
    from tenpy.algorithms.exact_diag import ExactDiag
    from tenpy.models.spins import SpinChain

    from pymablock.backends.tenpy import open_boundary_mpo

    model = SpinChain(
        {
            "L": 4,
            "S": 0.5,
            "Jx": 0.7,
            "Jy": 0.4,
            "Jz": 1.2,
            "hz": 0.3,
            "conserve": None,
            "bc_MPS": "finite",
        }
    )
    state = MPS.from_product_state(
        model.lat.mps_sites(),
        ["up", "down", "up", "down"],
        bc="finite",
        unit_cell_width=4,
    )

    converted = open_boundary_mpo(model.H_MPO)

    assert converted.chi[0] == converted.chi[-1] == 1
    np.testing.assert_allclose(
        converted.expectation_value(state),
        model.H_MPO.expectation_value(state),
        atol=1e-12,
    )
    exact = ExactDiag(model)
    exact.build_full_H_from_mpo()
    np.testing.assert_allclose(
        mpo_to_dense(converted),
        exact.full_H.to_ndarray(),
        atol=1e-12,
    )
