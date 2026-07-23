# ---
# jupyter:
#   jupytext:
#     formats: md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

"""Executable companion to the analytical Ising-chain MPO tutorial."""

# %% [markdown]
# # Analytical Ising-chain benchmark
#
# The longitudinal-field Ising chain is coupled to a detuned two-level
# sector. Since the Ising Hamiltonian commutes with the coupling operator,
# the effective Hamiltonian is known analytically.

# %%
# %%time

from functools import reduce
from itertools import product

import numpy as np
from tenpy.networks.site import SpinHalfSite
from tenpy_mpo_backend import TenpyMPOBackend, mpo_to_dense, product_mpo

from pymablock import block_diagonalize
from pymablock.mpo import BackendMPO, make_mpo_sylvester_solver

# %% [markdown]
# ## Build the Ising MPO

# %%
# %%time

L = 6
J = 0.7
h = 0.2
Delta = 5.0

sites = [SpinHalfSite(conserve=None) for _ in range(L)]
identity = np.eye(2)
z = np.diag([1.0, -1.0])

backend = TenpyMPOBackend(
    chi_max=64,
    svd_min=1e-12,
    krylov_dimension=8,
    max_restarts=10,
    solver_tolerance=1e-9,
)


def product_term(operators):
    """Construct a product operator from ``site: local_operator`` entries."""
    return product_mpo(
        sites,
        [operators.get(site, identity) for site in range(L)],
    )


def add_all(operators):
    """Add and compress a nonempty sequence of MPOs."""
    return reduce(backend.add, operators)


magnetization = add_all([product_term({site: z}) for site in range(L)])
ising_bonds = add_all([product_term({site: z, site + 1: z}) for site in range(L - 1)])
ising = backend.add(
    backend.scale(ising_bonds, -J),
    backend.scale(magnetization, -h),
)
identity_mpo = product_term({})
upper_block = backend.add(
    ising,
    backend.scale(identity_mpo, Delta),
)

# %% [markdown]
# ## Compute the effective Hamiltonian

# %%
# %%time

wrapped_ising = BackendMPO(ising, backend)
wrapped_upper = BackendMPO(upper_block, backend)
wrapped_magnetization = BackendMPO(magnetization, backend)

solve_sylvester = make_mpo_sylvester_solver(
    [wrapped_ising, wrapped_upper],
    backend.solve_sylvester,
    max_relative_residual=1e-8,
)

H_tilde, _, _ = block_diagonalize(
    [
        [[wrapped_ising, 0], [0, wrapped_upper]],
        [
            [0, wrapped_magnetization],
            [wrapped_magnetization, 0],
        ],
    ],
    solve_sylvester=solve_sylvester,
)
H_AA_2 = H_tilde[0, 0, 2]

# %% [markdown]
# Compare with the analytical coefficient.

# %%
# %%time

expected_h_aa_2 = backend.scale(
    backend.matmul(magnetization, magnetization),
    -1 / Delta,
)
computed_dense = mpo_to_dense(H_AA_2.operator)
expected_dense = mpo_to_dense(expected_h_aa_2)
coefficient_relative_error = np.linalg.norm(
    computed_dense - expected_dense
) / np.linalg.norm(expected_dense)
maximum_bond_dimension = max(
    max(record.output_bond_dimensions) for record in backend.compression_records
)
sylvester_residual = backend.solver_records[-1].relative_residuals[-1]

assert coefficient_relative_error < 1e-8
assert sylvester_residual < 1e-8
assert maximum_bond_dimension <= backend.chi_max

# %% [markdown]
# ## Check the fourth-order truncation error

# %%
# %%time

configurations = np.asarray(list(product((1.0, -1.0), repeat=L)))
magnetizations = configurations.sum(axis=1)
ising_energies = (
    -J
    * np.sum(
        configurations[:, :-1] * configurations[:, 1:],
        axis=1,
    )
    - h * magnetizations
)


def maximum_energy_error(coupling):
    """Return the largest second-order energy error over all spin states."""
    exact = (
        ising_energies
        + (Delta - np.sqrt(Delta**2 + 4 * coupling**2 * magnetizations**2)) / 2
    )
    second_order = ising_energies - coupling**2 * magnetizations**2 / Delta
    return np.max(np.abs(exact - second_order))


couplings = np.array([0.4, 0.2, 0.1, 0.05])
energy_errors = np.array([maximum_energy_error(coupling) for coupling in couplings])
observed_orders = np.log2(energy_errors[:-1] / energy_errors[1:])

assert observed_orders[-1] > 3.9
