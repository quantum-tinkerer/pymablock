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

"""Executable implicit-MPS perturbation tutorial for the transverse-field Ising chain."""

# %% [markdown]
# # Implicit MPS perturbation theory
#
# The transverse-field Ising chain is exactly solvable by a Jordan--Wigner
# transformation. Here it provides a check of perturbation theory around its
# two ferromagnetic product states without constructing the full unitary MPO.

# %% [markdown]
# ## Model and analytical result
#
# For an open chain,
#
# $$
# H(g)=-J\sum_{i=1}^{L-1}Z_iZ_{i+1}-g\sum_{i=1}^{L}X_i.
# $$
#
# At $g=0$, the all-up and all-down states span the degenerate ground space.
# Flipping an edge spin costs $2J$, while flipping an interior spin costs
# $4J$. Ordinary second-order perturbation theory therefore gives the same
# diagonal correction to both ground states,
#
# $$
# E^{(2)}
# =-\frac{2}{2J}-\frac{L-2}{4J}
# =-\frac{L+2}{4J}.
# $$
#
# The off-diagonal coefficient vanishes at second order for $L>2$.

# %%
# %%time

from functools import reduce

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy_implicit_backend import TenpyImplicitBackend
from tenpy_mpo_backend import product_mpo

from pymablock.implicit import block_diagonalize_implicit

# %% [markdown]
# ## Build the MPO and model space
#
# Only the two retained states are stored explicitly. The complement is
# represented through projected MPO applications.

# %%
# %%time

L = 8
J = 0.7

sites = [SpinHalfSite(conserve=None) for _ in range(L)]
identity = np.eye(2)
x = np.array([[0.0, 1.0], [1.0, 0.0]])
z = np.diag([1.0, -1.0])

backend = TenpyImplicitBackend(
    chi_max=32,
    svd_min=1e-12,
    max_sweeps=10,
    solver_tolerance=1e-9,
)


def product_term(operators):
    """Construct a Pauli string from ``site: local_operator`` entries."""
    return product_mpo(
        sites,
        [operators.get(site, identity) for site in range(L)],
    )


def add_all(operators):
    """Add a nonempty sequence of MPOs."""
    return reduce(backend.mpo_backend.add, operators)


h_0 = backend.mpo_backend.scale(
    add_all([product_term({site: z, site + 1: z}) for site in range(L - 1)]),
    -J,
)
perturbation = backend.mpo_backend.scale(
    add_all([product_term({site: x}) for site in range(L)]),
    -1,
)
references = [
    MPS.from_product_state(
        sites,
        [orientation] * L,
        bc="finite",
        unit_cell_width=L,
    )
    for orientation in ("up", "down")
]

# %% [markdown]
# ## Compute the perturbative coefficient
#
# Pymablock constructs its usual lazy series, but complement-space columns are
# MPSs. Whenever a new column is needed, the backend solves
#
# $$
# Q(H_0-E_0)Q|\eta\rangle=|S\rangle
# $$
#
# with constrained two-site sweeps.

# %%
# %%time

h_tilde, unitary, _ = block_diagonalize_implicit(
    [h_0, perturbation],
    references,
    backend,
    backend.solve_shifted,
    max_relative_residual=1e-8,
)
second_order = h_tilde[0, 0, 2].dense
first_order_columns = unitary[1, 0, 1].states

# %% [markdown]
# Compare the resulting $2\times2$ coefficient with the spin-flip
# denominators. This checks both shifted solves and the overlap products that
# feed the effective Hamiltonian.

# %%
# %%time

expected_second_order = -(L + 2) / (4 * J) * np.eye(2)
np.testing.assert_allclose(second_order, expected_second_order, atol=1e-10)

maximum_residual = max(record.relative_residuals[-1] for record in backend.solver_records)
maximum_orthogonality_error = max(
    record.orthogonality_errors[-1] for record in backend.solver_records
)
maximum_bond_dimension = max(max(column.chi) for column in first_order_columns)

assert maximum_residual < 1e-8
assert maximum_orthogonality_error < 1e-8
assert maximum_bond_dimension <= backend.chi_max

# %% [markdown]
# ## Compare with the exact free-fermion energy
#
# For the open chain, the exact ground-state energy is minus the sum of the
# singular values of the bidiagonal Jordan--Wigner matrix with diagonal $g$
# and subdiagonal $J$. If the second-order coefficient is correct, the error
# after adding $g^2E^{(2)}$ must decrease as $g^4$.

# %%
# %%time


def exact_ground_energy(field):
    """Return the finite-chain free-fermion ground-state energy."""
    jordan_wigner = np.diag(np.full(L, field))
    jordan_wigner += np.diag(np.full(L - 1, J), k=-1)
    return -np.linalg.svd(jordan_wigner, compute_uv=False).sum()


fields = np.array([0.2, 0.1, 0.05, 0.025])
unperturbed_energy = -J * (L - 1)
second_order_energy = expected_second_order[0, 0]
errors = np.array(
    [
        abs(
            exact_ground_energy(field)
            - (unperturbed_energy + field**2 * second_order_energy)
        )
        for field in fields
    ]
)
observed_orders = np.log2(errors[:-1] / errors[1:])

assert observed_orders[-1] > 3.99
