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

"""Executable companion to the TeNPy MPO tutorial."""

# %% [markdown]
# # MPO perturbation theory with TeNPy
#
# This executable companion contains the code cells from `tenpy_mpo.md`.

# %%
# %%time

import numpy as np
from tenpy.networks.site import SpinHalfSite
from tenpy_mpo_backend import TenpyMPOBackend, mpo_to_dense, product_mpo

from pymablock import block_diagonalize
from pymablock.mpo import BackendMPO, make_mpo_sylvester_solver

# %% [markdown]
# ## Construct the MPO blocks

# %%
# %%time

sites = [SpinHalfSite(conserve=None) for _ in range(2)]
identity = np.eye(2)
x = np.array([[0.0, 1.0], [1.0, 0.0]])
z = np.diag([1.0, -1.0])

backend = TenpyMPOBackend(
    chi_max=64,
    svd_min=1e-12,
    krylov_dimension=8,
    max_restarts=10,
    solver_tolerance=1e-9,
)

identity_mpo = product_mpo(sites, [identity, identity])
z_1 = product_mpo(sites, [z, identity])
z_2 = product_mpo(sites, [identity, z])
x_1 = product_mpo(sites, [x, identity])
x_2 = product_mpo(sites, [identity, x])

A = backend.scale(backend.add(z_1, z_2), 0.3)
B = backend.add(A, backend.scale(identity_mpo, 3))
T = backend.add(x_1, backend.scale(x_2, 0.2))

# %% [markdown]
# ## Connect the backend to Pymablock

# %%
# %%time

wrapped_a = BackendMPO(A, backend)
wrapped_b = BackendMPO(B, backend)
wrapped_t = BackendMPO(T, backend)
solve_sylvester = make_mpo_sylvester_solver(
    [wrapped_a, wrapped_b],
    backend.solve_sylvester,
    max_relative_residual=1e-8,
)

H_tilde_mpo, U_mpo, _ = block_diagonalize(
    [
        [[wrapped_a, 0], [0, wrapped_b]],
        [[0, wrapped_t], [wrapped_t.adjoint(), 0]],
    ],
    solve_sylvester=solve_sylvester,
)
U_AB_1 = U_mpo[0, 1, 1]
H_AA_2 = H_tilde_mpo[0, 0, 2]

# %% [markdown]
# ## Verify the minimal calculation

# %%
# %%time

dense_a = mpo_to_dense(A)
dense_b = mpo_to_dense(B)
dense_t = mpo_to_dense(T)
H_tilde_dense, U_dense, _ = block_diagonalize(
    [
        [[dense_a, 0], [0, dense_b]],
        [[0, dense_t], [dense_t.conj().T, 0]],
    ]
)

np.testing.assert_allclose(
    mpo_to_dense(U_AB_1.operator),
    U_dense[0, 1, 1],
    atol=1e-8,
    rtol=0,
)
np.testing.assert_allclose(
    mpo_to_dense(H_AA_2.operator),
    H_tilde_dense[0, 0, 2],
    atol=1e-8,
    rtol=0,
)
