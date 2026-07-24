# ---
# jupyter:
#   jupytext:
#     formats: md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Minimal full-MPO validation
#
# Conceptually, this tutorial demonstrates the complete interface between Pymablock and a tensor-network backend: TeNPy supplies MPO arithmetic and the Sylvester solver, while Pymablock organizes the perturbation series.
# Physically, we compute how virtual transitions through a detuned manifold shift and split the four spin levels in the retained manifold $A$ as the coupling increases.
#
# We use two spin-$\frac12$ sites so that the first-order transformation and second-order effective Hamiltonian can be checked against dense Pymablock.
# Because we want the complete low-energy spectrum, we represent these coefficients as full MPOs rather than only applying them to selected states.
# Dense matrices enter only as a small-system reference.
#
# ## Construct the MPO blocks
#
# We first import the TeNPy backend defined alongside this executable tutorial.
# The example uses no conserved charges because the adapter deliberately keeps its first implementation focused on the MPO algebra and Sylvester solver.

# %%
# %%time
import matplotlib.pyplot as plt
import numpy as np
from tenpy.networks.site import SpinHalfSite
from tenpy_mpo_backend import (
    TenpyMPOBackend,
    mpo_to_dense,
    product_mpo,
)

from pymablock import block_diagonalize
from pymablock.mpo import BackendMPO, make_mpo_sylvester_solver

# %% [markdown]
# We treat two spins with a low-energy manifold $A$ and a detuned excited manifold $B$.
# Both manifolds contain the same four spin states, and spin-flip tunnelling connects them.
# This is the elementary setting behind dispersive elimination: the excited manifold is barely occupied, yet virtual visits to it shift the spectrum measured within $A$.
# We retain $A$ and perturbatively decouple $B$.
#
# We expand in the dimensionless coupling $\lambda$:
#
# $$
# H(\lambda)=H_0+\lambda V,\qquad
# H_0=
# \begin{pmatrix}
# A&0\\
# 0&B
# \end{pmatrix},
# \qquad
# V=
# \begin{pmatrix}
# 0&T\\
# T^\dagger&0
# \end{pmatrix},
# $$
#
# with
#
# $$
# A=0.3(Z_1+Z_2),\qquad B=A+3I,
# \qquad
# T=X_1+0.2X_2.
# $$
#
# Here $X_i$ and $Z_i$ are Pauli operators on spin $i$, $I$ is the two-spin identity, and all coefficients use the same energy unit.
# The two sectors have the same longitudinal field, but $B$ lies three energy units above $A$.
# The perturbation changes sector while flipping spin 1 with amplitude $1$ or spin 2 with amplitude $0.2$.
#
# Our final physical quantity is the four-level spectrum within $A$ as a function of $\lambda$.
# The second-order effective Hamiltonian produces this spectrum, while the first-order transformation describes the accompanying admixture of $B$.
#
# At first order, Pymablock obtains the off-diagonal transformation from
#
# $$
# A U_{AB}^{(1)}-U_{AB}^{(1)}B=-T.
# $$
#
# The spectra of $A$ and $B$ are disjoint, so this Sylvester equation has a unique solution.
# At second order, $T^\dagger$ takes a state from $A$ to $B$, and $T$ brings it back.
# These virtual excursions shift and mix the four spin states within $A$, even though $V$ has no matrix element inside that manifold.

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
#
# We next wrap each native TeNPy MPO in {autolink}`~pymablock.mpo.BackendMPO`.
# The wrapper makes TeNPy's compressed arithmetic available to Pymablock without adding TeNPy as a core dependency.
# The backend aims for a relative Sylvester residual below $10^{-9}$, while the adapter independently rejects any result above $10^{-8}$.
# This second threshold prevents an unconverged compressed result from entering the perturbation series.
# For a generic equation $A\mathcal X-\mathcal X B=Y$, where $\mathcal X$ is the unknown operator and $Y$ is the source assembled by Pymablock at that order, both thresholds use
#
# $$
# \frac{\lVert Y-(A\mathcal X-\mathcal X B)\rVert_F}{\lVert Y\rVert_F}.
# $$

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

# %% [markdown]
# Pymablock stores each power of $\lambda$ separately, so $\lambda$ does not appear as a numerical variable in the code.
# In `U_mpo[0, 1, 1]`, the first two indices select the $AB$ block and the last selects first order.
# Likewise, `H_tilde_mpo[0, 0, 2]` selects the second-order coefficient in the target $AA$ block.
# Requesting the first term runs the Sylvester solver; requesting the second also requires MPO multiplication.

# %%
# %%time
U_AB_1 = U_mpo[0, 1, 1]
H_AA_2 = H_tilde_mpo[0, 0, 2]

# %% [markdown]
# ## Verify the minimal calculation
#
# We finally contract these two-site MPOs to dense matrices only for validation.
# An independent dense Pymablock calculation provides the reference coefficients.

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

first_order_error = np.linalg.norm(mpo_to_dense(U_AB_1.operator) - U_dense[0, 1, 1])
second_order_error = np.linalg.norm(
    mpo_to_dense(H_AA_2.operator) - H_tilde_dense[0, 0, 2]
)

assert first_order_error < 1e-8
assert second_order_error < 1e-8

{
    "first-order error": first_order_error,
    "second-order error": second_order_error,
    "Sylvester residual": backend.solver_records[-1].relative_residuals[-1],
}

# %% [markdown]
# ## Low-energy spectrum
#
# The coefficient comparison validates the implementation.
# To see the physics contained in the result, we compare the four low-energy eigenvalues of the full Hamiltonian with those of the second-order effective Hamiltonian
#
# $$
# H_{\mathrm{eff},A}^{(2)}(\lambda)
# =A+\lambda^2\widetilde H_{AA}^{(2)}.
# $$
#
# The four levels correspond to the four spin states in the retained manifold.
# We restrict $\lambda$ to a range where the two manifolds remain well separated.
# Dense matrices enter only to construct this small exact reference spectrum; the perturbative coefficient comes from the MPO calculation.

# %%
# %%time
couplings = np.linspace(0, 0.6, 61)
n_target = dense_a.shape[0]
dense_h_aa_2 = mpo_to_dense(H_AA_2.operator)

exact_energies = np.empty((len(couplings), n_target))
effective_energies = np.empty_like(exact_energies)

for index, coupling in enumerate(couplings):
    full_hamiltonian = np.block(
        [
            [dense_a, coupling * dense_t],
            [coupling * dense_t.conj().T, dense_b],
        ]
    )
    exact_energies[index] = np.linalg.eigvalsh(full_hamiltonian)[:n_target]
    effective_energies[index] = np.linalg.eigvalsh(dense_a + coupling**2 * dense_h_aa_2)

fig, ax = plt.subplots(figsize=(6, 4))
for level in range(n_target):
    color = f"C{level}"
    ax.plot(couplings, exact_energies[:, level], color=color, linewidth=2)
    ax.plot(
        couplings,
        effective_energies[:, level],
        color=color,
        linestyle="--",
        linewidth=1.5,
    )

ax.plot([], [], color="0.25", linewidth=2, label="Exact full model")
ax.plot(
    [],
    [],
    color="0.25",
    linestyle="--",
    linewidth=1.5,
    label=r"MPO PT through $\lambda^2$",
)
ax.set_xlabel(r"Perturbation strength $\lambda$")
ax.set_ylabel("Low-energy eigenvalue")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# Virtual transitions shift all four levels and split the degeneracy at zero energy.
# The dashed curves reproduce the spectrum at weak coupling, showing that the MPO coefficient describes the complete retained manifold rather than one selected state.
# Their gradual departure at larger $\lambda$ reflects the omitted fourth- and higher-order terms.
#
# ## Conclusion
#
# This calculation establishes the minimal end-to-end path from TeNPy MPOs to a Pymablock perturbation series.
# Agreement of $U_{AB}^{(1)}$ validates the Sylvester solve, while agreement of $\widetilde H_{AA}^{(2)}$ also validates MPO multiplication.
# The spectrum shows the physical content of that coefficient: virtual transitions shift and split every state in the retained manifold.
#
# The model is deliberately too small to demonstrate a scaling advantage.
# The [large-chain tutorial](tenpy_mpo_ising.md) shows the regime where a dense operator is impossible but the perturbative coefficients remain compact MPOs.
