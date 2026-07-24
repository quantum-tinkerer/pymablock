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
# # Large-chain effective interactions
#
# Conceptually, this tutorial demonstrates the main advantage of full-MPO perturbation theory: it constructs reusable effective operators when dense operator algebra is impossible.
# Physically, we compute the energy shift induced by a detuned ancilla as a function of the Ising chain's total magnetization, including both second- and fourth-order corrections.
#
# We work with 24 spins and retain the ancilla ground sector.
# The spin Hilbert space has $2^{24}$ states, but the effective operators have bond dimensions of only a few.
#
# ## Model and analytical result
#
# Consider an open chain of $L$ spin-$\frac12$ sites.
# Here $Z_i$ is the Pauli matrix on site $i$, $J>0$ is the ferromagnetic nearest-neighbor coupling, and $h$ is a longitudinal field.
# The chain Hamiltonian and total magnetization are
#
# $$
# H_\mathrm{I}
# =-J\sum_{i=1}^{L-1}Z_iZ_{i+1}
# -h\sum_{i=1}^{L}Z_i,
# \qquad
# M_z=\sum_{i=1}^{L}Z_i.
# $$
#
# A concrete interpretation is a spin chain coupled collectively to a detuned two-level ancilla.
# We call the ancilla ground and excited states $A$ and $B$.
# In sector $A$ the spins evolve with $H_\mathrm{I}$; in sector $B$ the same spin dynamics costs an additional energy $\Delta>0$.
# The coupling $gM_z$ flips the ancilla, so a spin configuration with magnetization $m$ couples the two sectors with amplitude $gm$:
#
# $$
# H(g)=H_0+gV=
# \begin{pmatrix}
# H_\mathrm{I} & gM_z\\
# gM_z & H_\mathrm{I}+\Delta I
# \end{pmatrix}.
# $$
#
# Here $I$ is the identity on the spin chain.
# We treat $g$ perturbatively, retain the ancilla ground sector $A$, and eliminate virtual excursions through $B$.
# We choose $\Delta$ below to make the full spectra of the two unperturbed sectors disjoint.
#
# All terms in $H_\mathrm{I}$ and $M_z$ contain only $Z_i$, so the two operators commute.
# For a common eigenstate with Ising energy $E$ and magnetization $m$, only the auxiliary $2\times2$ problem remains:
#
# $$
# H_{E,m}=E I_2+
# \begin{pmatrix}
# 0&gm\\
# gm&\Delta
# \end{pmatrix}.
# $$
#
# Its lower eigenvalue is $E+[\Delta-\sqrt{\Delta^2+4g^2m^2}]/2$.
# Replacing $E$ and $m$ by the commuting operators gives the exact block continuously connected to $A$:
#
# $$
# H_\mathrm{exact}^{AA}(g)
# =H_\mathrm{I}
# +\frac{\Delta-\sqrt{\Delta^2+4g^2M_z^2}}{2}.
# $$
#
# Its expansion is
#
# $$
# H_\mathrm{exact}^{AA}(g)
# =H_\mathrm{I}
# -\frac{g^2}{\Delta}M_z^2
# +\frac{g^4}{\Delta^3}M_z^4
# +\mathcal{O}(g^6).
# $$
#
# We ask Pymablock to recover both displayed coefficients and evaluate them for every allowed magnetization $m$.
# The resulting curve $\delta E(m)$ is the physical output of the tutorial: it shows which spin sectors the virtual ancilla lowers most strongly and how fourth order corrects the second-order prediction.
#
# Already at second order,
#
# $$
# M_z^2=LI+2\sum_{i<j}Z_iZ_j,
# $$
#
# so the effective Hamiltonian contains an all-to-all interaction.
# The fourth-order term also contains long-range four-spin interactions.
# Virtual excitation of one ancilla therefore mediates interactions between spins at arbitrary separations.

# %%
# %%time
from functools import reduce

import numpy as np
from matplotlib import pyplot as plt
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy_mpo_backend import TenpyMPOBackend, mpo_to_mps, product_mpo

from pymablock import block_diagonalize
from pymablock.mpo import BackendMPO, make_mpo_sylvester_solver

# %% [markdown]
# ## Build a problem beyond dense operator storage
#
# For $L=24$, one complex dense operator would require $4$ pebibytes (PiB) of memory.
# The code below constructs the same Hamiltonian from compressed sums of Pauli strings.

# %%
# %%time
L = 24
J = 0.07
h = 0.02
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
detuned_block = backend.add(
    ising,
    backend.scale(identity_mpo, Delta),
)

hilbert_dimension = 2**L
dense_storage_pib = np.dtype(complex).itemsize * hilbert_dimension**2 / 2**50
spectral_radius_bound = J * (L - 1) + h * L
assert Delta > 2 * spectral_radius_bound

{
    "Hilbert-space dimension": hilbert_dimension,
    "dense complex operator [PiB]": dense_storage_pib,
    "upper bound on the Ising spectral width": 2 * spectral_radius_bound,
    "sector detuning": Delta,
    "Ising MPO bond dimension": max(ising.chi),
    "magnetization MPO bond dimension": max(magnetization.chi),
}

# %% [markdown]
# The bound $\lVert H_\mathrm{I}\rVert\leq J(L-1)+hL$ places its spectrum in an interval of width at most $2[J(L-1)+hL]$.
# Our detuning is larger than this width, so the two unperturbed block spectra are disjoint.
#
# ## Compute two perturbative orders
#
# Pymablock treats $g$ as the formal perturbative parameter, so the off-diagonal input is $M_z$ rather than $gM_z$.
# Requesting the fourth-order term evaluates all lower orders lazily.
# The calculation uses only compressed MPO algebra and matrix-free Sylvester solves.

# %%
# %%time
wrapped_ising = BackendMPO(ising, backend)
wrapped_detuned = BackendMPO(detuned_block, backend)
wrapped_magnetization = BackendMPO(magnetization, backend)

solve_sylvester = make_mpo_sylvester_solver(
    [wrapped_ising, wrapped_detuned],
    backend.solve_sylvester,
    max_relative_residual=1e-8,
)

H_tilde, _, _ = block_diagonalize(
    [
        [[wrapped_ising, 0], [0, wrapped_detuned]],
        [
            [0, wrapped_magnetization],
            [wrapped_magnetization, 0],
        ],
    ],
    solve_sylvester=solve_sylvester,
)
H_AA_2 = H_tilde[0, 0, 2]
H_AA_4 = H_tilde[0, 0, 4]

# %% [markdown]
# ## Verify the operators without dense matrices
#
# Vectorizing an MPO as an MPS preserves its Frobenius norm.
# We therefore measure each operator error from the MPS norm of the difference, without dense contraction.

# %%
# %%time
magnetization_squared = backend.matmul(magnetization, magnetization)
expected_h_aa_2 = backend.scale(
    magnetization_squared,
    -1 / Delta,
)
expected_h_aa_4 = backend.scale(
    backend.matmul(magnetization_squared, magnetization_squared),
    1 / Delta**3,
)


def relative_frobenius_error(computed, expected):
    """Return ``||computed - expected||_F / ||expected||_F``."""
    computed_vector = mpo_to_mps(computed)
    expected_vector = mpo_to_mps(expected)
    difference = computed_vector.add(
        expected_vector,
        alpha=1,
        beta=-1,
        cutoff=0,
    )
    difference_norm_squared = float(np.real(difference.overlap(difference)))
    expected_norm_squared = float(np.real(expected_vector.overlap(expected_vector)))
    return np.sqrt(max(difference_norm_squared, 0) / expected_norm_squared)


second_order_error = relative_frobenius_error(
    H_AA_2.operator,
    expected_h_aa_2,
)
fourth_order_error = relative_frobenius_error(
    H_AA_4.operator,
    expected_h_aa_4,
)
maximum_sylvester_residual = max(
    record.relative_residuals[-1] for record in backend.solver_records
)

assert second_order_error < 1e-8
assert fourth_order_error < 1e-8
assert maximum_sylvester_residual < 1e-8

{
    "second order": {
        "relative Frobenius error": second_order_error,
        "bond dimension": max(H_AA_2.operator.chi),
    },
    "fourth order": {
        "relative Frobenius error": fourth_order_error,
        "bond dimension": max(H_AA_4.operator.chi),
    },
    "Sylvester iterations": [record.iterations for record in backend.solver_records],
    "maximum Sylvester residual": maximum_sylvester_residual,
}

# %% [markdown]
# ## Magnetization-dependent energy shift
#
# The effective MPOs describe how virtual ancilla excitations shift the energy of a spin state.
# For a $Z$-basis product state with total magnetization $m$, the exact shift is
#
# $$
# \delta E_\mathrm{exact}(m)
# =\frac{\Delta-\sqrt{\Delta^2+4g^2m^2}}{2}.
# $$
#
# We evaluate the second- and fourth-order MPOs on one product MPS for every allowed $m$.
# This samples the effective operator across the chain Hilbert space using tensor-network contractions.
# The square-root expansion converges for every spin state when $2|g|L/\Delta<1$.
# We choose $g=0.08$, for which this ratio is $0.768$: close enough to show the truncation error, but inside the convergence radius.

# %%
# %%time
plot_coupling = 0.08
magnetizations = np.arange(-L, L + 1, 2)
product_states = [
    MPS.from_product_state(
        sites,
        ["up"] * ((L + magnetization) // 2) + ["down"] * ((L - magnetization) // 2),
        bc="finite",
        unit_cell_width=L,
    )
    for magnetization in magnetizations
]


def real_expectation(operator, state):
    """Evaluate a Hermitian MPO on an MPS."""
    value = operator.expectation_value(state)
    assert abs(np.imag(value)) < 1e-10
    return float(np.real(value))


second_order_shifts = plot_coupling**2 * np.array(
    [real_expectation(H_AA_2.operator, state) for state in product_states]
)
fourth_order_shifts = second_order_shifts + plot_coupling**4 * np.array(
    [real_expectation(H_AA_4.operator, state) for state in product_states]
)
exact_shifts = (Delta - np.sqrt(Delta**2 + 4 * plot_coupling**2 * magnetizations**2)) / 2

expansion_parameter = 2 * plot_coupling * L / Delta
second_order_max_error = np.max(np.abs(second_order_shifts - exact_shifts))
fourth_order_max_error = np.max(np.abs(fourth_order_shifts - exact_shifts))
assert expansion_parameter < 1
assert fourth_order_max_error < second_order_max_error

figure, axis = plt.subplots(figsize=(6, 4))
axis.plot(
    magnetizations / L,
    exact_shifts / L,
    color="black",
    label="exact",
)
axis.plot(
    magnetizations / L,
    second_order_shifts / L,
    "o--",
    label="second order",
)
axis.plot(
    magnetizations / L,
    fourth_order_shifts / L,
    "s:",
    label="fourth order",
)
axis.set(
    xlabel=r"Magnetization density $m/L$",
    ylabel=r"Induced energy shift $\delta E/L$",
)
axis.legend()
figure.tight_layout()
plt.show()

# %% [markdown]
# The virtual process lowers the energy most strongly for states with large $|m|$.
# The quadratic second-order term lowers these states too much, while the positive fourth-order term bends the result back toward the exact curve.
# Here the fourth-order term reduces the maximum error from about $0.085$ to $0.024$.
# Each point uses the same two effective MPOs; evaluating an entangled MPS would use the same contraction and would still require no dense operator.
#
# ## Conclusion
#
# For a Hilbert space of more than sixteen million states, Pymablock obtains the long-range operators $-M_z^2/\Delta$ and $M_z^4/\Delta^3$ with bond dimensions $3$ and $5$.
# These MPOs can be reused directly in later DMRG, dynamics, or observable calculations; no dense operator is formed.
# The energy-shift curve shows how these reusable MPOs capture the ancilla-mediated interaction across all magnetization sectors.
#
# The one-step GMRES convergence in this example follows from the commuting coupling and constant detuning.
# Generic models require more iterations, and the calculation remains useful only while the residuals and bond dimensions converge as the numerical tolerances are tightened.
