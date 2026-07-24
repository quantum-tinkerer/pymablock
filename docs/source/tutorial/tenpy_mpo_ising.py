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
# This tutorial demonstrates the main advantage of full-MPO perturbation theory: it constructs reusable effective operators when dense operator algebra is impossible.
# We eliminate a detuned auxiliary sector coupled to a 24-site Ising chain and compute second- and fourth-order corrections to the low-energy sector as MPOs.
# The Hilbert space has $2^{24}$ states, but the resulting operators have bond dimensions of only a few.
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
# We couple the chain to two auxiliary sectors, $A$ and $B$.
# Sector $A$ contains $H_\mathrm{I}$ and is the subspace whose effective Hamiltonian we want.
# Sector $B$ has the same spin dynamics but costs an additional energy $\Delta>0$.
# The weak amplitude $g$ changes the auxiliary sector with a strength set by $M_z$:
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
# We treat $g$ perturbatively and eliminate $B$ to obtain corrections acting entirely within $A$.
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
# We ask Pymablock to recover both displayed coefficients.
# Already at second order,
#
# $$
# M_z^2=LI+2\sum_{i<j}Z_iZ_j,
# $$
#
# so the effective Hamiltonian contains an all-to-all interaction.
# The fourth-order term also contains long-range four-spin interactions.
# Thus a local-looking virtual sector change lets every spin talk to every other spin in the effective theory.

# %%
# %%time
from functools import reduce

import numpy as np
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
# ## Conclusion
#
# For a Hilbert space of more than sixteen million states, Pymablock obtains the long-range operators $-M_z^2/\Delta$ and $M_z^4/\Delta^3$ with bond dimensions $3$ and $5$.
# These MPOs can be reused directly in later DMRG, dynamics, or observable calculations; no dense operator is formed.
#
# The one-step GMRES convergence in this example follows from the commuting coupling and constant detuning.
# Generic models require more iterations, and the calculation remains useful only while the residuals and bond dimensions converge as the numerical tolerances are tightened.
