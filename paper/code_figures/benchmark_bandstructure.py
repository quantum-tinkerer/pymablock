# %%
import tinyarray as ta
import scipy.linalg
from scipy.sparse.linalg import eigsh
import numpy as np
import kwant
import matplotlib
import matplotlib.pyplot as plt

color_cycle = ["#5790fc", "#f89c20", "#e42536"]

from pymablock import block_diagonalize

# %%
figwidth = matplotlib.rcParams["figure.figsize"][0]
# %%
sigma_z = ta.array([[1, 0], [0, -1]], float)
sigma_x = ta.array([[0, 1], [1, 0]], float)

syst = kwant.Builder()
lat = kwant.lattice.square(norbs=2)
L = 200
W = L / 4


def normal_onsite(site, mu_n, delta_mu, t):
    return (-mu_n + delta_mu * site.pos[0] / L + 4 * t) * sigma_z


def sc_onsite(site, mu_sc, delta_mu, Delta, t):
    return (-mu_sc + delta_mu * site.pos[0] / L + 4 * t) * sigma_z + Delta * sigma_x


syst[lat.shape((lambda pos: abs(pos[1]) < W and abs(pos[0]) < L), (0, 0))] = (
    normal_onsite
)
syst[lat.shape((lambda pos: abs(pos[1]) < W and abs(pos[0]) < L / 3), (0, 0))] = (
    sc_onsite
)

syst[lat.neighbors()] = lambda site1, site2, t: -t * sigma_z


def barrier(site1, site2):
    return (abs(site1.pos[0]) - L / 3) * (abs(site2.pos[0]) - L / 3) < 0


syst[(hop for hop in syst.hoppings() if barrier(*hop))] = (
    lambda site1, site2, t_barrier: -t_barrier * sigma_z
)

# %%
fig, ax = plt.subplots()
kwant.plot(
    syst,
    fig_size=(10, 6),
    site_color=(lambda site: abs(site.pos[0]) < L / 3),
    colorbar=False,
    cmap="seismic",
    hop_lw=0,
    ax=ax,
)
ax.set_aspect("equal")
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])

ax.text(
    0.5,
    0.5,
    r"$\mu_{\textrm{sc}}, \Delta_{\textrm{sc}}$",
    ha="center",
    va="center",
    transform=ax.transAxes,
)
ax.text(
    0.2,
    0.5,
    r"$\mu_{\textrm{N}}, \Delta_{\textrm{N}}$",
    ha="center",
    va="center",
    transform=ax.transAxes,
)
ax.text(
    0.8,
    0.5,
    r"$\mu_{\textrm{N}}, \Delta_{\textrm{N}}$",
    ha="center",
    va="center",
    transform=ax.transAxes,
)

ax.text(0.5, -0.05, r"$L$", ha="center", va="center", transform=ax.transAxes)
ax.text(0, 0.5, r"$L/4$", ha="center", va="center", transform=ax.transAxes)

fig.savefig("../figures/benchmark_lattice.pdf", bbox_inches="tight")
# %%
sysf = syst.finalized()
# %%
default_params = dict(
    mu_n=0.05,
    mu_sc=0.3,
    Delta=0.05,
    t=1.0,
)

params = {**default_params, "t_barrier": 0.0, "delta_mu": 0.0}

h_0 = sysf.hamiltonian_submatrix(params=params, sparse=True).real

barrier = sysf.hamiltonian_submatrix(
    params={**{p: 0 for p in params.keys()}, "t_barrier": 1}, sparse=True
).real
delta_mu = sysf.hamiltonian_submatrix(
    params={**{p: 0 for p in params.keys()}, "delta_mu": 1}, sparse=True
).real
# %%
vals, vecs = eigsh(h_0, k=4, sigma=0)
vecs, _ = scipy.linalg.qr(vecs, mode="economic")  # orthogonalize the vectors
# %%

H_tilde, *_ = block_diagonalize([h_0, barrier, delta_mu], subspace_eigenvectors=[vecs])
# %%

# Combine all the perturbative terms into a single 4D array
fill_value = np.zeros((), dtype=object)
fill_value[()] = np.zeros_like(H_tilde[0, 0, 0, 0])

ns = range(1, 4)
pt_results = []
barrier_vals = np.array([0, 0.25, 0.5])
delta_mu_vals = np.linspace(0, 0.6 * 10e-4, num=101)
for n in ns:
    h_tilde = np.ma.filled(H_tilde[0, 0, :n, :n], fill_value).tolist()

    def effective_energies(h_tilde, barrier, delta_mu):
        barrier_powers = barrier ** np.arange(n).reshape(-1, 1, 1, 1)
        delta_mu_powers = delta_mu ** np.arange(n).reshape(1, -1, 1, 1)
        return scipy.linalg.eigvalsh(
            np.sum(h_tilde * barrier_powers * delta_mu_powers, axis=(0, 1))
        )

    pt_results.append(
        [
            np.array([effective_energies(h_tilde, bar, dmu) for dmu in delta_mu_vals])
            for bar in barrier_vals
        ]
    )
# %%
skip = 40

sparse_results = [
    np.array(
        [
            eigsh(
                syst.hamiltonian_submatrix(
                    params={**default_params, "t_barrier": bar, "delta_mu": dmu},
                    sparse=True,
                ),
                k=4,
                sigma=0,
            )[0]
            for dmu in delta_mu_vals[0::skip]
        ]
    )
    for bar in barrier_vals
]
# %%
results = pt_results + [sparse_results]
# %%
fig, axs = plt.subplots(
    1, len(pt_results) + 1, figsize=(figwidth, figwidth / 2), sharex=True, sharey=True
)

for i, ax in enumerate(axs[:-1]):
    [
        ax.plot(
            delta_mu_vals,
            result,
            color=color,
            label=[f"$t_b={barrier}$"] + 3 * [None],
            linestyle="--",
        )
        for result, color, barrier in zip(results[i], color_cycle, barrier_vals)
    ]
    ax.set_title(rf"$n={n}$")
    ax.set_xlabel(r"$\delta_\mu$")
    ax.set_xticks([delta_mu_vals[0], delta_mu_vals[-1]])
    ax.set_xticklabels([r"$0$", r"$6 \times 10^{-4}$"])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

axs[0].set_ylabel(r"$E$")
axs[0].set_yticks([-0.001, -0.0005, 0, 0.0005, 0.001])
axs[0].set_yticklabels(
    [r"$-10^{-3}$", r"$-5 \times 10^{-4}$", r"$0$", r"$5 \times 10^{-4}$", r"$10^{-3}$"]
)
axs[0].legend(frameon=False, loc="center left")

[
    axs[-1].plot(
        delta_mu_vals[0::skip],
        result,
        color=color,
        label=[f"$t_b={barrier}$"] + 3 * [None],
        marker=".",
        linestyle="",
    )
    for result, color, barrier in zip(sparse_results, color_cycle, barrier_vals)
]
axs[-1].set_title(r"$\textrm{Sparse}$")
fig.savefig("../figures/benchmark_bandstructure.pdf", bbox_inches="tight")
# %%
