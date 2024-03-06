# %%
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh
import tinyarray
import kwant
import matplotlib
import matplotlib.pyplot as plt
from pymablock import block_diagonalize

color_cycle = ["#5790fc", "#f89c20", "#e42536"]
# %%
figwidth = matplotlib.rcParams["figure.figsize"][0]
# %%
sigma_z = tiniarray.array([[1, 0], [0, -1]], float)
sigma_x = tiniarray.array([[0, 1], [1, 0]], float)

syst = kwant.Builder()
lat = kwant.lattice.square(norbs=2)
L = 200
W = 40


def normal_onsite(site, mu_n, t):
    return (-mu_n + 4 * t) * sigma_z


def sc_onsite(site, mu_sc, Delta, t):
    return (-mu_sc + 4 * t) * sigma_z + Delta * sigma_x


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
sysf = syst.finalized()
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

ax.text(0.5, -0.1, rf"$L={L}$", ha="center", va="center", transform=ax.transAxes)
ax.text(
    0.02,
    0.5,
    rf"$W={W}$",
    ha="center",
    va="center",
    transform=ax.transAxes,
    rotation=90,
)

fig.savefig("../figures/QD_lattice.pdf", bbox_inches="tight")
# %%
params = dict(mu_n=0.05, mu_sc=0.3, Delta=0.05, t=1.0, t_barrier=0.0)
h_0 = sysf.hamiltonian_submatrix(params=params, sparse=True).real

barrier = sysf.hamiltonian_submatrix(
    params={**{p: 0 for p in params.keys()}, "t_barrier": 1}, sparse=True
).real
delta_mu = (
    kwant.operator.Density(sysf, (lambda site: sigma_z * site.pos[0] / L)).tocoo().real
)
# %%
vals, vecs = eigsh(h_0, k=4, sigma=0)
vecs, _ = scipy.linalg.qr(vecs, mode="economic")  # orthogonalize the vectors

H_tilde, *_ = block_diagonalize([h_0, barrier, delta_mu], subspace_eigenvectors=[vecs])
# %%
# Combine all the perturbative terms into a single 4D array
fill_value = np.zeros((), dtype=object)
fill_value[()] = np.zeros_like(H_tilde[0, 0, 0, 0])
h_tilde = np.array(np.ma.filled(H_tilde[0, 0, :3, :3], fill_value).tolist())


def effective_energies(h_tilde, barrier, delta_mu):
    barrier_powers = barrier ** np.arange(3).reshape(-1, 1, 1, 1)
    delta_mu_powers = delta_mu ** np.arange(3).reshape(1, -1, 1, 1)
    return scipy.linalg.eigvalsh(
        np.sum(h_tilde * barrier_powers * delta_mu_powers, axis=(0, 1))
    )


barrier_vals = np.array([0, 0.5, 0.75])
delta_mu_vals = np.linspace(0, 10e-4, num=101)
results = [
    np.array([effective_energies(h_tilde, bar, dmu) for dmu in delta_mu_vals])
    for bar in barrier_vals
]
# %%
fig, ax = plt.subplots(figsize=(figwidth, figwidth / 3))

[
    ax.plot(
        delta_mu_vals,
        result,
        color=color,
        label=[f"$t_b={barrier}$"] + 3 * [None],
        lw=1,
    )
    for result, color, barrier in zip(results, color_cycle, barrier_vals)
]
ax.set_xlabel(r"$\delta_\mu$")
ax.set_ylabel(r"$E$")
ax.legend(frameon=False, bbox_to_anchor=(0, -0.2), loc="upper left", ncol=3)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("../figures/QD_spectrum.pdf", bbox_inches="tight")
# %%
