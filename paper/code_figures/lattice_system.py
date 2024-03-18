# %%
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh
import tinyarray
import kwant
import matplotlib
import matplotlib.pyplot as plt
from pymablock import block_diagonalize

color_cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
# %%
figwidth = matplotlib.rcParams["figure.figsize"][0]
# %%
sigma_z = tinyarray.array([[1, 0], [0, -1]], float)
sigma_x = tinyarray.array([[0, 1], [1, 0]], float)


def system(L, W, params):
    syst = kwant.Builder()
    lat = kwant.lattice.square(norbs=2)

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

    h_0 = sysf.hamiltonian_submatrix(params=params, sparse=True).real

    barrier = sysf.hamiltonian_submatrix(
        params={**{p: 0 for p in params.keys()}, "t_barrier": 1}, sparse=True
    ).real
    dmu = (
        kwant.operator.Density(sysf, (lambda site: sigma_z * site.pos[0] / L))
        .tocoo()
        .real
    )
    return h_0, barrier, dmu, syst


# %%
L, W = 200, 40
params = {"mu_n": 0.05, "mu_sc": 0.3, "Delta": 0.05, "t": 1.0, "t_barrier": 0.0}
h_0, barrier, dmu, syst = system(L, W, params)
# %%
vals, vecs = eigsh(h_0, k=4, sigma=0)
vecs, _ = scipy.linalg.qr(vecs, mode="economic")  # orthogonalize the vectors

H_tilde, *_ = block_diagonalize([h_0, barrier, dmu], subspace_eigenvectors=[vecs])
# %%
# Combine all the perturbative terms into a single 4D array
fill_value = np.zeros((), dtype=object)
fill_value[()] = np.zeros_like(H_tilde[0, 0, 0, 0])
h_tilde = np.array(np.ma.filled(H_tilde[0, 0, :3, :3], fill_value).tolist())


def effective_energies(h_tilde, barrier, dmu):
    barrier_powers = barrier ** np.arange(3).reshape(-1, 1, 1, 1)
    dmu_powers = dmu ** np.arange(3).reshape(1, -1, 1, 1)
    return scipy.linalg.eigvalsh(
        np.sum(h_tilde * barrier_powers * dmu_powers, axis=(0, 1))
    )


barrier_vals = np.array([0, 0.5, 0.75])
dmu_vals = np.linspace(0, 10e-4, num=101)
results = [
    np.array([effective_energies(h_tilde, bar, dmu) for dmu in dmu_vals])
    for bar in barrier_vals
]
# %%
mosaic = [["A", "B"]]

fig, ax = plt.subplot_mosaic(
    mosaic, figsize=(figwidth, figwidth / 2.2), width_ratios=[1, 1]
)

# PANEL A
h_0_plot, barrier_plot, dmu_plot, _ = system(8, 2, params)

ax["A"].spy(h_0_plot, markersize=0.5, aspect="equal", c=color_cycle[0], label=r"$H_0$")
ax["A"].spy(
    barrier_plot,
    markersize=0.5,
    aspect="equal",
    c=color_cycle[2],
    label=r"$\propto t_b$",
)
ax["A"].spy(
    dmu_plot,
    markersize=0.5,
    aspect="equal",
    c=color_cycle[1],
    label=r"$\propto \delta \mu$",
)
ax["A"].set_title(r"$H_0 + H_{1} + H_{2}$")
ax["A"].set_xticks([])
ax["A"].set_yticks([])
ax["A"].legend(frameon=False, loc="lower left", markerscale=3)

# PANEL B
alpha_values = [0.2, 0.5, 1]
[
    ax["B"].plot(
        dmu_vals,
        result,
        color=color_cycle[2],
        label=[f"$t_b={barrier}$"] + 3 * [None],
        lw=1,
        alpha=alpha_val,
    )
    for result, barrier, alpha_val in zip(results, barrier_vals, alpha_values)
]
ax["B"].set_xlabel(r"$\delta_\mu$")
ax["B"].set_ylabel(r"$E$")
# ax["B"].legend(frameon=False, bbox_to_anchor=(0, -0.2), loc="upper left", ncol=3)
ax["B"].spines[["top", "right"]].set_visible(False)
ax["B"].set_ylim(-0.0013, 0.0015)
ax["B"].set_xticks([])
ax["B"].set_yticks([])
ax["B"].set_xticklabels([])
ax["B"].set_yticklabels([])
ax["B"].set_aspect(0.4)

# ax_inset = fig.add_axes([0.25, 0.53, 0.22, 0.2])
# kwant.plot(
#     syst,
#     fig_size=(10, 6),
#     site_color=(lambda site: abs(site.pos[0]) < L / 3),
#     colorbar=False,
#     cmap="seismic",
#     hop_lw=0,
#     ax=ax_inset,
# )
# ax_inset.set_aspect("equal")
# ax_inset.set_frame_on(False)
# ax_inset.set_xticks([])
# ax_inset.set_yticks([])

# ax_inset.text(
#     0.5,
#     0.5,
#     r"$\mu_{\textrm{sc}}, \Delta$",
#     ha="center",
#     va="center",
#     transform=ax_inset.transAxes,
#     backgroundcolor="white",
#     fontsize=7,
#     alpha=0.8,
# )
# ax_inset.text(
#     0.2,
#     0.5,
#     r"$\mu_{\textrm{N}}$",
#     ha="center",
#     va="center",
#     transform=ax_inset.transAxes,
#     backgroundcolor="white",
#     fontsize=7,
#     alpha=0.8,
# )
# ax_inset.text(
#     0.8,
#     0.5,
#     r"$\mu_{\textrm{N}}$",
#     ha="center",
#     va="center",
#     transform=ax_inset.transAxes,
#     backgroundcolor="white",
#     fontsize=7,
#     alpha=0.8,
# )

# ax_inset.text(
#     0.5,
#     -0.1,
#     rf"$L={L}$",
#     ha="center",
#     va="center",
#     transform=ax_inset.transAxes,
#     fontsize=7,
# )
# ax_inset.text(
#     0.02,
#     0.5,
#     rf"$W={W}$",
#     ha="center",
#     va="center",
#     transform=ax_inset.transAxes,
#     rotation=90,
#     fontsize=7,
# )

fig.savefig("../figures/lattice_system.pdf", bbox_inches="tight", dpi=300)
# %%
