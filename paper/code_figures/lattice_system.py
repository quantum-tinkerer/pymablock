# %%
import tinyarray as ta
import kwant
import matplotlib
import matplotlib.pyplot as plt

color_cycle = ["#5790fc", "#f89c20", "#e42536"]
# %%
figwidth = matplotlib.rcParams["figure.figsize"][0]
# %%
sigma_z = ta.array([[1, 0], [0, -1]], float)
sigma_x = ta.array([[0, 1], [1, 0]], float)

syst = kwant.Builder()
lat = kwant.lattice.square(norbs=2)
L = 200
W = 40


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

fig.savefig("../figures/benchmark_lattice.pdf", bbox_inches="tight")
# %%
