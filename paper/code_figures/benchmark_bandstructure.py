# %%
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
import numpy as np
import kwant
import matplotlib
import matplotlib.pyplot as plt
from pymablock import block_diagonalize

from timer import Timer

# %%
color_cycle = ["#5790fc", "#f89c20", "#e42536"]
color_cycle_grays = matplotlib.cm.get_cmap("Greys")(np.linspace(0.1, 0.8, 20))
figwidth = matplotlib.rcParams["figure.figsize"][0]

times = Timer()


# %%
# Functions to sort the eigenvalues so that the spectrum is continuous
def best_match(evecs_1, evecs_2, threshold=None):
    """Find the best match of two sets of eigenvectors.


    Parameters:
    -----------
    evecs_1, evecs_2 : numpy 2D complex arrays
        Arrays of initial and final eigenvectors.
    threshold : float, optional
        Minimal overlap when the eigenvectors are considered belonging to the same band.
        The default value is :math:`1/(2N)^{1/4}`, where :math:`N` is the length of each eigenvector.

    Returns:
    --------
    sorting : numpy 1D integer array
        Permutation to apply to ``evecs_2`` to make the optimal match.
    disconnects : numpy 1D bool array
        The levels with overlap below the ``threshold`` that should be considered disconnected.
    """
    if threshold is None:
        threshold = (2 * evecs_1.shape[0]) ** -0.25
    Q = np.abs(evecs_1.T.conj() @ evecs_2)  # Overlap matrix
    orig, perm = linear_sum_assignment(-Q)
    return perm, Q[orig, perm] < threshold


def sort_evals(hamiltonians, evals_0, evecs_0, tol=1e-10, maxiter=20, name=""):
    sorted_levels = [evals_0]
    evals, evecs = evals_0, evecs_0
    for h in hamiltonians[1:]:
        if h.shape[1] == evecs.shape[1]:
            with times(name + " eigh"):
                evals_2, evecs_2 = np.linalg.eigh(h)
        else:
            with times(name + " eigsh"):
                evals_2, evecs_2 = eigsh(h, k=evecs.shape[1], sigma=-5)
        perm, line_breaks = best_match(evecs, evecs_2)
        evals_2 = evals_2[perm]
        # intermediate = (evals + evals_2) / 2
        # intermediate[line_breaks] = None
        evecs = evecs_2[:, perm]
        evals = evals_2
        # sorted_levels.append(intermediate)
        sorted_levels.append(evals)
    return np.array(sorted_levels)


# %%
# Define the Kwant system
syst = kwant.Builder()
lat = kwant.lattice.square(norbs=1)
L = 52
W = 52
r = 35


def onsite(site, mu, V, dmu, t):
    x, y = site.pos
    potential = (
        4 * t
        + (mu - dmu / 2) * kwant.digest.gauss(repr(site), salt="1")
        + dmu * kwant.digest.gauss(repr(site), salt="17")  # 11
    )

    if x**2 + y**2 < r**2:
        return potential + V
    return potential


def nearest_neighbor(site1, site2, t):
    return t


syst[lat.shape((lambda pos: np.abs(pos[1]) < W and np.abs(pos[0]) < L), (0, 0))] = (
    onsite
)
syst[lat.neighbors()] = nearest_neighbor
sysf = syst.finalized()

# Generate the Hamiltonian and the perturbation
default_params = {"mu": 0.45, "V": -0.017, "t": 1, "dmu": 0}
h_0 = sysf.hamiltonian_submatrix(params=default_params, sparse=True).real
h_p = sysf.hamiltonian_submatrix(
    params={**{p: 0 for p in default_params.keys()}, "dmu": 1}, sparse=True
).real
# %%
# Choose the number of states to consider
occupied_states = 10
extra_states = 10  # Unoccupied states for reference
total_states = occupied_states + extra_states
# %%
with times("PT occupied eigsh"):
    evals_0, evecs_0 = eigsh(h_0, k=occupied_states, sigma=-5)
with times("PT QR"):
    evecs_0, _ = scipy.linalg.qr(evecs_0, mode="economic")  # orthogonalize the vectors
with times("PT H_tilde"):
    H_tilde, *_ = block_diagonalize([h_0, h_p], subspace_eigenvectors=[evecs_0])
# %%
mu_max = 0.2
n_pt = 45
n_total = 25
n_occupied = 1
spectra = {}


# %%
# Perturbation theory
def pt_spectrum(n):
    fill_value = np.zeros((), dtype=object)
    fill_value[()] = np.zeros_like(H_tilde[0, 0, 0])
    with times(f"PT {n}"):
        H_tilde[0, 0, : (n + 1)]
    h_tilde = np.ma.filled(H_tilde[0, 0, : (n + 1)], fill_value).tolist()
    hamiltonians = [
        np.sum([h * dmu**i for i, h in zip(range(n + 1), h_tilde)], axis=0)
        for dmu in np.linspace(0, mu_max, num=45)
    ]
    evals = sort_evals(
        hamiltonians,
        *np.linalg.eigh(h_tilde[0]),
        name=f"PT {n}",
    )
    return np.linspace(0, mu_max, num=45), evals


for n in range(0, 4):
    spectra["PT " + str(n)] = pt_spectrum(n)
# %%
# Sparse diagonalization for background bands


def sparse_spectrum(num_states, num_dmu_points, name):
    with times(name + " eigsh"):
        evals_0, evecs_0 = eigsh(h_0, k=num_states, sigma=-5)

    hamiltonians = [
        sysf.hamiltonian_submatrix(
            params={**default_params, "dmu": dmu},
            sparse=True,
        ).real
        for dmu in np.linspace(0, mu_max, num=num_dmu_points)
    ]
    sparse_spectrum = sort_evals(
        hamiltonians,
        evals_0,
        evecs_0,
        tol=1e-10,
        name=name,
    )
    return np.linspace(0, mu_max, num=num_dmu_points), sparse_spectrum


for num_states, num_dmu_points, name in [
    (occupied_states, n_occupied, "sparse occupied"),
    (total_states, n_total, "sparse total"),
]:
    spectra[name] = sparse_spectrum(num_states, num_dmu_points, name)
# %%
# Shift all bands by the energy of the lowest state
shifted_spectra = {}
for key, (dmu_points, spectrum) in spectra.items():
    num_states = spectrum.shape[1]
    shifted_spectrum = spectrum - np.repeat(spectrum[:, 0], num_states).reshape(
        -1, num_states
    )
    shifted_spectra[key] = (dmu_points, shifted_spectrum)
# %%
mosaic = [
    ["PT 0", "PT 1", "PT 2", "PT 3"],
    [".", ".", ".", "."],
    ["time", "time", "time", "time"],
]
fig, axs = plt.subplot_mosaic(
    mosaic,
    figsize=(figwidth, 2 * figwidth / 3),
    constrained_layout=True,
    height_ratios=[1, 0.1, 0.2],
)

for label in mosaic[0]:
    axs[label].plot(
        *shifted_spectra["sparse total"],
        linestyle="-",
        c=color_cycle_grays[5],
        linewidth=1.2,
    )
    axs[label].plot(
        *shifted_spectra[label],
        linestyle="-",
        c=color_cycle_grays[15],
        linewidth=1.2,
    )
    axs[label].set_title(rf"$n = {label[-1]}$")
    axs[label].set_ylim(-0.002, 0.075)
    axs[label].set_xlim(0, mu_max)
    axs[label].set_xlabel(r"$\delta \mu$")
    axs[label].set_yticks([])
    axs[label].set_yticklabels([])
    axs[label].set_xticks([])
    axs[label].set_xticklabels([])
    axs[label].spines[["top", "right"]].set_visible(False)
    if label != "PT 0":
        axs[label].set_yticklabels([])
    else:
        axs[label].set_ylabel(r"$E$")

left = 0
for t_label, label in zip(
    ["PT H_tilde", "PT 2", "PT 3"], [r"$\mathrm{PT}$", r"$n=2$", r"$n=3$"]
):
    t = times.times[t_label]
    p = axs["time"].barh("PT", np.mean(t), height=0.5, left=left, alpha=0.6)
    left += np.mean(t)
    axs["time"].bar_label(p, labels=(label,), label_type="center")

p = axs["time"].barh(
    "sparse", times.times["sparse occupied eigsh"], height=0.5, alpha=0.6
)
axs["time"].bar_label(p, labels=(r"$\mathrm{Sparse}$",), label_type="center")

axs["time"].spines[["top", "right"]].set_visible(False)
axs["time"].set_yticks([])
axs["time"].set_yticklabels([])
axs["time"].set_xticks([0, 0.5, 1, 1.5])
axs["time"].set_xticklabels([r"$0$", r"$1/2$", r"$1$", r"$3/2$"])
axs["time"].set_xlabel(r"$\textrm{Time (s)}$")
axs["time"].set_xlim(0, times.times["sparse occupied eigsh"][0])

fig.savefig("../figures/benchmark_bandstructure.pdf", bbox_inches="tight")


# %%
fig, ax = plt.subplots(figsize=(figwidth, figwidth / 2))
for t_label, t in times.times.items():
    ax.barh(t_label, np.mean(t), label=t_label, height=0.5)
ax.set_xlim(0, 1.5)


# %%
