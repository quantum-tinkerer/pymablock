# %%
import scipy.linalg
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.optimize import linear_sum_assignment
import numpy as np
import kwant
import matplotlib
import matplotlib.pyplot as plt

color_cycle = ["#5790fc", "#f89c20", "#e42536"]
color_cycle_grays = matplotlib.cm.get_cmap("Greys")(np.linspace(0.1, 0.8, 20))

from pymablock import block_diagonalize

# %%
figwidth = matplotlib.rcParams["figure.figsize"][0]
# %%
syst = kwant.Builder()
lat = kwant.lattice.square(norbs=1)
L = 50
W = 50
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

default_params = {"mu": 0.45, "V": -0.017, "t": 1, "dmu": 0}
h_0 = sysf.hamiltonian_submatrix(params=default_params, sparse=True).real
h_p = sysf.hamiltonian_submatrix(
    params={**{p: 0 for p in default_params.keys()}, "dmu": 1}, sparse=True
).real
# %%
occupied_states = 10
extra_states = 10
total_states = occupied_states + extra_states
evals_0, evecs_0 = eigsh(h_0, k=occupied_states, sigma=-5)
evecs_0, _ = scipy.linalg.qr(evecs_0, mode="economic")  # orthogonalize the vectors
H_tilde, *_ = block_diagonalize([h_0, h_p], subspace_eigenvectors=[evecs_0])


# %%
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


def sort_evals(hamiltonians, evals_0, evecs_0, tol=1e-10, maxiter=20, use_lobpcg=True):
    sorted_levels = [evals_0]
    evals, evecs = evals_0, evecs_0
    for h in hamiltonians[1:]:
        if h.shape[1] == evecs.shape[1]:
            evals_2, evecs_2 = np.linalg.eigh(h)
        elif use_lobpcg:
            try:
                evals_2, evecs_2 = lobpcg(
                    h, X=evecs, largest=False, tol=tol, maxiter=maxiter
                )
            except UserWarning:
                use_lobpcg = False
                evals_2, evecs_2 = eigsh(h, k=evecs.shape[1], sigma=-5)
        else:
            evals_2, evecs_2 = eigsh(h, k=evecs.shape[1], sigma=-5)
        perm, line_breaks = best_match(evecs, evecs_2)
        evals_2 = evals_2[perm]
        intermediate = (evals + evals_2) / 2
        intermediate[line_breaks] = None
        evecs = evecs_2[:, perm]
        evals = evals_2
        sorted_levels.append(intermediate)
        sorted_levels.append(evals)
    return sorted_levels


# %%
# Perturbation theory
ns = range(1, 4)  # PT orders
pt_spectra = []
dmu_vals = np.linspace(0, 0.2, num=45)


def pt_spectrum(n):
    fill_value = np.zeros((), dtype=object)
    fill_value[()] = np.zeros_like(H_tilde[0, 0, 0])
    h_tilde = np.ma.filled(H_tilde[0, 0, :n], fill_value).tolist()
    return sort_evals(
        [
            np.sum([h * dmu**i for i, h in zip(range(n), h_tilde)], axis=0)
            for dmu in dmu_vals
        ],
        *np.linalg.eigh(h_tilde[0]),
    )


pt_spectra = [pt_spectrum(n) for n in ns]
# %%
evals_0, evecs_0 = eigsh(h_0, k=total_states, sigma=-5)

# Sparse diagonalization
dmu_vals_sparse = np.linspace(dmu_vals[0], dmu_vals[-1], num=19)
sparse_spectrum = sort_evals(
    [
        sysf.hamiltonian_submatrix(
            params={**default_params, "dmu": dmu},
            sparse=True,
        )
        for dmu in dmu_vals_sparse
    ],
    evals_0,
    evecs_0,
    tol=1e-10,
    use_lobpcg=False,
)

# %%
sparse_spectrum = np.array(sparse_spectrum)
sparse_spectrum_a = sparse_spectrum[::2] - np.repeat(
    sparse_spectrum[::2][:, 0], total_states
).reshape(-1, total_states)
pt_spectra_a = []
for result in np.array(pt_spectra):
    pt_spectra_a.append(
        result[::2]
        - np.repeat(result[::2][:, 0], occupied_states).reshape(-1, occupied_states)
    )
pt_spectra_a = np.array(pt_spectra_a)
# %%
fig, axs = plt.subplots(
    1, len(pt_spectra) + 1, figsize=(figwidth, figwidth / 2), sharey=True
)

for i in range(len(pt_spectra) + 1):
    for j in range(total_states):
        axs[i].plot(
            dmu_vals_sparse,
            sparse_spectrum_a[:, j],
            linestyle="-",
            c=color_cycle_grays[5],
        )
    if i < len(pt_spectra):
        axs[i].plot(
            dmu_vals,
            pt_spectra_a[i],
            linestyle="-",
        )

axs[0].set_ylabel(r"$E$")
axs[0].legend(
    frameon=False, bbox_to_anchor=(0.75, 1, 0, 0), bbox_transform=axs[0].transAxes
)

for i in range(len(pt_spectra)):
    axs[i].set_title(rf"$n={i}$")
# axs[0].set_ylim(-0.001, 0.011)
axs[-1].set_title(r"$\textrm{Sparse}$")

fig.savefig("../figures/benchmark_bandstructure.pdf", bbox_inches="tight")
# %%
fig, ax = plt.subplots()
ax.plot(dmu_vals_sparse, sparse_spectrum[::2], linestyle="-", alpha=0.5, c="r")
ax.plot(dmu_vals, pt_spectra[2][::2], linestyle="-", alpha=0.5, c="b")
# ax.set_ylim(-0.13, -0.05)
# %%
fig, ax = plt.subplots()
sparse_spectrum = np.array(sparse_spectrum)
pt_spectra = np.array(pt_spectra)

plot_1 = sparse_spectrum[::2] - np.repeat(
    sparse_spectrum[::2][:, 0], total_states
).reshape(-1, total_states)
plot_2 = pt_spectra[2][::2] - np.repeat(
    pt_spectra[2][::2][:, 0], occupied_states
).reshape(-1, occupied_states)

ax.plot(dmu_vals_sparse, plot_1, linestyle="-", alpha=0.5, c="r")
ax.plot(dmu_vals, plot_2, linestyle="-", alpha=0.5, c="b")

# %%
