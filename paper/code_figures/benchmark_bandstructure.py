# %%
import tinyarray as ta
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
import numpy as np
import kwant
import matplotlib
import matplotlib.pyplot as plt

color_cycle = ["#5790fc", "#f89c20", "#e42536"]

from pymablock import block_diagonalize

# %%
figwidth = matplotlib.rcParams["figure.figsize"][0]
# %%
syst = kwant.Builder()
lat = kwant.lattice.square(norbs=1)
L = 100
W = 100


def onsite(site, mu, dmu, t):
    return (
        4 * t
        + (mu - dmu / 2) * kwant.digest.gauss(repr(site), salt="1")
        + dmu * kwant.digest.gauss(repr(site), salt="20")
    )


def nearest_neighbor(site1, site2, t):
    return t


syst[lat.shape((lambda pos: 0 <= pos[1] < W and 0 <= pos[0] < L), (0, 0))] = onsite
syst[lat.neighbors()] = nearest_neighbor
sysf = syst.finalized()

default_params = {"mu": 0.47, "t": 1, "dmu": 0}
h_0 = sysf.hamiltonian_submatrix(params=default_params, sparse=True).real
h_p = sysf.hamiltonian_submatrix(
    params={**{p: 0 for p in default_params.keys()}, "dmu": 1}, sparse=True
).real
# %%
occupied_states = 8
extra_states = 8
vals, vecs = eigsh(h_0, k=occupied_states, sigma=-5)
vecs, _ = scipy.linalg.qr(vecs, mode="economic")  # orthogonalize the vectors
H_tilde, *_ = block_diagonalize([h_0, h_p], subspace_eigenvectors=[vecs])


# %%
def best_match(psi1, psi2, threshold=None):
    """Find the best match of two sets of eigenvectors.


    Parameters:
    -----------
    psi1, psi2 : numpy 2D complex arrays
        Arrays of initial and final eigenvectors.
    threshold : float, optional
        Minimal overlap when the eigenvectors are considered belonging to the same band.
        The default value is :math:`1/(2N)^{1/4}`, where :math:`N` is the length of each eigenvector.

    Returns:
    --------
    sorting : numpy 1D integer array
        Permutation to apply to ``psi2`` to make the optimal match.
    disconnects : numpy 1D bool array
        The levels with overlap below the ``threshold`` that should be considered disconnected.
    """
    if threshold is None:
        threshold = (2 * psi1.shape[0]) ** -0.25
    Q = np.abs(psi1.T.conj() @ psi2)  # Overlap matrix
    orig, perm = linear_sum_assignment(-Q)
    return perm, Q[orig, perm] < threshold


def sort_evals(hamiltonians):
    evals, evecs = eigsh(hamiltonians[0], k=occupied_states + extra_states, sigma=-5)
    sorted_levels = [evals]
    for ham in hamiltonians[1:]:
        evals2, evecs2 = eigsh(ham, k=occupied_states + extra_states, sigma=-5)
        perm, line_breaks = best_match(evecs, evecs2)
        evals2 = evals2[perm]
        intermediate = (evals + evals2) / 2
        intermediate[line_breaks] = None
        evecs = evecs2[:, perm]
        evals = evals2
        sorted_levels.append(intermediate)
        sorted_levels.append(evals)
    return sorted_levels


# %%
# Perturbation theory
ns = range(1, 6)  # PT orders
pt_spectra = []
dmu_vals = np.linspace(0, 0.13, num=35)


def pt_spectrum(n):
    fill_value = np.zeros((), dtype=object)
    fill_value[()] = np.zeros_like(H_tilde[0, 0, 0])
    h_tilde = np.ma.filled(H_tilde[0, 0, :n], fill_value).tolist()
    pt_hamiltonians = [
        np.sum([h * dmu**i for i, h in zip(range(n), h_tilde)], axis=0)
        for dmu in dmu_vals
    ]
    return sort_evals(pt_hamiltonians)
    # return [np.linalg.eigh(h)[0] for h in pt_hamiltonians]


pt_spectra = [pt_spectrum(n) for n in ns]
# %%
# Sparse diagonalization
dmu_vals_sparse = np.linspace(dmu_vals[0], dmu_vals[-1], num=9)
sparse_spectrum = sort_evals(
    [
        sysf.hamiltonian_submatrix(
            params={**default_params, "dmu": dmu},
            sparse=True,
        )
        for dmu in dmu_vals_sparse
    ]
)

# %%
fig, ax = plt.subplots()
ax.plot(
    sparse_spectrum,
    linestyle="-",
)
# %%
fig, axs = plt.subplots(
    1, len(pt_spectra) + 1, figsize=(figwidth, figwidth / 2), sharey=True
)

for i in range(len(pt_spectra)):
    axs[i].plot(
        dmu_vals,
        pt_spectra[i][::2],
        linestyle="-",
    )
axs[-1].plot(
    dmu_vals_sparse,
    sparse_spectrum[::2],
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
ax.plot(dmu_vals, pt_spectra[3][::2], linestyle="-", alpha=0.5, c="b")
# ax.set_ylim(-0.13, -0.05)
# %%
