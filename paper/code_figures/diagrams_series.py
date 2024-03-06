# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.linalg import block_diag
from pymablock import block_diagonalize

# %%
figwidth = matplotlib.rcParams["figure.figsize"][0]
color_cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
# %%
# Diagonal unperturbed Hamiltonian
H_0 = np.diag([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0])


# Random Hermitian matrix as a perturbation
def random_hermitian(n):
    H = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H += H.conj().T
    return H


H_1 = 0.2 * random_hermitian(len(H_0))

H_tilde, U, U_adjoint = block_diagonalize(
    [H_0, H_1], subspace_indices=[0, 0, 1, 1, 1, 1]
)
# %%
mosaic = [["A", "B", "C", "D"]]
fig, ax = plt.subplot_mosaic(
    mosaic,
    figsize=(figwidth, figwidth / 4),
    width_ratios=[1, 1, 0.06, 1],
    sharey=True,
    sharex=True,
)


ax["A"].imshow(H_0.real, cmap="PuOr", vmin=-1.7, vmax=1.7)
ax["A"].set_title(r"$H_0$")
ax["A"].set_xticks([])
ax["A"].set_yticks([])

ax["B"].imshow(H_1.real, cmap="PuOr", vmin=-1.7, vmax=1.7)
ax["B"].set_title(r"$H_1$")
ax["B"].set_xticks([])
ax["B"].set_yticks([])
ax["B"].annotate(r"$+$", xy=(0, 2.5), xytext=(-1.7, 2.5))

ax["C"].spines[["top", "right", "left", "bottom"]].set_visible(False)
ax["D"].annotate(
    "", xy=(-0.5, 2.5), xytext=(-3.2, 2.5), arrowprops=dict(arrowstyle="->")
)
ax["D"].annotate(r"$\mathcal{U}$", xy=(0, 2.1), xytext=(-2.2, 2.1))

transformed_H = ma.sum(H_tilde[:2, :2, :3], axis=2)
block = block_diag(transformed_H[0, 0], transformed_H[1, 1])

ax["D"].imshow(block.real, cmap="PuOr", vmin=-1.7, vmax=1.7)
ax["D"].set_title(r"$\tilde{H}_0 + \tilde{H}_1 + \tilde{H}_2$")
ax["D"].set_xticks([])
ax["D"].set_yticks([])

fig.savefig("../figures/diagrams_H.pdf", bbox_inches="tight")

# %%
