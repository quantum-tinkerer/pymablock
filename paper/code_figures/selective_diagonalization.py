# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import pymablock

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
figwidth = matplotlib.rcParams["figure.figsize"][0]
# %%
np.random.seed(0)
N = 16
H_0 = np.diag(np.arange(N) + 0.2 * np.random.randn(N))
H_1 = 0.1 * np.random.randn(N, N)
H_1 += H_1.T

smiley_binary = np.array(
    [
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ],
    dtype=bool,
)

mask = np.zeros((N, N), dtype=bool)
mask[1:smiley_binary.shape[0] + 1, -smiley_binary.shape[1] - 1:-1] = smiley_binary
mask = ~(mask | mask.T)
np.fill_diagonal(mask, False)

H_tilde, *_ = pymablock.block_diagonalize([H_0, H_1], fully_diagonalize={0: mask})
Heff = H_tilde[0, 0, 0] + H_tilde[0, 0, 1] + H_tilde[0, 0, 2]
# %%
fig, axs = plt.subplots(1, 3, figsize=(figwidth, figwidth / 3.5))
axs[0].imshow(H_0 + H_1, cmap='seismic', norm=TwoSlopeNorm(vcenter=0))
axs[0].set_title(r'$H_0 + H_1$')
axs[1].imshow(mask, cmap='gray')
axs[1].set_title('Mask')
axs[2].imshow(Heff, cmap='seismic', norm=TwoSlopeNorm(vcenter=0))
axs[2].set_title(r'$\tilde{H}$')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig('../figures/selective_diagonalization.pdf', bbox_inches='tight')
# %%
