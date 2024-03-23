# %%
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt

from pymablock.block_diagonalization import block_diagonalize
from pymablock.series import BlockSeries, AlgebraElement, zero

# %%
figwidth = matplotlib.rcParams["figure.figsize"][0]


# %%
def solve_sylvester(A):
    return AlgebraElement(f"S({A})")


def eval_dense_first_order(*index):
    if index[0] != index[1] and sum(index[2:]) == 0:
        return zero
    elif index[2] > 1 or any(index[3:]):
        return zero
    return AlgebraElement(f"H{index}")


def eval_dense_every_order(*index):
    if index[0] != index[1] and sum(index[2:]) == 0:
        return zero
    return AlgebraElement(f"H{index}")


def eval_offdiagonal_first_order(*index):
    if index[0] != index[1] and sum(index[2:]) == 1:
        return AlgebraElement(f"H{index}")
    return zero


def eval_offdiagonal_every_order(*index):
    if index[0] != index[1] and sum(index[2:]) == 0:
        return zero
    elif index[0] == index[1] and sum(index[2:]) != 0:
        return zero
    return AlgebraElement(f"H{index}")


evals = {
    r"$\mathcal{H}'_a$": eval_dense_first_order,
    r"$\mathcal{H}'_b$": eval_dense_every_order,
    r"$\mathcal{H}'_c$": eval_offdiagonal_first_order,
    r"$\mathcal{H}'_d$": eval_offdiagonal_every_order,
}
# %%
multiplication_counts = {}
for structure in evals.keys():
    multiplication_counts[structure] = {}
    H = BlockSeries(
        eval=evals[structure],
        shape=(2, 2),
        n_infinite=1,
    )

    for order in range(10):
        AlgebraElement.log = []
        H_tilde, *_ = block_diagonalize(
            H,
            solve_sylvester=solve_sylvester,
        )
        H_tilde[0, 0, order]
        multiplication_counts[structure][order] = Counter(
            call[1] for call in AlgebraElement.log
        )["__mul__"]
# %%
mosaic = [["A", "B"]]
fig, ax = plt.subplot_mosaic(mosaic, figsize=(figwidth, figwidth / 3))

for structure, counts in multiplication_counts.items():
    ax["A"].plot(
        list(counts.keys())[2:],
        list(counts.values())[2:],
        label=None,
        linestyle="--",
        alpha=0.3,
        linewidth=1,
    )
    ax["A"].scatter(
        list(counts.keys())[2:],
        list(counts.values())[2:],
        label=structure,
        s=20,
    )

ax["A"].set_xlabel(r"$\textrm{Order of } \tilde{\mathcal{H}}^{AA}$")
ax["A"].set_ylabel(r"$\# \textrm{ Matrix products}$")
ax["A"].legend(frameon=False, loc="upper left")
ax["A"].spines["right"].set_visible(False)
ax["A"].spines["top"].set_visible(False)
fig.savefig("../figures/benchmark_matrix_products.pdf", bbox_inches="tight")
# %%
