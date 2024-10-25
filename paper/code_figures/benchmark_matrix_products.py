# %%
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt

from pymablock.block_diagonalization import block_diagonalize
from pymablock.series import BlockSeries, AlgebraElement, zero

# %%
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
figwidth = matplotlib.rcParams["figure.figsize"][0]


# %%
def solve_sylvester(A):
    return AlgebraElement(f"G({A})")


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
    r"$H_{1}$": eval_dense_first_order,
    r"$\mathcal{H}^{'}$": eval_dense_every_order,
    r"$H_{\textrm{offdiag}, 1}$": eval_offdiagonal_first_order,
    r"$\mathcal{H}^{'}_{\textrm{offdiag}}$": eval_offdiagonal_every_order,
}
# %%
multiplication_counts = {}
for structure in evals.keys():
    multiplication_counts[structure] = {}
    H = BlockSeries(
        data={
            (0, 0, 0): AlgebraElement("H000"),
            (1, 1, 0): AlgebraElement("H110"),
        },
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
fig, ax = plt.subplot_mosaic(mosaic, figsize=(figwidth, figwidth / 3.8), sharey=True)

for (structure, counts), panel, color in zip(
    multiplication_counts.items(), ["A", "B", "A", "B"], ["C0", "C2", "C1", "C7"]
):
    ax[panel].plot(
        list(counts.keys())[2:],
        list(counts.values())[2:],
        label=None,
        linestyle="--",
        alpha=0.3,
        linewidth=1,
        color=color,
    )
    ax[panel].scatter(
        list(counts.keys())[2:],
        list(counts.values())[2:],
        label=structure,
        s=20,
        color=color,
    )

    ax[panel].set_xlabel(r"$n$", labelpad=0)
    ax[panel].legend(frameon=False, loc="upper left")
    ax[panel].spines["right"].set_visible(False)
    ax[panel].spines["top"].set_visible(False)

ax["A"].set_ylabel(r"$\# \textrm{ Matrix products}$")
ax["A"].set_yticks([0, 100, 200, 300])
ax["A"].set_yticklabels([r"$0$", r"$100$", r"$200$", r"$300$"])
fig.savefig("../figures/benchmark_matrix_products.pdf", bbox_inches="tight")
# %%
