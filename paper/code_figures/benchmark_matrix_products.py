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
    "dense_first_order": eval_dense_first_order,
    "dense_every_order": eval_dense_every_order,
    "offdiagonal_first_order": eval_offdiagonal_first_order,
    "offdiagonal_every_order": eval_offdiagonal_every_order,
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
fig, ax = plt.subplots(figsize=(figwidth / 2, figwidth))
for structure, counts in multiplication_counts.items():
    ax.plot(
        list(counts.keys()),
        list(counts.values()),
        label=None,
        linestyle="--",
        alpha=0.3,
    )
    ax.scatter(
        list(counts.keys()),
        list(counts.values()),
        label=structure,
    )
ax.set_xlabel(r"$\textrm{Order of } \tilde{\mathcal{H}}^{AA}$")
ax.set_ylabel(r"$\# \textrm{ Multiplications}$")
ax.legend(frameon=False, loc="upper left")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
fig.savefig("../figures/benchmark_matrix_products.pdf", bbox_inches="tight")
# %%
