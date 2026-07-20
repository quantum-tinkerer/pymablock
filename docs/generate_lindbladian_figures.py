"""Generate the matrix-pattern figures for the Lindbladian PT note."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

OUTPUT = Path(__file__).parent / "source" / "_static"
TEXT = "#263640"
GRID = "#cad2cd"
EMPTY = "#f7f8f5"
TRACE = "#b9dbe8"
MASK = "#e98b68"
COHERENT = "#7fbd96"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "svg.fonttype": "none",
        "svg.hashsalt": "pymablock-lindbladian-figures",
    }
)


def save_svg(fig, filename):
    """Save deterministic SVG without Matplotlib's trailing path whitespace."""
    path = OUTPUT / filename
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    normalized = "\n".join(line.rstrip() for line in path.read_text().splitlines())
    path.write_text(normalized + "\n")


def draw_mask(ax, values, labels, colors, title, *, cell_labels=False):
    """Draw a square categorical mask with operator-pair tick labels."""
    size = values.shape[0]
    ax.imshow(
        values,
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=len(colors) - 0.5,
        interpolation="none",
    )
    ax.set_xticks(range(size), labels=labels)
    ax.set_yticks(range(size), labels=labels)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, weight="semibold", pad=13)
    if cell_labels:
        for row in range(size):
            for column in range(size):
                ax.text(
                    column, row, labels[row] + labels[column], ha="center", va="center"
                )


def mask_pairing_figure():
    """Draw the trace- and Hermiticity-compatible mask orbits."""
    labels = ["trace", "p", "p*", "r", "s"]
    values = np.zeros((5, 5), dtype=int)
    values[0, :] = 1
    values[1, 3] = 2
    values[2, 3] = 2

    fig, ax = plt.subplots(figsize=(5.2, 4.25), layout="constrained")
    draw_mask(
        ax,
        values,
        labels,
        [EMPTY, TRACE, MASK],
        "One- and two-element dagger orbits",
    )
    ax.set_xlabel("input mode", labelpad=9)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("output mode", labelpad=9)
    ax.plot([3.55, 3.7, 3.7, 3.55], [0.55, 0.55, 2.45, 2.45], color=MASK, lw=1.5)
    ax.text(3.8, 1.5, "paired entries", ha="left", va="center")
    ax.legend(
        handles=[
            Patch(facecolor=TRACE, label="protected trace row"),
            Patch(facecolor=MASK, label="select both or neither"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.29),
        ncol=2,
        frameon=False,
    )
    save_svg(fig, "lindblad_mask_pairing.svg")
    plt.close(fig)


def unitary_lift_figure():
    """Draw a Hamiltonian mask and its lift to Liouville space."""
    n = 3
    operator_labels = [str(index) for index in range(n)]
    liouville_labels = [f"{i}{j}" for i in range(n) for j in range(n)]

    hamiltonian_mask = np.zeros((n, n), dtype=bool)
    hamiltonian_mask[0, 2] = hamiltonian_mask[2, 0] = True

    or_lift = np.zeros((n**2, n**2), dtype=bool)
    commutator = np.zeros_like(or_lift)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ell in range(n):
                    row, column = n * i + j, n * k + ell
                    or_lift[row, column] = (
                        hamiltonian_mask[i, k] or hamiltonian_mask[j, ell]
                    )
                    commutator[row, column] = (j == ell and hamiltonian_mask[i, k]) or (
                        i == k and hamiltonian_mask[ell, j]
                    )

    assert np.all(commutator <= or_lift)
    assert np.array_equal(or_lift, or_lift.T)
    assert np.array_equal(commutator, commutator.T)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.2, 4.0),
        gridspec_kw={"width_ratios": [0.55, 1, 1], "wspace": 0.3},
    )
    draw_mask(
        axes[0],
        hamiltonian_mask.astype(int),
        operator_labels,
        [EMPTY, MASK],
        "Hamiltonian mask m",
    )
    axes[0].set_xlabel(r"$m_{02}=m_{20}=1$", labelpad=8)
    draw_mask(
        axes[1],
        or_lift.astype(int),
        liouville_labels,
        [EMPTY, MASK],
        "OR lift",
    )
    axes[1].set_xlabel("ket or bra index changes", labelpad=8)
    draw_mask(
        axes[2],
        commutator.astype(int),
        liouville_labels,
        [EMPTY, COHERENT],
        "Commutator support",
    )
    axes[2].set_xlabel("one index changes at a time", labelpad=8)
    for ax in axes[1:]:
        ax.tick_params(labelsize=7)

    fig.legend(
        handles=[
            Patch(facecolor=MASK, label="gauge-compatible mask"),
            Patch(facecolor=COHERENT, label="coherent subset"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
    )
    save_svg(fig, "lindblad_unitary_lift.svg")
    plt.close(fig)


if __name__ == "__main__":
    mask_pairing_figure()
    unitary_lift_figure()
