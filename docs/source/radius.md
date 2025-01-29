---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Comparison to generalized Schrieffer-Wolff transformation

Pymablock computes perturbative series using the least action principle and minimizes $\Vert \mathcal{U} - 1\Vert$.
In the $2 \times 2$ block-diagonalization case, this is equivalent to the Schrieffer-Wolff transformation.
In the more general case of multi-block diagonalization, or the problem of eliminating an arbitrary subset of off-diagonal elements, the Schrieffer-Wolff transformation [does not satisfy](https://doi.org/10.48550/arXiv.2408.14637) the least action principle.

In [introducing](algorithms.md) the algorithm, we demonstrate that Pymablock algorithm is more efficient and that it generalizes to multiple perturbations.
However this leaves open the question whether two algorithms have comparable stability and convergence properties.

Let us:

- Implement a prototype selective diagonalization that uses the Schrieffer-Wolff transformation.
- Compare its convergence properties with Pymablock.

## Selective diagonalization using Schrieffer-Wolff transformation

The Schrieffer-Wolff transformation uses a unitary in the form $\exp(\mathcal{S})$ with each order of $\mathcal{S}$ being offdiagonal.
Similar to the Pymablock setting, we set the *remaining* part of $\tilde{\mathcal{H}} = \exp(\mathcal{S}) \mathcal{H} \exp(-\mathcal{S})$ to zero at each order, and use the Sylvester equation to solve for $\mathcal{S}$.

To obtain the transformed Hamiltonian, we apply the Baker-Campbell-Hausdorff formula to $\exp(\mathcal{S}) \mathcal{H} \exp(-\mathcal{S})$:

:::{math}
\exp(\mathcal{S}) \mathcal{H} \exp(-\mathcal{S}) = \mathcal{H} + [\mathcal{S}, \mathcal{H}] + \frac{1}{2!} [\mathcal{S}, [\mathcal{S}, \mathcal{H}]] + \ldots.
:::

Because $S_0 = 0$, the $n$-th order of $\tilde{H}_n$ only contains $S_n$ in the second term in the form $[S_n, H_0]$.
To find $S_n$ we then compute the $n$-th order of all other terms and solve for $S_n$ using the Sylvester equation.

We start from importing the necessary modules.

```{code-cell} ipython3
:tags: [hide-input]
from math import factorial

import numpy as np
from matplotlib import pyplot as plt

import pymablock
from pymablock.series import BlockSeries, zero
from pymablock.block_diagonalization import solve_sylvester_diagonal, is_diagonal

np.set_printoptions(precision=2, suppress=True)

def zero_sum(*terms):
    """Sum that returns a singleton zero if empty and omits zero terms."""
    return sum((term for term in terms if term is not zero), start=zero)

```

To be able to compute high orders of the Baker-Campbell-Hausdorff series efficiently, it is crucial to avoid expanding the nested commutators directly, as this would result in an exponential number of terms.
Instead, we compute them recursively: given 1d series $\mathcal{A}$ and $\mathcal{B}$, we define a 2d series $\mathcal{C}$ where the entry $C_{n, m}$ is the $m$-th order of the $n$-nested commutator of $\mathcal{A}$ and $\mathcal{B}$.
We then use $C_{n, m} = \sum_{k=1}^m A_k C_{n-1, m-k} + \textrm{h.c.}$ (where h.c. stands for Hermitian conjugate), valid when $A$ is antihermitian and $B$ is hermitian.
As an extra technical complication, we need to handle skipping the term containing $[S_n, H_0]$, which we implement using `skip_last` argument.


```{code-cell} ipython3
def baker_camphell_hausdorff(
    A: BlockSeries, B: BlockSeries, skip_last=False,
) -> BlockSeries:
    """Compute the series expansion of exp(A) @ B @ exp(-A) using BCH formula."""
    if not A.n_infinite == B.n_infinite == 1 or not A.shape == B.shape == ():
        raise ValueError("Only 1x1 infinite blocks are supported")

    # Series where the entry (n, m) is the m-th order of the n-nested
    # commutator of A and B

    # We use recursion because the n-th order can be defined as the commutator
    # of n-1-st order with A
    nested_commutators = BlockSeries(
        shape=(),
        n_infinite=2,
    )

    def nested_commutators_eval(*index):
        n, m = index
        if n > m:
            return zero
        if n == 0:
            return B[m]

        product = zero_sum(
            *(
                A[k] @ nc
                for k in range(1, m + 1)
                if (nc := nested_commutators[n - 1, m - k]) is not zero
                and A[k] is not zero
            )
        )
        # We assume that A is antihermitian and B is hermitian
        return product + product.T.conj() if product is not zero else zero

    nested_commutators.eval = nested_commutators_eval

    def bch_eval(m):
        return zero_sum(
            *(nested_commutators[n, m] * (1 / factorial(n)) for n in range(2 * skip_last, m + 1))
        )

    return BlockSeries(
        eval=bch_eval,
        shape=(),
        n_infinite=1,
    )
```

We are now ready to implement the generalized Schrieffer-Wolff transformation, which eliminates terms of of the Hamiltonian specified by a binary `mask` array. Compared to the Pymablock algorithm it is still rather minimal: it only handles numpy arrays, and only a single first-order perturbation.

```{code-cell} ipython3
def schrieffer_wolff(H_0, H_1, mask):
    if not is_diagonal(H_0):
        raise ValueError("H_0 must be diagonal")
    solve_sylvester = solve_sylvester_diagonal((np.diag(H_0),))

    H = BlockSeries(data={(0,): H_0, (1,): H_1}, n_infinite=1, shape=())
    H_0 = BlockSeries(data={(0,): H_0}, n_infinite=1, shape=())
    H_p = BlockSeries(data={(0,): zero, (1,): H_1}, n_infinite=1, shape=())
    S = BlockSeries(data={(0,): zero}, n_infinite=1, shape=())

    exp_S_Hp_exp_mS = baker_camphell_hausdorff(S, H_p)
    exp_S_H0_exp_mS = baker_camphell_hausdorff(S, H_0, skip_last=True)

    def S_eval(m):
        return solve_sylvester(
            zero_sum(exp_S_Hp_exp_mS[m], exp_S_H0_exp_mS[m]) * mask, [0, 0, m]
        )

    S.eval = S_eval

    return baker_camphell_hausdorff(S, H), S
```

## Comparison of the algorithms

Let us first check that the algorithm is correct by confirming that the result is the same in the $2 \times 2$ block-diagonalization case.

```{code-cell} ipython3

# Hamiltonian
np.random.seed(0)
N = 10
H_0 = np.diag(np.arange(N))
H_1 = np.random.rand(N, N) + 1j * np.random.rand(N, N)
H_1 += H_1.conj().T

# Test that 2x2 block-diagonalization agrees with pymablock
mask = np.zeros((N, N), dtype=bool)
mask[: N // 2, N // 2 :] = mask[N // 2 :, : N // 2] = True

H_tilde, *_ = pymablock.block_diagonalize([H_0, H_1], fully_diagonalize={0: mask})
H_tilde_sw, S = schrieffer_wolff(H_0, H_1, mask)

n = 5
np.testing.assert_almost_equal(H_tilde_sw[n], H_tilde[0, 0, n])
```

To compare the convergence properties of the two algorithms, we do the following:

1. Define a simple diagonal $H_0$, a random Gaussian $H_1$, and a random mask for which elements to eliminate.
2. Apply both algorithms to $H_0 + \alpha H_1$ for varying $\alpha$ and the order of the perturbation.
3. Compute the total error as the difference between eigenvalues of $\sum_{n=0}^{N} \tilde{H}_n \alpha^n$ and the exact eigenvalues of $H_0 + \alpha H_1$.
4. Compute the ratio of errors.
5. Plot the results.

```{code-cell} ipython3
:tags: [hide-input]
def compare_schrieffer_wolff_pymablock(H_0, H_1, mask, n_max=100, alpha_max=0.5):
    H_tilde, *_ = pymablock.block_diagonalize([H_0, H_1], fully_diagonalize={0: mask})
    H_tilde_sw, S = schrieffer_wolff(H_0, H_1, mask)

    alpha = np.linspace(0, alpha_max, 100).reshape(1, 1, 1, -1)
    powers = np.arange(n_max).reshape(-1, 1, 1, 1)
    H_orders = H_tilde[0, 0, :n_max]
    H_orders[0] = H_0
    H_orders = np.array(list(H_orders))
    H_orders_sw = np.array(list(H_tilde_sw[:n_max]))

    eigvals_la = np.linalg.eigvalsh(np.cumsum(H_orders[..., None] * alpha ** powers, axis=0).transpose(0, 3, 1, 2))
    eigvals_sw = np.linalg.eigvalsh(np.cumsum(H_orders_sw[..., None] * alpha ** powers, axis=0).transpose(0, 3, 1, 2))
    eigvals_exact = np.linalg.eigvalsh((H_0[..., None] + H_1[..., None] * alpha[0, 0, 0]).transpose(2, 0, 1))[None, ...]

    fig = plt.figure()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    for order in range(n_max):
        ax1.plot(
            alpha.flatten()[1:],
            np.linalg.norm(eigvals_la[order] - eigvals_exact, axis=(0, -1))[1:],
            c=plt.cm.inferno(order / n_max),
            label=f"Least action, order {order}",
        )
        ax1.set_ylabel(r"$\|E_\mathrm{LA} - E_{\mathrm{exact}}\|$")
        ax2.plot(
            alpha.flatten()[1:],
            np.linalg.norm(eigvals_sw[order] - eigvals_exact, axis=(0, -1))[1:]
            / np.linalg.norm(eigvals_la[order] - eigvals_exact, axis=(0, -1))[1:],
            c=plt.cm.inferno(order / n_max),
            label=f"Order {order}",
        )
        ax2.set_ylabel(r"$\|E_{\mathrm{SW}} - E_{\mathrm{exact}}\| / \|E_{\mathrm{LA}} - E_{\mathrm{exact}}\|$")
    for ax in (ax1, ax2):
        ax.semilogy()
        ax.set_xlabel(r"$\alpha$")
    inset_ax = fig.add_axes([0.15, 0.75, 0.2, 0.2])
    inset_ax.imshow(mask, cmap='gray', interpolation='none')
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title('Mask')
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=0, vmax=n_max)), ax=[ax1, ax2], orientation='vertical')
    cbar.set_label('Order')
```

```{code-cell} ipython3
# Random Hamiltonian and mask
np.random.seed(0)
N = 9
H_0 = np.diag(10 * np.random.randn(N))
H_1 = np.random.rand(N, N) + 1j * np.random.rand(N, N)
H_1 += H_1.conj().T

# random mask
mask = np.random.rand(N, N) < 0.3
mask = mask | mask.T
mask = mask & ~np.eye(N, dtype=bool)

# width-3 banded mask
banded_mask = np.ones((N, N), dtype=bool)
banded_mask ^= np.eye(N, dtype=bool) + np.eye(N, k=1, dtype=bool) + np.eye(N, k=-1, dtype=bool)

# mask for 3x3 block-diagonalization
three_block_mask = np.ones((N, N), dtype=bool)
slices = [slice(None, N // 3), slice(N // 3, 2 * N // 3), slice(2 * N // 3, None)]
three_block_mask[slices[0], slices[0]] = three_block_mask[slices[1], slices[1]] = three_block_mask[slices[2], slices[2]] = False

# Comparisons
compare_schrieffer_wolff_pymablock(H_0, H_1, mask, n_max=100, alpha_max=1)
plt.title("Random mask")
compare_schrieffer_wolff_pymablock(H_0, H_1, banded_mask, n_max=40, alpha_max=0.5)
plt.title("Banded mask")
compare_schrieffer_wolff_pymablock(H_0, H_1, three_block_mask, n_max=40, alpha_max=0.5)
plt.title("3x3 block-diagonalization mask")
```

We conclude that regardless of the mask the convergence radius and the error behavior are similar.
