---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Getting started

## Pymablock workflow

Getting started with Pymablock is simple, let's start by importing it
together with `numpy`.

```{code-cell} ipython3
import pymablock
import numpy as np
```

Let's apply perturbation theory to a diagonal Hamiltonian with two subspaces
$A$ and $B$, coupled by a perturbation.

### 1. Define a Hamiltonian

We begin by defining a Hamiltonian and a random perturbation.

```{code-cell} ipython3
# Diagonal unperturbed Hamiltonian
H_0 = np.diag([-1., -1., 1., 1.])

# Random Hermitian matrix as a perturbation
def random_hermitian(n):
    H = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H += H.conj().T
    return H

H_1 = 0.2 * random_hermitian(4)
```

While $H_0$ has two subspaces separated in energy, $H_1$ couples them.

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

fig, (ax_0, ax_1) = plt.subplots(ncols=2)

ax_0.imshow(H_0.real, cmap='seismic', vmin=-2, vmax=2)
ax_0.set_title(r'Unperturbed Hamiltonian $H_0$')
ax_0.set_xticks([])
ax_0.set_yticks([])

ax_1.imshow(H_1.real, cmap='seismic', vmin=-2, vmax=2)
ax_1.set_title(r'Perturbation $H_1$')
ax_1.set_xticks([])
ax_1.set_yticks([]);
```

:::{admonition} Subspaces must be separated
:class: warning
For the perturbation theory to work, the spectra of the two subspaces must
differ.
The larger the smallest energy difference between the two subspaces, the
better the perturbation theory works.
:::

### 2. Define the perturbative series

Most Pymablock users will need only one function:
{autolink}`~pymablock.block_diagonalize`.
It takes all the possible types of input and defines a solution of the
perturbation theory problem as infinite series of the transformed Hamiltonian
$\tilde{H}$ and the corresponding transformation $U$.

```{code-cell} ipython3
from pymablock import block_diagonalize

hamiltonian = [H_0, H_1]

H_tilde, U, U_adjoint = block_diagonalize(hamiltonian, subspace_indices=[0, 0, 1, 1])
```

Here the first term in the `hamiltonian` list is the unperturbed Hamiltonian
$H_0$, and the following terms are the perturbations. The `subspace_indices`
argument defines to which subspace each diagonal term of $H_0$ belongs.

This does do any computations yet, and only defines the answer as an object
that we can query.

Most users will only ever need the diagonal blocks of `H_tilde`, however
Pymablock returns the extra information in case it is needed.

### 3. Get the perturbative results

To get perturbative corrections to the diagonal blocks of the Hamiltonian, we
query `H_tilde` with the block and the order of the
correction.

For example, we obtain a second order correction to the first subspace as

```{code-cell} ipython3
H_tilde[0, 0, 2]
```

where `(0, 0)` is the first subspace ($AA$ block), and `2` means the second
order correction.

Let us also check that the off-diagonal blocks of the Hamiltonian are $0$ to
any order.

```{code-cell} ipython3
H_tilde[0, 1, 3]
```

where `(0, 1)` is the $AB$ block, and `3` is the third order correction.

Just like `H_tilde`, `U` and `U_adjoint` are
{autolink}`~pymablock.series.BlockSeries` objects too.
In most situations these are not necessary, but they can be useful to transform
any other observable to the basis of the `H_tilde`.

To get more than one perturbative correction at a time, we can query `H_tilde`
using `numpy`'s [indexing](https://numpy.org/devdocs/user/basics.indexing.html)
convention.
For example, we query the corrected Hamiltonian of the first subspace up to
second order using

```{code-cell} ipython3
H_tilde[0, 0, :3]
```

The output is a {autolink}`~numpy.ma.MaskedArray` with the same block structure
as if we queried a numpy array with the same indices.
The entries of this array are the Hamiltonian terms themselves, and therefore they may be of different types: here we see that the unperturbed Hamiltonian is stored as a sparse matrix, while the higher orders are numpy arrays.

The final block-diagonalized Hamiltonian up to second order looks like this:

```{code-cell} ipython3
:tags: [hide-input]

import numpy.ma as ma
from scipy.linalg import block_diag

transformed_H = ma.sum(H_tilde[:2, :2, :3], axis=2)
block = block_diag(transformed_H[0, 0], transformed_H[1, 1])

fix, ax_2 = plt.subplots()
ax_2.imshow(block.real, cmap='seismic', vmin=-2, vmax=2)
ax_2.set_title(r'Transformed Hamiltonian $\tilde{H}$')
ax_2.set_xticks([])
ax_2.set_yticks([]);
```

:::{admonition} Masked arrays skip entries
:class: dropdown tip
NumPy's masked arrays are arrays where some entries are masked out because they
are undefined, or because they are not needed, such that they are skipped in
various operations.
In Pymablock, we use masked arrays to skip the terms that are zero, so that
they are skipped in summation and multiplication throughout the algorithm.
To sum over the entries of a masked array, use `np.ma.sum(array)`, for example.
:::

## Further capabilities

Let us now consider a more complex example, where:
- The unperturbed Hamiltonian is not diagonal
- There are multiple perturbative parameters
- Some perturbations are not first order

Because diagonalization is both standard, and not our focus, Pymablock won't do it for us.
However, it will properly treat a non-diagonal unperturbed Hamiltonian if we provide its eigenvectors.

### General Hamiltonians
Let's define a problem with two perturbative parameters:
\begin{equation}
H = H_{00} + \lambda_1 H_{10} + \lambda_1^2 H_{20}  + \lambda_2 H_{01}
\end{equation}

```{code-cell} ipython3
H_00 = random_hermitian(5)  # Unperturbed Hamiltonian
H_10 = random_hermitian(5)  # Linear term in the first perturbative parameter
H_20 = random_hermitian(5)  # Quadratic term in the first perturbative parameter
H_01 = random_hermitian(5)  # Linear term in the second perturbative parameter
hamiltonian = {(0, 0): H_00, (1, 0): H_10, (2, 0): H_20, (0, 1): H_01}
```

The keys of the `hamiltonian` dictionary are tuples of integers, where $i$-th
integer is the order of the $i$-th perturbative parameter.

:::{admonition} Efficiency hint
:class: hint
The Hamiltonian does not contain values of $\lambda_1$ and $\lambda_2$.
Instead, to evaluate $\tilde{H}$, we will provide the values of the perturbative parameters at the last step.

If you want to vary the perturbation strength, providing its values last is more
efficient than recomputing the perturbative series.
:::

Differently from the first example, $H_{00}$ is not diagonal anymore

```{code-cell} ipython3
:tags: [hide-input]
plt.figure(figsize=(3, 3))
plt.imshow(H_00.real, cmap='seismic', vmin=-2, vmax=2)
plt.title(r'Unperturbed $H_{00}$')
plt.xticks([])
plt.yticks([]);
```

### Specifying the subspaces

To define the perturbative series we compute the eigenvectors of $H_{00}$
and split them into two groups that define the $A$ and $B$ subspaces.

```{code-cell} ipython3
_, evecs = np.linalg.eigh(H_00)
subspace_eigenvectors = [evecs[:, :3], evecs[:, 3:]]

H_tilde, U, U_adjoint = block_diagonalize(
  hamiltonian=hamiltonian, subspace_eigenvectors=subspace_eigenvectors
)
```

:::{important}
{autolink}`~pymablock.block_diagonalize` transforms everything to the basis of
`subspace_vectors`, such that, for example, the unperturbed Hamiltonian becomes
diagonal.

Accordingly `U` is the unitary transformation that block-diagonalizes
the Hamiltonian in the eigenbasis of $H_0$.
:::

```{code-cell} ipython3
:tags: [hide-input]

from scipy.sparse import block_diag

H_0 = block_diag(H_tilde[[0, 1], [0, 1], 0, 0]).toarray()

fix, ax_2 = plt.subplots()
ax_2.imshow(H_0.real, cmap='seismic', vmin=-2, vmax=2)
ax_2.set_title(r'$H_{00}$ in its eigenbasis')
ax_2.set_xticks([])
ax_2.set_yticks([]);
```

```{code-cell} ipython3
:tags: [remove-cell]
# Empty the data to illustrate caching clearer
H_tilde._data = {}
one = pymablock.series.one
U._data = {key: one for key, value in U._data.items() if value is one}
U_adjoint = {key: one for key, value in U_adjoint._data.items() if value is one}
```

### Querying the perturbative series

Let us examine how the perturbative series is stored in `H_tilde`, which is a {autolink}`~pymablock.series.BlockSeries` object.

It has a $2\times 2$ block structure corresponding to the $A$ and $B$ subspaces.
The number of its infinite size dimensions is the number of perturbative parameters.

```{code-cell} ipython3
f"{H_tilde.shape=}, {H_tilde.n_infinite=}"
```

{autolink}`~pymablock.series.BlockSeries` defines a way to compute its entries, which are stored in the
internal `_data` attribute.
Before we did any computation, `_data` is empty

```{code-cell} ipython3
f"{H_tilde._data=}, {U._data=}"
```

Querying a multivariate {autolink}`~pymablock.series.BlockSeries` requires specifying the orders of all its perturbations. For example, here we compute a term of $\tilde{H}^{AA}$ of the order $\lambda_1^2\lambda_2^3$

```{code-cell} ipython3
%time H_tilde[0, 0, 2, 3]
```

Computing this term required also computing several orders of `U`, which are now stored in `U._data`

```{code-cell} ipython3
f"{len(H_tilde._data.keys())=}, {len(U._data.keys())=}"
```

That means that querying the same term of $\tilde{H}^{AA}$ is now nearly instantaneous

```{code-cell} ipython3
%time H_tilde[0, 0, 2, 3]
```
