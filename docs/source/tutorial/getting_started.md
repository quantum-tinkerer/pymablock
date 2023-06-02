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

## pymablock basics

Getting started with {{Pymablock}} is simple, let's start by importing it
together with `numpy`.

```{code-cell} ipython3
import pymablock
import numpy as np
```

## Minimal example

Let's apply perturbation theory to a diagonal Hamiltonian with two subspaces
$A$ and $B$, coupled by a perturbation.

### 1. Define a Hamiltonian

We by defining a Hamiltonian and a random perturbation.

```{code-cell} ipython3
# Diagonal unperturbed Hamiltonian
H_0 = np.diag([-1., -1., 1., 1.]) # shape (4, 4)

# Random Hermitian matrix as a perturbation
H_1 = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
H_1 += H_1.conj().T
H_1 = 0.2 * H_1
```

While $H_0$ has subspaces separated in energy, $H_1$ couples them.

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

```{important}
The unperturbed Hamiltonian **must** have at least two subspaces separated by
an energy gap.
```

### 2. Define the perturbative series

Most {{Pymablock}} users will need only one function: {autolink}`~pymablock.block_diagonalize`.
It converts all types of input into a solution of the perturbation theory problem:
infinite series of $\tilde{H}$ and $U$.

```{code-cell} ipython3
from pymablock import block_diagonalize

hamiltonian = [H_0, H_1]

H_tilde, U, U_adjoint = block_diagonalize(hamiltonian, subspace_indices=[0, 0, 1, 1])
```

This does do any computations yet, and only defines the answer as an object
that we can query.

### 3. Get the perturbative results

To get perturbative corrections to the Hamiltonian, we index the transformed
Hamiltonian indicating the block and order of the correction.

For example, we obtain a second order correction to the occupied subspace as

```{code-cell} ipython3
H_tilde[0, 0, 2]
```

where `(0, 0)` is the $AA$ block, and `2` refers to the second order
correction.

{{Pymablock}} uses `numpy`'s convention on
[indexing](https://numpy.org/devdocs/user/basics.indexing.html).
Therefore, all the corrections to the occupied subspace to 2th order can be
computed like

```{code-cell} ipython3
H_tilde[0, 0, :3]
```

The output is a {autolink}`~numpy.ma.MaskedArray` where `zero` values are masked.

The final block-diagonalized Hamiltonian up to second order looks like

```{code-cell} ipython3
:tags: [hide-input]

import numpy.ma as ma

transformed_H = ma.sum(H_tilde[:2, :2, :3], axis=2)
block = np.block([
    [transformed_H[0, 0], transformed_H[0, 1]],
    [transformed_H[1, 0], transformed_H[1, 1]]
])

fix, ax_2 = plt.subplots()
ax_2.imshow(block.real, cmap='seismic', vmin=-2, vmax=2)
ax_2.set_title(r'Transformed Hamiltonian $\tilde{H}$')
ax_2.set_xticks([])
ax_2.set_yticks([]);
```

## A non-diagonal Hamiltonian with multivariate perturbations

Let us now consider a more complex example, where:
- The unperturbed Hamiltonian is not diagonal
- We have a multivariate perturbation

Because diagonalization is both standard, and not our focus, {{Pymablock}} won't do it for us.
However, it will properly treat a non-diagonal unperturbed Hamiltonian if we provide its eigenvectors.

### 1. Define a Hamiltonian
Let's initialize a random Hamiltonian and two perturbations

```{code-cell} ipython3
# Define a random Hamiltonian
H_0 = np.random.random((5, 5)) + 1j * np.random.random((5, 5))
H_0 += H_0.conj().T

# Define two random perturbations
H_1 = np.random.random((5, 5)) + 1j * np.random.random((5, 5))
H_1 += H_1.conj().T
H_1 = 0.2 * H_1

H_2 = np.random.random((5, 5)) + 1j * np.random.random((5, 5))
H_2 += H_2.conj().T
H_2 = 0.2 * H_2
```

This time, $H_0$ is not in its eigenbasis

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

fig, (ax_0, ax_1, ax_2) = plt.subplots(ncols=3)

ax_0.imshow(H_0.real, cmap='seismic', vmin=-2, vmax=2)
ax_0.set_title(r'Unperturbed $H_0$')
ax_0.set_xticks([])
ax_0.set_yticks([])

ax_1.imshow(H_1.real, cmap='seismic', vmin=-2, vmax=2)
ax_1.set_title(r'Perturbation $H_1$')
ax_1.set_xticks([])
ax_1.set_yticks([]);

ax_2.imshow(H_2.real, cmap='seismic', vmin=-2, vmax=2)
ax_2.set_title(r'Perturbation $H_2$')
ax_2.set_xticks([])
ax_2.set_yticks([]);
```

### 2. Define the perturbative series

To set up the block diagonalization routine we compute the eigenvectors of $H_0$
and split them into two groups that define the $A$ and $B$ subspaces.

```{code-cell} ipython3
_, evecs = np.linalg.eigh(H_0)
subspace_eigenvectors = [evecs[:, :3], evecs[:, 3:]]

H_tilde, U, U_adjoint = block_diagonalize(
  hamiltonian=[H_0, H_1, H_2], subspace_eigenvectors=subspace_eigenvectors
)
```

```{important}
The Hamiltonian will be projected using `subspace_vectors`, such that it
becomes diagonal. Therefore, the perturbation theory is carried out in this
basis. In this case, `U` is the unitary transformation that block-diagonalizes
the Hamiltonian in the eigenbasis of the projected diagonal `H_0`.
```

### 3. Get the perturbative results

While the number of subspaces in the Hamiltonian is finite, the order of a
perturbative correction is infinite. `H_tilde` is a
{autolink}`~pymablock.series.BlockSeries` object, and we can access the number of its
subspaces by calling

```{code-cell} ipython3
H_tilde.shape
```

and the number of its infinite dimensions by calling

```{code-cell} ipython3
H_tilde.n_infinite
```

Once again, `H_tilde` is empty since none of the corrections have been requested

```{code-cell} ipython3
H_tilde._data
```

We can get the corrections to the $AA$ block by specifying the individual
order of each perturbation. For example, we can request the second order
correction on the first perturbation and third order correction on the
second perturbation as,

```{code-cell} ipython3
H_tilde[0, 0, 2, 3]
```

The final block-diagonalized Hamiltonian up to second order looks like

```{code-cell} ipython3
:tags: [hide-input]

import numpy.ma as ma

transformed_H = ma.sum(ma.sum(H_tilde[:2, :2, :3, :3], axis=-1), axis=-1)
block = np.block([
    [transformed_H[0, 0], transformed_H[0, 1]],
    [transformed_H[1, 0], transformed_H[1, 1]]
])

fix, ax_2 = plt.subplots()
ax_2.imshow(block.real, cmap='seismic', vmin=-2, vmax=2)
ax_2.set_title(r'Transformed Hamiltonian $\tilde{H}$')
ax_2.set_xticks([])
ax_2.set_yticks([]);
```
