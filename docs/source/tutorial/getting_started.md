---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
mystnb:
    execution_mode: 'inline'
---
# Getting started

This tutorial demonstrates how to use _Lowdin_ on numerical inputs.

## Minimal example

The most minimal example is a diagonal Hamiltonian with two subspaces
{math}`A` and {math}`B`, coupled by a perturbation.

### 1. Define a Hamiltonian

Let's start by defining these.

```{code-cell} ipython3
import numpy as np

# Diagonal unperturbed Hamiltonian
H_0 = np.diag([-1., -1., 1., 1.]) # shape (4, 4)

# Random Hermitian matrix as a perturbation
H_1 = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
H_1 += H_1.conj().T
H_1 = 0.2 * H_1
```

While {math}`H_0` has subspaces separated in energy, {math}`H_1` couples these.

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

### 2. Set up _Lowdin_

We define
```{code-cell} ipython3

from lowdin import block_diagonalize

hamiltonian = [H_0, H_1]

H_tilde, U, U_adjoint = block_diagonalize(hamiltonian, subspace_indices=[0, 0, 1, 1])
```

### 3. Get the perturbative results

We obtain a second order correction to the occupied subspace as
```{code-cell} ipython3
H_tilde[0, 0, 2]
```

All corrections to the occupied subspace to 2th order can be computed like
```{code-cell} ipython3
H_tilde[0, 0, :3]
```
which is a `numpy.MaskedArray`.

The final block-diagonalized Hamiltonian up to second order looks is
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

## A non-diagonal Hamiltonian

In general, {math}`H_0` is not in its eigenbasis, and if this is the case, we have
several options.
One option is to provide a customized `solve_sylvester` function to `block_diagonalize`.
A better option is to bring {math}`H_0` and {math}`H_1` to the eigenbasis of
{math}`H_0` by providing the eigenvectors to `block_diagonalize` directly.

### 1. Define a Hamiltonian
Let's initialize a random Hamiltonian and a perturbation
```{code-cell} ipython3
# Define a random Hamiltonian
H_0 = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
H_0 += H_0.conj().T

# Define a random.random perturbation
H_1 = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
H_1 += H_1.conj().T
H_1 = 0.2 * H_1
```

This time, {math}`H_0` is not in its eigenbasis
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

### 2. Set up _Lowdin_

We define
```{code-cell} ipython3

from lowdin import block_diagonalize

hamiltonian = [H_0, H_1]
_, evecs = np.linalg.eigh(H_0)
subspace_eigenvectors = [evecs[:, :2], evecs[:, 2:]]

H_tilde, U, U_adjoint = block_diagonalize(
  hamiltonian, subspace_eigenvectors=subspace_eigenvectors
)
```

```{important}
The Hamiltonian will be projected using `subspace_vectors`, such that it
becomes diagonal. Therefore, the perturbation theory is carried out in this
basis. In this case, `U` is the unitary transformation that block-diagonalizes
the Hamiltonian from the eigenbasis of the projected diagonal `H_0`.
```

### 3. Get the perturbative results

The final block-diagonalized Hamiltonian up to second order looks is
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
