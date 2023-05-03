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

Here we demonstrate how to use Lowdin on numerical inputs.
## Minimal example

The most minimal example is a diagonal Hamiltonian with two subspaces
{math}`AA` (occupied states) and {math}`BB` (unoccupied states), coupled by a perturbation.
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

Let's now plot the Hamiltonians to visualize the subspaces,
```{code-cell} ipython3
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
Here we see that while {math}`H_0` has subspaces clearly separated in energy, {math}`H_1`
couples these.

```{important}
The unperturbed Hamiltonian **must** have at least two subspaces separated by an energy
gap to apply Lowdin.
```

We identify the occupied and unoccupied subspaces of the unperturbed Hamiltonian
```{code-cell} ipython3
H_0_AA = H_0[:2, :2] # occupied subspace
H_0_BB = H_0[2:, :2] # unoccupied subspace
```
and the respective blocks of the perturbation
```{code-cell} ipython3
H_p_AA = {(1, ): H_1[:2, :2]}
H_p_BB = {(1, ): H_1[2:, 2:]}
H_p_AB = {(1, ): H_1[:2, 2:]} # mixes subspaces
```
Here we use `(1, )` to indicate that this is a first order perturbation.

Now we can write the total Hamiltonian as a Block Series,
```{code-cell} ipython3
import lowdin
H = lowdin.to_BlockSeries(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB)
```
from which we can access the occupied block of the {math}`0`-th order
Hamiltonian, `H_0_AA`, as
```{code-cell} ipython3
H[0, 0, 0]
```
Here the first two indices `0, 0` corespond to `A, A`, while the third `0` corresponds
to the {math}`0`-th order.

Similarly, the occupied block of the {math}`1`-st order Hamiltonian is
```{code-cell} ipython3
H[0, 0, 1]
```
which corresponds to the original perturbation.

We can now set up Lowdin by defining
```{code-cell} ipython3
H_tilde, U, U_adjoint = lowdin.general(H)
```
and obtain a second order correction to the occupied subspace as
```{code-cell} ipython3
H_tilde[0, 0, 2]
```

All corrections to the occupied subspace to 2th order can be computed like
```{code-cell} ipython3
H_tilde[0, 0, :3]
```
which is a `numpy.MaskedArray`.

We can obtain the final block-diagonalized Hamiltonian up to second order as
```{code-cell} ipython3
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

**TODO: Finish this section**

**Important aspects:**
* Show alternatives: diagonalize first, provide `solve_sylvester`, use KPM
* Show that one can use different algorithms

In general, we do not have {math}`H_0` in its eigenbasis, and if this is the case, we have
several options.
One option is to bring {math}`H_0` and {math}`H_1` to the eigenbasis of {math}`H_0` by
numerically diagonalizing {math}`H_0` and applying the transformation to {math}`H_1`.
Another option is to provide a customized function to solve Sylvester's Equation.

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