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

# Induced gap in a double quantum dot

This tutorial demonstrates how to efficiently build effective models from big numerical
Hamiltonians.
It also shows how _Pymablock_ integrates with [Kwant](https://kwant-project.org/).
As an example, we consider a tight binding model of a quantum dot and
a superconductor with a tunnel barrier in between.

Let's start by importing the necessary packages

```{code-cell} ipython3
import tinyarray as ta
import matplotlib.backends
import scipy.linalg
from scipy.sparse.linalg import eigsh
import numpy as np
import kwant
import matplotlib.pyplot as plt
color_cycle= ["#5790fc", "#f89c20", "#e42536"]

from pymablock import block_diagonalize
```

Following [Kwant's tutorials](https://kwant-project.org/doc/1/tutorial/) we
start by defining the lattice

```{code-cell} ipython3
sigma_z = ta.array([[1, 0], [0, -1]], float)
sigma_x = ta.array([[0, 1], [1, 0]], float)

syst = kwant.Builder()
lat = kwant.lattice.square(norbs=2)
L, W = 200, 40
```

Next, we define the onsite potential for the quantum dot and superconductor

```{code-cell} ipython3
def normal_onsite(site, mu_n, t):
    return (-mu_n + 4 * t) * sigma_z

def sc_onsite(site, mu_sc, Delta, t):
    return (-mu_sc + 4 * t) * sigma_z + Delta * sigma_x

syst[lat.shape((lambda pos: abs(pos[1]) < W and abs(pos[0]) < L), (0, 0))] = normal_onsite
syst[lat.shape((lambda pos: abs(pos[1]) < W and abs(pos[0]) < L / 3), (0, 0))] = sc_onsite
```

and the hoppings

```{code-cell} ipython3
syst[lat.neighbors()] = lambda site1, site2, t: -t * sigma_z

def barrier(site1, site2):
    return (abs(site1.pos[0]) - L / 3) * (abs(site2.pos[0]) - L / 3) < 0

syst[(hop for hop in syst.hoppings() if barrier(*hop))] = lambda site1, site2, t_barrier: -t_barrier * sigma_z
```

We can now finalize the system and visualize it

```{code-cell} ipython3
syst = syst.finalized()

kwant.plot(syst, fig_size=(10, 6), site_color=(lambda site: abs(syst.sites[site].pos[0]) < L/3), colorbar=False)
plt.show()
```

The quantum dots are at the right and left, in the dark regions.
The superconductor is at the center, in the light region.
To get the Hamiltonian, we use the following values for $\mu_n$,
$\mu_{sc}$, $\Delta$, $t$, and $t_{\text{barrier}}$.

```{code-cell} ipython3
params = dict(
    mu_n=0.05,
    mu_sc=0.3,
    Delta=0.05,
    t=1.,
    t_barrier=0.1,
)
```

However, the Hamiltonian for these values is **too large**, we need more than
60 GiB of memory to alllocate it.
We can instead get the unperturbed Hamiltonian and use of the `implicit` mode of
_Pymablock_ in order to block diagonalize the full Hamiltonian for an
interesting subspace.

The unperturbed Hamiltonian is that where $\mu_n = \mu_{sc} = \Delta = t = 0$
and $t_{\text{barrier}} = 1$.

```{code-cell} ipython3
h_0 = syst.hamiltonian_submatrix(params={**params, "t_barrier": 0}, sparse=True).real
barrier = syst.hamiltonian_submatrix(
    params={**{p: 0 for p in params.keys()}, "t_barrier": 1}, sparse=True
).real
delta_mu = kwant.operator.Density(syst, (lambda site: sigma_z * site.pos[0] / L)).tocoo().real
```

We see that it is indeed large, more than what diagonalization can handle
without extra effort

```{code-cell} ipython3
h_0.size  # number of non-zero entries
```

Therefore, we will use _Pymablock_ and consider the low energy degrees of freedom.
For this, we need the orthonormal eigenvectors of the relevant subspace,
associated with the $4$ eigenvalues closest to $E=0$.

```{code-cell} ipython3
vals, vecs = eigsh(h_0, k=4, sigma=0)
vecs, _ = scipy.linalg.qr(vecs, mode="economic")  # orthogonalize
```

We can now define the block diagonalization routine and compute the few lowest
orders of the effective Hamiltonian.
The barrier and applied dot asymmetry are treated perturbatively.

```{code-cell} ipython3
%%time

H_tilde, *_ = block_diagonalize([h_0, barrier, delta_mu], subspace_eigenvectors=[vecs])
```

We see that we have obtained the effective model in only a few seconds.
For convenience, we collect the first three orders on each parameter in an appropriately sized tensor.

```{code-cell} ipython3
%%time

# Combine all the perturbative terms into a single 4D array
fill_value = np.zeros((), dtype=object)
fill_value[()] = np.zeros_like(H_tilde[0, 0, 0, 0])
h_tilde = np.array(np.ma.filled(H_tilde[0, 0, :3, :3], fill_value).tolist())
```

We can now compute the low energy spectrum

```{code-cell} ipython3
def effective_energies(h_tilde, barrier, delta_mu):
    barrier_powers = barrier ** np.arange(3).reshape(-1, 1, 1, 1)
    delta_mu_powers = delta_mu ** np.arange(3).reshape(1, -1, 1, 1)
    return scipy.linalg.eigvalsh(
        np.sum(h_tilde * barrier_powers * delta_mu_powers, axis=(0, 1))
    )
```

and plot it

```{code-cell} ipython3
:tags: [hide-input]

barrier_vals = np.array([0, 0.5, .75])
delta_mu_vals = np.linspace(0, 10e-4, num=101)
results = [
    np.array([effective_energies(h_tilde, bar, dmu) for dmu in delta_mu_vals])
    for bar in barrier_vals
]

plt.figure(figsize=(10, 6), dpi=200)
[
    plt.plot(delta_mu_vals, result, color=color, label=[f"$t_b={barrier}$"] + 3 * [None])
    for result, color, barrier in zip(results, color_cycle, barrier_vals)
]
plt.xlabel(r"$\delta_\mu$")
plt.ylabel(r"$E$")
plt.legend();
```

As expected, the crossing at $E=0$ due to the dot asymmetry is lifted when the dots are coupled to the superconductor. In addition, we observe how the proximity gap of the dots increases with the coupling strength.
