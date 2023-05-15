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
---

# Induced gap and crossed Andreev reflection

This tutorial demonstrates how to efficiently use big numerical Hamiltonians,
and how _Lowdin_ integrates with [Kwant](https://kwant-project.org/).
As an example, we will consider a tight binding model of a quantum dot and
a superconductor with a tunnel barrier in between.

Let's start by importing the necessary packages

```{code-cell} ipython3
import tinyarray as ta
import matplotlib.backends
import scipy.linalg
from scipy.sparse.linalg import eigsh
import numpy as np
import kwant

from lowdin import block_diagonalize
```

Following [Kwant's tutorials](https://kwant-project.org/doc/1/tutorial/) we
start by defining the lattice

```{code-cell} ipython3
sigma_z = ta.array([[1, 0], [0, -1]], float)
sigma_x = ta.array([[0, 1], [1, 0]], float)

syst = kwant.Builder()
lat = kwant.lattice.square(norbs=2)
L, W = 100, 20
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

kwant.plot(syst);
```

To get the Hamiltonian, we use the following values for $\mu_n$, $\mu_{sc}$,
$\Delta$, $t$, and $t_{\text{barrier}}$

```{code-cell} ipython3
params = dict(
    mu_n=0.05,
    mu_sc=0.3,
    Delta=0.05,
    t=1.,
    t_barrier=0.1,
)

h_0 = syst.hamiltonian_submatrix(params={**params, "t_barrier": 0}, sparse=True).real
barrier = syst.hamiltonian_submatrix(params={**{p: 0 for p in params.keys()}, "t_barrier": 1}, sparse=True).real
delta_mu = kwant.operator.Density(syst, (lambda site: sigma_z * site.pos[0] / L)).tocoo().real
```

The Hamiltonian is large

```{code-cell} ipython3
h_0.size
```

Therefore, we will use of the implicit mode of _Lowdin_ in order to block
diagonalize the Hamiltonian.
For this, we need orthonormal eigenvectors of the relevant subspace, associated
with the $4$ eigenvalues closest to $E=0$.

```{code-cell} ipython3
vals, vecs = eigsh(h_0, k=4, sigma=0)
vecs, _ = scipy.linalg.qr(vecs, mode="economic")  # orthogonalize
```

We can now define the block diagonalization routine

```{code-cell} ipython3
H_tilde, *_ = block_diagonalize([h_0, barrier, delta_mu], subspace_eigenvectors=[vecs])
```


```{code-cell} ipython3
fill_value = np.zeros((), dtype=object)
fill_value[()] = np.zeros_like(H_tilde[0, 0, 0, 0])
H_tilde = np.array(np.ma.filled(H_tilde[0, 0, :3, :2], fill_value).tolist())
```

```{code-cell} ipython3
H_tilde[np.abs(H_tilde) < 1e-15] = 0
H_tilde
```
