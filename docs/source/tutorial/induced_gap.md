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

This tutorial demonstrates how to efficiently use big numerical Hamiltonians,
and how _Pymablock_ integrates with [Kwant](https://kwant-project.org/).
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
import matplotlib.pyplot as plt

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

To get the Hamiltonian, we use the following values for $\mu_n$, $\mu_{sc}$,
$\Delta$, $t$, and $t_{\text{barrier}}$. We extract the barrier on its own to treat its presense perturbatively. Additionally, we apply a linear change of the chemical potential along the length of the system to model a weak electric field.

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

The Hamiltonian is large, more than what diagonalization can handle without extra effort

```{code-cell} ipython3
h_0.size
```

Therefore, we will use of the implicit mode of _Pymablock_ in order to block
diagonalize the Hamiltonian.
To consider the low energy degrees of freedom, we need orthonormal eigenvectors of the relevant subspace, associated
with the $4$ eigenvalues closest to $E=0$.

```{code-cell} ipython3
vals, vecs = eigsh(h_0, k=4, sigma=0)
vecs, _ = scipy.linalg.qr(vecs, mode="economic")  # orthogonalize
```

We can now define the block diagonalization routine and compute the few lowest orders of the effective Hamiltonian.

```{code-cell} ipython3
%%time

H_tilde, *_ = block_diagonalize([h_0, barrier, delta_mu], subspace_eigenvectors=[vecs])
```

We see that we have obtained the effective model in only a few seconds. For convenience, we collect the first three orders in each parameter in an appropriately sized tensor.

```{code-cell} ipython3
:tags: []

# Combine all the perturbative terms into a single 4D array
fill_value = np.zeros((), dtype=object)
fill_value[()] = np.zeros_like(H_tilde[0, 0, 0, 0])
H_tilde = np.array(np.ma.filled(H_tilde[0, 0, :3, :3], fill_value).tolist())
```

We can now, for instance, calculate the gap energy in the dot depending on the barrier coupling

```{code-cell} ipython3
:tags: []

def effective_energies(barrier_value, delta_mu_value):
    parms = barrier_value**np.arange(3).reshape(-1,1,1,1)*delta_mu_value**np.arange(3).reshape(1,-1,1,1)
    h_evaluate = np.sum(H_tilde*parms,axis=(0,1))
    return scipy.linalg.eigh(h_evaluate)[0]
```

```{code-cell} ipython3
:tags: []

barrier_vals = np.array([0,0.5,1])
delta_mu_vals = np.linspace(0,10e-4,num=101)

results = [np.array([effective_energies(bar, dmu) for dmu in delta_mu_vals]) for bar in barrier_vals]

plt.figure(figsize=(10,6),dpi=200)
color_cycle= ["#5790fc", "#f89c20", "#e42536"]
[[plt.plot(delta_mu_vals, results[j][:,i], color=color_cycle[j]) for i in range(4)]for j in range(3)]
plt.xlabel(r'$\delta_\mu$')
plt.ylabel(r'$E$')
plt.show()
```

As expected, the degeneracy because of the asymetry is lifted when the dots are coupled to the superconductor. In addition, we recognize how the proximity gap of the dots increases with the coupling strength.
