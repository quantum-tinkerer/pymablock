# Finding an effective model

## Pymablock workflow

**The workflow of Pymablock consists of three steps.**

**Depending on the input Hamiltonian, Pymablock uses specific routines to find the effective model, so that symbolic expressions are compact and numerics are efficient.**

## k.p model of bilayer graphene

**We use bilayer graphene to illustrate how to use Pymablock with analytic models.**
To illustrate how to use Pymablock with analytic models, we consider two layers of graphene stacked on top of each other.
Our goal is to find the low energy model near the $K$ point, like in Ref. [McCann_2013](doi:10.1088/0034-4885/76/5/056503).
To start, we construct the k.p Hamiltonian of bilayer graphene from the tight-binding model shown in the figure below.

```{figure} figures/bilayer.svg
:name: bilayer
:alt: Crystal structure and hopping of bilayer graphene
:width: 250px
:align: center

Crystal structure and hopping of bilayer graphene
```

The physics of this system is not crucial for us, but here are the main features:

- The unit cell is spanned by vectors $\mathbf{a}_1 = (1/2, \sqrt{3}/2)$ and $\mathbf{a}_2=(-1/2, \sqrt{3}/2)$.
- The unit cell contains 4 atoms with wave functions $\phi_{A,1}, \phi_{B,1}, \phi_{A,2}, \phi_{B,2}$.
- The hoppings within each layer are $t_1$.
- The hopping between atoms that are on top of each other is $t_2$.
- The layers have an onsite potential $\pm m$.


### Define a symbolic Hamiltonian

We define the Hamiltonian using Sympy [sympy](10.7717/peerj-cs.103), a Python package for symbolic computations.

```{code-cell} ipython3
import numpy as np
from sympy import symbols, Matrix, sqrt, Eq, exp, I, pi, Add, MatAdd
from sympy.physics.vector import ReferenceFrame
import sympy

t_1, t_2, m = symbols("t_1 t_2 m", real=True)
alpha = symbols(r"\alpha")

H = Matrix(
    [[m, t_1 * alpha, 0, 0],
     [t_1 * alpha.conjugate(), m, t_2, 0],
     [0, t_2, -m, t_1 * alpha],
     [0, 0, t_1 * alpha.conjugate(), -m]]
)
Eq(symbols("H"), H, evaluate=False)
```

where `\alpha` groups the momentum dependent terms in the Hamiltonian.
We define it by making $\mathbf{K}=(4\pi/3, 0)$ the reference point for the $\mathbf{k}$-vector, making $k_x$ and $k_y$ the perturbative parameters.

```{code-cell} ipython3
k_x, k_y = symbols("k_x k_y", real=True)
N = ReferenceFrame("N")
a_1 = (sqrt(3) * N.y + N.x) / 2
a_2 = (sqrt(3) * N.y - N.x) / 2
k = (4 * pi / 3 + k_x) * N.x + k_y * N.y

alpha_k = (1 + exp(I * k.dot(a_1)) + exp(I * k.dot(a_2)))
alpha_k = alpha_k.expand(complex=True, trig=True)
Eq(alpha, alpha_k, evaluate=False)
```

### Define the perturbative series

Now we obtain the eigenvectors of the unperturbed Hamiltonian by substituting the unperturbed values (`sympy.core.basic.Basic.subs`) and diagonalizing (`~sympy.matrices.matrices.MatrixEigen.diagonalize`).

```{code-cell} ipython3
vecs = H.subs({alpha: 0, m: 0}).diagonalize(normalize=True)[0]
vecs
```

After substituting the full expression for $\alpha(k)$ into the Hamiltonian, we are ready to `block_diagonalize` it.
For that we specify which symbols are the perturbative parameters using `symbols` argument. The order of `symbols` is important: it defines the order of variables in the perturbative series.


```{code-cell} ipython3
from pymablock import block_diagonalize

H_tilde = block_diagonalize(
    H.subs({alpha: alpha_k}),
    symbols=(k_x, k_y, m),
    subspace_eigenvectors=[vecs[:, :2], vecs[:, 2:]]
)[0]
```

The names of `symbols` specifying the perturbative parameters are stored in the
`dimension_names` attribute of the result:

```{code-cell} ipython3
H_tilde.dimension_names
```

Now we are ready to specify which calculation to perform.

To compute the standard quadratic dispersion of bilayer graphene and trigonal warping, we need corrections up to third order in momentum.
Let us then group the terms by total power of momentum.
For now this requires an explicit definition of all components, but in the future we plan to automate this step.

```{code-cell} ipython3
k_square = np.array([[0, 1, 2], [2, 1, 0]])
k_cube = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
```

The above manual definition of `k_square` and `k_cube` becomes cumbersome for higher orders or dimensions.
Instead, we can use the `np.mgrid` and select the terms we need by total power like this:
```python
k_powers = np.mgrid[:4, :4]
k_square = k_powers[..., np.sum(k_powers, axis=0) == 2]
k_cube = k_powers[..., np.sum(k_powers, axis=0) == 3]
```

Before we saw that querying `H_tilde` returns the results in a numpy array.
To gather different entries into one symbolic expression, we define a convenience function that sums several orders together.
This uses the `~numpy.ma.MaskedArray.compressed` method of masked numpy arrays, and simplifies the resulting expression.

```{code-cell} ipython3
def H_tilde_AA(*orders):
    return Add(*H_tilde[0, 0, orders[0], orders[1], orders[2]].compressed()).simplify()
```

Finally, we are ready to obtain the result.

```{code-cell} ipython3
mass_term = H_tilde_AA([0], [0], [1])
kinetic = H_tilde_AA(*k_square, 0)
mass_correction = H_tilde_AA(*k_square, 1)
cubic = H_tilde_AA(*k_cube, 0)
MatAdd(mass_term + kinetic, mass_correction + cubic, evaluate=False)
```

The first term contains the standard quadratic dispersion of bilayer graphene with a gap.
The second term contains trigonal warping and the coupling between the gap and momentum.

## Induced gap in a double quantum dot

**Large systems pose an additional challenge due to the scaling of linear algebra routines for large matrices.**
Large systems pose an additional challenge due to the scaling of linear algebra routines for large matrices.
Pymablock handles large systems by using sparse matrices and avoiding the construction of the full Hamiltonian.
We illustrate its efficiency with a model of a double quantum dot and a superconductor with a tunnel barrier in between.

_(Include figure with scheme of the system)_

### Building the Hamiltonian with Kwant

**We use Kwant to build the Hamiltonian of the system.**
We use the Kwant package [kwant](doi:10.1088/1367-2630/16/6/063065) to build the Hamiltonian of the system.
In the following code, we define a square lattice of $L \times W = 200 \times 40$ sites with $2$ orbitals per unit cell with the superconducting region in the middle and the quantum dots on the sides.

```{code-cell} ipython3

import tinyarray as ta
import matplotlib.backends
import scipy.linalg
from scipy.sparse.linalg import eigsh
import numpy as np
import kwant
import matplotlib.pyplot as plt
color_cycle = ["#5790fc", "#f89c20", "#e42536"]

from pymablock import block_diagonalize


sigma_z = ta.array([[1, 0], [0, -1]], float)
sigma_x = ta.array([[0, 1], [1, 0]], float)

syst = kwant.Builder()
lat = kwant.lattice.square(norbs=2)
L, W = 200, 40

def normal_onsite(site, mu_n, t):
    return (-mu_n + 4 * t) * sigma_z

def sc_onsite(site, mu_sc, Delta, t):
    return (-mu_sc + 4 * t) * sigma_z + Delta * sigma_x

syst[lat.shape((lambda pos: abs(pos[1]) < W and abs(pos[0]) < L), (0, 0))] = normal_onsite
syst[lat.shape((lambda pos: abs(pos[1]) < W and abs(pos[0]) < L / 3), (0, 0))] = sc_onsite
syst[lat.neighbors()] = lambda site1, site2, t: -t * sigma_z

def barrier(site1, site2):
    return (abs(site1.pos[0]) - L / 3) * (abs(site2.pos[0]) - L / 3) < 0

syst[(hop for hop in syst.hoppings() if barrier(*hop))] = (
    lambda site1, site2, t_barrier: -t_barrier * sigma_z
)
```

Here `mu_n` and `mu_sc` are the chemical potentials of the normal and superconducting regions, respectively, `Delta` is the superconducting gap, and `t` is the hopping amplitude within each region.
The barrier strength between the quantum dots and the superconductor is `t_barrier`.

We can now plot the system and finalize it

```{code-cell} ipython3

kwant.plot(
    syst,
    fig_size=(10, 6),
    site_color=(lambda site: abs(site.pos[0]) < L / 3),
    colorbar=False,
    cmap="seismic",
    hop_lw=0,
)

syst = syst.finalized()
f"The system has {len(syst.sites)} sites."
```

In the plot the blue regions are the left and right quantum dots, while the
superconductor is the red region in the middle.

We see that the system is large: with this many sites even storing all the eigenvectors would take 60 GB of memory. We must therefore use sparse matrices, and may only compute a few eigenvectors.
In this case, perturbation theory allows us to compute the effective Hamiltonian of the low energy degrees of
freedom.

To get the unperturbed Hamiltonian, we use the following values for $\mu_n$,
$\mu_{sc}$, $\Delta$, $t$, and $t_{\text{barrier}}$.

```{code-cell} ipython3
params = dict(
    mu_n=0.05,
    mu_sc=0.3,
    Delta=0.05,
    t=1.,
    t_barrier=0.,
)

h_0 = syst.hamiltonian_submatrix(params=params, sparse=True).real
```

The barrier strength and the asymmetry of the dot potentials are the two perturbations
that we vary.

```{code-cell} ipython3
barrier = syst.hamiltonian_submatrix(
    params={**{p: 0 for p in params.keys()}, "t_barrier": 1}, sparse=True
).real
delta_mu = (
    kwant.operator.Density(syst, (lambda site: sigma_z * site.pos[0] / L)).tocoo().real
)
```

### Define the perturbative series

In the implicit mode, Pymablock computes the perturbative series without
knowing the eigenvectors of one of the Hamiltonian subspaces.

Therefore we compute 4 eigenvectors of the unperturbed Hamiltonian, which
correspond to the 4 lowest eigenvalues closest to $E=0$.
These are the lowest energy Andreev states in two quantum dots.

```{code-cell} ipython3
%%time

vals, vecs = eigsh(h_0, k=4, sigma=0)
vecs, _ = scipy.linalg.qr(vecs, mode="economic")  # orthogonalize the vectors
```

The orthogonalization is often necessary to do manually because `~scipy.sparse.linalg.eigsh` does not return orthogonal eigenvectors if the matrix is complex and eigenvalues are degenerate.

We now define the block diagonalization routine and compute the few lowest orders of the effective Hamiltonian.
Here we only provide the set of vectors of the interesting subspace.
This selects the `~pymablock.implicit` method that uses efficient sparse solvers for Sylvester's equation.

```{code-cell} ipython3
%%time

H_tilde, *_ = block_diagonalize([h_0, barrier, delta_mu], subspace_eigenvectors=[vecs])
```

Block diagonalization is now the most time consuming step because it requires
pre-computing several decomposition of the full Hamiltonian. It is, however,
manageable and it only produces a constant overhead.

### Get results

For convenience, we collect the lowest three orders in each parameter in an
appropriately sized tensor.

```{code-cell} ipython3
%%time

# Combine all the perturbative terms into a single 4D array
fill_value = np.zeros((), dtype=object)
fill_value[()] = np.zeros_like(H_tilde[0, 0, 0, 0])
h_tilde = np.array(np.ma.filled(H_tilde[0, 0, :3, :3], fill_value).tolist())
```

We see that we have obtained the effective model in only a few seconds.
We can now compute the low energy spectrum after rescaling the perturbative
corrections by the magnitude of each perturbation.

```{code-cell} ipython3
def effective_energies(h_tilde, barrier, delta_mu):
    barrier_powers = barrier ** np.arange(3).reshape(-1, 1, 1, 1)
    delta_mu_powers = delta_mu ** np.arange(3).reshape(1, -1, 1, 1)
    return scipy.linalg.eigvalsh(
        np.sum(h_tilde * barrier_powers * delta_mu_powers, axis=(0, 1))
    )
```

Finally, we plot the spectrum

```{code-cell} ipython3
:tags: [hide-input]
%%time

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

As expected, the crossing at $E=0$ due to the dot asymmetry is lifted when the dots are coupled to the superconductor.
In addition, we observe how the proximity gap of the dots increases with the coupling strength.

We also see that computing the spectrum perturbatively is faster than repeatedly using sparse diagonalization for a set of parameters.
In this example the total runtime of Pymablock would only allow us to compute the  eigenvectors at around 5 points in the parameter space.
