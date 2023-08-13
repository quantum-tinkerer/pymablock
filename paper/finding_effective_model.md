# Constructing an effective model

## Pymablock workflow

**The workflow of Pymablock consists of three steps.**
Building an effective model using Pymablock is a three step process:

* Define a Hamiltonian
* Call `pymablock.block_diagonalize`
* Request the desired order of the effective Hamiltonian


The following code snippet shows how to use Pymablock to compute the fourth
order correction to an effective Hamiltonian $\tilde{H}$ from a perturbation
$H_p$ to an unperturbed Hamiltonian $H_0$.

```python
from pymablock import block_diagonalize

# Define perturbation theory
H_tilde, *_ = block_diagonalize([h_0, h_p], subspace_eigenvectors=[vecs_A, vecs_B])

# Request 4th order correction to the effective Hamiltonian
H_AA_4 = H_tilde[0, 0, 4]
```

<!-- **Depending on the input Hamiltonian, Pymablock uses specific routines to find
the effective model, so that symbolic expressions are compact and numerics are
efficient.** -->
The function `block_diagonalize` interprets the Hamiltonian and calls the block
diagonalization routines depending on the type and sparsity of the input, so
that symbolic expressions are compact and numerics are efficient.
This is the main function of Pymablock, and it is the only one that the user
needs to call.
It first output is a multivariate series whose terms are different blocks and
orders of the effective Hamiltonian.
Calling `block_diagonalize` is not computationally expensive, because the
terms of the series are only computed when requested.

## k.p model of bilayer graphene

<!-- **We use bilayer graphene to illustrate how to use Pymablock with analytic models.** -->
To illustrate how to use Pymablock with analytic models, we consider two layers
of graphene stacked on top of each other.
Our goal is to find the low energy model near the $\mathbf{K}$ point, following Ref.
[McCann_2013](doi:10.1088/0034-4885/76/5/056503).

First, we construct the k.p Hamiltonian of bilayer graphene from the
tight-binding model shown in the figure below.

```{figure} figures/bilayer.svg
:name: bilayer
:alt: Crystal structure and hopping of bilayer graphene
:width: 250px
:align: center

Crystal structure and hoppings of bilayer graphene.
```

The main features of the model are:

* The unit cell is spanned by vectors $\mathbf{a}_1 = (1/2, \sqrt{3}/2)$ and $\mathbf{a}_2=(-1/2, \sqrt{3}/2)$.
* The unit cell contains 4 atoms with wave functions $\phi_{A,1}, \phi_{B,1}, \phi_{A,2}, \phi_{B,2}$.
* The hoppings within each layer are $t_1$.
* The hopping between atoms that are on top of each other is $t_2$.
* The layers have an onsite potential $\pm m$.

### Defining a symbolic Hamiltonian

We define the Bloch Hamiltonian using the Sympy package for symbolic Python
[sympy](10.7717/peerj-cs.103).

```{embed} # cell-1-finding_effective_model
```

where $\alpha(\mathbf{k}) = 1 + e^{i \mathbf{k'} \cdot (\mathbf{a}_1 +
\mathbf{a}_2)}$ and $\mathbf{k'} = (4\pi/3 + k_x, k_y)$, because we choose
$\mathbf{K}=(4\pi/3, 0)$ as the reference point for the k.p effective model.

### Defining the perturbative series

<!-- **We define the perturbative series** -->
To call `block_diagonalize`, we use the eigenvectors of the unperturbed
Hamiltonian$H(\alpha = m = 0)$.

\begin{align}
v_{A,1} &= \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix} &
v_{A,2} &= \begin{pmatrix} 0 \\ 1 \\ 0 \\ 1 \end{pmatrix} &
v_{B,1} &= \frac{1}{\sqrt{2}} \begin{pmatrix} 0 \\ 0 \\ -1 \\ 1 \end{pmatrix} &
v_{B,2} &= \frac{1}{\sqrt{2}} \begin{pmatrix} 0 \\ 0 \\ 1 \\ 1 \end{pmatrix}
\end{align}

These determine the basis on which the perturbative corrections are computed
and $A$, the subspace of interest for the effective model.
Then, we substitute $\alpha(\mathbf{k})$ into the Hamiltonian, and define the
block diagonalization routine using that $k_x$, $k_y$, and $m$ are perturbative
parameters.

```{embed} # cell-4-finding_effective_model
```

Here `symbols` specifies the perturbative parameters in the order of variables
in the perturbative series.

### Requesting the effective Hamiltonian

We need corrections up to third order in momentum to compute the standard
quadratic dispersion of bilayer graphene and trigonal warping.
Therefore, we define second and third order terms in momentum and group them
total power of momentum.

```{embed} # cell-6-finding_effective_model
```

:::{admonition} Grouping higher order terms
:class: dropdown info

The above manual definition of `k_square` and `k_cube` becomes cumbersome for
higher orders or dimensions.
Instead, we can use the `np.mgrid` and select the terms we need by total power
like this:

```{embed} # cell-7-finding_effective_model
```

:::

Before we saw that querying `H_tilde` returns the results in a numpy array.
To gather different entries into one symbolic expression, we define a
convenience function that sums several orders together.
This uses the `numpy.ma.MaskedArray.compressed` method of masked numpy arrays,
and simplifies the resulting expression.

```{embed} # cell-8-finding_effective_model
```

Finally, we are ready to obtain the result.

```{embed} # cell-9-finding_effective_model
```

The first term contains the standard quadratic dispersion of bilayer graphene with a gap.
The second term contains trigonal warping and the coupling between the gap and momentum.

## Induced gap in a double quantum dot

**Large systems pose an additional challenge due to the scaling of linear algebra routines for large matrices.**
Large systems pose an additional challenge due to the scaling of linear algebra
routines for large matrices.
Pymablock handles large systems by using sparse matrices and avoiding the
construction of the full Hamiltonian.
We illustrate its efficiency with a model of a double quantum dot and a
superconductor with a tunnel barrier in between.

_(Include figure with scheme of the system)_

### Building the Hamiltonian with Kwant

**We use Kwant to build the Hamiltonian of the system.**
We use the Kwant package [kwant](doi:10.1088/1367-2630/16/6/063065) to build
the Hamiltonian of the system.
In the following code, we define a square lattice of $L \times W = 200 \times
40$ sites with $2$ orbitals per unit cell with the superconducting region in
the middle and the quantum dots on the sides.

```{embed} # cell-10-finding_effective_model
```

Here `mu_n` and `mu_sc` are the chemical potentials of the normal and
superconducting regions, respectively, `Delta` is the superconducting gap, and
`t` is the hopping amplitude within each region.
The barrier strength between the quantum dots and the superconductor is `t_barrier`.

We can now plot the system and finalize it

```{embed} # cell-11-finding_effective_model
```

In the plot the blue regions are the left and right quantum dots, while the
superconductor is the red region in the middle.

We see that the system is large: with this many sites even storing all the
eigenvectors would take 60 GB of memory.
We must therefore use sparse matrices, and may only compute a few eigenvectors.
In this case, perturbation theory allows us to compute the effective
Hamiltonian of the low energy degrees of freedom.

To get the unperturbed Hamiltonian, we use the following values for $\mu_n$,
$\mu_{sc}$, $\Delta$, $t$, and $t_{\text{barrier}}$.

```{embed} # cell-12-finding_effective_model
```

The barrier strength and the asymmetry of the dot potentials are the two
perturbations that we vary.

```{embed} # cell-13-finding_effective_model
```

### Define the perturbative series

In the implicit mode, Pymablock computes the perturbative series without
knowing the eigenvectors of one of the Hamiltonian subspaces.

Therefore we compute 4 eigenvectors of the unperturbed Hamiltonian, which
correspond to the 4 lowest eigenvalues closest to $E=0$.
These are the lowest energy Andreev states in two quantum dots.

```{embed} # cell-14-finding_effective_model
```

The orthogonalization is often necessary to do manually because
`scipy.sparse.linalg.eigsh` does not return orthogonal eigenvectors if the
matrix is complex and eigenvalues are degenerate.

We now define the block diagonalization routine and compute the few lowest
orders of the effective Hamiltonian.
Here we only provide the set of vectors of the interesting subspace.
This selects the `pymablock.implicit` method that uses efficient sparse
solvers for Sylvester's equation.

```{embed} # cell-15-finding_effective_model
```

Block diagonalization is now the most time consuming step because it requires
pre-computing several decomposition of the full Hamiltonian.
It is, however, manageable and it only produces a constant overhead.

### Get results

For convenience, we collect the lowest three orders in each parameter in an
appropriately sized tensor.

```{embed} # cell-16-finding_effective_model
```

We see that we have obtained the effective model in only a few seconds.
We can now compute the low energy spectrum after rescaling the perturbative
corrections by the magnitude of each perturbation.

```{embed} # cell-17-finding_effective_model
```

Finally, we plot the spectrum

```{embed} # cell-18-finding_effective_model
```

As expected, the crossing at $E=0$ due to the dot asymmetry is lifted when the
dots are coupled to the superconductor. In addition, we observe how the
proximity gap of the dots increases with the coupling strength.

We also see that computing the spectrum perturbatively is faster than
repeatedly using sparse diagonalization for a set of parameters.
In this example the total runtime of Pymablock would only allow us to compute
the  eigenvectors at around 5 points in the parameter space.
