# Implementation

**We need a block series to implement the algorithms efficiently.**
To implement the algorithms, we need a data structure that represents the
Hamiltonian and the unitary transformation in a convenient way.
For this, we have several requirements.
Firstly, because the block-diagonalization algorithms are recursive, we need to
store the data for each perturbative order separately.
Secondly, because the effective Hamiltonian is only one block of the transformed
Hamiltonian, it is more efficient to manipulate the data for each block
separately.
This is useful because the $BB$ block is usually much larger than the $AA$
block, and computing it requires expensive matrix products.
Moreover, Hamiltonians may have several perturbations, so we need to generalize
the algorithms previously described to $H$ and $U$ with multiple indices.
These requirements are met by a tensor-like object whose indices label the
blocks and orders of a matrix series.
We call this object a `BlockSeries`, the main object in Pymablock.

**We add and multiply these tensors in a block-wise fashion.**
To compute the unitary transformation, we need to implement the generalization
of the algorithms described in the previous section using the `BlockSeries`.
Therefore, we need to compute the Cauchy product between the series of
$U^\dagger$, $H$, and $U$ excluding the terms that involve the desired order of
$U$ and $H$ while keeping the block structure.
We implement this by defining a function that computes the block-wise Cauchy
product of a series of `BlockSeries` objects, forming a new `BlockSeries`
with the result.
We call this function `cauchy_dot_product`, and we use it to compute the
transformed Hamiltonian too.

**We manage to reduce the number of matrix products by a factor of two by
exploiting Hermiticity.**
Manipulating the data at a block level brings an additional advantage:
we can use Hermiticity to halve the number of matrix products.
We implement this at every order of the block-diagonalization algorithm, and
in two steps.
Firstly, since $\tilde{H}$ and $U$ are defined as a product of series whose
diagonal blocks are Hermitian, we only need to compute half of the matrix
products from the Cauchy product to find the $AA$ and $BB$ blocks.
The remaining terms are defined as conjugate transposes of the ones we compute,
which is a cheap operation.
Secondly, since the off-diagonal blocks of $U$ are related by anti-Hermiticity,
we only need to compute $V^{AB}$ and $V^{BA} = -(V^{AB})^{\dagger}$.
Once again, this saves us half of the matrix products, a trick that we apply
to $\tilde{H}^{AB} = \tilde{H}^{BA}^{\dagger}$ as well.

**We use functions to compute data on demand and cache the results in a
dictionary.**

**To call the cached data, we implement numpy array element access.**

<!-- Things to mention:

- Cache, orders of Htilde independent, and requesting more without having to restart
- Linear operators
- Use of hermiticity in cauchy products
- Function to compute data on demand
- Skip zero entries, call several entries at once, masked arrays -->
