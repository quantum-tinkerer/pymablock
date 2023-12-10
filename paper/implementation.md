# Implementation

**We need a block series to implement the algorithms efficiently.**
To implement the algorithms, we need a data structure that represents the
Hamiltonian and the unitary transformation in a convenient way.
For this, we have several requirements.
Firstly, because the block-diagonalization algorithms are recursive, we need to
store the data for each perturbative order separately.
Secondly, because the effective Hamiltonian is only one block of the transformed
Hamiltonian, we benefit from storing the data for each block separately,
such that we can compute the effective Hamiltonian without computing the
uninteresting subspace.
This is useful because the $BB$ block is usually much larger than the $AA$
block, and computing it requires expensive matrix products.
Moreover, Hamiltonians may have several perturbations, so we need to generalize
the algorithms previously described to $H$ and $U$ with multiple indices.
These requirements are met by a tensor-like object whose indices label the
blocks and orders of a matrix series.
By implementing addition and multiplication of these objects, we can implement
Pymablock's algorithms avoiding unnecessary computations.
We call this object a `BlockSeries`, the main object in Pymablock.

**BlockSeries may have structure, so we use a function to compute the data on
demand.**
While a tensor-like object is convenient for storing the elements of the
$\tilde{H}$ and $U$ series, it does not take full advantage of the block
structure of these operators.

Things to mention:

- Cache, orders of Htilde independent, and requesting more without having to restart
- Linear operators
- Use of hermiticity in cauchy products
- Function to compute data on demand
- Skip zero entries, call several entries at once, masked arrays
