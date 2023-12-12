# Implementation

**To implement the algorithms, we need a data structure that represents a
multidimensional series of block matrices.**
To implement the algorithms, we need a data structure that represents a
multidimensional series of operators, where dimensions label independent
perturbations.
Additionally, the data structure needs to label blocks, so that the algorithm
supports several forms of input, e.g. dense arrays, sparse matrices, symbolic
expressions, an implicit subspace, or a custom Python object.
Manipulating blocks also allows to compute the effective Hamiltonian without
explicitly constructing the full Hamiltonian, which is useful for Hamiltonians
with a large $BB$ subspace that is costly to store and compute.
To run the recursion, the series needs to be queryable by order and block.
This is useful in cases where the user may want terms that combine different
perturbations, or when the user wants to compute more terms than originally
requested.
Lastly, the data structure needs to support a block-wise multivariate Cauchy
product, which is the main operation in the recursion and is used to compute
the transformed Hamiltonian.

**We address this by defining a `BlockSeries` class.**
To address these requirements, we define a `BlockSeries` Python class and use
it to represent the series of $U$, $H$, and $\tilde{H}$.
A `BlockSeries` is a Python object equipped with a function to compute its
elements and a dictionary to cache the results.
For example, the `BlockSeries` for $\tilde{H}$ has a function that computes
the block-wise multivariate Cauchy product of $U^\dagger H U$.
To get the elements of the series, we implement Numpy array indexing,
which allows us to request several elements at once by using tuples and slices.
In this case, the `BlockSeries` returns a masked array that only contains
non-zero elements, a feature that we use each time we compute the Cauchy
products between series, avoiding unnecessary operations.

**Using the BlockSeries interface allows us to implement a range of
optimizations that go beyond directly implementing the polynomial
parametrization**
Not only does the `BlockSeries` interface allow us to implement the polynomial
parametrization of the unitary transformation, but it allows us to implement
several other optimizations.
By wrapping $BB$ blocks in a SciPy LinearOperator, we can exploit
the associativity of matrix multiplication to first multiply small matrices
and then the large ones, saving computational time and memory.
For example, the term $V_{n-i}^{AB} H_0^{BB} W_i^{BB}$ in $Y_n$ is
systematically computed as $(V_{n-i}^{AB} H_0^{BB}) W_i^{BB}$ instead of
$V_{n-i}^{AB} (H_0^{BB} W_i^{BB})$.
Moreover, when forming the Cauchy product of a Hermitian block, e.g.
diagonal blocks of $U$ and $\tilde{H}$, we only compute half of the matrix
products, and then complex conjugate the result to obtain the rest.
Similarly, we avoid computing $BA$ blocks of $U$ and $\tilde{H}$ by providing
a function to the `BlockSeries` that returns the transpose of the $AB$ blocks.
This way, whenever $BA$ blocks are requested, we first compute the $AB$ blocks,
store them, and then compute the $BA$ blocks directly.
This is only one example of how the `BlockSeries` interface allows us to
implement a symmetrized algorithm, and we leave other symmetries for future
work.
This is an extension that would be useful for systems where $U$ or $\tilde{H}$
vanish due to symmetries, so that the zero blocks can be skipped.

**To deal with an implicit $B$ subspace, we use MUMPS and LinearOperators.**

**Finally, we implement an overall function that interprets the user inputs and
returns a BlockSeries for the transformed Hamiltonian.**
