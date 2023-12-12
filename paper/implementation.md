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
To address these requirements, we define a `BlockSeries` class and use
it to represent the series of $U$, $H$, and $\tilde{H}$.
A `BlockSeries` is a Python object equipped with a function to compute its
elements and a dictionary to cache them.
For example, the `BlockSeries` for $\tilde{H}$ has a function that computes
the block-wise multivariate Cauchy product of $U^\dagger H U$.
To get the elements of the series, we implement Numpy array indexing,
which allows us to request several elements at once by using tuples and slices.
In this case, the `BlockSeries` returns a masked array that only contains
non-zero elements, a feature that we use each time we request elements to
compute the Cauchy products, avoiding unnecessary operations.

**Using the BlockSeries interface allows us to implement a range of
optimizations that go beyond directly implementing the polynomial
parametrization**

**To deal with an implicit $B$ subspace, we use MUMPS and LinearOperators.**

**Finally, we implement an overall function that interprets the user inputs and
returns a BlockSeries for the transformed Hamiltonian.**
