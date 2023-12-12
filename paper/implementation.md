# Implementation

**To implement the algorithms, we need a data structure that represents a
multidimensional series of block matrices.**
-> Linear algebra (formats: implicit)
-> Queryable by order and block -> complicated to give orders beforehand
-> Cauchy product

**We address this by defining a `BlockSeries` class.**
-> Definition of how to compute new elements
-> Cache in dictionary
-> Numpy array element access
-> Skip zeros when requesting arrays of data

**Using the BlockSeries interface allows us to implement a range of
optimizations that go beyond directly implementing the polynomial
parametrization**

**To deal with an implicit $B$ subspace, we use MUMPS and LinearOperators.**

**Finally, we implement an overall function that interprets the user inputs and
returns a BlockSeries for the transformed Hamiltonian.**
