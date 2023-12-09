# Implementation

**BlockSeries is the building block of all algorithms.**
The main building block of all algorithms is a `BlockSeries` object, which
represents a series of $2\times 2$ block operators that can add and multiply
with each other, forming new series with a block structure.
The elements of a `BlockSeries` object are labeled by integer indices, like
a tensor.
The first $2$ indices label finite dimensions, the $A$ and $B$
blocks, and the rest label infinite dimensions, the perturbative orders.
`BlockSeries` may be initialized with data stored for each index, or without data.
In the latter case, the data is computed on demand by a function that is passed
as an argument, and it is cached for future use.

**Hamiltonians and unitary transformations are `BlockSeries` objects.**
The first `BlockSeries` object that is defined when `block_diagonalize` is called
is the Hamiltonian, which is parsed into a `BlockSeries` object by projecting
it onto its two subspaces.
This `BlockSeries` is then passed to one of the algorithms, which return
a `BlockSeries` object for the unitary transformation $U$ and a `BlockSeries`
object for the transformed Hamiltonian $\tilde{H}$.
These two `BlockSeries` objects are initialized without data, such that their
entries are computed following the recursive block-diagonalization procedure.
Consequently, whenever a perturbative order of the transformed Hamiltonian is
requested, the `BlockSeries` object for $\tilde{H}$ computes its entry by
performing a Cauchy product of the `BlockSeries` objects for $U$, $H$, and
$U^\dagger$.
In turn, new entries of $U$ are computed by solving Sylvester's equation
and ensuring that $U$ is unitary.
The entries are then stored in the `BlockSeries` object for $U$, so that
they can be reused in future computations for $\tilde{H}$.

**If BlockSeries is Hermitian, we can use the conjugate transpose to speed up
computations.**

**Another source of speedup is the use of LinearOperator.**
