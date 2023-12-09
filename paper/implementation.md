# Implementation

**BlockSeries is the building block of all algorithms.**
The building block of all algorithms are `BlockSeries` objects.
These are series of block operators that can add and multiply
with each other, forming new series with a block structure.
The elements of the series are labeled by integer indices, like
a tensor.
The first $2$ indices label finite dimensions, the $A$ and $B$
blocks, and the rest label infinite dimensions, the perturbative orders.
A `BlockSeries` object may be initialized with data stored for each index, or
without data.
In the latter case, a function that computes the data is passed as an argument,
and the output is cached for future use after an entry is requested.

**Hamiltonians and unitary transformations are `BlockSeries` objects.**
The Hamiltonian is the first `BlockSeries` object that is defined when
`block_diagonalize` is called.
The input Hamiltonian is parsed into a `BlockSeries` object by projecting
it onto its two subspaces, $A$ and $B$.
The parsed Hamiltonian is then passed to a block-diagonalization algorithm:
numerical inputs are passed to `general` or `implicit`, and symbolic inputs are
passed to `expanded`.
The algorithms return `BlockSeries` objects for the unitary transformation $U$
and for the transformed Hamiltonian $\tilde{H}$.
These outputs do not contain data, but their entries are computed on demand
following the recursive block-diagonalization procedure.
Whenever a perturbative order of the transformed Hamiltonian is requested, the
entry of $\tilde{H}$ is computed as a Cauchy product of the `BlockSeries`
objects for $U$, $H$, and $U^\dagger$.
In turn, new entries of $U$ are computed by solving Sylvester's equation
and ensuring that $U$ is unitary.
The entries are then stored in the `BlockSeries` object for $U$, so that
they can be reused in future computations for $\tilde{H}$.

**Computing entries on demand is more efficient than computing all entries at
once.**

**If BlockSeries is Hermitian, we can use the conjugate transpose to speed up
computations.**

**Another source of speedup is the use of LinearOperator.**

**BlockSeries skip zero entries.**
A user may request a specific entry of a `BlockSeries` object, or several
entries at once using numpy's slicing syntax.
