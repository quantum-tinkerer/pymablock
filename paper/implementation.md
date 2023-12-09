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
One advantage of computing entries on demand is that the user may skip
computations that are not needed.
This is because different orders of the transformed Hamiltonian are
independent of each other.
Therefore requesting a specific order does not require computing previous
orders of $\tilde{H}$, but only of $U$.
This is particularly useful when the user knows beforehand that some orders
vanish because of symmetries, for example.
This feature is also useful when the user desires to compute more orders than
originally requested, because only the missing orders need to be computed.

**We use block structure to speed up computations.**
Another advantage of using `BlockSeries` objects is that we can use the
block structure of every order to speed up computations.
On the one hand, since the $AA$ and $BB$ blocks of the Hamiltonian are
independent of each other, requesting $\tilde{H}_i^{AA}$ does not require
computing $\tilde{H}_i^{BB}$.
This is useful because the $BB$ block is usually much larger than the $AA$
block, and computing it is expensive.
On the other hand, since the algorithm deals with Hermitian and anti-Hermitian
matrices, we can avoid computing matrix products by conjugating and
transposing.
This is useful when computing the entries of $U$, whose off-diagonal blocks are
anti-Hermitian, and to speed up the computation of $\tilde{H}$, whose
diagonal blocks are Hermitian.

**Another source of speedup is the use of LinearOperator.**

**BlockSeries skip zero entries.**
A user may request a specific entry of a `BlockSeries` object, or several
entries at once using numpy's slicing syntax.
