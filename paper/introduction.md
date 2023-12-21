# Introduction

## Effective models

**Effective models enable the study of complex physical systems by reducing the
space of interest to a low energy one.**
Effective models enable the study of complex physical systems by reducing the
dimensionality of the Hilbert space.
The effective subspace and the remaining Hilbert space are decoupled and
separated by an energy gap.
As a consequence, the physics of the effective model are sufficient to describe
the low energy properties of the original system.

**To find an effective Hamiltonian, we use perturbative approaches, like a SW
transformation or Lowdin perturbation theory.**
Common approaches to construct an effective Hamiltonian are the Schrieffer–Wolff
Schrieffer–Wolff transformation
[Schrieffer1966](doi:10.1103/PhysRev.149.491), [Bravyi2011](doi:10.1016/j.aop.2011.06.004)
and Lowdin partitioning [Lowdin1951](doi:10.1063/1.1745671).
Both methods are perturbative and, as input, they take a Hamiltonian and a
perturbation, together with the subspaces to decouple.
Then, they find the unitary transformation that block-diagonalizes the
Hamiltonian for each perturbative order recursively.
These methods are standard when working with superconducting circuits,
quantum dot physics, density functional theory, k.p models, and other
systems where the physics of interest lies in the low energy states.

**Even though these methods are standard, their algorithm is computationally
expensive, scaling poorly for large systems and high orders.**
Constructing an effective Hamiltonian, is however, a computationally expensive
task.
This is a consequence of the exponential parametrization of the unitary
transformation in a Schrieffer–Wolff transformation, which requires computing
an exponentially growing number of matrix products per order.
Big systems, like those of many-body physics, bosonic Hamiltonians, and
otherwise large Hilbert spaces, are thus expensive to compute.
Similarly, high orders and combined perturbations are also costly, because they
require computing all the terms of the previous orders too.
Aside from the scaling, a Schrieffer–Wolff transformation also requires
truncating the results, effectively wasting computational resources.

**We develop an efficient algorithm capable of symbolic and numeric
computations and make it available in Pymablock.**
In this work, we introduce an algorithm to construct effective models
efficiently.
Our algorithm scales linearly with the perturbative order, does not require
truncating the outputs, and treats multiple perturbations independently.
Its performance makes it possible to find effective Hamiltonians for a variety
of systems, numerical and symbolic, and with several perturbations.
We make the algorithm available via the open source package Pymablock, for
Python matrix block diagonalization of Hamiltonians, a versatile tool for
the study of complex physical systems.

**Pymablock considers a Hamiltonian as a series of $2 \times 2$ block operators
and finds a minimal unitary transformation that separates its subspaces.**
Pymablock considers Hamiltonians as series of $2\times 2$ block operators.
The zeroth order is block-diagonal, and the perturbative orders couple
blocks to each other and within themselves.
To carry out the block-diagonalization procedure, Pymablock finds a minimal
unitary transformation $U$ that cancels the off-diagonal block of the
Hamiltonian order by order:

\begin{equation}
H = \begin{pmatrix}H_0^{AA} & 0 \\ 0 & H_0^{BB}\end{pmatrix} + \sum_{i\geq 1} H_i,\quad
U = \sum_{i=0}^\infty U_n,
\end{equation}

where $H_i$ and $U_i$ are proportional to an $i$-th order contribution on the
perturbative parameter.
Throughout this work, we use $A$ and $B$ to denote the low and high energy
subspaces, respectively.
The result of this procedure is a perturbative series of the transformed
block-diagonal Hamiltonian.

:::{math}
:label: eq:transformed_hamiltonian
\tilde{H} = U^\dagger H U=\sum_{i=0}^{\infty}
\begin{pmatrix}
\tilde{H}_i^{AA} & 0 \\
0 & \tilde{H}_i^{BB}
\end{pmatrix}.
:::

**Pymablock offers the same solution as traditional methods.**
Similar to Lowdin perturbation theory or the Schrieffer–Wolff transformation,
Pymablock solves Sylvester's equation and ensures that the transformation
$U$ is unitary order by order.
However, differently from other approaches, Pymablock uses efficient algorithms
by choosing an appropriate parametrization of the series of the unitary
transformation.
As a consequence, the computational cost of every order scales linearly with
the order, while the effective Hamiltonians are still mathematically equivalent.
