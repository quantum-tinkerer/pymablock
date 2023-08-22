# Introduction

## Effective models

**Effective models enable the study of complex physical systems by reducing the space of interest to a low energy one.**

**To find an effective Hamiltonian, we use perturbative approaches, like a SW transformation or Lowdin perturbation theory.**

**Even though these methods are standard, their algorithm is computationally expensive, scaling poorly for large systems and high orders.**
Similar to Lowdin perturbation theory or the Schrieffer–Wolff transformation,
Pymablock solves Sylvester's equation and imposes unitarity at every order.
However, differently from other approaches, Pymablock uses efficient algorithms
by choosing an appropriate parametrization of the series of the unitary
transformation.
As a consequence, the computational cost of every order scales linearly with
the order, while the algorithms are still mathematically equivalent.

**We develop an efficient algorithm capable of symbolic and numeric computations and make it available in Pymablock.**

**Pymablock considers a Hamiltonian as a series of $2 \times 2$ block operators
and finds a minimal unitary transformation that separates its subspaces.**
Pymablock considers a Hamiltonian as a series of $2\times 2$ block operators
with the zeroth order block-diagonal.
To carry out the block-diagonalization procedure, Pymablock finds a minimal
unitary transformation $U$ that cancels the off-diagonal block of the
Hamiltonian order by order.

\begin{equation}
H = \begin{pmatrix}H_0^{AA} & 0 \\ 0 & H_0^{BB}\end{pmatrix} + \sum_{i\geq 1} H_i,\quad
U = \sum_{i=0}^\infty U_n
\end{equation}

The result of this procedure is a perturbative series of the transformed
block-diagonal Hamiltonian.

\begin{equation}
\tilde{H} = U^\dagger H U=\sum_{i=0}
\begin{pmatrix}
\tilde{H}_i^{AA} & 0 \\
0 & \tilde{H}_i^{BB}
\end{pmatrix}.
\end{equation}

**Building an effective model with Pymablock is easy, its core is a versatile block diagonalization routine.**
