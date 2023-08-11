---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Algorithms for block diagonalization

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

Similar to Lowdin perturbation theory or the Schrieffer–Wolff transformation,
Pymablock solves Sylvester's equation and imposes unitarity at every order.
However, differently from other approaches, Pymablock uses efficient algorithms
by choosing an appropriate parametrization of the series of the unitary
transformation.
As a consequence, the computational cost of every order scales linearly with
the order, while the algorithms are still mathematically equivalent.

**Pymablock has tailored algorithms for block diagonalization of Hamiltonians.**

## General algorithm

**The algorithms of Pymablock rely on decomposing $U$ into two parts.**
The algorithms of Pymablock rely on decomposing $U$, the unitary transformation
that block diagonalizes the Hamiltonian, as a series of Hermitian
block diagonal $W$ and skew-Hermitian and block off-diagonal $V$ terms.
The transformed Hamiltonian is a Cauchy product between the series of
$U^\dagger$, $H$, and $U$.

For brevity we use a single first order perturbation $H_1$ throughout this
document. The generalization to multiple perturbations is straightforward.

**The result of this procedure is a perturbative series of the transformed
block-diagonal Hamiltonian.**
The transformed Hamiltonian at order $n$ is

:::{math}
:label: h_tilde
\begin{align}
\tilde{H}_{n} = \sum_{i=0}^n (W_{n-i} - V_{n-i}) H_0 (W_i + V_i) +
\sum_{i=0}^{n-1} (W_{n-i-1} - V_{n-i-1}) H_1 (W_i + V_i).
\end{align}
:::


**Pymablock finds the unitary transformation recursively, using unitarity and
solving Sylvester's equation at every order.**
To block diagonalize $H_0 + H_1$, Pymablock finds the orders of $W$
such that $U$ is unitary

:::{math}
:label: unitarity
\begin{equation}
W_{n} = - \frac{1}{2} \sum_{i=1}^{n-1}(W_{n-i}W_i - V_{n-i}V_i),
\end{equation}
:::

:::{admonition} Derivation
:class: dropdown info
We evaluate the series
$U^\dagger U + UU^\dagger=2$ and use that $W=W^\dagger$ and $V=-V^\dagger$ to obtain

```{math}
\begin{equation}
\sum_{i=0}^n \left[(W_{n-i} - V_{n-i})(W_i + V_i) +
(W_{n-i} + V_{n-i})(W_i - V_i)\right] = 0.
\end{equation}
```

Using that $W_0=1$, $V_0=0$, expanding, and solving for $W_n$ gives the Eq. {eq}`unitarity`.
:::

and the orders of $V$ by requiring that $\tilde{H}^{AB}_n=0$

```{math}
:label: sylvester
H_0^{AA} V_{n}^{AB} - V_{n}^{AB} H_0^{BB} = Y_{n}.
```

This is known as Sylvester's equation and $Y_{n}$ is a combination of lower
order terms in $H$ and $U$.

::::{admonition} Full expression
:class: dropdown info
The full expression for $Y_n$ is cumbersome already in our simplest case:

:::{math}
:label: y_n
\begin{align}
Y_n=&-
\sum_{i=1}^{n-1}\left[W_{n-i}^{AA}H_0^{AA}V_i^{AB}-V_{n-i}^{AB}
H_0^{BB}W_i^{BB}\right] \\
&-\sum_{i=0}^{n-1}\bigg[W_{n-i-1}^{AA}H_1^{AA}V_i^{AB}+W_{n-i-1}^{AA}
H_1^{AB}W_i^{BB} \\
&\quad \quad \quad -V_{n-i-1}^{AB}(H_1^{AB})^\dagger V_i^{AB} -V_{n-i-1}^{AB}
H_1^{BB}W_i^{BB}\bigg]
\end{align}
:::
::::

## Expanded algorithm for analytic Hamiltonians

Pymablock has two algorithms, `pymablock.general` and `pymablock.expanded`.
The `pymablock.general` algorithm implements the procedure outlined here directly.
On the other hand, `pymablock.expanded` simplifies the expressions
for $\tilde{H}_{n}$ (Eq. {eq}`h_tilde`) such that it only depends on $V$ and the
perturbation $H_1$, but not explicitly on $H_0$.

:::{admonition} How this works
:class: dropdown info
The `pymablock.expanded` algorithm first uses
`pymablock.general` with a symbolic input to derive the general
symbolic form for $Y_n$ and $\tilde{H}_n$.
Then it uses the Sylvester's equation for lower orders of $V_n$ to eliminate
$H_0$ from these expressions.
Finally, `pymablock.expanded` replaces the problem-specific $H$ into
the simplified $\tilde{H}$.
:::

As an example, the corrections to the effective Hamiltonian up to fourth
order using `pymablock.expanded` are

```{code-cell} ipython3
:tags: [remove-input]

from operator import mul

from sympy import Symbol, Eq

from pymablock.block_diagonalization import BlockSeries, symbolic

H = BlockSeries(
    data={
        (0, 0, 0): Symbol('{H_{0}^{AA}}'),
        (1, 1, 0): Symbol('{H_{0}^{BB}}'),
        (0, 0, 1): Symbol('{H_{1}^{AA}}'),
        (0, 1, 1): Symbol('{H_{1}^{AB}}'),
        (1, 1, 1): Symbol('{H_{1}^{BB}}'),
    },
    shape=(2, 2),
    n_infinite=1,
)

max_order = 5
hamiltonians = {
  Symbol(f'H_{{{index}}}'): value for index, value in H._data.items()
}
offdiagonals = {
  Symbol(f'V_{{({order},)}}'): Symbol(f'V_{order}') for order in range(max_order)
}

H_tilde, *_ = symbolic(H)

for order in range(max_order):
    result = Symbol(fr'\tilde{{H}}_{order}^{{AA}}')
    display(Eq(result, H_tilde[0, 0, order].subs({**hamiltonians, **offdiagonals})))
```

Here we omitted the superscript $AB$ on all the $V$'s for brevity.

At lower orders, `pymablock.expanded` performs fewer operator
products than `pymablock.general`, and with analytic Hamiltonians
the resulting expressions are simpler.
At high orders, however, `pymablock.expanded` requires exponentially
many terms, unlike `pymablock.general` which only requires a linear
number of terms.

## Implicit algorithm for large Hamiltonians

Solving Sylvester's equation and computing the matrix products are the most
expensive steps of the algorithms for large Hamiltonians.
Pymablock can efficiently construct an effective Hamiltonian of a small
subspace even when the full Hamiltonian is a sparse matrix that is too costly to
diagonalize.
It does so by avoiding explicit computation of operators in $B$ subspace, and by
utilizing the sparsity of the Hamiltonian to compute the Green's function.
To do so, Pymablock uses either the MUMPS sparse solver
[mumps1](doi:10.1137/S0895479899358194),
[mumps2](doi:10.1016/j.parco.2005.07.004) or the KPM method
[kpm](doi:10.1103/RevModPhys.78.275).

This approach was originally introduced in Ref.
[hybrid-kpm](doi:10.48550/arXiv.1909.09649).

:::{admonition} Implementation details
:class: dropdown info
We use the matrix $\Psi_A$ of the eigenvectors of the $A$ subspace to rewrite
the Hamiltonian as

```{math}
H \to \begin{pmatrix}
\Psi_A^\dagger H \Psi_A & \Psi_A^\dagger H P_B \\
P_B H \Psi_A & P_B H P_B
\end{pmatrix},
```

where $P_B = 1 - \Psi_A \Psi_A^\dagger$ is the projector onto the $B$ subspace.
This Hamiltonian is larger in size than the original one because the $B$ block has
additional null vectors corresponding to the $A$ subspace.
This, however, allows to preserve the sparsity structure of the Hamiltonian by applying
$P_B$ and $H$ separately.
Additionally, applying $P_B$ is efficient because $\Psi_A$ is a low rank matrix.
We then perform perturbation theory of the rewritten $H$.

To solve the Sylvester's equation for the modified Hamiltonian, we write it for
every row of $V_n^{AB}$ separately:

```{math}
V_{n, ij}^{AB} (E_i - H_0) = Y_{n, j}
```

This equation is well-defined despite $E_i - H_0$ is not invertible because
$Y_{n}$ has no components in the $A$ subspace.
:::

## Proof of equivalence to Schrieffer-Wolff transformation

**The transformed Hamiltonian is equivalent to that of other perturbative
methods, but the algorithm is efficient.**
Both the Pymablock algorithm and the more commonly used Schrieffer-Wolff
transformation find a unitary transformation $U$ such that $\tilde{H}^{AB}=0$.
They are therefore equivalent up to a gauge choice on each subspace.
We establish the correspondence between the two by demonstrating that this gauge
choice is the same for both algorithms.

Pymablock chooses $U=W+V$, where $W$ is block diagonal Hermitian and
$V$ is block off-diagonal anti-Hermitian.
Then requiring that $U$ is unitary and $\tilde{H}^{AB}=0$ to all orders defines
a unique value for $W$ and $V$.

The Schrieffer-Wolff transformation parameterizes $U = \exp S$, where $S =
\sum_n S_n$ is a series of anti-Hermitian block off-diagonal operators:

```{math}
:label: exp_s_expansion
\begin{align}
U = \exp{\left(S\right)}=\exp{\left(\sum_{n=0}^\infty
S_n\right)} = 1+\sum_{j=1}^\infty \left[\frac{1}{j!}
\left(\sum_{n=1}^\infty S_n\right)^j\right]
\end{align}
```

Here we consider a single perturbation for brevity.

Because both the above approach and Schrieffer-Wolff produce a unique answer, it
is sufficient to show that they solve the same problem under the same
conditions.
Some conditions are straightforwardly the same:

- Both algorithms guarantee that $\tilde{H}^{AB} = 0$ to all orders.
- Both algorithms guarantee that $U$ is unitary to all orders:
  Pymablock by construction, and Schrieffer-Wolff by the
  definition of the exponential and anti-Hermiticity of $S$.

We are left to show that the diagonal blocks of $\exp S$ are Hermitian, while
off-diagonal blocks are anti-Hermitian because this is the only remaining
property of the Pymablock algorithm.
To do so, we expand all terms in Eq. {eq}`exp_s_expansion` using the multinomial theorem.
The result contains all possible products of $S_n$ of all lengths with fractional prefactors.
Furthermore, for every term $S_{k_1}S_{k_2}\cdots S_{k_n}$, there is a
corresponding term $S_{k_n}S_{k_{n-1}}\cdots S_{k_1}$ with the same prefactor.
If the number of $S_{k_n}$ is even, then both terms are block-diagonal since
each $S_n$ is block off-diagonal.
Because $S_n$ are anti-Hermitian, the two terms are Hermitian conjugates of each
other, and therefore their sum is Hermitian.
On the other hand, if the number of $S_{k_n}$ is odd, then the two terms are
block off-diagonal and their sum is anti-Hermitian by the same reasoning.

This concludes the proof.
