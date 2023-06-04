# The algorithms

The algorithms of {{Pymablock}} rely on decomposing $U$, the unitary transformation
that block diagonalizes the Hamiltonian, as a series of Hermitian
block diagonal $W$ and skew-Hermitian and block off-diagonal $V$ terms.
The transformed Hamiltonian is a
[Cauchy product](https://en.wikipedia.org/wiki/Cauchy_product)
between the series of $U^\dagger$, $H$, and $U$.

For example, for a single first order perturbation $H_p$, the transformed
Hamiltonian at order $n$ is

:::{math}
:label: h_tilde
\begin{align}
\tilde{H}_{n} = \sum_{i=0}^n (W_{n-i} - V_{n-i}) H_0 (W_i + V_i) +
\sum_{i=0}^{n-1} (W_{n-i-1} - V_{n-i-1}) H_p (W_i + V_i).
\end{align}
:::

To block diagonalize $H_0 + H_p$, {{Pymablock}} finds the orders of $W$
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
(W_{n-i} + V_{n-i})(W_i - V_i)\right] = 0.\\
\end{equation}
```
Using that $W_0=1$, $V=0$, expanding, and solving for $W_n$ gives the Eq. {eq}`unitarity`.
:::

and the orders of $V$ by requiring that $\tilde{H}^{AB}_n=0$

\begin{equation}
H_0^{AA} V_{n}^{AB} - V_{n}^{AB} H_0^{BB} = Y_{n}.
\end{equation}

This is known as [Sylvester's equation](https://en.wikipedia.org/wiki/Sylvester_equation)
and $Y_{n}$ is a combination of lower order terms in $H$ and $U$.

::::{admonition} Full expression
:class: dropdown info
The full expression for $Y_n$ is cumbersome already in our simplest case:

:::{math}
:label: y_n
\begin{align}
Y_n=&-
\sum_{i=1}^{n-1}\left[W_{n-i}^{AA}H_0^{AA}V_i^{AB}-V_{n-i}^{AB}
H_0^{BB}W_i^{BB}\right] \\
&-\sum_{i=0}^{n-1}\bigg[W_{n-i-1}^{AA}H_p^{AA}V_i^{AB}+W_{n-i-1}^{AA}
H_p^{AB}W_i^{BB}
-V_{n-i-1}^{AB}(H_p^{AB})^\dagger V_i^{AB} -V_{n-i-1}^{AB}
H_p^{BB}W_i^{BB}\bigg]
\end{align}
:::
::::

{{Pymablock}} has two algorithms, {autolink}`~pymablock.general` and {autolink}`~pymablock.expanded`.
The {autolink}`~pymablock.general` algorithm implements the procedure outlined here directly.
On the other hand, {autolink}`~pymablock.expanded` simplifies the expressions
for $\tilde{H}_{n}$ (Eq. {eq}`h_tilde`) such that it only depends on $V$ and the
perturbation $H_p$, but not explicitly on $H_0$.

:::{admonition} How this works
:class: dropdown info
The {autolink}`~pymablock.expanded` algorithm first uses
{autolink}`~pymablock.general` with a symbolic input to derive the general
symbolic form for $Y_n$ and $\tilde{H}_n$.
Then it uses the Sylvester's equation for lower orders of $V_n$ to eliminate
$H_0$ from these expressions.
Finally, {autolink}`~pymablock.expanded` replaces the problem-specific $H$ into
the simplified $\tilde{H}$.
:::

As an example, the corrections to the effective Hamiltonian up to fourth
order using {autolink}`~pymablock.expanded` are

```{code-cell} ipython3
:tags: [remove-input]

from operator import mul

from sympy import Symbol, Eq

from pymablock.block_diagonalization import BlockSeries, symbolic

H = BlockSeries(
    data={
        (0, 0, 0): Symbol('{H_{0}^{AA}}'),
        (1, 1, 0): Symbol('{H_{0}^{BB}}'),
        (0, 0, 1): Symbol('{H_{p}^{AA}}'),
        (0, 1, 1): Symbol('{H_{p}^{AB}}'),
        (1, 1, 1): Symbol('{H_{p}^{BB}}'),
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

At lower orders, {autolink}`~pymablock.expanded` performs fewer operator
products than {autolink}`~pymablock.general`, and with analytic Hamiltonians
the resulting expressions are simpler.
At high orders, however, {autolink}`~pymablock.expanded` requires exponentially
many terms, unlike {autolink}`~pymablock.general` which only requires a linear
number of terms.

:::{admonition} Proof of equivalence to Schrieffer-Wolff transformation
:class: dropdown info

Schrieffer-Wolff transformation parameterizes $U = \exp{S}$, where
$S$ is a series of anti-hermitian block off-diagonal operators.

Assume $S$ solves {eq}`unitarity` and
Eq. {eq}`v_condition` for some $H$. We rewrite

```{math}
:label: exp_s_expansion
\begin{align}
U=\exp{\left(S\right)}=\exp{\left(\sum_{i=0}^\infty
S_n\right)}=1+\sum_{n=1}^\infty \left[\frac{1}{n!}
\left(\sum_{j=1}^\infty S_n\right)^n\right]
\end{align}
```

where $S_n$ inherits the anti Hermiticity from $S$. We truncate the
series at some finite $n$.

To establish coincidence of the two formulations it suffices to show that said
truncated series only containes terms that are either block diagonal and
hermitian or block offdiagonal and anti hermitian as presented in Eq.
{eq}`W_V_block`. Through the multinomial theorem, each generated term of the
truncated expansion at a given order consists out of itself and its order
reversed partner.
Furthermore observe how

```{math}
:label: s_relation
\begin{align}
\prod_{\sum_i k_i=N}S_{k_i} = (-1)^N\left(\prod_{\sum_ik_i=N}
S_{k_{N-i}}\right)^\dagger,
\end{align}
```

for all $n\in\mathbb{N}$. The indexation refers to all
vectors $\vec{k}\in\{\mathbb{N}^N:\sum_ik_i=N\}$ that are permissible by the
multinomial theorem.

To see that Eq. {eq}`s_relation` is true observe that the
adjoint operation, on one hand, maps $k_i\rightarrow k_{N-i}$ reversing the
order of the terms, and, on the other hand, leads to a minus
for each factor in the product due to the anti Hermiticity. Since each term
comes with its reversed partner and even number products of purely block
offdiagonal matrices yield a purely block diagonal matrix, we conclude that
a truncation of the series {eq}`exp_s_expansion` only contains purely block
diagonal unitaries or purely block offdiagonal anti hermitian matrices.
Since at $\lambda=0$ both parametrizations must be proportional to the
identity we can conclude coincidence of both forumulations up to a trivial
global phase of the unitary $U$.

We want to point out that this proof establishes coincidence of the two
parametrizations given the same basis ordering of the original Hamiltonian
$\mathcal{H}$. Basis reordering pertains to gauges in block matrix space of the
form

\begin{align}
\tilde{U}=\begin{pmatrix}
\tilde{U}^{AA} & 0 \\
0 & \tilde{U}^{BB}
\end{pmatrix}
\end{align}

Since this class of gauges is constraint to be block diagonal (basis reordering
does not lead to coupling of the $A$ and $B$ spaces) and therefore proportional
to identity in block matrix space the statement of the proof remains valid.
:::

##  How to use {{Pymablock}} on large numerical Hamiltonians?

Solving Sylvester's equation and computing the matrix products are the most
expensive steps of the algorithms for large Hamiltonians.
{{Pymablock}} can efficiently construct an effective Hamiltonian of a small
subspace even when the full Hamiltonian is a sparse matrix that is too costly to
diagonalize.
It does so by avoiding explicit computation of operators in $B$ subspace, and by
utilizing the sparsity of the Hamiltonian  to compute the Green's function.

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

To solve it efficiently, {{Pymablock}} uses either the [MUMPS sparse
solver](https://mumps-solver.org/) or the [KPM
method](https://doi.org/10.1103/RevModPhys.78.275).

This approach was originally introduced in [this
work](https://arxiv.org/abs/1909.09649).
:::
