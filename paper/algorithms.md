# Algorithms for block diagonalization

**Pymablock's algorithm does not use the Schrieffer-Wolff transformation,
because the former is inefficient.**
A common approach to construct effective Hamiltonians is to use a
Schrieffer-Wolff transformation:
:::{math}
\begin{equation}
\tilde{\mathcal{H}} = e^\mathcal{S} \mathcal{H} e^{-\mathcal{S}},
\end{equation}
:::
where $\mathcal{S} = \sum_n S_n$ is an antihermitian polynomial series in the
perturbative parameter, making $e^\mathcal{S}$ a unitary transformation.
In this approach, $\mathcal{S}$ is found by ensuring unitarity and the block
diagonalization of the Hamiltonian to every order, a procedure that amounts to
solving a recursive equation whose terms are nested commutators between series.
Moreover, the transformed Hamiltonian is also given by a series of nested
commutators
:::{math}
\begin{equation}
\tilde{\mathcal{H}} = \sum_{j=0}^\infty \frac{1}{j!} \Big [\mathcal{H}, \sum_{n=0}^{\infty} \mathcal{S}_n \Big ]^{(j)},
\end{equation}
:::
a computationally expensive expression because it requires computing
exponentially many matrix products.
This expression also requires truncating the series at the same order
to which $\mathcal{S}$ is computed, which is a waste of computational resources.
Finally, generalizing the Schrieffer-Wolff transformation to multiple
perturbations is only straightforward if the perturbations are bundled
together.
However, this makes it impossible to request individual order combinations,
making it necessary to compute more terms than needed.

**There are algorithms that use different parametrizations for $\mathcal{U}$, a
difference that is crucial for efficiency, even though the results are
equivalent.**
The algorithm used to block diagonalize a Hamiltonian perturbatively is,
however, not unique.
Alternative parametrizations of the unitary transformation $\mathcal{U}$, for example,
using hyperbolic functions or a polynomial series directly, give rise to
different algorithms.
These also require solving unitarity and block diagonalization conditions,
and achieve the same effective Hamiltonian as the Schrieffer-Wolff
transformation
[VanVleck1929](doi:10.1103/PhysRev.33.467), [Lowdin1964](doi:10.1063/1.1724312)
[Klein1974][doi:10.1063/1.1682018], [Suzuki1983](doi:10.1143/PTP.70.439),
[Shavitt1980](doi:10.1063/1.440050).
Despite the conceptual equivalence of the algorithms and the agreement of
their results, there is a crucial difference in their computational efficiency:
a Schrieffer-Wolff transformation has an exponential scaling with the
perturbative order, but it can be reduced to a linear one.
Reference [Li2022](doi:10.1103/PRXQuantum.3.030313), for example, introduces an
algorithm with linear scaling for diagonalization of a single state by
reformulating the recursive steps of the Schrieffer-Wolff transformation.
Block diagonalization of a Hamiltonian, however, recovers the exponential
scaling.
To design the algorithms of Pymablock, we choose the parametrization that is
most computationally efficient: linear scaling with a polynomial series for
$\mathcal{U} = \sum_{n=0}^{\infty} \mathcal{U}_n$.

## General algorithm

**The algorithms of Pymablock rely on decomposing $\mathcal{U}$ into two parts.**
The algorithms of Pymablock rely on decomposing $\mathcal{U}$, the unitary transformation
that block diagonalizes the Hamiltonian, as a series of Hermitian
block diagonal $\mathcal{W}$ and skew-Hermitian and block off-diagonal $\mathcal{V}$ terms.
The transformed Hamiltonian is a Cauchy product between the series of
$\mathcal{U}^{\dagger}$, $\mathcal{H}$, and $\mathcal{U}$, a product between
series defined as
:::{math}
:label: cauchy_product
\begin{equation}
(\mathcal{A} \star \mathcal{B})_{n} = \sum_{i=0}^{n} \mathcal{A}_{i} \mathcal{B}_{n-i}.
\end{equation}
:::

For brevity we use a single first order perturbation $\mathcal{H}_1$ throughout this
document. The generalization to multiple perturbations is follows naturally
by including more indices.

**The result of this procedure is a perturbative series of the transformed
block-diagonal Hamiltonian.**
The transformed Hamiltonian at order $n$ is
:::{math}
:label: h_tilde
\begin{align} \tilde{\mathcal{H}}_{n} = (\mathcal{U}^{\dagger} \mathcal{H}
\mathcal{U})_{n} = \Big ( (\mathcal{W}-\mathcal{V}) \star \mathcal{H} \star
(\mathcal{W}+\mathcal{V}) \Big)_{n}.
\end{align}
:::

**Pymablock finds the unitary transformation recursively, using unitarity and
solving Sylvester's equation at every order.**
To block diagonalize $\mathcal{H}_0 + \mathcal{H}_1$, Pymablock finds the
orders of $\mathcal{W}$ such that $\mathcal{U}$ is unitary
:::{math}
:label: unitarity
\begin{align}
\mathcal{W}_{n}^{AA} &= - \frac{1}{2} (\mathcal{U}^{\dagger} \star \mathcal{U})_{\cancel{n}}^{AA}, \\
\mathcal{W}_{n}^{BB} &= - \frac{1}{2} (\mathcal{U}^{\dagger} \star \mathcal{U})_{\cancel{n}}^{BB},
\end{align}
:::
where the subscript $\cancel{n}$ indicates that the $n$th term of the first
and last series are omitted from the Cauchy product.

:::{admonition} Derivation
:class: dropdown info
We evaluate the series $\mathcal{U}^\dagger \mathcal{U} +
\mathcal{U}\mathcal{U}^\dagger=2$ and use that
$\mathcal{W}=\mathcal{W}^\dagger$ and $\mathcal{V}=-\mathcal{V}^{\dagger}$
to obtain
\begin{equation}
\sum_{i=0}^n \left[(\mathcal{W}_{n-i} - \mathcal{V}_{n-i})(\mathcal{W}_i +
\mathcal{V}_i) + (\mathcal{W}_{n-i} + \mathcal{V}_{n-i})(\mathcal{W}_i -
\mathcal{V}_i)\right] = 0.
\end{equation}
Using that $\mathcal{W}_0=1$, $\mathcal{V}_0=0$, expanding, and solving for
$\mathcal{W}_n$ gives
\begin{equation}
\mathcal{W}_{n} &= - \frac{1}{2}
\sum_{i=1}^{n-1}(\mathcal{W}_{n-i}\mathcal{W}_i -
\mathcal{V}_{n-i}\mathcal{V}_i),
\end{equation}
a sum of Cauchy products that misses the $n \textsuperscript{th}$ term of each
series.
For every order, $\mathcal{W}$ remains block-diagonal and Hermitian.
:::
Similarly, Pymablock finds the terms of $\mathcal{V}$ by requiring that
$\tilde{\mathcal{H}}^{AB}_n=0$
```{math}
:label: sylvester
\mathcal{H}_0^{AA} \mathcal{V}_{n}^{AB} - V_{n}^{AB} \mathcal{H}_0^{BB} = \mathcal{Y}_{n}.
```
This is known as Sylvester's equation and $\mathcal{Y}_{n}$ is a combination of lower
order terms in $\mathcal{H}$ and $\mathcal{U}$ defined as
\begin{equation}
\mathcal{Y}_n = (\mathcal{U}^\dagger \star H \star \mathcal{U})_{\cancel{n}}^{AB}.
\end{equation}

::::{admonition} Full expression
:class: dropdown info
The full expression for $\mathcal{Y}_n$ is cumbersome already in our simplest case:

:::{math}
:label: y_n
\begin{align}
\mathcal{Y}_n=&-
\sum_{i=1}^{n-1}\left[\mathcal{W}_{n-i}^{AA}\mathcal{H}_0^{AA}\mathcal{V}_i^{AB}-\mathcal{V}_{n-i}^{AB}
\mathcal{H}_0^{BB}\mathcal{W}_i^{BB}\right] \\
&-\sum_{i=0}^{n-1}\bigg[W_{n-i-1}^{AA}\mathcal{H}_1^{AA}\mathcal{V}_i^{AB}+W_{n-i-1}^{AA}
\mathcal{H}_1^{AB}\mathcal{W}_i^{BB} \\
&\quad \quad \quad -\mathcal{V}_{n-i-1}^{AB}(\mathcal{H}_1^{AB})^\dagger \mathcal{V}_i^{AB} -V_{n-i-1}^{AB}
\mathcal{H}_1^{BB}\mathcal{W}_i^{BB}\bigg]
\end{align}
:::
::::
It follows that for every order, Sylvester's equation needs to be solved
only once, a requirement of any algorithm that block diagonalizes a Hamiltonian.

### Proof of equivalence to Schrieffer-Wolff transformation

**The transformed Hamiltonian is equivalent to that of other perturbative
methods, but the algorithm is efficient.**
Both the Pymablock algorithm and the more commonly used Schrieffer-Wolff
transformation find a unitary transformation $\mathcal{U}$ such that
$\tilde{\mathcal{H}}^{AB}=0$.
They are therefore equivalent up to a gauge choice on each subspace.
We establish the correspondence between the two by demonstrating that this gauge
choice is the same for both algorithms.

Pymablock chooses $\mathcal{U}=\mathcal{W}+\mathcal{V}$, where $\mathcal{W}$ is
block diagonal Hermitian and $\mathcal{V}$ is block off-diagonal
anti-Hermitian.
Then requiring that $\mathcal{U}$ is unitary and $\tilde{\mathcal{H}}^{AB}=0$
to all orders defines a unique value for $\mathcal{W}$ and $\mathcal{V}$.

The Schrieffer-Wolff transformation parameterizes $\mathcal{U} = \exp
\mathcal{S}$, where $\mathcal{S} = \sum_n \mathcal{S}_n$ is a series of
anti-Hermitian block off-diagonal operators:
```{math}
:label: exp_s_expansion
\begin{align}
\mathcal{U} = \exp{\left(\mathcal{S}\right)}=\exp{\left(\sum_{n=0}^\infty
\mathcal{S}_n\right)} = 1+\sum_{j=1}^\infty \left[\frac{1}{j!}
\left(\sum_{n=1}^\infty \mathcal{S}_n\right)^j\right]
\end{align}
```
Here we consider a single perturbation for brevity.

Because both the above approach and Schrieffer-Wolff produce a unique answer, it
is sufficient to show that they solve the same problem under the same
conditions.
Some conditions are straightforwardly the same:

- Both algorithms guarantee that $\tilde{\mathcal{H}}^{AB} = 0$ to all orders.
- Both algorithms guarantee that $\mathcal{U}$ is unitary to all orders:
  Pymablock by construction, and Schrieffer-Wolff by the
  definition of the exponential and anti-Hermiticity of $\mathcal{S}$.

We are left to show that the diagonal blocks of $\exp \mathcal{S}$ are
Hermitian, while off-diagonal blocks are anti-Hermitian because this is the
only remaining property of the Pymablock algorithm.
To do so, we expand all terms in Eq. {eq}`exp_s_expansion` using the multinomial theorem.
The result contains all possible products of $\mathcal{S}_n$ of all lengths with fractional prefactors.
Furthermore, for every term $\mathcal{S}_{k_1}\mathcal{S}_{k_2}\cdots \mathcal{S}_{k_n}$, there is a
corresponding term $\mathcal{S}_{k_n}\mathcal{S}_{k_{n-1}}\cdots \mathcal{S}_{k_1}$ with the same prefactor.
If the number of $\mathcal{S}_{k_n}$ is even, then both terms are block-diagonal since
each $\mathcal{S}_n$ is block off-diagonal.
Because $\mathcal{S}_n$ are anti-Hermitian, the two terms are Hermitian conjugates of each
other, and therefore their sum is Hermitian.
On the other hand, if the number of $\mathcal{S}_{k_n}$ is odd, then the two terms are
block off-diagonal and their sum is anti-Hermitian by the same reasoning.

This concludes the proof.

## Expanded algorithm for analytic Hamiltonians

The `pymablock.general` algorithm implements the procedure outlined here directly.
However, this may not be the most efficient algorithm for analytic Hamiltonians,
where the priority is to obtain compact expressions right away, instead of
simplifying them afterwards.
To overcome this, Pymablock also has the `pymablock.expanded` algorithm, which
returns simplified expressions for $\tilde{\mathcal{H}}_{n}$ (Eq. {eq}`h_tilde`) such
that $\tilde{\mathcal{H}}_{n}$ only depends on $\mathcal{V}$ and the perturbation $\mathcal{H}_1$, but not
explicitly on $\mathcal{H}_0$.
This way, the expressions do not contain fractions that must cancel, reducing
the number of terms in the final result.

:::{admonition} How this works
:class: dropdown info
The `pymablock.expanded` algorithm first uses
`pymablock.general` with a symbolic input to derive the general
symbolic form for $\mathcal{Y}_n$ and $\tilde{\mathcal{H}}_n$.
Then it uses the Sylvester's equation for lower orders of $\mathcal{V}_n$ to eliminate
$\mathcal{H}_0$ from these expressions.
Finally, `pymablock.expanded` replaces the problem-specific $\mathcal{H}$ into
the simplified $\tilde{\mathcal{H}}$.
:::

As an example, the corrections to the effective Hamiltonian up to fourth
order using `pymablock.expanded` are

```{embed} # expanded
```

Here we omitted the superscript $AB$ on all the $\mathcal{V}$'s for brevity.

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

This approach was originally introduced in Ref.
[hybrid-kpm](doi:10.48550/arXiv.1909.09649).

:::{admonition} Implementation details
:class: dropdown info
We use the matrix $\Psi_A$ of the eigenvectors of the $A$ subspace to rewrite
the Hamiltonian as
```{math}
:label: H_implicit
\begin{equation}
\mathcal{H} \to \begin{pmatrix}
\Psi_A^\dagger \mathcal{H} \Psi_A & \Psi_A^\dagger \mathcal{H} P_B \\
P_B \mathcal{H} \Psi_A & P_B \mathcal{H} P_B
\end{pmatrix},
\end{equation}
```
where $P_B = 1 - \Psi_A \Psi_A^\dagger$ is the projector onto the $B$ subspace.
This Hamiltonian is larger in size than the original one because the $B$ block has
additional null vectors corresponding to the $A$ subspace.
This, however, allows to preserve the sparsity structure of the Hamiltonian by applying
$P_B$ and $\mathcal{H}$ separately.
Additionally, applying $P_B$ is efficient because $\Psi_A$ is a low rank matrix.
We then perform perturbation theory of the rewritten $\mathcal{H}$.

To solve the Sylvester's equation for the modified Hamiltonian, we write it for
every row of $\mathcal{V}_n^{AB}$ separately:
```{math}
\mathcal{V}_{n, ij}^{AB} (E_i - \mathcal{H}_0) = \mathcal{Y}_{n, j}
```
This equation is well-defined despite $E_i - \mathcal{H}_0$ is not invertible because
$\mathcal{Y}_{n}$ has no components in the $A$ subspace.
:::
