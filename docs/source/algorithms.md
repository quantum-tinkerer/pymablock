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
# The algorithm

## Problem formulation

Pymablock finds a series of the unitary transformation $\mathcal{U}$ (we use
callilgraphic letters to denote series) that block-diagonalizes the Hamiltonian

:::{math}
:label: eq:hamiltonian
\mathcal{H} = H_0 + \mathcal{H}',\quad H_0 = \begin{pmatrix}
H_0^{AA} & 0\\
0 & H_0^{BB}
\end{pmatrix},
:::

with $\mathcal{H}'$ containing an arbitrary number and orders of perturbations.
The series here may be multivariate, and they represent sums of the form

$$
\mathcal{A} = \sum_{n_1=0}^\infty \sum_{n_2=0}^\infty \cdots \sum_{n_k=0}^\infty \lambda_1^{n_1} \lambda_2^{n_2} \cdots \lambda_k^{n_k} A_{n_1, n_2, \ldots, n_k},
$$

where $\lambda_i$ are the perturbation parameters and $A_{n_1, n_2, \ldots,
n_k}$ are linear operators.

The problem statement, therefore, is finding $\mathcal{U}$ and
$\tilde{\mathcal{H}}$ such that:

:::{math}
:label: eq:problem_definition
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U},\quad \tilde{\mathcal{H}}^{AB} = 0,\quad \mathcal{U}^\dagger \mathcal{U} = 1, \quad \tilde{\mathcal{H}}^{\dagger} = \tilde{\mathcal{H}},
:::

where series multiply according to the [Cauchy
product](https://en.wikipedia.org/wiki/Cauchy_product):

$$
\mathcal{C} = \mathcal{A}\mathcal{B} \Leftrightarrow C_\mathbf{n} = \sum_{\mathbf{m} + \mathbf{p} = \mathbf{n}} A_\mathbf{m} B_\mathbf{p}.
$$

:::{admonition} Computational complexity of the Cauchy product
:class: dropdown info
The Cauchy product is the most expensive operation in perturbation theory,
because it involves a large number of multiplications between potentially large
matrices, so let us discuss its complexity.
Evaluating $\mathbf{n}$-th order of $\mathcal{C}$ requires $\sim\prod_i n_i =
N$ multiplications of the series elements.
A direct computation of all the possible index combinations in a product
between three series $\mathcal{A}\mathcal{B}\mathcal{C}$ would have a higher
cost $\sim N^2$, however if we use associativity of the product and compute
this as $(\mathcal{A}\mathcal{B})\mathcal{C}$, then the scaling of the cost
stays the same.
The scaling does become slower if we need to compute a Taylor expansion of a series:

$$
f(\mathcal{A}) = \sum_{n=0}^\infty a_n \mathcal{A}^n.
$$

Once again, a direct computation would require $\sim \exp N$ multiplications,
however defining new series as $\mathcal{A}^{n+1} = \mathcal{A}\mathcal{A}^{n}$
and reusing the previously computed results brings these costs down to $\sim
N^2$.
:::

There are many ways to solve this problem that give identical expressions for
$\mathcal{U}$ and $\tilde{\mathcal{H}}$.
We are searching for a procedure that satisfies two additional constraints:

- It only requires computing Cauchy products and therefore has the lowest
  possible scaling of complexity.
- It does not require multiplications by $H_0$. This is because $n$-th order
  corrections to $\tilde{\mathcal{H}}$ in perturbation theory carry $n$ powers
  of energy denominators. Therefore, any additional multiplications by $H_0$
  must cancel with additional energy denominators. Muliplying by $H_0$ is
  therefore unnecessary work, and it gives longer intermediate expressions.

## Solution

To find $\mathcal{U}$, let us separate it into an identity, $\mathcal{W}$, and
$\mathcal{V}$:

:::{math}
:label: eq:U
\mathcal{U} = 1 + \mathcal{U}' = 1 + \mathcal{W} + \mathcal{V},\quad \mathcal{W}^\dagger = \mathcal{W},\quad \mathcal{V}^\dagger = -\mathcal{V}.
:::

Substituting $\mathcal{U}'$ into the unitarity condition
$(1/2)(\mathcal{U}^\dagger \mathcal{U} + \mathcal{U}^\dagger \mathcal{U}) = 1$ gives
that $\mathcal{U}'$ has no 0-th order term, and

$$
2\mathcal{W} = \mathcal{U}' + \mathcal{U}'^\dagger = -\mathcal{U}'^\dagger \mathcal{U}' = -\mathcal{W}^2 + \mathcal{V}^2.
$$

We can use this to define $\mathcal{W}$ as a Taylor series in $\mathcal{V}$:

$$
\mathcal{W} = \sqrt{1 + \mathcal{V}^2} - 1 \equiv f(\mathcal{V}) \equiv \sum_n a_n \mathcal{V}^{2n},
$$

however this is both computationally expensive and unnecessary.
Instead, because $\mathcal{U}'$ has no 0th order term, $(\mathcal{U}'^\dagger
\mathcal{U}')_\mathbf{n}$ does not depend on the $\mathbf{n}$-th order of
$\mathcal{U}$, and therefore we recursively compute $\mathcal{W}$ as

:::{math}
:label: eq:W
\mathcal{W} = -\mathcal{U}'^\dagger \mathcal{U}'/2,
:::

which is fully defined by the unitarity condition.
*This recursive definition is the first secret ingredient of Pymablock✨*

On the other hand, $\mathcal{V}$ is defined by the requirement
$\tilde{\mathcal{H}}^{AB} = 0$.
Additionally, we constrain it to be block off-diagonal:
$\mathcal{V}^{AA} = \mathcal{V}^{BB} = 0$,
so that the unitary transformation is equivalent to the Schrieffer-Wolff
transformation.
In turn, it follows that $\mathcal{W}$ is block-diagonal and that the norm
of $\mathcal{U}$ is minimal.

:::{admonition} Equivalence to Schrieffer-Wolff transformation
:class: dropdown info
Both the Pymablock algorithm and the more commonly used Schrieffer-Wolff
transformation find a unitary transformation $\mathcal{U}$ such that
$\tilde{\mathcal{H}}^{AB}=0$.
They are therefore equivalent up to a gauge choice in each subspace, $A$ and
$B$.
We establish the equivalence between the two by demonstrating that this gauge
choice is the same for both algorithms.

The Schrieffer-Wolff transformation uses $\mathcal{U} = \exp \mathcal{S}$,
where $\mathcal{S} = -\mathcal{S}^\dagger$ and $\mathcal{S}^{AA} =
\mathcal{S}^{BB} = 0$.

The series $\exp\mathcal{S}$ contains all possible products of $S_n$ of all
lengths with fractional prefactors.
For every term $S_{k_1}S_{k_2}\cdots S_{k_n}$, there is a corresponding term
$S_{k_n}S_{k_{n-1}}\cdots S_{k_1}$ with the same prefactor.
If the number of $S_{k_n}$ is even, then both terms are block-diagonal since
each $S_n$ is block off-diagonal.
Because $S_n$ are anti-Hermitian, the two terms are Hermitian conjugates of each
other, and therefore their sum is Hermitian.
On the other hand, if the number of $S_{k_n}$ is odd, then the two terms are
block off-diagonal and their sum is anti-Hermitian by the same reasoning.

Therefore, just like in our algorithm, the diagonal blocks of $\exp S$ are
Hermitian, while off-diagonal blocks are anti-Hermitian.
Schieffer-Wolff transformation produces a unique answer and satisfies the same
diagonalization requirements as our algorithm, which means that the two are
equivalent.
:::

To find $\mathcal{V}$, we look for a recursive procedure that does not involve
multiplications by $H_0$, and that follows from $\tilde{\mathcal{H}}^{AB} = 0$.
Let us then look at the unitary transformation of the Hamiltonian:

$$
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U} = H_0 + \mathcal{U}'^\dagger H_0 + H_0 \mathcal{U}' + \mathcal{U}'^\dagger H_0 \mathcal{U}' + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U}.
$$

To get rid of the terms with $H_0$, we define

:::{math}
:label: eq:XYX
\begin{align}
\mathcal{X} &\equiv [\mathcal{U}', H_0] = \mathcal{Y} + \mathcal{Z}, \\
\mathcal{Y} &\equiv [\mathcal{V}, H_0] = \mathcal{Y}^\dagger,\\
\mathcal{Z} &\equiv [\mathcal{W}, H_0] = -\mathcal{Z}^\dagger,
\end{align}
:::

where $\mathcal{Y}$ is therefore block off-diagonal and $\mathcal{Z}$ is block
diagonal.
It follows that

:::{math}
:label: eq:H_tilde
\tilde{\mathcal{H}} = H_0 - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U},
:::

where the terms multiplied by $H_0$ cancel by unitarity.
Then, we use Hermiticity

$$
0 = \tilde{\mathcal{H}}-\tilde{\mathcal{H}}^\dagger = -\mathcal{X}-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{X}^\dagger + \mathcal{X}^\dagger\mathcal{U}',
$$

and find a recursive definition for $\mathcal{Z}$:

:::{math}
:label: eq:Z
\mathcal{Z} = \frac{1}{2}(\mathcal{X} - \mathcal{X}^\dagger) = (-\mathcal{U}'^\dagger\mathcal{X} - \textrm{h.c.})/2,
:::

which defines the diagonal blocks of $\mathcal{X}$ by using all blocks of the
previous orders of $\mathcal{X}$ in the right hand side.
*This is our second secret ingredient✨*

Finally, we compute the off-diagonal blocks of $\mathcal{X}$ by requiring that
$\tilde{\mathcal{H}}^{AB} = 0$.
Thus, from {eq}`H_tilde` it follows that

:::{math}
:label: eq:Y
\mathcal{X}^{AB} = (\mathcal{U}^\dagger \mathcal{H}' \mathcal{U} - \mathcal{U}'^\dagger \mathcal{X})^{AB}.
:::

Once again, despite $\mathcal{X}$ enters the right hand side, because all the
terms lack 0-th order, this defines a recursive relation for $\mathcal{X}^{AB}$,
and therefore $\mathcal{Y}$.
*This is our last secret ingredient✨*

The final part is standard: the definition of $\mathcal{Y}$ in {eq}`XYX` fixes
$\mathcal{V}$ as a solution of:

:::{math}
:label: eq:sylvester
\mathcal{V}^{AB}H_0^{BB} - H_0^{AA} \mathcal{V}^{AB} = \mathcal{Y}^{AB},
:::

a [Sylvester's equation](https://en.wikipedia.org/wiki/Sylvester_equation).

## The algorithm

We now have a complete algorithm:

1. Define series $\mathcal{U}'$ and $\mathcal{X}$ and make use of their block structure and Hermiticity.
2. To define the diagonal blocks of $\mathcal{U}'$, use $\mathcal{W} = -\mathcal{U}'^\dagger\mathcal{U}'/2$, see {eq}`W`.
3. To find the off-diagonal blocks of $\mathcal{U}'$, solve Sylvester's equation  $\mathcal{V}^{AB}H_0^{AA} - H_0^{BB}\mathcal{V}^{AB} = \mathcal{Y}^{AB}$. Note that {eq}`sylvester` requires $\mathcal{X}$.
4. To find the diagonal blocks of $\mathcal{X}$, define $\mathcal{Z} = (-\mathcal{U}'^\dagger\mathcal{X} - \textrm{h.c.})/2$, see {eq}`Z`.
5. For the off-diagonal blocks of $\mathcal{X}$, use $\mathcal{Y}^{AB} =
 (-\mathcal{U}'^\dagger\mathcal{X} +
  \mathcal{U}^\dagger\mathcal{H}'\mathcal{U})^{AB}$, see {eq}`Y`.
6. Compute the effective Hamiltonian as $\tilde{\mathcal{H}}_{\textrm{diag}} = H_0 + \mathcal{U}^\dagger\mathcal{H}'\mathcal{U} - (\mathcal{U}'^\dagger \mathcal{X} +\textrm{h.c.})/2$, see {eq}`H_tilde`.

##  How to use Pymablock on large numerical Hamiltonians?

Solving Sylvester's equation and computing the matrix products are the most
expensive steps of the algorithms for large Hamiltonians.
Pymablock can efficiently construct an effective Hamiltonian of a small
subspace even when the full Hamiltonian is a sparse matrix that is too costly to
diagonalize.
It does so by avoiding explicit computation of operators in $B$ subspace, and by
utilizing the sparsity of the Hamiltonian to compute the Green's function.
To do so, Pymablock uses either the [MUMPS sparse
solver](https://mumps-solver.org/) or the [KPM
method](https://doi.org/10.1103/RevModPhys.78.275).

This approach was originally introduced in [this
work](https://arxiv.org/abs/1909.09649).

::::{admonition} Implementation details
:class: dropdown info
We use the matrix $\Psi_A$ of the eigenvectors of the $A$ subspace to rewrite
the Hamiltonian as

:::{math}
H \to \begin{pmatrix}
\Psi_A^\dagger H \Psi_A & \Psi_A^\dagger H P_B \\
P_B H \Psi_A & P_B H P_B
\end{pmatrix},
:::

where $P_B = 1 - \Psi_A \Psi_A^\dagger$ is the projector onto the $B$ subspace.
This Hamiltonian is larger in size than the original one because the $B$ block has
additional null vectors corresponding to the $A$ subspace.
This, however, allows to preserve the sparsity structure of the Hamiltonian by applying
$P_B$ and $H$ separately.
Additionally, applying $P_B$ is efficient because $\Psi_A$ is a low rank matrix.
We then perform perturbation theory of the rewritten $H$.

To solve the Sylvester's equation for the modified Hamiltonian, we write it for
every row of $V_n^{AB}$ separately:

:::{math}
V_{n, ij}^{AB} (E_i - H_0) = Y_{n, j}
:::

This equation is well-defined despite $E_i - H_0$ is not invertible because
$Y_{n}$ has no components in the $A$ subspace.
::::
