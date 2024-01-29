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
calligraphic letters to denote series) that block-diagonalizes the Hamiltonian

:::{math}
:label: hamiltonian
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
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U},\quad \tilde{\mathcal{H}}^{AB} = 0,\quad \mathcal{U}^\dagger \mathcal{U} = 1,
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
stays $\sim N$.
:::

## How to solve the problem?

There are many ways to solve this problem that give identical expressions for
$\mathcal{U}$ and $\tilde{\mathcal{H}}$.
We are searching for a procedure that satisfies two additional constraints:

- It has the same complexity scaling as a Cauchy product.
- It does not require multiplications by $H_0$.
  This is because in perturbation theory, $n$-th order  corrections to
  $\tilde{\mathcal{H}}$ carry $n$ energy denominators $1/(E_i - E_j)$
  (see [here](https://en.wikipedia.org/wiki/Perturbation_theory_(quantum_mechanics)#Time-independent_perturbation_theory)).
  Therefore, any additional multiplications by $H_0$ must cancel with
  additional energy denominators.
  Multiplying by $H_0$ is therefore unnecessary work, and it gives longer
  intermediate expressions.

The goal of our algorithm is thus to be efficient and to produce compact
results that do not require further simplifications.

## Our solution

To find $\mathcal{U}$, let us separate it into an identity and $\mathcal{U}' =
\mathcal{W} + \mathcal{V}$:

:::{math}
:label: U
\mathcal{U} = 1 + \mathcal{U}' = 1 + \mathcal{W} + \mathcal{V},\quad \mathcal{W}^\dagger = \mathcal{W},\quad \mathcal{V}^\dagger = -\mathcal{V}.
:::

First, we use the unitarity condition
$\mathcal{U}^\dagger \mathcal{U} = 1$ by substituting $\mathcal{U}'$ into it
and obtain

:::{math}
:label: W
\toggle{
  \mathcal{W} = \texttip{\color{red}{\ldots}}{click to expand} = -\frac{1}{2}
  \mathcal{U}'^\dagger \mathcal{U}'.
}{
  \begin{align}
    W &= \frac{1}{2}(\mathcal{U}'^\dagger + \mathcal{U}') \\
      &= \frac{1}{2} \Big[(1 + \mathcal{U}'^\dagger)(1+\mathcal{U}') - 1 - \mathcal{U}'^\dagger \mathcal{U}' \Big] \\
      &= -\frac{1}{2} \mathcal{U}'^\dagger \mathcal{U}'.
  \end{align}
}
\endtoggle
:::

Because $\mathcal{U}'$ has no $0$-th order term, $(\mathcal{U}'^\dagger
\mathcal{U}')_\mathbf{n}$ does not depend on the $\mathbf{n}$-th order of
$\mathcal{U}'$ nor $\mathcal{W}$.
More generally, a Cauchy product $\mathcal{A}\mathcal{B}$ where $\mathcal{A}$
and $\mathcal{B}$ have no $0$-th order terms depends on $\mathcal{A}_1, \ldots,
\mathcal{A}_{n-1}$ and $\mathcal{B}_1, \ldots, \mathcal{B}_{n-1}$.
This allows us to use Cauchy products to define recurrence relations, which
we apply throughout the algorithm.
Therefore, we compute $\mathcal{W}$ as a Cauchy product of $\mathcal{U}'$ with
itself.
*This recurrence relation is the first secret ingredient of Pymablock✨*

:::{admonition} Choosing the right definition for $\mathcal{W}$
:class: dropdown info

Using the definition of $\mathcal{W}$ as the Hermitian part of $\mathcal{U}'$,
and the unitarity condition:

$$
\begin{align}
2\mathcal{W}
= \mathcal{U}' + \mathcal{U}'^\dagger
= -\mathcal{U}'^\dagger \mathcal{U}'
= -\mathcal{W}^2 + \mathcal{V}^2.
\end{align}
$$

we see that we could alternatively define $\mathcal{W}$ as a Taylor series in
$\mathcal{V}$:

$$
\mathcal{W} = \sqrt{1 + \mathcal{V}^2} - 1 \equiv f(\mathcal{V}) \equiv \sum_n a_n \mathcal{V}^{2n},
$$

however the scaling of such a Cauchy product becomes slower if we need to
compute a Taylor expansion of a series:

$$
f(\mathcal{A}) = \sum_{n=0}^\infty a_n \mathcal{A}^n.
$$

However, evaluating a Taylor expansion of a given series has a higher scaling
of complexity.
A direct computation of all possible products of terms would require $\sim \exp
N$ multiplications.
We improve on this by defining a new series as $\mathcal{A}^{n+1} =
\mathcal{A}\mathcal{A}^{n}$ and reusing the previously computed results, which
brings these costs down to $\sim N^2$.
Using the Taylor expansion approach is therefore both more complicated and more
computationally expensive than the recurrent definition in {eq}`W`.
:::

To compute $\mathcal{U}'$ we also need to find $\mathcal{V}$, which is defined
by the requirement $\tilde{\mathcal{H}}^{AB} = 0$.
Additionally, we constrain $\mathcal{V}$ to be block off-diagonal:
$\mathcal{V}^{AA} = \mathcal{V}^{BB} = 0$,
so that the resulting unitary transformation is equivalent to the
Schrieffer-Wolff transformation.
In turn, this means that $\mathcal{W}$ is block-diagonal and that the norm
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
Schrieffer-Wolff transformation produces a unique answer and satisfies the same
diagonalization requirements as our algorithm, which means that the two are
equivalent.
:::

To find $\mathcal{V}$, we need to first look at the transformed Hamiltonian:

$$
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U} = H_0 +
\mathcal{U}'^\dagger H_0 + H_0 \mathcal{U}' + \mathcal{U}'^\dagger H_0
\mathcal{U}' + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U},
$$
where we used $\mathcal{U}=1+\mathcal{U}'$ and $\mathcal{H} = H_0 +
\mathcal{H'}$.

Because we want to avoid unnecessary products by $H_0$, we need to get rid of
the terms that contain it by replacing them with an alternative expression.
Our strategy is to define an auxiliary operator $\mathcal{X}$ that we can
compute without ever multiplying by $H_0$.
Like $\mathcal{U}'$, $\mathcal{X}$ needs to be defined via a recurrence
relation, which we will find later.
Because the expression above has $H_0$ multiplied by $\mathcal{U}'$ by the left
and by the right, we get rid of these terms by making sure that $H_0$
multiplies terms from one side only.
To achieve this, we choose $\mathcal{X}$ to be the commutator between
$\mathcal{U}'$ and $H_0$:

:::{math}
:label: XYZ
\mathcal{X} \equiv [\mathcal{U}', H_0] = \mathcal{Y} + \mathcal{Z}, \quad
\mathcal{Y} \equiv [\mathcal{V}, H_0] = \mathcal{Y}^\dagger,\quad
\mathcal{Z} \equiv [\mathcal{W}, H_0] = -\mathcal{Z}^\dagger,
:::

where $\mathcal{Y}$ is therefore block off-diagonal and $\mathcal{Z}$, block
diagonal.
We use $H_0 \mathcal{U}' = \mathcal{U}' H_0 -\mathcal{X}$ to move $H_0$ through
to the right and find

:::{math}
:label: H_tilde
\toggle{
  \tilde{\mathcal{H}} = \texttip{\color{red}{\ldots}}{click to expand} = H_0 - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U},
}{
  \begin{align*}
  \tilde{\mathcal{H}}
  &= H_0 + \mathcal{U}'^\dagger H_0 + (H_0 \mathcal{U}') + \mathcal{U}'^\dagger H_0
  \mathcal{U}' + \mathcal{U}^\dagger(\mathcal{H'}\mathcal{U})
  \\
  &= H_0 + \mathcal{U}'^\dagger H_0 + \mathcal{U}'H_0 - \mathcal{X} + \mathcal{U}'^\dagger (\mathcal{U}' H_0 - \mathcal{X}) + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U}\\
  &= H_0 + (\mathcal{U}'^\dagger + \mathcal{U}' + \mathcal{U}'^\dagger \mathcal{U}')H_0 - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U}\\
  &= H_0 - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U},
  \end{align*}
}
\endtoggle
:::

where the terms multiplied by $H_0$ cancel by unitarity.

The transformed Hamiltonian does not contain products by $H_0$ anymore, but it
does depend on $\mathcal{X}$, an auxiliary operator whose recurrent definition
we do not know yet.
To find it, we first focus on its anti-Hermitian part, $\mathcal{Z}$.
Since recurrence relations are expressions whose right hand side contains
Cauchy products between series, we need to find a way to make a product appear.
This is where the unitarity condition $\mathcal{U}'^\dagger + \mathcal{U} =
-\mathcal{U}'^\dagger \mathcal{U}$ comes in handy and gives:

:::{math}
:label: Z
\toggle{
  \mathcal{Z} = \texttip{\color{red}{\ldots}}{click to expand} = \frac{1}{2}(-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{X}^\dagger\mathcal{U}').
}{
  \begin{align}
  \mathcal{Z}
  &= \frac{1}{2} (\mathcal{X} + \mathcal{X}^{\dagger}) \\
  &= \frac{1}{2}\Big[ (\mathcal{U}' + \mathcal{U}'^{\dagger}) H_0 - H_0 (\mathcal{U}' + \mathcal{U}'^{\dagger}) \Big] \\
  &= \frac{1}{2} \Big[ - \mathcal{U}'^{\dagger} (\mathcal{U}'H_0 - H_0 \mathcal{U}') + (\mathcal{U}'H_0 - H_0 \mathcal{U}') \mathcal{U}' \Big] \\
  &= \frac{1}{2} (- \mathcal{U}'^{\dagger} \mathcal{X} + \mathcal{X}^{\dagger} \mathcal{U}').
  \end{align}
}
\endtoggle
:::

Similar to computing $\mathcal{W_n}$, computing $\mathcal{Z_n}$ requires lower
orders of $\mathcal{X}$ and $\mathcal{U}'$, all blocks included.
*This is our second secret ingredient✨*

Then, we compute the Hermitian part of $\mathcal{X}$ by requiring that
$\tilde{\mathcal{H}}^{AB} = 0$ and find

:::{math}
:label: Y
\mathcal{X}^{AB} = (\mathcal{U}^\dagger \mathcal{H}' \mathcal{U} -
\mathcal{U}'^\dagger \mathcal{X})^{AB}.
:::

Once again, despite $\mathcal{X}$ enters the right hand side, because all the
terms lack 0-th order, this defines a recursive relation for $\mathcal{X}^{AB}$,
and therefore $\mathcal{Y}$.
*This is our last secret ingredient✨*

The final part is standard: the definition of $\mathcal{Y}$ in {eq}`XYZ` fixes
$\mathcal{V}$ as a solution of:

:::{math}
:label: sylvester
\mathcal{V}^{AB}H_0^{BB} - H_0^{AA} \mathcal{V}^{AB} = \mathcal{Y}^{AB},
:::

a [Sylvester's equation](https://en.wikipedia.org/wiki/Sylvester_equation),
which we only need to solve once for every new order.
In the eigenbasis of $H_0$, the solution of Sylvester's equation is
$V^{AB}_{ij} = Y^{AB}_{ij}/(E_i - E_j)$, where $E_i$ are the eigenvalues of
$H_0$.
However, even if the eigenbasis of $H_0$ is not available, there are efficient
algorithms to solve Sylvester's equation, see [below](#implicit).

## The algorithm

We now have a complete algorithm:

1. Define series $\mathcal{U}'$ and $\mathcal{X}$ and make use of their block structure and Hermiticity.
2. To define the diagonal blocks of $\mathcal{U}'$, use $\mathcal{W} = -\mathcal{U}'^\dagger\mathcal{U}'/2$.
3. To find the off-diagonal blocks of $\mathcal{U}'$, solve Sylvester's equation  $\mathcal{V}^{AB}H_0^{AA} - H_0^{BB}\mathcal{V}^{AB} = \mathcal{Y}^{AB}$. This requires $\mathcal{X}$.
4. To find the diagonal blocks of $\mathcal{X}$, define $\mathcal{Z} = (-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{X}^\dagger\mathcal{U}')/2$.
5. For the off-diagonal blocks of $\mathcal{X}$, use $\mathcal{Y}^{AB} =
 (-\mathcal{U}'^\dagger\mathcal{X} +
  \mathcal{U}^\dagger\mathcal{H}'\mathcal{U})^{AB}$.
6. Compute the effective Hamiltonian as $\tilde{\mathcal{H}}_{\textrm{diag}} = H_0 - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U}$.

(implicit)=
## How to use Pymablock on large numerical Hamiltonians?

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
