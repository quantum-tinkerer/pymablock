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

In the simplest setting, Pymablock finds a series of the unitary transformation $\mathcal{U}$ (we use calligraphic letters to denote series) that block-diagonalizes the Hamiltonian

:::{math}
:label: hamiltonian
\mathcal{H} = H_0 + \mathcal{H}',\quad H_0 = \begin{pmatrix}
H_0^{AA} & 0\\
0 & H_0^{BB}
\end{pmatrix},
:::

with $\mathcal{H}' = \mathcal{H}'_{S} + \mathcal{H}'_{R}$ containing an arbitrary number and orders of perturbations with block-diagonal components $\mathcal{H}'_{S}$ and block-offdiagonal components $\mathcal{H}'_{R}$, respectively.
Pymablock's algorithm, however, does not just work with $2 \times 2$ block matrices: it can perturbatively cancel any set of offdiagonal elements of $\mathcal{H}$.
For that we define the *selected* part of an operator (subscript $S$) as the subset of its matrix elements that we want to keep, and the *remaining* part (subscript $R$) as the rest.
For example, selected and remaining parts may look like this:

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

np.random.seed(0)
N = 16
H_1 = np.random.randn(N, N)
H_1 += H_1.T
# Take a square root of the absolute value to make the elements bigger
H_1 = np.sqrt(np.abs(H_1)) * np.sign(H_1)

smiley_binary = np.array(
    [
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ],
    dtype=bool,
)

mask = np.zeros((N, N), dtype=bool)
mask[1:smiley_binary.shape[0] + 1, -smiley_binary.shape[1] - 1:-1] = smiley_binary
mask[[3, 5, 12], [4, 3, 14]] = True
mask = ~(mask | mask.T)
np.fill_diagonal(mask, False)
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9, 3))
ax0.imshow(H_1, cmap='seismic', norm=TwoSlopeNorm(vcenter=0), origin='upper')
ax0.set_title('$H$')
ax1.imshow(H_1 * ~mask, cmap='seismic', norm=TwoSlopeNorm(vcenter=0), origin='upper')
ax1.set_title('$H_S$')
ax2.imshow(H_1 * mask, cmap='seismic', norm=TwoSlopeNorm(vcenter=0), origin='upper')
ax2.set_title('$H_R$')
for ax in (ax0, ax1, ax2):
    ax.axis('off')
```

As another generalization compared to the simplest case, the series here may be multivariate, and they represent sums of the form

$$
\mathcal{A} = \sum_{n_1=0}^\infty \sum_{n_2=0}^\infty \cdots \sum_{n_k=0}^\infty \lambda_1^{n_1} \lambda_2^{n_2} \cdots \lambda_k^{n_k} A_{n_1, n_2, \ldots, n_k},
$$

where $\lambda_i$ are the perturbation parameters and $A_{n_1, n_2, \ldots, n_k}$ are linear operators.

The problem statement, therefore, is finding $\mathcal{U}$ and $\tilde{\mathcal{H}}$ such that:

:::{math}
:label: eq:problem_definition
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U},\quad \tilde{\mathcal{H}}_{R} = 0,\quad \mathcal{U}^\dagger \mathcal{U} = 1,
:::

where series multiply according to the [Cauchy product](https://en.wikipedia.org/wiki/Cauchy_product):

$$
\mathcal{C} = \mathcal{A}\mathcal{B} \Leftrightarrow C_\mathbf{n} = \sum_{\mathbf{m} + \mathbf{p} = \mathbf{n}} A_\mathbf{m} B_\mathbf{p}.
$$

:::{admonition} Computational complexity of the Cauchy product
:class: dropdown info
The Cauchy product is the most expensive operation in perturbation theory, because it involves a large number of multiplications between potentially large matrices, so let us discuss its complexity.
Evaluating $\mathbf{n}$-th order of $\mathcal{C}=\mathcal{AB}$ requires $\sim\prod_i n_i = N$ multiplications of the series elements.
A direct computation of all the possible index combinations in a product between three series $\mathcal{A}\mathcal{B}\mathcal{C}$ would have a higher cost $\sim N^2$, however if we use associativity of the product and compute this as $(\mathcal{A}\mathcal{B})\mathcal{C}$, then the scaling of the cost stays $\sim N$.
:::

## How to solve the problem?

There are many ways to solve this problem that give identical expressions for $\mathcal{U}$ and $\tilde{\mathcal{H}}$.
We are searching for a procedure that satisfies three additional constraints:

- It has the same complexity scaling as a Cauchy product.
- It does not require multiplications by $H_0$.
  This is because in perturbation theory, $n$-th order corrections to $\tilde{\mathcal{H}}$ carry $n$ energy denominators $1/(E_i - E_j)$ (see [here](https://en.wikipedia.org/wiki/Perturbation_theory_(quantum_mechanics)#Time-independent_perturbation_theory)).
  Therefore, any additional multiplications by $H_0$ must cancel with additional energy denominators.
  Multiplying by $H_0$ is therefore unnecessary work, and it gives longer intermediate expressions.
- It requires only one Cauchy product by $\mathcal{H}_{S}$, the selected part of $\mathcal{H}$.
  By considering $\mathcal{H}_{R}=0$, we observe that $\mathcal{H}_{S}$ must at least enter $\tilde{\mathcal{H}}$ as an
  added term, without any products.
  Moreover, because $\mathcal{U}$ depends on the entire Hamiltonian, there must be at least one Cauchy product by $\mathcal{H}'_{S}$.
  This lower bound turns out to be possible to achieve.

The goal of our algorithm is thus to be efficient and to produce compact results that do not require further simplifications.

## Our solution

To find $\mathcal{U}$, let us separate it into an identity and $\mathcal{U}' =
\mathcal{W} + \mathcal{V}$:

:::{math}
:label: U
\mathcal{U} = 1 + \mathcal{U}' = 1 + \mathcal{W} + \mathcal{V},\quad \mathcal{W}^\dagger = \mathcal{W},\quad \mathcal{V}^\dagger = -\mathcal{V}.
:::

First, we use the unitarity condition $\mathcal{U}^\dagger \mathcal{U} = 1$ by substituting $\mathcal{U}'$ into it
and obtain

:::{math}
:label: W
\toggle{
  \mathcal{W} = \texttip{\color{red}{\ldots}}{click to expand} = -\frac{1}{2}
  \mathcal{U}'^\dagger \mathcal{U}'.
}{
  \begin{align}
    \mathcal{W} &= \frac{1}{2}(\mathcal{U}'^\dagger + \mathcal{U}') \\
      &= \frac{1}{2} \Big[(1 + \mathcal{U}'^\dagger)(1+\mathcal{U}') - 1 - \mathcal{U}'^\dagger \mathcal{U}' \Big] \\
      &= -\frac{1}{2} \mathcal{U}'^\dagger \mathcal{U}'.
  \end{align}
}
\endtoggle
:::

Because $\mathcal{U}'$ has no $0$-th order term, $(\mathcal{U}'^\dagger \mathcal{U}')_\mathbf{n}$ does not depend on the $\mathbf{n}$-th order of $\mathcal{U}'$ nor $\mathcal{W}$.
More generally, a Cauchy product $\mathcal{A}\mathcal{B}$ where $\mathcal{A}$ and $\mathcal{B}$ have no $0$-th order terms depends on $\mathcal{A}_1, \ldots, \mathcal{A}_{n-1}$ and $\mathcal{B}_1, \ldots, \mathcal{B}_{n-1}$.
This allows us to use Cauchy products to define recurrence relations, which we apply throughout the algorithm.
Therefore, we compute $\mathcal{W}$ as a Cauchy product of $\mathcal{U}'$ with itself.
*This recurrence relation is the first secret ingredient of Pymablock✨*

:::{admonition} Choosing the right definition for $\mathcal{W}$
:class: dropdown info

Using the definition of $\mathcal{W}$ as the Hermitian part of $\mathcal{U}'$, and the unitarity condition:

$$
\begin{align}
2\mathcal{W}
= \mathcal{U}' + \mathcal{U}'^\dagger
= -\mathcal{U}'^\dagger \mathcal{U}'
= -\mathcal{W}^2 + \mathcal{V}^2.
\end{align}
$$

we see that we could alternatively define $\mathcal{W}$ as a Taylor series in $\mathcal{V}$:

$$
\mathcal{W} = \sqrt{1 + \mathcal{V}^2} - 1 \equiv f(\mathcal{V}) \equiv \sum_n a_n \mathcal{V}^{2n},
$$

however the scaling of this definition is worse than that of a Cauchy product because we need to compute a Taylor expansion of a series:

$$
f(\mathcal{A}) = \sum_{n=0}^\infty a_n \mathcal{A}^n.
$$

A direct computation of all possible products of terms would require $\sim \exp N$ multiplications.
We improve on this by defining a new series as $\mathcal{A}^{n+1} = \mathcal{A}\mathcal{A}^{n}$ and reusing the previously computed results, which brings these costs down to $\sim N^2$—still worse than a cost of a single Cauchy product.
Using the Taylor expansion approach is therefore both more complicated and more computationally expensive than the recurrent definition in {eq}`W`.
:::

To compute $\mathcal{U}'$ we also need to find $\mathcal{V}$, which is defined by the requirement $\tilde{\mathcal{H}}_{R} = 0$.
Additionally, we constrain $\mathcal{V}$ to have no selected part: $\mathcal{V}_{S} = 0$, so that its norm and the norm of $\mathcal{U}'$ are minimal.
This also makes our transformation equivalent to Schrieffer-Wolff transformation in the $2\times 2$ block case.

:::{admonition} Equivalence to Schrieffer-Wolff transformation
:class: dropdown info
Both the Pymablock algorithm applied to a $2\times 2$ block matrix and the Schrieffer-Wolff transformation find a unitary transformation $\mathcal{U}$ such that $\tilde{\mathcal{H}}^{AB}=0$.
They are therefore equivalent up to a gauge choice in each subspace, $A$ and $B$.
We establish the equivalence between the two by demonstrating that this gauge choice is the same for both algorithms.

The Schrieffer-Wolff transformation uses $\mathcal{U} = \exp \mathcal{S}$, where $\mathcal{S} = -\mathcal{S}^\dagger$ and $\mathcal{S}^{AA} = \mathcal{S}^{BB} = 0$.

The series $\exp\mathcal{S}$ contains all possible products of $S_n$ of all lengths with fractional prefactors.
For every term $S_{k_1}S_{k_2}\cdots S_{k_n}$, there is a corresponding term $S_{k_n}S_{k_{n-1}}\cdots S_{k_1}$ with the same prefactor.
If the number of $S_{k_n}$ is even, then both terms are block-diagonal since each $S_n$ is block off-diagonal.
Because $S_n$ are anti-Hermitian, the two terms are Hermitian conjugates of each other, and therefore their sum is Hermitian.
On the other hand, if the number of $S_{k_n}$ is odd, then the two terms are block off-diagonal and their sum is anti-Hermitian by the same reasoning.

Therefore, just like in our algorithm, the diagonal blocks of $\exp S$ are Hermitian, while off-diagonal blocks are anti-Hermitian.
Schrieffer-Wolff transformation produces a unique answer and satisfies the same diagonalization requirements as our algorithm, which means that the two are equivalent.
:::

To find $\mathcal{V}$, we need to first look at the transformed Hamiltonian:

$$
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U} = \mathcal{H}_{S} +
\mathcal{U}'^\dagger \mathcal{H}_{S} + \mathcal{H}_{S} \mathcal{U}' + \mathcal{U}'^\dagger \mathcal{H}_{S}
\mathcal{U}' + \mathcal{U}^\dagger\mathcal{H}'_{R}\mathcal{U},
$$
where we used $\mathcal{U}=1+\mathcal{U}'$ and $\mathcal{H} = \mathcal{H}_{S} + \mathcal{H}'_{R}$, since $(H_0)_{R}=0$ by definition.

Because we want to avoid unnecessary products by $\mathcal{H}_{S}$, we need to get rid of the terms that contain it by replacing them with an alternative expression.
Our strategy is to define an auxiliary operator $\mathcal{X}$ that we can compute without ever multiplying by $\mathcal{H}_{S}$.
Like $\mathcal{U}'$, $\mathcal{X}$ needs to be defined via a recurrence relation, which we will find later.
Because the expression above has $\mathcal{H}_{S}$ multiplied by $\mathcal{U}'$ from the left and from the right, we get rid of these terms by making sure that $\mathcal{H}_{S}$ multiplies terms from one side only.
To achieve this, we choose $\mathcal{X}=\mathcal{Y}+\mathcal{Z}$ to be the commutator between $\mathcal{U}'$ and $\mathcal{H}_{S}$:

:::{math}
:label: XYZ
\mathcal{X} \equiv [\mathcal{U}', \mathcal{H}_{S}] = \mathcal{Y} + \mathcal{Z}, \quad
\mathcal{Y} \equiv [\mathcal{V}, \mathcal{H}_{S}] = \mathcal{Y}^\dagger,\quad
\mathcal{Z} \equiv [\mathcal{W}, \mathcal{H}_{S}] = -\mathcal{Z}^\dagger.
:::

In the common case, when  $\mathcal{A}_{S}$ is a block-diagonal part of a matrix, $\mathcal{Y}$ is block off-diagonal. Additionally, in the case of $2\times 2$ block diagonalization, $\mathcal{Z}$ is block-diagonal.
We use $\mathcal{H}_{S} \mathcal{U}' = \mathcal{U}' \mathcal{H}_{S} -\mathcal{X}$ to move $\mathcal{H}_{S}$ through
to the right and find

:::{math}
:label: H_tilde
\toggle{
  \tilde{\mathcal{H}} = \texttip{\color{red}{\ldots}}{click to expand} = \mathcal{H}_{S} - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H}'_{R}\mathcal{U},
}{
  \begin{align*}
  \tilde{\mathcal{H}}
  &= \mathcal{H}_{S} + \mathcal{U}'^\dagger \mathcal{H}_{S} + (\mathcal{H}_{S} \mathcal{U}') + \mathcal{U}'^\dagger \mathcal{H}_{S}
  \mathcal{U}' + \mathcal{U}^\dagger\mathcal{H}'_{R}\mathcal{U}
  \\
  &= \mathcal{H}_{S} + \mathcal{U}'^\dagger \mathcal{H}_{S} + \mathcal{U}'\mathcal{H}_{S} - \mathcal{X} + \mathcal{U}'^\dagger (\mathcal{U}' \mathcal{H}_{S} - \mathcal{X}) + \mathcal{U}^\dagger\mathcal{H}_{R}\mathcal{U}\\
  &= \mathcal{H}_{S} + (\mathcal{U}'^\dagger + \mathcal{U}' + \mathcal{U}'^\dagger \mathcal{U}')\mathcal{H}_{S} - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H}'_{R}\mathcal{U}\\
  &= \mathcal{H}_{S} - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H}'_{R}\mathcal{U},
  \end{align*}
}
\endtoggle
:::

where the terms multiplied by $\mathcal{H}_{S}$ cancel by unitarity.

The transformed Hamiltonian does not contain products by $\mathcal{H}_{S}$ anymore, but it does depend on $\mathcal{X}$, an
auxiliary operator whose recurrent definition we do not know yet.
To find it, we first focus on its anti-Hermitian part, $\mathcal{Z}$.
Since recurrence relations are expressions whose right hand side contains Cauchy products between series, we need to find a way to make a product appear.
This is where the unitarity condition $\mathcal{U}'^\dagger + \mathcal{U}' = -\mathcal{U}'^\dagger \mathcal{U}'$ comes in handy and gives:

:::{math}
:label: Z
\toggle{
  \mathcal{Z} = \texttip{\color{red}{\ldots}}{click to expand} = \frac{1}{2}(-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{X}^\dagger\mathcal{U}').
}{
  \begin{align}
  \mathcal{Z}
  &= \frac{1}{2} (\mathcal{X} - \mathcal{X}^{\dagger}) \\
  &= \frac{1}{2}\Big[ (\mathcal{U}' + \mathcal{U}'^{\dagger}) \mathcal{H}_{S} - \mathcal{H}_{S} (\mathcal{U}' + \mathcal{U}'^{\dagger}) \Big] \\
  &= \frac{1}{2} \Big[ - \mathcal{U}'^{\dagger} (\mathcal{U}'\mathcal{H}_{S} - \mathcal{H}_{S} \mathcal{U}') + (\mathcal{U}'\mathcal{H}_{S} - \mathcal{H}_{S} \mathcal{U}')^{\dagger} \mathcal{U}' \Big] \\
  &= \frac{1}{2} (-\mathcal{U}'^{\dagger} \mathcal{X} + \mathcal{X}^{\dagger} \mathcal{U}').
  \end{align}
}
\endtoggle
:::

Similar to computing $W_\mathbf{n}$, computing $Z_\mathbf{n}$ requires lower orders of $\mathcal{X}$ and $\mathcal{U}'$, all blocks included.
*This is our second secret ingredient✨*

Then, we compute $\mathcal{Y}$ (the Hermitian part of $\mathcal{X}$) by requiring that $\tilde{\mathcal{H}}_{R} = 0$ and find

:::{math}
:label: Y
\mathcal{Y}_{R} = (\mathcal{U}^\dagger \mathcal{H}'_{R} \mathcal{U} -
\mathcal{U}'^\dagger \mathcal{X} - \mathcal{Z})_{R}.
:::

Once again, despite $\mathcal{X}$ enters the right hand side, because all the terms lack 0-th order, this defines a recursive relation for $\mathcal{Y}_{R}$.
To find $\mathcal{Y}_{S}$ we use the definition of $\mathcal{Y}$ in {eq}`XYZ`, which fixes $\mathcal{V}$ as a solution of:

:::{math}
:label: sylvester
[\mathcal{V}, H_0] = \mathcal{Y} - [\mathcal{V}, \mathcal{H}'_{S}],
:::

which is a [continuous-time Lyapunov equation](https://en.wikipedia.org/wiki/Lyapunov_equation) for $\mathcal{V}$.
In order for this equation to be satisfiable, the selected part of the right hand side must vanish, since the left hand side only has the remaining part.
Therefore we find the selected part of $\mathcal{Y}$ as:

:::{math}
:label: Y_S
\mathcal{Y}_{S} = [\mathcal{V}, \mathcal{H}'_{S}]_{S},
:::

and it vanishes if $\mathcal{A}_{S}$ corresponds to a block-diagonal matrix.
*This is our last secret ingredient✨*

The final part is straightforward. Finding $\mathcal{V}$ from $\mathcal{Y}$ amounts to solving a [Sylvester's equation](https://en.wikipedia.org/wiki/Sylvester_equation), which we only need to solve once for every new order.
This is the only step in the algorithm that requires a direct multiplication by $\mathcal{H}'_{S}$.
In the eigenbasis of $H_0$, the solution of Sylvester's equation is $V_{\mathbf{n}, ij} = (\mathcal{Y}_{R} - [\mathcal{V},
\mathcal{H}'_{S}]_{R})_{\mathbf{n}, ij}/(E_i - E_j)$, where $E_i$ are the eigenvalues of $H_0$.
However, even if the eigenbasis of $H_0$ is not available, there are efficient algorithms to solve Sylvester's equation, see [below](#implicit).

## Actual algorithm

We now have the complete algorithm:

1. Define series $\mathcal{U}'$ and $\mathcal{X}$ and make use of their block structure and Hermiticity.
2. To define the Hermitian part $\mathcal{U}'$, use $\mathcal{W} = -\mathcal{U}'^\dagger\mathcal{U}'/2$.
3. To find the antihermitian part of $\mathcal{U}'$, solve Sylvester's equation  $[\mathcal{V}, H_0] = (\mathcal{Y} - [\mathcal{V}, \mathcal{H}'_{S}])_{R}$. This requires $\mathcal{X}$.
4. To find the antihermitian part of $\mathcal{X}$, define $\mathcal{Z} = (-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{X}^\dagger\mathcal{U}')/2$.
5. For the Hermitian part of $\mathcal{X}$, use $\mathcal{Y} = (-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{U}^\dagger\mathcal{H}'_{R}\mathcal{U})_{R} + [\mathcal{V}, \mathcal{H}'_{S}]_{S}$.
6. Compute the effective Hamiltonian as $\tilde{\mathcal{H}}_{S} = \mathcal{H}_{S} - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H}'_{R}\mathcal{U}$.

::::{admonition} Extra optimization: common subexpression elimination
:class: dropdown info

We further optimize the algorithm by reusing products that are needed in several places.

Firstly, we rewrite the expressions for $\mathcal{Z}$ and $\tilde{\mathcal{H}}$ by utilizing the Hermitian conjugate of $\mathcal{U}'^\dagger \mathcal{X}$ without recomputing it:

$$
\begin{gather*}
\mathcal{Z} = \frac{1}{2}[(-\mathcal{U}'^\dagger \mathcal{X})- \textrm{h.c.}],\\
\tilde{\mathcal{H}} = \mathcal{H}_{S} + \mathcal{U}^\dagger \mathcal{H}'_{R} \mathcal{U} - (\mathcal{U}'^\dagger \mathcal{X} + \textrm{h.c.})/2 - \mathcal{Y}_{S},
\end{gather*}
$$

where $\textrm{h.c.}$ is the Hermitian conjugate, and $\mathcal{Z}$ drops out from  $\tilde{\mathcal{H}}_{S}$ because $\mathcal{Z}$ is anti-Hermitian.

To compute $\mathcal{U}^\dagger \mathcal{H}'_{R} \mathcal{U}$ faster, we express it using $\mathcal{A} \equiv \mathcal{H}'_{R}\mathcal{U}'$:

$$
\mathcal{U}^\dagger \mathcal{H}'_{R} \mathcal{U} = \mathcal{H}'_{R} + \mathcal{A} + \mathcal{A}^\dagger + \mathcal{U}'^\dagger \mathcal{A}.
$$

To further optimize the computations, we observe that some products appear both in $\mathcal{U}'^\dagger \mathcal{X}$ and $\mathcal{U}^\dagger \mathcal{H}'_{R} \mathcal{U}$.
We reuse these products by introducing $\mathcal{B} = \mathcal{X} - \mathcal{H}'_{R} - \mathcal{A}$.


Using this definition, we first express the remaining part of $\mathcal{B}$ as follows:

$$
\toggle{
  \mathcal{B}_{R} = \texttip{\color{red}{\ldots}}{click to expand} = -(\mathcal{U'}^\dagger \mathcal{B})_{R},
}{
  \begin{align*}
  \mathcal{B}_{R} &= \left[\mathcal{X} - \mathcal{H}'_{R} - \mathcal{A} \right]_{R}\\
  &= \left[\mathcal{A}^\dagger + \mathcal{U}'^\dagger\mathcal{A} - \mathcal{U}'^\dagger \mathcal{X} \right]_{R}\\
  &= \left[\mathcal{U}'^\dagger\mathcal{H}'_{R} + \mathcal{U}'^\dagger\mathcal{A} - \mathcal{U}'^\dagger \mathcal{X} \right]_{R}\\
  &= -(\mathcal{U'}^\dagger \mathcal{B})_{R},
  \end{align*}
}
\endtoggle
$$
where we also used Eq. {eq}`Y` and the definition of $\mathcal{A}$.

The selected part of $\mathcal{B}$ is then

$$
\toggle{
  \mathcal{B}_{S} = \texttip{\color{red}{\ldots}}{click to expand} = \left[\frac{1}{2}[(-\mathcal{U}'^\dagger \mathcal{B})- \textrm{h.c.}] + [\mathcal{VH}'_{S} +\textrm{h.c.}] - \frac{1}{2}[\mathcal{A}^\dagger + \mathcal{A} ]\right]_{S},
}{
\begin{align*}
  \mathcal{B}_{S} &= \left[\mathcal{Y} + \mathcal{Z} - \mathcal{H}'_{R} - \mathcal{A}\right]_{S} \\
  &= \left[\frac{1}{2}[(-\mathcal{U}'^\dagger \mathcal{X})- \textrm{h.c.}] + \mathcal{Y} - \mathcal{A}\right]_{S} \\
  &= \left[\frac{1}{2}[(-\mathcal{U}'^\dagger [\mathcal{X} - \mathcal{H}'_{R} - \mathcal{A}])- \textrm{h.c.}] + \mathcal{Y} - \frac{1}{2}[\mathcal{A}^\dagger + \mathcal{A} ] + {\frac{1}{2}[( - \mathcal{U}'^\dagger\mathcal{A} ) - \textrm{h.c.}]}\right]_{S}, \\
  &= \left[\frac{1}{2}[(-\mathcal{U}'^\dagger \mathcal{B})- \textrm{h.c.}] + [\mathcal{VH}'_{S} +\textrm{h.c.}] - \frac{1}{2}[\mathcal{A}^\dagger + \mathcal{A} ]\right]_{S},
  \end{align*}
}
\endtoggle
$$
where we used Eq. {eq}`Z` and that the $\mathcal{U}'^\dagger \mathcal{A}$ is Hermitian.

Using $\mathcal{B}$ changes the relation for $\mathcal{V}$ in Eq. {eq}`sylvester` to

:::{math}
:label: sylvester_optimized
[\mathcal{V},H_0] = \left(\mathcal{B} - \mathcal{H}' - \mathcal{A} - [\mathcal{V}, \mathcal{H}'_{S}]\right)_{R}.
:::

Finally, we compute the selected part of $\tilde{\mathcal{H}}$ as

$$
\tilde{\mathcal{H}}_{S} = \mathcal{H}_{S} + (\mathcal{A} - \mathcal{U}^\dagger \mathcal{B} + 2\mathcal{VH}'_{S} + \textrm{h.c.})_{S}/2.
$$

::::

(implicit)=
## How to use Pymablock on large numerical Hamiltonians?

Solving Sylvester's equation and computing the matrix products are the most expensive steps of the algorithms for large Hamiltonians.
Pymablock can efficiently construct an effective Hamiltonian of a small subspace even when the full Hamiltonian is a sparse matrix that is too costly to diagonalize.
It does so by avoiding forming a matrix representation of operators in the implicit subspace—the one without known eigenvectors, and by utilizing the sparsity of the Hamiltonian to compute the Green's function.
To do so, Pymablock uses either the [MUMPS sparse solver](https://mumps-solver.org/) via the [python-mumps](https://gitlab.kwant-project.org/kwant/python-mumps/) wrapper or the [KPM method](https://doi.org/10.1103/RevModPhys.78.275).

This approach was originally introduced in [this work](https://arxiv.org/abs/1909.09649).

::::{admonition} Implementation details
:class: dropdown info
We use the matrix $\Psi_E$ of the known eigenvectors of the explicit subspace to rewrite the Hamiltonian as

:::{math}
\mathcal{H} \to \begin{pmatrix}
\Psi_E^\dagger \mathcal{H} \Psi_E & \Psi_E^\dagger \mathcal{H} P_I \\
P_I \mathcal{H} \Psi_E & P_I \mathcal{H} P_I
\end{pmatrix},
:::

where $P_I = 1 - \Psi_E \Psi_E^\dagger$ is the projector onto the implicit subspace.
This Hamiltonian is larger in size than the original one because the $B$ block has additional null vectors corresponding to the $A$ subspace.
This, however, allows to preserve the sparsity structure of the Hamiltonian by applying $P_I$ and $\mathcal{H}$ separately.
Additionally, applying $P_I$ is efficient because $\Psi_E$ is a low rank matrix.
We then perform perturbation theory of the rewritten $\mathcal{H}$.

To solve the Sylvester's equation for the modified Hamiltonian, we write it for every row of $V_{\mathbf{n}}^{EI}$ separately:

:::{math}
V_{\mathbf{n}, ij}^{EI} (E_i - H_0) = Y_{\mathbf{n}, j}
:::

This equation is well-defined despite $E_i - H_0$ is not invertible because $Y_{\mathbf{n}}$ has no components in the explicit subspace.
::::
