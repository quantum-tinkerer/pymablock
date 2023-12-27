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

$$
\mathcal{H} = H_0 + \mathcal{H}',\quad H_0 = \begin{pmatrix}
H_0^{AA} & 0\\
0 & H_0^{BB}
\end{pmatrix},
$$

with $\mathcal{H}'$ containing an arbitrary number and orders of perturbations.
The series here may be multivariate, and they represent sums of the form

$$
\mathcal{A} = \sum_{n_1=0}^\infty \sum_{n_2=0}^\infty \cdots \sum_{n_k=0}^\infty \lambda_1^{n_1} \lambda_2^{n_2} \cdots \lambda_k^{n_k} A_{n_1, n_2, \ldots, n_k},
$$

where $\lambda_i$ are the perturbation parameters and $A_{n_1, n_2, \ldots,
n_k}$ are linear operators.

The problem statement, therefore, is finding $\mathcal{U}$ and
$\tilde{\mathcal{H}}$ such that:

$$
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U},\quad \tilde{\mathcal{H}}^{AB} = 0,\quad \mathcal{U}^\dagger \mathcal{U} = 1,
$$

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
  of energy denominators. Therefore, any additional multiplcations by $H_0$
  must be canceled by additional energy denominators. Muliplying by $H_0$ is
  therefore unnecessary work that gives longer intermediate expressions.

## Solution

Let us separate the unitary transformation into more parts

$$
\mathcal{U} = 1 + \mathcal{U}' = 1 + \mathcal{W} + \mathcal{V},\quad \mathcal{W}^\dagger = \mathcal{W},\quad \mathcal{V}^\dagger = -\mathcal{V}.
$$

Substituting $\mathcal{U}'$ into the unitarity condition
$(1/2)(\mathcal{U}^\dagger \mathcal{U} + \mathcal{U}^\dagger \mathcal{U}) = 1$ gives

$$
2\mathcal{W} = \mathcal{U}' + \mathcal{U}'^\dagger = -\mathcal{U}'^\dagger \mathcal{U}' = -\mathcal{W}^2 + \mathcal{V}^2.
$$

We can use this to define $\mathcal{W}$ as a Taylor series in $\mathcal{V}$:

$$
\mathcal{W} = \sqrt{1 + \mathcal{V}^2} - 1 \equiv f(\mathcal{V}) \equiv \sum_n a_n \mathcal{V}^{2n},
$$

however this is both computationally expensive and unnecessary.
Indeed, because $\mathcal{U}'$ has no 0th order term, $(\mathcal{U}'^\dagger
\mathcal{U}')_\mathbf{n}$ does not depend on the $\mathbf{n}$-th order of
$\mathcal{U}$, and therefore $\mathcal{W}$ can be computed recursively by using
the identity above.
We also see that $\mathcal{W}$ is fully defined by the unitarity condition,
while we can choose $\mathcal{V}$ to block-diagonalize the Hamiltonian.
*This recursive definition is the first secret ingredient of Pymablock✨*

We now additionally constrain $\mathcal{V}$ to be block off-diagonal:
$\mathcal{V}^{AA} = \mathcal{V}^{BB} = 0$.
This choice means that the unitary transformation has the smallest norm, and is
equivalent to the Schrieffer-Wolff transformation.

:::{admonition} Equivalence to Schrieffer-Wolff transformation
:class: dropdown info
Both the Pymablock algorithm and the more commonly used Schrieffer-Wolff
transformation find a unitary transformation $\mathcal{U}$ such that $\tilde{\mathcal{H}}^{AB}=0$.
They are therefore equivalent up to a gauge choice in each subspace.
We establish the correspondence between the two by demonstrating that this gauge
choice is the same for both algorithms.

The Schrieffer-Wolff transformation uses $\mathcal{U} = \exp \mathcal{S}$, where $\mathcal{S} = -\mathcal{S}^\dagger$ and $\mathcal{S}^{AA} = \mathcal{S}^{BB} = 0$.

The series $\exp\mathcal{S}$ contains all possible products of $S_n$ of all lengths with fractional prefactors.
For every term $S_{k_1}S_{k_2}\cdots S_{k_n}$, there is a corresponding term
$S_{k_n}S_{k_{n-1}}\cdots S_{k_1}$ with the same prefactor.
If the number of $S_{k_n}$ is even, then both terms are block-diagonal since
each $S_n$ is block off-diagonal.
Because $S_n$ are anti-Hermitian, the two terms are Hermitian conjugates of each
other, and therefore their sum is Hermitian.
On the other hand, if the number of $S_{k_n}$ is odd, then the two terms are
block off-diagonal and their sum is anti-Hermitian by the same reasoning.

Therefore just like in our algorithm, the diagonal blocks of $\exp S$ are
Hermitian, while off-diagonal blocks are anti-Hermitian.
Schieffer-Wolff transformation produces a unique answer and satisfies the same
diagonalization requirements as our algorithm, which means that the two are
equivalent.
:::

Now let us look at the unitary transformation of the Hamiltonian

$$
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U} = (1 + \mathcal{U}'^\dagger) H_0 (1 + \mathcal{U}') + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U}.
$$

Our goal is to define a recursive definition for $\mathcal{U}'$ that does not
involve multiplications by $H_0$.
We do so by defining the commutator $\mathcal{X} \equiv [\mathcal{U}', H_0]$ as
an auxiliary varitable.
Somewhat similarly to $\mathcal{U}'$, $\mathcal{X} = \mathcal{Y} + \mathcal{Z}$,
where $\mathcal{Y} = [\mathcal{V}, H_0]$ is Hermitian block-offdiagonal and $\mathcal{Z} = [\mathcal{W}, H_0]$ is anti-Hermitian block-diagonal.
Our goal is to eliminate products of $H_0$ from $\tilde{\mathcal{H}}$.
In other words, we want to find an expression for $\tilde{\mathcal{H}}$ that
depends on $\mathcal{X}$ and $\mathcal{U}'$, but does not contain
multiplications by $H_0$.

To utilize the commutator, we commute all $H_0$ to the right:

$$
\begin{align*}
\mathcal{U}^\dagger H \mathcal{U}
&= H_0 + \mathcal{U}'^\dagger H_0 + H_0 \mathcal{U}' + \mathcal{U}'^\dagger H_0 \mathcal{U}' + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U}\\
&= H_0 + \mathcal{U}'^\dagger H_0 + \mathcal{U}'H_0 - \mathcal{X} + \mathcal{U}'^\dagger (\mathcal{U}' H_0 - \mathcal{X}) + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U}\\
&= H_0 + (\mathcal{U}'^\dagger + \mathcal{U}' + \mathcal{U}'^\dagger \mathcal{U}')H_0 - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U}\\
&= H_0 - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U},
\end{align*}
$$

Where the terms multiplied by $H_0$ cancel by unitarity.
The expression for $\tilde{\mathcal{H}}$ must be Hermitian (we started with a
Hermitian one after all).
We can exploit this to find the recurrence relation for $\mathcal{Z}$:

$$
0 = \tilde{\mathcal{H}}-\tilde{\mathcal{H}}^\dagger = -\mathcal{X}-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{X}^\dagger + \mathcal{X}^\dagger\mathcal{U}'.
$$

Therefore

$$
\mathcal{Z} = \frac{1}{2}(\mathcal{X} - \mathcal{X}^\dagger) = (-\mathcal{U}'^\dagger\mathcal{X} - \textrm{h.c.})/2.
$$

*This is our second secret ingredient✨*

Finally, we compute the Hermitian part of $\mathcal{X}$ by requiring that $\tilde{\mathcal{H}}$ is block-diagonal and the diagonal blocks are zero, so

$$
\mathcal{Y}^{AB} = (\mathcal{Y}^{BA})^\dagger=\mathcal{X}^{AB} = (\mathcal{U}^\dagger \mathcal{H}' \mathcal{U} - \mathcal{U}'^\dagger \mathcal{X})^{AB}.
$$

Once again, despite $\mathcal{X}$ enters the right hand side, because all the terms lack 0-th order, this defines a recurrence relation for $\mathcal{Y}$ similar to the recursive definition of $\mathcal{W}$. *This is our last secret ingredient✨*

The final part is standard: the definition of $\mathcal{Y}$ fixes $\mathcal{V}$ as a solution of Sylvester equation:

$$
\mathcal{V}^{AB}H_0^{BB} - H_0^{AA} \mathcal{V}^{AB} = \mathcal{Y}^{AB}.
$$

## The algorithm

We now have a complete algorithm:
1. Define series $\mathcal{U}'$ and $\mathcal{X}$ and make use of their block structure and Hermiticity.
2. Use $\mathcal{W} = -\mathcal{U}'^\dagger\mathcal{U}'/2$ to define diagonal blocks of $\mathcal{U}'$.
3. Solve the Sylvester's equation $\mathcal{V}^{AB}H_0^{AA} - H_0^{BB}\mathcal{V}^{AB} = \mathcal{Y}^{AB}$ to find offdiagonal blocks of $\mathcal{U}'$.
4. Use $\mathcal{Z} = (-\mathcal{U}'^\dagger\mathcal{X} - \textrm{h.c.})/2$ for diagonal blocks of $\mathcal{X}$.
5. Use $\mathcal{Y}^{AB} = (-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{U}^\dagger\mathcal{H}'\mathcal{U})^{AB}$
6. Use $\tilde{\mathcal{H}}_{\textrm{diag}} = H_0 + \mathcal{U}^\dagger\mathcal{H}'\mathcal{U} - (\mathcal{U}'^\dagger \mathcal{X} +\textrm{h.c.})/2$.

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
