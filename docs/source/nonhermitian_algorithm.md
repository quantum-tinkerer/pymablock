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

# Non-Hermitian algorithm

This page describes the non-Hermitian extension of Pymablock's [main algorithm](algorithms.md).
Using it requires an extra flag:

```python
H_tilde, U, U_inv = block_diagonalize(..., hermitian=False)
```

Here the third returned series is the perturbative inverse of $U$.
This is because unlike in Hermitian block-diagonalization, the inverse transformation is not the adjoint of the forward transformation and we track it explicitly.

The overall structure is the same:

- split the Hamiltonian into selected and remaining parts,
- organize perturbation theory through multivariate Cauchy products,
- avoid unnecessary products by $H_0$,
- reduce each perturbative order to one Sylvester solve.

Throughout this page, we use the notation from [the main algorithm page](algorithms.md).


## Problem statement

We seek a perturbative similarity transform

:::{math}
:label: nh:setup
\tilde{\mathcal{H}} = \mathcal{U}^{-1}\mathcal{H}\mathcal{U},
\qquad
\tilde{\mathcal{H}}_{R}=0,
\qquad
\mathcal{U}^{-1}\mathcal{U}=1,
:::

with

:::{math}
:label: nh:H_split
\mathcal{H}=\mathcal{H}_S+\mathcal{H}'_R,
\qquad
\mathcal{H}_S \equiv H_0+\mathcal{H}'_S.
:::

Here $S$ denotes the selected part and $R$ the remainder to eliminate, exactly as in [the main algorithm](algorithms.md).
Since $\mathcal{U}^{-1}\neq \mathcal{U}^{\dagger}$ in general, the left and right eigenvectors need not coincide.

## Working variables

Like in the Hermitian case, we separate the transformation into identity at zeroth ordwer and a correction, which allows us to define recursive relation expressing all series as a Cauchy product of other series.
Specifically, we introduce $\mathcal{U}'$ as the correction of the transformation $\mathcal{U}$ and $\mathcal{G}$ as the correction of its inverse $\mathcal{U}^{-1}$:

:::{math}
:label: nh:UG_def
\mathcal{U}=1+\mathcal{U}',
\qquad
\mathcal{U}^{-1}=1+\mathcal{G},
\qquad
\mathcal{U}'_0=\mathcal{G}_0=0.
:::

The inverse constraint then becomes

:::{math}
:label: nh:G_rec
\mathcal{G}=-\mathcal{U}'-\mathcal{G}\mathcal{U}'.
:::

Since both series $\mathcal{U}'$ and $\mathcal{G}$ start at first order, this is a closed recurrence for $\mathcal{G}$ once $\mathcal{U}'$ is known.

Similar to the Hermitian case, the block-diagonalizing transformation is not unique.
We fix the gauge by requiring that the selected part of $\mathcal{U}-\mathcal{U}^{-1}$ vanishes:

:::{math}
:label: nh:gauge
(\mathcal{U}'-\mathcal{G})_S=0.
:::

Equations {eq}`nh:G_rec` and {eq}`nh:gauge` together fix the selected part of the correction:

:::{math}
:label: nh:Uprime_S
\mathcal{U}'_S=-\frac{1}{2}(\mathcal{G}\mathcal{U}')_S.
:::

This matches the role played by the selected Hermitian part of the transformation in the Hermitian algorithm.

::::{admonition} Equivalence to the Hermitian algorithm
:class: dropdown info
If $\mathcal{H}$ is Hermitian and $\mathcal{U}^{-1}=\mathcal{U}^{\dagger}$, the construction reduces to the Hermitian algorithm.

In that case we set

:::{math}
:label: nh:herm_limit_assumption
\mathcal{G}=\mathcal{U}'^{\dagger},
:::

Equation {eq}`nh:G_rec` then becomes

:::{math}
:label: nh:herm_limit_unitarity
\mathcal{U}'^{\dagger}+\mathcal{U}'+\mathcal{U}'^{\dagger}\mathcal{U}'=0,
:::

This is exactly the Hermitian unitarity recursion from [the main algorithm page](algorithms.md).

We now decompose

:::{math}
:label: nh:WV_def
\mathcal{U}'=\mathcal{W}+\mathcal{V},
\qquad
\mathcal{W}^{\dagger}=\mathcal{W},
\qquad
\mathcal{V}^{\dagger}=-\mathcal{V},
:::

Equation {eq}`nh:herm_limit_unitarity` then gives

:::{math}
:label: nh:herm_limit_W
\mathcal{W}=-\frac{1}{2}\mathcal{U}'^{\dagger}\mathcal{U}',
:::

The gauge condition becomes

:::{math}
:label: nh:herm_limit_V
(\mathcal{U}'-\mathcal{G})_S=0
\quad\Longleftrightarrow\quad
\mathcal{V}_S=0.
:::

So, in the Hermitian limit, the non-Hermitian construction gives the same gauge choice and recurrence for the selected part.
::::

## Optimized transformed Hamiltonian

As in [the main algorithm](algorithms.md), the implementation avoids unnecessary products by $H_0$.
Here we skip the intermediate steps from the Hermitian derivation and derive the optimized form directly.

We define

:::{math}
:label: nh:XAB_defs
\mathcal{X}\equiv[\mathcal{H}_S,\mathcal{U}'],
\qquad
\mathcal{A}\equiv\mathcal{H}'_R\mathcal{U}',
\qquad
\mathcal{B}\equiv\mathcal{X}+\mathcal{H}'_R+\mathcal{A}.
:::

Starting from $\tilde{\mathcal{H}}=(1+\mathcal{G})(\mathcal{H}_S+\mathcal{H}'_R)(1+\mathcal{U}')$, we substitute $\mathcal{H}_S\mathcal{U}'=\mathcal{U}'\mathcal{H}_S+\mathcal{X}$ and use Eq. {eq}`nh:G_rec` to cancel the terms multiplied by $\mathcal{H}_S$.
This gives

:::{math}
:label: nh:Htilde_B
\tilde{\mathcal{H}}=\mathcal{H}_S+\mathcal{B}+\mathcal{G}\mathcal{B}.
:::

Once $\mathcal{X}$, $\mathcal{A}$, and $\mathcal{B}$ are known, the effective Hamiltonian can be assembled without extra products by $H_0$.

## Elimination condition and Sylvester solve

The condition $\tilde{\mathcal{H}}_R=0$ implies that

:::{math}
:label: nh:XR_rec
\mathcal{X}_R=-(\mathcal{H}'_R+\mathcal{A}+\mathcal{G}\mathcal{B})_R.
:::

The selected part of $\mathcal{X}$ follows directly from its definition.
Since $H_0$ is selected and diagonal in the unperturbed basis, $[H_0,\mathcal{U}']$ has no selected part, so

:::{math}
:label: nh:XS_def
\mathcal{X}_S=[\mathcal{H}'_S,\mathcal{U}']_S.
:::

For the remaining part, we split the commutator $\mathcal{X}=[\mathcal{H}_S,\mathcal{U}']=[H_0,\mathcal{U}']+[\mathcal{H}'_S,\mathcal{U}']$.
This gives the Sylvester equation

:::{math}
:label: nh:Sylvester_Uprime
[H_0,\mathcal{U}']_R
=\mathcal{X}_R-[\mathcal{H}'_S,\mathcal{U}']_R.
:::

So the nontrivial linear solve still appears only once per perturbative order.

## Implementation summary

At order $\mathbf{n}$, this part of the implementation is easiest to read in three steps:

1. Introduce the series that appear repeatedly.
2. Evaluate the recurrence from top to bottom using Cauchy products.
3. Use the result to obtain $\tilde{\mathcal{H}}_{\mathbf{n},S}$.

The first block defines the composite quantities.

:::{math}
:label: nh:closed_defs
\begin{aligned}
\mathcal{H} &\equiv \mathcal{H}_S + \mathcal{H}'_R, \qquad
\mathcal{H}_S \equiv H_0 + \mathcal{H}'_S, \\
\mathcal{U} &\equiv 1+\mathcal{U}', \\
\mathcal{U}^{-1} &\equiv 1+\mathcal{G}, \\
\mathcal{X} &\equiv [\mathcal{H}_S,\mathcal{U}'], \\
\mathcal{A} &\equiv \mathcal{H}'_R\mathcal{U}', \\
\mathcal{B} &\equiv \mathcal{X}+\mathcal{H}'_R+\mathcal{A}, \\
\tilde{\mathcal{H}}_S &\equiv \mathcal{H}_S + (\mathcal{B}+\mathcal{G}\mathcal{B})_S,
\qquad
\tilde{\mathcal{H}}_R \equiv 0.
\end{aligned}
:::

With this notation, the order-by-order recurrence is

:::{math}
:label: nh:closed_recs
\begin{aligned}
\mathcal{U}'_0 &= 0,\qquad \mathcal{G}_0 = 0,\qquad \mathcal{X}_0=0, \\
\mathcal{U}'_S &= -\frac{1}{2}(\mathcal{G}\mathcal{U}')_S, \\
\mathcal{G} &= -\mathcal{U}'-\mathcal{G}\mathcal{U}', \\
\mathcal{A} &= \mathcal{H}'_R\mathcal{U}', \\
\mathcal{X}_R &= -(\mathcal{H}'_R+\mathcal{A}+\mathcal{G}\mathcal{B})_R, \\
\mathcal{X}_S &= [\mathcal{H}'_S,\mathcal{U}']_S, \\
[H_0,\mathcal{U}']_R &= \mathcal{X}_R-[\mathcal{H}'_S,\mathcal{U}']_R.
\end{aligned}
:::

At each perturbative order, Eq. {eq}`nh:closed_recs` is closed in
$\{\mathcal{U}',\mathcal{G},\mathcal{A},\mathcal{B},\mathcal{X}\}$
and uses one Sylvester solve (last line); Eq. {eq}`nh:closed_defs` then yields
$\tilde{\mathcal{H}}_{\mathbf{n},S}$.

## Structure-Preserving Constraints (Trace and Hermiticity)

For Lindbladians (or general Liouville-space generators), we often want the
transformed generator to keep:

- trace preservation, and
- Hermiticity preservation.

This is not automatic for an arbitrary selective split; it follows from extra
constraints on $S/R$ and the Sylvester solver.

### Liouville-Space Conventions

Use row-major vectorization:

:::{math}
A\rho B \;\mapsto\; (A\otimes B^T)\,\mathrm{vec}(\rho).
:::

Let $\tau$ be the trace row functional:

:::{math}
\mathrm{tr}(\rho)=\tau\,\mathrm{vec}(\rho).
:::

Let $K$ be the permutation implementing $\mathrm{vec}(\rho^\dagger)=K\,\mathrm{vec}(\rho)^*$.
Define the involution

:::{math}
X^\sharp \equiv K X^* K.
:::

A superoperator preserves Hermiticity iff $X^\sharp=X$.

### Additional Assumptions

In addition to Eq. {eq}`nh:closed_recs`, assume:

1. $S$ and $R$ are complementary projectors:
   $S+R=I$, $S^2=S$, $R^2=R$, $SR=RS=0$.
2. Trace-compatible split:
   :::{math}
   \tau R(X)=0 \;\;\forall X
   \quad\Longleftrightarrow\quad
   \tau S(X)=\tau X.
   :::
3. Hermiticity-compatible split:
   :::{math}
   S(X^\sharp)=S(X)^\sharp,\qquad R(X^\sharp)=R(X)^\sharp.
   :::
4. Sylvester solver compatibility on $R$:
   if $Y=R(Y)$, then
   :::{math}
   S(\mathrm{Sylv}(Y))=0,\quad
   \tau\,\mathrm{Sylv}(Y)=0,\quad
   \mathrm{Sylv}(Y^\sharp)=\mathrm{Sylv}(Y)^\sharp.
   :::
5. Input coefficients are trace/Hermiticity preserving:
   :::{math}
   \tau \mathcal{H}_{\mathbf{n}}=0,\qquad
   \mathcal{H}_{\mathbf{n}}^\sharp=\mathcal{H}_{\mathbf{n}}
   \quad\text{for all orders }\mathbf{n}.
   :::

### Proposition 1: Trace Preservation

Under assumptions 1, 2, 4, and 5:

:::{math}
\tau\mathcal{U}=\tau,\qquad
\tau\mathcal{U}^{-1}=\tau,\qquad
\tau\tilde{\mathcal{H}}=0.
:::

**Proof (order by order).**

Base order: $\mathcal{U}'_0=\mathcal{G}_0=0$, hence
$\tau\mathcal{U}_0=\tau\mathcal{U}^{-1}_0=\tau$.

Induction step at order $\mathbf{n}>0$:

- From $\mathcal{G}=-\mathcal{U}'-\mathcal{G}\mathcal{U}'$,
  :::{math}
  \tau\mathcal{G}_{\mathbf{n}}
  =-\tau\mathcal{U}'_{\mathbf{n}}
   -\tau(\mathcal{G}\mathcal{U}')_{\mathbf{n}}.
  :::
  The product term uses only lower orders, so by induction it vanishes:
  $\tau(\mathcal{G}\mathcal{U}')_{\mathbf{n}}=0$.
  Therefore $\tau\mathcal{G}_{\mathbf{n}}=-\tau\mathcal{U}'_{\mathbf{n}}$.

- Gauge condition $(\mathcal{U}'-\mathcal{G})_S=0$ implies
  $\tau(\mathcal{U}'-\mathcal{G})_{\mathbf{n}}=0$ because $\tau R=0$.
  Hence $\tau\mathcal{U}'_{\mathbf{n}}=\tau\mathcal{G}_{\mathbf{n}}$.

Combining both equalities gives
$\tau\mathcal{U}'_{\mathbf{n}}=\tau\mathcal{G}_{\mathbf{n}}=0$.
Thus $\tau\mathcal{U}=\tau$ and $\tau\mathcal{U}^{-1}=\tau$ at all orders.

Finally,

:::{math}
\tau\tilde{\mathcal{H}}
=\tau\,\mathcal{U}^{-1}\mathcal{H}\mathcal{U}
=\tau\,\mathcal{H}\mathcal{U}
=0,
:::

since $\tau\mathcal{H}=0$ order by order. $\square$

### Proposition 2: Hermiticity Preservation

Under assumptions 1, 3, 4, and 5:

:::{math}
\mathcal{U}^\sharp=\mathcal{U},\qquad
(\mathcal{U}^{-1})^\sharp=\mathcal{U}^{-1},\qquad
\tilde{\mathcal{H}}^\sharp=\tilde{\mathcal{H}}.
:::

**Proof (order by order).**

Base order: $\mathcal{U}'_0=\mathcal{G}_0=0$, so trivially sharp-invariant.

Induction step:

- If lower orders are sharp-invariant, then products and commutators built from
  them are also sharp-invariant because
  $(AB)^\sharp=A^\sharp B^\sharp$ and
  $[A,B]^\sharp=[A^\sharp,B^\sharp]$.
- Since $\mathcal{H}_{\mathbf{n}}^\sharp=\mathcal{H}_{\mathbf{n}}$ and $S,R$
  commute with $\sharp$, all projected terms
  ($\mathcal{H}'_S,\mathcal{H}'_R,\mathcal{A},\mathcal{B},\mathcal{X}_{S/R}$)
  at order $\mathbf{n}$ are sharp-invariant.
- The Sylvester right-hand side for $\mathcal{U}'_R$ is therefore
  sharp-invariant; by solver equivariance,
  $\mathcal{U}'_{R,\mathbf{n}}$ is sharp-invariant.
- $\mathcal{G}_{\mathbf{n}}$ from
  $\mathcal{G}=-\mathcal{U}'-\mathcal{G}\mathcal{U}'$
  is then sharp-invariant, and the gauge relation fixes
  $\mathcal{U}'_{S,\mathbf{n}}$ accordingly.

Hence $\mathcal{U}'_{\mathbf{n}}$ and $\mathcal{G}_{\mathbf{n}}$ are
sharp-invariant for all $\mathbf{n}$.
Therefore $\mathcal{U}$ and $\mathcal{U}^{-1}$ are sharp-invariant.

Now

:::{math}
\tilde{\mathcal{H}}^\sharp
=(\mathcal{U}^{-1}\mathcal{H}\mathcal{U})^\sharp
=(\mathcal{U}^{-1})^\sharp \mathcal{H}^\sharp \mathcal{U}^\sharp
=\mathcal{U}^{-1}\mathcal{H}\mathcal{U}
=\tilde{\mathcal{H}}.
:::

So $\tilde{\mathcal{H}}$ preserves Hermiticity. $\square$

### Practical Mask Rules (One-Block Selective Case)

For a boolean elimination mask in a single Liouville block, assumptions 2 and 3
translate to:

- do not eliminate entries in trace-support rows ($\tau$-support),
- close the mask under the dagger index map $(i,j)\leftrightarrow(j,i)$
  (in Liouville indexing).

These are exactly the structural constraints used in the non-Hermitian tests.

:::{note}
These constraints preserve trace and Hermiticity, not complete positivity.
Outside perturbative convergence, positivity can still fail.
:::

## Implicit mode

The [Hermitian implicit construction](algorithms.md) assumes that the explicit
subspace is described by one orthonormal basis $\Psi_E$. The missing block is
then represented by the orthogonal complement

:::{math}
:label: nh:implicit_herm_projector
Q = 1 - \Psi_E \Psi_E^\dagger.
:::

For a genuinely non-Hermitian $H_0$, we instead use biorthogonal right and left
bases:

:::{math}
:label: nh:implicit_biorth_basis
R_E,\;L_E,
\qquad
L_E^\dagger R_E = 1,
:::

The columns of $R_E$ span the explicit subspace, and the columns of $L_E$ span
its dual. The projector onto this subspace and its complement are

:::{math}
:label: nh:implicit_oblique_projector
P_E = R_E L_E^\dagger,
\qquad
Q = 1 - R_E L_E^\dagger.
:::

In general this projector is oblique rather than orthogonal, so it is not
self-adjoint. The block projections become

:::{math}
:label: nh:implicit_block_projections
H_{ij} = L_i^\dagger H R_j,
\qquad
H_{iQ} = L_i^\dagger H Q,
\qquad
H_{Qi} = Q H R_i,
\qquad
H_{QQ} = Q H Q.
:::

The Sylvester equations keep the same structure as in the Hermitian implicit
derivation, but they use this oblique $Q$ and the explicit energies

:::{math}
:label: nh:implicit_biorth_energies
L_i^\dagger H_0 R_i.
:::

So the direct implicit solver needs the same two ingredients:

- right subspace bases to define the retained states,
- left dual bases to define the projection and the complementary block.
