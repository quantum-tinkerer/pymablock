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

Like in the Hermitian case, we separate the transformation into identity at zeroth order and a correction, which allows us to define recurrence relations through Cauchy products.
We first introduce $\mathcal{U}'$ as the correction of the transformation $\mathcal{U}$ and $\mathcal{G}$ as the correction of its inverse $\mathcal{U}^{-1}$:

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

To harmonize the notation with the Hermitian algorithm, we now define

:::{math}
:label: nh:WV_general
\mathcal{V}\equiv\frac{\mathcal{U}'-\mathcal{G}}{2},
\qquad
\mathcal{W}\equiv\frac{\mathcal{U}'+\mathcal{G}}{2}.
:::

Using Eq. {eq}`nh:G_rec`, the second variable can also be written as

:::{math}
:label: nh:W_rec
\mathcal{W}=-\frac{1}{2}\mathcal{G}\mathcal{U}'.
:::

So the forward and inverse corrections are decomposed exactly as in the Hermitian page:

:::{math}
:label: nh:UG_from_WV
\mathcal{U}'=\mathcal{W}+\mathcal{V},
\qquad
\mathcal{G}=\mathcal{W}-\mathcal{V}.
:::

Unlike in the Hermitian case, $\mathcal{W}$ need not be Hermitian and
$\mathcal{V}$ need not be anti-Hermitian.
What survives is the gauge condition:

:::{math}
:label: nh:Vgauge
\mathcal{V}_S=0.
:::

Equations {eq}`nh:W_rec` and {eq}`nh:Vgauge` are the non-Hermitian analogues of the
Hermitian recurrences for $\mathcal{W}$ and $\mathcal{V}$.

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

Equations {eq}`nh:WV_general` and {eq}`nh:UG_from_WV` reduce to the Hermitian split

:::{math}
:label: nh:WV_def
\mathcal{U}'=\mathcal{W}+\mathcal{V},
\qquad
\mathcal{W}^{\dagger}=\mathcal{W},
\qquad
\mathcal{V}^{\dagger}=-\mathcal{V},
:::

and Eq. {eq}`nh:W_rec` becomes

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

So, in the Hermitian limit, the non-Hermitian construction gives the same variables,
gauge choice, and recurrence for the selected part as the Hermitian algorithm.
::::

## Optimized transformed Hamiltonian

As in [the main algorithm](algorithms.md), the implementation avoids unnecessary products by $H_0$.
Here we skip the intermediate steps from the Hermitian derivation and derive the optimized form directly.

We define

:::{math}
:label: nh:XAB_defs
\mathcal{X}\equiv[\mathcal{U}',\mathcal{H}_S],
\qquad
\mathcal{A}\equiv\mathcal{H}'_R\mathcal{U}',
\qquad
\mathcal{B}\equiv\mathcal{X}-\mathcal{H}'_R-\mathcal{A}.
:::

Starting from $\tilde{\mathcal{H}}=(1+\mathcal{G})(\mathcal{H}_S+\mathcal{H}'_R)(1+\mathcal{U}')$, we substitute $\mathcal{H}_S\mathcal{U}'=\mathcal{U}'\mathcal{H}_S-\mathcal{X}$ and use Eq. {eq}`nh:G_rec` to cancel the terms multiplied by $\mathcal{H}_S$.
This gives

:::{math}
:label: nh:Htilde_B
\tilde{\mathcal{H}}=\mathcal{H}_S-\mathcal{B}-\mathcal{G}\mathcal{B}.
:::

Once $\mathcal{X}$, $\mathcal{A}$, and $\mathcal{B}$ are known, the effective Hamiltonian can be assembled without extra products by $H_0$.

## Elimination condition and Sylvester solve

The condition $\tilde{\mathcal{H}}_R=0$ implies that

:::{math}
:label: nh:BR_rec
\mathcal{B}_R=-(\mathcal{G}\mathcal{B})_R.
:::

To harmonize the derivation with the Hermitian algorithm, we split

:::{math}
:label: nh:YZ_def
\mathcal{X}\equiv[\mathcal{U}',\mathcal{H}_S]=\mathcal{Y}+\mathcal{Z},
\qquad
\mathcal{Y}\equiv[\mathcal{V},\mathcal{H}_S],
\qquad
\mathcal{Z}\equiv[\mathcal{W},\mathcal{H}_S].
:::

Using Eq. {eq}`nh:W_rec`, the $\mathcal{W}$ commutator can be written entirely in
terms of reusable Cauchy products. It is convenient to define

:::{math}
:label: nh:Bplus_def
\mathcal{B}_+ \equiv \mathcal{B}+\mathcal{G}\mathcal{B}.
:::

Then

:::{math}
:label: nh:Z_rec
\mathcal{Z}
= \frac{1}{2}\left(
- \mathcal{G}\mathcal{H}'_R
+ \mathcal{A}
- \mathcal{G}\mathcal{B}
- \mathcal{B}_+\mathcal{G}
\right).
:::

Since $\mathcal{B}=\mathcal{X}-\mathcal{H}'_R-\mathcal{A}$, the remaining part
$\mathcal{Y}=[\mathcal{V},\mathcal{H}_S]$ is

:::{math}
:label: nh:Y_rec
\mathcal{Y}=\mathcal{B}+\mathcal{H}'_R+\mathcal{A}-\mathcal{Z}.
:::

The gauge condition {eq}`nh:Vgauge` implies $[\mathcal{V},H_0]_S=0$.
Therefore the selected part is fixed exactly as in the Hermitian algorithm:

:::{math}
:label: nh:Y_S
\mathcal{Y}_S=[\mathcal{V},\mathcal{H}'_S]_S.
:::

Equivalently, since $\mathcal{B}=\mathcal{Y}+\mathcal{Z}-\mathcal{H}'_R-\mathcal{A}$,

:::{math}
:label: nh:BS_rec
\mathcal{B}_S=\left([\mathcal{V},\mathcal{H}'_S] + \mathcal{Z} - \mathcal{A}\right)_S.
:::

For the remaining part we use

:::{math}
:label: nh:Sylvester_V
[\mathcal{V},H_0]_R
= \mathcal{Y}_R-[\mathcal{V},\mathcal{H}'_S]_R
= \left(\mathcal{B}+\mathcal{H}'_R+\mathcal{A}-\mathcal{Z}-[\mathcal{V},\mathcal{H}'_S]\right)_R.
:::

Equation {eq}`nh:Sylvester_V` is the only Sylvester solve.
All other terms are Cauchy products between series that start at first order, so the
algorithm still avoids extra products by $H_0$ outside the Sylvester step itself.

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
\mathcal{V} &\equiv \frac{\mathcal{U}'-\mathcal{G}}{2}, \\
\mathcal{W} &\equiv \frac{\mathcal{U}'+\mathcal{G}}{2}
= -\frac{1}{2}\mathcal{G}\mathcal{U}', \\
\mathcal{X} &\equiv [\mathcal{U}',\mathcal{H}_S]
= \mathcal{Y} + \mathcal{Z}, \\
\mathcal{A} &\equiv \mathcal{H}'_R\mathcal{U}', \\
\mathcal{Y} &\equiv [\mathcal{V},\mathcal{H}_S], \\
\mathcal{Z} &\equiv [\mathcal{W},\mathcal{H}_S], \\
\mathcal{B} &\equiv \mathcal{U}\mathcal{H}_S - \mathcal{H}\mathcal{U}
= \mathcal{X} - \mathcal{H}'_R - \mathcal{A}, \\
\mathcal{B}_+ &\equiv \mathcal{B}+\mathcal{G}\mathcal{B}, \\
\tilde{\mathcal{H}}_S &\equiv \mathcal{H}_S - \mathcal{B}_+,
\qquad
\tilde{\mathcal{H}}_R \equiv 0.
\end{aligned}
:::

With this notation, the order-by-order recurrence is

:::{math}
:label: nh:closed_recs
\begin{aligned}
\mathcal{U}'_0 &= 0,\qquad \mathcal{G}_0 = 0,\qquad \mathcal{V}_0 = 0,\qquad
\mathcal{B}_0 = 0, \\
\mathcal{W} &= -\frac{1}{2}\mathcal{G}\mathcal{U}', \\
\mathcal{U}' &= \mathcal{W}+\mathcal{V}, \\
\mathcal{G} &= \mathcal{W}-\mathcal{V}, \\
\mathcal{A} &= \mathcal{H}'_R\mathcal{U}', \\
\mathcal{B}_R &= -(\mathcal{G}\mathcal{B})_R, \\
\mathcal{B}_+ &= \mathcal{B}+\mathcal{G}\mathcal{B}, \\
\mathcal{Z} &=
\frac{1}{2}
\left(
+ \mathcal{A}
- \mathcal{G}\mathcal{H}'_R
- \mathcal{G}\mathcal{B}
- \mathcal{B}_+\mathcal{G}
\right), \\
\mathcal{X} &= \mathcal{B}+\mathcal{H}'_R+\mathcal{A}, \\
\mathcal{Y} &= \mathcal{X}-\mathcal{Z}, \\
\mathcal{B}_S &= \left([\mathcal{V},\mathcal{H}'_S] + \mathcal{Z} - \mathcal{A}\right)_S, \\
[\mathcal{V},H_0]_R &= \left(\mathcal{Y} - [\mathcal{V},\mathcal{H}'_S]\right)_R.
\end{aligned}
:::

The last line is the only Sylvester solve.
At each perturbative order, Eq. {eq}`nh:closed_recs` is closed in
$\{\mathcal{U}',\mathcal{G},\mathcal{V},\mathcal{W},\mathcal{X},\mathcal{A},\mathcal{Y},\mathcal{Z},\mathcal{B},\mathcal{B}_+\}$
and determines these quantities from lower orders.
Equation {eq}`nh:closed_defs` then yields $\tilde{\mathcal{H}}_{\mathbf{n},S}$.

## Implicit mode

The [Hermitian implicit construction](algorithms.md) assumes that the explicit subspace is described by one orthonormal basis $\Psi_E$.
The missing block is then represented by the orthogonal complement

:::{math}
:label: nh:implicit_herm_projector
Q = 1 - \Psi_E \Psi_E^\dagger.
:::

For a genuinely non-Hermitian $H_0$, we instead use biorthogonal right and left bases:

:::{math}
:label: nh:implicit_biorth_basis
R_E,\;L_E,
\qquad
L_E^\dagger R_E = 1,
:::

The columns of $R_E$ span the explicit subspace, and the columns of $L_E$ span its dual.
The projector onto this subspace and its complement are

:::{math}
:label: nh:implicit_oblique_projector
P_E = R_E L_E^\dagger,
\qquad
Q = 1 - R_E L_E^\dagger.
:::

In general this projector is oblique rather than orthogonal, so it is not self-adjoint.
The block projections become

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

The Sylvester equations keep the same structure as in the Hermitian implicit derivation, but they use this oblique $Q$ and the explicit energies

:::{math}
:label: nh:implicit_biorth_energies
L_i^\dagger H_0 R_i.
:::

So the direct implicit solver needs the same two ingredients:

- right subspace bases to define the retained states,
- left dual bases to define the projection and the complementary block.
