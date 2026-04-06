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

This page describes the non-Hermitian extension of Pymablock's [main algorithm](algorithms.md). Enable it with

```python
H_tilde, U, U_inv = block_diagonalize(..., hermitian=False)
```

The third returned series is the perturbative inverse of $U$. In the non-Hermitian case, this inverse is not the adjoint of the forward transformation, so it must be computed explicitly.

The overall structure is the same as in the Hermitian algorithm. We split the Hamiltonian into selected and remaining parts, organize perturbation theory through Cauchy products, avoid unnecessary products by $H_0$, and reduce each perturbative order to one Sylvester solve.

Throughout, we use the notation from [the main algorithm page](algorithms.md).


## Setup

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

Here $S$ denotes the selected part and $R$ the remainder to eliminate, as in [the main algorithm](algorithms.md).
Since $\mathcal{U}^{-1}\neq \mathcal{U}^{\dagger}$ in general, the left and right eigenvectors need not coincide.

## Variables and gauge

We write the forward and inverse transformations as identity plus first-order corrections:

:::{math}
:label: nh:UG_def
\mathcal{U}=1+\mathcal{U}',
\qquad
\mathcal{U}^{-1}=1+\mathcal{G},
\qquad
\mathcal{U}'_0=\mathcal{G}_0=0.
:::

As in the Hermitian algorithm, it is convenient to separate the terms that enter $\mathcal{U}$ and $\mathcal{U}^{-1}$ with the same sign from those that enter with opposite signs:

:::{math}
:label: nh:UG_from_WV
\mathcal{U}=1+\mathcal{W}+\mathcal{V},
\qquad
\mathcal{U}^{-1}=1+\mathcal{W}-\mathcal{V}.
:::

In the Hermitian case, this split gives the Hermitian and anti-Hermitian parts of $\mathcal{U}'$. In the non-Hermitian setting, $\mathcal{W}$ and $\mathcal{V}$ are only auxiliary series and need not have those symmetries.

The inverse constraint $\mathcal{U}^{-1}\mathcal{U}=1$ then gives

:::{math}
:label: nh:W_rec
\mathcal{W}=-\frac{1}{2}\mathcal{G}\mathcal{U}'.
:::

Because both series start at first order, Eq. {eq}`nh:W_rec` determines $\mathcal{W}$ from lower orders.

The least-action principle, which in the Hermitian case fixes the non-uniqueness of $\mathcal{U}$ by minimizing $\|\mathcal{U}-1\|$, is not available in the non-Hermitian setting because there is no canonical norm. We therefore fix the gauge by the pragmatic choice:

:::{math}
:label: nh:Vgauge
\mathcal{V}_S=0.
:::

This choice is simple, sparse, and close to the Hermitian construction.

::::{admonition} Equivalence to the Hermitian algorithm
:class: dropdown info
If $\mathcal{H}$ is Hermitian and $\mathcal{U}^{-1}=\mathcal{U}^{\dagger}$, then

:::{math}
:label: nh:herm_limit_assumption
\mathcal{G}=\mathcal{U}'^{\dagger},
:::

Substituting $\mathcal{G}=\mathcal{U}'^{\dagger}$ into Eq. {eq}`nh:UG_from_WV` gives the Hermitian and anti-Hermitian parts of $\mathcal{U}'$:

:::{math}
:label: nh:WV_def
\mathcal{W}=\frac{\mathcal{U}'^{\dagger}+\mathcal{U}'}{2},
\qquad
\mathcal{V}=\frac{\mathcal{U}'-\mathcal{U}'^{\dagger}}{2},
\qquad
\mathcal{W}^{\dagger}=\mathcal{W},
\qquad
\mathcal{V}^{\dagger}=-\mathcal{V},
:::

Equation {eq}`nh:W_rec` becomes

:::{math}
:label: nh:herm_limit_W
\mathcal{W}=-\frac{1}{2}\mathcal{U}'^{\dagger}\mathcal{U}',
:::

The gauge condition becomes

:::{math}
:label: nh:herm_limit_V
\mathcal{V}_S=0.
:::

This is the Hermitian parameterization, together with the same gauge choice and selected-part recurrence.
::::

## Derivation

As in [the main algorithm](algorithms.md), we derive the non-Hermitian recurrence in a form that avoids unnecessary products by $H_0$.

We introduce the shorthand

:::{math}
:label: nh:XAB_defs
\mathcal{X}\equiv[\mathcal{U}',\mathcal{H}_S],
\qquad
\mathcal{A}\equiv\mathcal{H}'_R\mathcal{U}',
\qquad
\mathcal{B}\equiv\mathcal{X}-\mathcal{H}'_R-\mathcal{A}.
:::

Starting from $\tilde{\mathcal{H}}=(1+\mathcal{G})(\mathcal{H}_S+\mathcal{H}'_R)(1+\mathcal{U}')$, we substitute $\mathcal{H}_S\mathcal{U}'=\mathcal{U}'\mathcal{H}_S-\mathcal{X}$ and use $\mathcal{U}'+\mathcal{G}=-\mathcal{G}\mathcal{U}'$ to cancel the terms multiplied by $\mathcal{H}_S$:

:::{math}
:label: nh:Htilde_B
\tilde{\mathcal{H}}=\mathcal{H}_S-\mathcal{B}-\mathcal{G}\mathcal{B}.
:::

This form lets us assemble the effective Hamiltonian from $\mathcal{X}$, $\mathcal{A}$, and $\mathcal{B}$ without extra products by $H_0$.

The elimination condition $\tilde{\mathcal{H}}_R=0$ implies

:::{math}
:label: nh:BR_rec
\mathcal{B}_R=-(\mathcal{G}\mathcal{B})_R.
:::

We split $\mathcal{X}$ into the contributions from $\mathcal{V}$ and $\mathcal{W}$:

:::{math}
:label: nh:YZ_def
\mathcal{X}\equiv[\mathcal{U}',\mathcal{H}_S]=\mathcal{Y}+\mathcal{Z},
\qquad
\mathcal{Y}\equiv[\mathcal{V},\mathcal{H}_S],
\qquad
\mathcal{Z}\equiv[\mathcal{W},\mathcal{H}_S].
:::

Equation {eq}`nh:W_rec` rewrites the $\mathcal{W}$ commutator in terms of Cauchy products. We also define

:::{math}
:label: nh:Bplus_def
\mathcal{B}_+ \equiv \mathcal{B}+\mathcal{G}\mathcal{B}.
:::

This yields

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

Since $\mathcal{B}=\mathcal{X}-\mathcal{H}'_R-\mathcal{A}$, the remaining part $\mathcal{Y}=[\mathcal{V},\mathcal{H}_S]$ is

:::{math}
:label: nh:Y_rec
\mathcal{Y}=\mathcal{B}+\mathcal{H}'_R+\mathcal{A}-\mathcal{Z}.
:::

The gauge condition {eq}`nh:Vgauge` implies $[\mathcal{V},H_0]_S=0$, so the selected part only involves $\mathcal{H}'_S$:

:::{math}
:label: nh:Y_S
\mathcal{Y}_S=[\mathcal{V},\mathcal{H}'_S]_S.
:::

Using $\mathcal{B}=\mathcal{Y}+\mathcal{Z}-\mathcal{H}'_R-\mathcal{A}$, we obtain

:::{math}
:label: nh:BS_rec
\mathcal{B}_S=\left([\mathcal{V},\mathcal{H}'_S] + \mathcal{Z} - \mathcal{A}\right)_S.
:::

For the remaining part, we solve the Sylvester equation

:::{math}
:label: nh:Sylvester_V
[\mathcal{V},H_0]_R
= \mathcal{Y}_R-[\mathcal{V},\mathcal{H}'_S]_R
= \left(\mathcal{B}+\mathcal{H}'_R+\mathcal{A}-\mathcal{Z}-[\mathcal{V},\mathcal{H}'_S]\right)_R.
:::

Equation {eq}`nh:Sylvester_V` is the only Sylvester solve. Every earlier step is a Cauchy product between series that start at first order, so the algorithm never multiplies by $H_0$ outside that solve.

## Closed recurrence

At order $\mathbf{n}$, the implementation reduces to the following closed recurrence.

:::{math}
:label: nh:closed_defs
\begin{aligned}
\mathcal{H} &\equiv \mathcal{H}_S + \mathcal{H}'_R, \qquad
\mathcal{H}_S \equiv H_0 + \mathcal{H}'_S, \\
\mathcal{U} &\equiv 1+\mathcal{U}' \equiv 1+\mathcal{W}+\mathcal{V}, \\
\mathcal{U}^{-1} &\equiv 1+\mathcal{G} \equiv 1+\mathcal{W}-\mathcal{V}, \\
\mathcal{W} &\equiv -\frac{1}{2}\mathcal{G}\mathcal{U}', \\
\mathcal{A} &\equiv \mathcal{H}'_R\mathcal{U}', \\
\mathcal{Z} &\equiv [\mathcal{W},\mathcal{H}_S], \\
\mathcal{Y} &\equiv \mathcal{B}+\mathcal{H}'_R+\mathcal{A}-\mathcal{Z}, \\
\mathcal{B} &\equiv \mathcal{U}\mathcal{H}_S - \mathcal{H}\mathcal{U}, \\
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
\mathcal{Y} &= \mathcal{B}+\mathcal{H}'_R+\mathcal{A}-\mathcal{Z}, \\
\mathcal{B}_S &= \left([\mathcal{V},\mathcal{H}'_S] + \mathcal{Z} - \mathcal{A}\right)_S, \\
[\mathcal{V},H_0]_R &= \left(\mathcal{Y} - [\mathcal{V},\mathcal{H}'_S]\right)_R.
\end{aligned}
:::

The last line is the only Sylvester solve. All earlier lines are Cauchy products between series that start at first order, so order $\mathbf{n}$ depends only on lower orders. Eq. {eq}`nh:closed_defs` then gives $\tilde{\mathcal{H}}_{\mathbf{n},S}$.

## Implicit mode

In the [Hermitian implicit construction](algorithms.md), one orthonormal basis $\Psi_E$ describes the explicit subspace. The complementary block is then represented by the orthogonal projector

:::{math}
:label: nh:implicit_herm_projector
Q = 1 - \Psi_E \Psi_E^\dagger.
:::

For non-Hermitian $H_0$, we instead use biorthogonal right and left bases:

:::{math}
:label: nh:implicit_biorth_basis
R_E,\;L_E,
\qquad
L_E^\dagger R_E = 1,
:::

The columns of $R_E$ span the explicit subspace, and the columns of $L_E$ span its dual. The corresponding projector and its complement are

:::{math}
:label: nh:implicit_oblique_projector
P_E = R_E L_E^\dagger,
\qquad
Q = 1 - R_E L_E^\dagger.
:::

This projector is generally oblique rather than orthogonal, so it is not self-adjoint. The block projections are

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

The Sylvester equations keep the same structure as in the Hermitian implicit derivation, but they use this oblique projector $Q$ and the explicit energies

:::{math}
:label: nh:implicit_biorth_energies
L_i^\dagger H_0 R_i.
:::

The direct implicit solver therefore needs the same two ingredients: right subspace bases for the retained states, and left dual bases for the projection and complementary block.
