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

# Non-Hermitian Algorithm

This page summarizes a non-Hermitian generalization of the perturbative block-diagonalization strategy, where
$\mathcal{U}^{\dagger}$ is replaced by $\mathcal{U}^{-1}$.

## Setup

We consider the perturbative similarity transform

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
\mathcal{H}=H_0+\mathcal{H}'_S+\mathcal{H}'_R,
:::

where $S$ and $R$ denote selected and remaining parts, and $H_0$ is purely selected.

## Sum-Difference Parameterization

Define

:::{math}
:label: nh:PM_def
\mathcal{P}\equiv \mathcal{U}+\mathcal{U}^{-1},
\qquad
\mathcal{M}\equiv \mathcal{U}-\mathcal{U}^{-1},
:::

so that

:::{math}
:label: nh:U_from_PM
\mathcal{U}=\frac{\mathcal{P}+\mathcal{M}}{2},
\qquad
\mathcal{U}^{-1}=\frac{\mathcal{P}-\mathcal{M}}{2}.
:::

Since $\mathcal{U}_0=\mathcal{U}^{-1}_0=1$, write

:::{math}
:label: nh:Pprime_def
\mathcal{P}=2+\mathcal{P}',
\qquad
\mathcal{M}_0=\mathcal{P}'_0=0.
:::

To fix the gauge we impose the minimal-difference condition

:::{math}
:label: nh:min_diff
\mathcal{M}_S=0
:::

(order by order).

### Constraint From $\mathcal{U}^{-1}\mathcal{U}=1$

Using

:::{math}
:label: nh:inv_constraint_1
(\mathcal{P}-\mathcal{M})(\mathcal{P}+\mathcal{M})=4,
:::

we get

:::{math}
:label: nh:inv_constraint_2
\mathcal{P}^2-\mathcal{M}^2+[\mathcal{P},\mathcal{M}]=4.
:::

With $\mathcal{P}=2+\mathcal{P}'$,

:::{math}
:label: nh:Pprime_rec
\mathcal{P}'=-\frac{1}{4}\Big(\mathcal{P}'^2-\mathcal{M}^2+[\mathcal{P}',\mathcal{M}]\Big).
:::

Because $\mathcal{P}'_0=\mathcal{M}_0=0$, the right-hand side at order $\mathbf{n}$ only uses lower orders, so Eq. {eq}`nh:Pprime_rec` is recursive.

### Constraint From $\tilde{\mathcal{H}}_R=0$

Expand

:::{math}
:label: nh:Htilde_expanded
\tilde{\mathcal{H}}=
\frac{1}{4}(\mathcal{P}-\mathcal{M})\mathcal{H}(\mathcal{P}+\mathcal{M})
=\mathcal{H}
+\frac{1}{2}\{\mathcal{P}',\mathcal{H}\}
+\frac{1}{2}[\mathcal{H},\mathcal{M}]
+\frac{1}{4}(\mathcal{P}'-\mathcal{M})\mathcal{H}(\mathcal{P}'+\mathcal{M}).
:::

Hence

:::{math}
:label: nh:Htilde_R_zero
0=\tilde{\mathcal{H}}_R=
\Big(
\mathcal{H}'_R
+\frac{1}{2}\{\mathcal{P}',\mathcal{H}\}
+\frac{1}{2}[\mathcal{H},\mathcal{M}]
+\frac{1}{4}(\mathcal{P}'-\mathcal{M})\mathcal{H}(\mathcal{P}'+\mathcal{M})
\Big)_R.
:::

Isolating the linear Sylvester part gives

:::{math}
:label: nh:M_rec
[H_0,\mathcal{M}]_R
=-\Big(
2\mathcal{H}'_R
+\{\mathcal{P}',\mathcal{H}\}
+[\mathcal{H}'_S+\mathcal{H}'_R,\mathcal{M}]
+\frac{1}{2}(\mathcal{P}'-\mathcal{M})\mathcal{H}(\mathcal{P}'+\mathcal{M})
\Big)_R.
:::

This is recursive because every product on the right contains at least one primed series with zero order absent.

## Toward an Optimized Algorithm

To avoid multiplications by $H_0$ in reusable Cauchy products, rewrite the transform as

:::{math}
:label: nh:Htilde_X
\tilde{\mathcal{H}}
=\mathcal{U}^{-1}\mathcal{H}_S\mathcal{U}
+\mathcal{U}^{-1}\mathcal{H}'_R\mathcal{U}
=\mathcal{H}_S+\mathcal{U}^{-1}\mathcal{X}
+\mathcal{U}^{-1}\mathcal{H}'_R\mathcal{U},
:::

where

:::{math}
:label: nh:X_def
\mathcal{X}\equiv [\mathcal{H}_S,\mathcal{U}].
:::

Indeed,
$\mathcal{U}^{-1}\mathcal{H}_S\mathcal{U}
=\mathcal{U}^{-1}(\mathcal{U}\mathcal{H}_S+[\mathcal{H}_S,\mathcal{U}])
=\mathcal{H}_S+\mathcal{U}^{-1}\mathcal{X}$.

Define

:::{math}
:label: nh:SD_def
\mathcal{U}=1+\frac{\mathcal{S}}{2},
\qquad
\mathcal{U}^{-1}=1+\frac{\mathcal{D}}{2},
:::

so that the inverse condition gives

:::{math}
:label: nh:D_rec
\mathcal{D}=-\mathcal{S}-\frac{1}{2}\mathcal{D}\mathcal{S}.
:::

The minimal-difference gauge $\mathcal{M}_S=0$, with $\mathcal{M}=(\mathcal{S}-\mathcal{D})/2$, implies

:::{math}
:label: nh:Ss_gauge
\mathcal{S}_S=-\frac{1}{4}(\mathcal{D}\mathcal{S})_S.
:::

Now introduce products that do not contain $H_0$:

:::{math}
:label: nh:AYB_defs
\mathcal{A}\equiv \mathcal{H}'_R+\frac{1}{2}\mathcal{H}'_R\mathcal{S},
\qquad
\mathcal{Y}\equiv \mathcal{X}+\mathcal{A},
\qquad
\mathcal{B}\equiv \mathcal{D}\mathcal{Y}.
:::

Using Eq. {eq}`nh:Htilde_X`,

:::{math}
:label: nh:Htilde_opt
\tilde{\mathcal{H}}=\mathcal{H}_S+\mathcal{Y}+\frac{1}{2}\mathcal{B},
:::

and therefore

:::{math}
:label: nh:XR_rec
\tilde{\mathcal{H}}_R=0
\quad\Longleftrightarrow\quad
\mathcal{X}_R=-\left(\mathcal{A}+\frac{1}{2}\mathcal{B}\right)_R.
:::

To recover $\mathcal{S}_R$ with a single Sylvester solve per order, use

:::{math}
:label: nh:Sylvester_S
[H_0,\mathcal{S}]_R
=2\mathcal{X}_R-[\mathcal{H}'_S,\mathcal{S}]_R.
:::

Also

:::{math}
:label: nh:XS_def
\mathcal{X}_S
=\frac{1}{2}[\mathcal{H}'_S,\mathcal{S}]_S.
:::

### Order-by-order Evaluation

At order $\mathbf{n}$:

1. Compute $\mathcal{S}_{\mathbf{n},S}$ from Eq. {eq}`nh:Ss_gauge`.
2. Compute $\mathcal{D}_{\mathbf{n}}$ from Eq. {eq}`nh:D_rec`.
3. Compute $\mathcal{A}_{\mathbf{n}}$ and $\mathcal{B}_{\mathbf{n}}$ from Eq. {eq}`nh:AYB_defs`.
4. Compute $\mathcal{X}_{\mathbf{n},R}$ from Eq. {eq}`nh:XR_rec`, and $\mathcal{X}_{\mathbf{n},S}$ from Eq. {eq}`nh:XS_def`.
5. Solve Eq. {eq}`nh:Sylvester_S` for $\mathcal{S}_{\mathbf{n},R}$.
6. Evaluate $\tilde{\mathcal{H}}_{\mathbf{n},S}$ from Eq. {eq}`nh:Htilde_opt`.

This has the same key structural property as the optimized Hermitian algorithm: $H_0$ appears only in the Sylvester solve and not in Cauchy-product building blocks.

## Closed Recursive System

### Definitions

:::{math}
:label: nh:closed_defs
\begin{aligned}
\mathcal{H} &\equiv H_0 + \mathcal{H}'_S + \mathcal{H}'_R, \\
\mathcal{M} &\equiv \frac{\mathcal{S}-\mathcal{D}}{2}, \\
\mathcal{X} &\equiv \mathcal{X}_S+\mathcal{X}_R, \qquad
\mathcal{Y}\equiv \mathcal{X}+\mathcal{A}, \\
\tilde{\mathcal{H}}_S &\equiv
\mathcal{H}_S + \left(\mathcal{Y}+\frac{1}{2}\mathcal{B}\right)_S,
\qquad
\tilde{\mathcal{H}}_R \equiv 0.
\end{aligned}
:::

### Recursive Relations

:::{math}
:label: nh:closed_recs
\begin{aligned}
\mathcal{S}_0 &= 0,\qquad \mathcal{D}_0 = 0,\qquad \mathcal{X}_0=0,\qquad \mathcal{M}_S=0, \\
\mathcal{D} &= -\mathcal{S}-\frac{1}{2}\mathcal{D}\mathcal{S}, \\
\mathcal{S}_S &= -\frac{1}{4}(\mathcal{D}\mathcal{S})_S, \\
\mathcal{A} &= \mathcal{H}'_R+\frac{1}{2}\mathcal{H}'_R\mathcal{S},\qquad
\mathcal{B}=\mathcal{D}(\mathcal{X}+\mathcal{A}), \\
\mathcal{X}_R &= -\left(\mathcal{A}+\frac{1}{2}\mathcal{B}\right)_R,\qquad
\mathcal{X}_S = \frac{1}{2}[\mathcal{H}'_S,\mathcal{S}]_S, \\
[H_0,\mathcal{S}]_R &= 2\mathcal{X}_R-[\mathcal{H}'_S,\mathcal{S}]_R.
\end{aligned}
:::

At each perturbative order, the recursive block Eq. {eq}`nh:closed_recs` forms a closed system for
$\mathcal{S}_{\mathbf{n}}$, $\mathcal{D}_{\mathbf{n}}$, $\mathcal{A}_{\mathbf{n}}$, $\mathcal{B}_{\mathbf{n}}$, and $\mathcal{X}_{\mathbf{n}}$,
with one Sylvester solve in the last line; Eq. {eq}`nh:closed_defs` then yields
$\tilde{\mathcal{H}}_{\mathbf{n},S}$.
