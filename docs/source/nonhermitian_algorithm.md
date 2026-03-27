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

This page describes the variant used by
`block_diagonalize(..., hermitian=False)`.
It is the non-Hermitian analogue of the [main algorithm](algorithms.md), and it
reuses the same general setup:

- split the Hamiltonian into selected and remaining parts,
- organize perturbation theory through multivariate Cauchy products,
- avoid unnecessary products by $H_0$,
- reduce each perturbative order to one Sylvester solve.

It uses the same notation and general motivation as
[the main algorithm page](algorithms.md).
The discussion below introduces the additional ingredients needed when the
inverse transformation has to be tracked explicitly.

In the API, this is the path selected by

```python
H_tilde, U, U_inv = block_diagonalize(..., hermitian=False)
```

where the third returned series is the perturbative inverse of $U$.

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

Here $S$ denotes the selected part and $R$ the remainder to eliminate, exactly
as in [the main algorithm](algorithms.md).

## Working variables

Since both $\mathcal{U}$ and $\mathcal{U}^{-1}$ have identity zeroth order, we
write

:::{math}
:label: nh:UG_def
\mathcal{U}=1+\mathcal{U}',
\qquad
\mathcal{U}^{-1}=1+\mathcal{G},
\qquad
\mathcal{U}'_0=\mathcal{G}_0=0.
:::

The inverse constraint becomes

:::{math}
:label: nh:G_rec
\mathcal{G}=-\mathcal{U}'-\mathcal{G}\mathcal{U}'.
:::

Because both $\mathcal{U}'$ and $\mathcal{G}$ start at first order, this is a
closed recurrence for $\mathcal{G}$ once $\mathcal{U}'$ is known.

To fix the gauge, we require the selected part of
$\mathcal{U}-\mathcal{U}^{-1}$ to vanish:

:::{math}
:label: nh:gauge
(\mathcal{U}'-\mathcal{G})_S=0.
:::

Combining Eqs. {eq}`nh:G_rec` and {eq}`nh:gauge` gives the selected part of the
correction directly:

:::{math}
:label: nh:Uprime_S
\mathcal{U}'_S=-\frac{1}{2}(\mathcal{G}\mathcal{U}')_S.
:::

This is the non-Hermitian counterpart of the Hermitian recurrence for the
selected/Hermitian part of the transformation.

## Relation to the Hermitian algorithm

The construction above is designed so that it reduces back to the Hermitian
algorithm when $\mathcal{H}$ is Hermitian and
$\mathcal{U}^{-1}=\mathcal{U}^{\dagger}$.

If we set

:::{math}
:label: nh:herm_limit_assumption
\mathcal{G}=\mathcal{U}'^{\dagger},
:::

then Eq. {eq}`nh:G_rec` becomes

:::{math}
:label: nh:herm_limit_unitarity
\mathcal{U}'^{\dagger}+\mathcal{U}'+\mathcal{U}'^{\dagger}\mathcal{U}'=0,
:::

which is exactly the Hermitian unitarity recursion from
[the main algorithm page](algorithms.md).

If we further decompose

:::{math}
:label: nh:WV_def
\mathcal{U}'=\mathcal{W}+\mathcal{V},
\qquad
\mathcal{W}^{\dagger}=\mathcal{W},
\qquad
\mathcal{V}^{\dagger}=-\mathcal{V},
:::

then Eq. {eq}`nh:herm_limit_unitarity` gives

:::{math}
:label: nh:herm_limit_W
\mathcal{W}=-\frac{1}{2}\mathcal{U}'^{\dagger}\mathcal{U}',
:::

while the gauge condition becomes

:::{math}
:label: nh:herm_limit_V
(\mathcal{U}'-\mathcal{G})_S=0
\quad\Longleftrightarrow\quad
\mathcal{V}_S=0.
:::

So the non-Hermitian construction reduces to the same gauge choice and the same
selected-part recursion as in the Hermitian path.

## Optimized transformed Hamiltonian

The code is organized to avoid unnecessary products by $H_0$.
As in [the main algorithm](algorithms.md), the useful object is not the raw
expansion of $\tilde{\mathcal{H}}$, but a rearranged version in which $H_0$
appears only inside the Sylvester solve.

Define

:::{math}
:label: nh:XAB_defs
\mathcal{X}\equiv[\mathcal{H}_S,\mathcal{U}'],
\qquad
\mathcal{A}\equiv\mathcal{H}'_R\mathcal{U}',
\qquad
\mathcal{B}\equiv\mathcal{X}+\mathcal{H}'_R+\mathcal{A}.
:::

Starting from
$\tilde{\mathcal{H}}=(1+\mathcal{G})(\mathcal{H}_S+\mathcal{H}'_R)(1+\mathcal{U}')$,
use
$\mathcal{H}_S\mathcal{U}'=\mathcal{U}'\mathcal{H}_S+\mathcal{X}$ and
Eq. {eq}`nh:G_rec` to cancel the terms multiplied by $\mathcal{H}_S$.
This gives the compact expression

:::{math}
:label: nh:Htilde_B
\tilde{\mathcal{H}}=\mathcal{H}_S+\mathcal{B}+\mathcal{G}\mathcal{B}.
:::

This is the non-Hermitian analogue of the optimized Hermitian formula:
once $\mathcal{X}$, $\mathcal{A}$, and $\mathcal{B}$ are known, the effective
Hamiltonian can be assembled without extra products by $H_0$.

## Elimination condition and Sylvester solve

The remaining-part condition $\tilde{\mathcal{H}}_R=0$ implies

:::{math}
:label: nh:XR_rec
\mathcal{X}_R=-(\mathcal{H}'_R+\mathcal{A}+\mathcal{G}\mathcal{B})_R.
:::

The selected part of $\mathcal{X}$ is fixed directly by its definition.
Because $H_0$ is selected and diagonal in the unperturbed basis,
$[H_0,\mathcal{U}']$ has no selected part, so

:::{math}
:label: nh:XS_def
\mathcal{X}_S=[\mathcal{H}'_S,\mathcal{U}']_S.
:::

For the remaining part, split the commutator
$\mathcal{X}=[\mathcal{H}_S,\mathcal{U}']=
[H_0,\mathcal{U}']+[\mathcal{H}'_S,\mathcal{U}']$.
This yields the Sylvester equation

:::{math}
:label: nh:Sylvester_Uprime
[H_0,\mathcal{U}']_R
=\mathcal{X}_R-[\mathcal{H}'_S,\mathcal{U}']_R.
:::

So, just as in the Hermitian algorithm, the nontrivial linear solve appears
only once per perturbative order.

## Implementation summary

The implementation forms a closed tail-recursive system.
At order $\mathbf{n}$, the equations below determine
$\{\mathcal{U}',\mathcal{G},\mathcal{A},\mathcal{B},\mathcal{X}\}_{\mathbf{n}}$
from lower orders, except for the single Sylvester solve for
$\mathcal{U}'_{\mathbf{n},R}$.
Once that solve is done, the remaining quantities at the same order are fixed by
the same relations.

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
