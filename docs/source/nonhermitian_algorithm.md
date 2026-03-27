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
reuses the same basic setup:

- split the Hamiltonian into selected and remaining parts,
- organize perturbation theory through multivariate Cauchy products,
- avoid reusable products by $H_0$,
- reduce each perturbative order to one Sylvester solve.

We therefore do not repeat the general motivation and notation from
[the main algorithm page](algorithms.md).
This page only explains what changes once the Hermitian relation
$\mathcal{U}^{\dagger} = \mathcal{U}^{-1}$ is no longer available.

In the API, this is the path selected by

```python
H_tilde, U, U_inv = block_diagonalize(..., hermitian=False)
```

where the third returned series is the perturbative inverse of $U$.

:::{admonition} What is different from the Hermitian algorithm?
:class: note

The Hermitian algorithm can work with a single correction series because the
inverse transformation is obtained for free as the adjoint.
In the non-Hermitian case, the inverse must be propagated explicitly.
The main task of this page is therefore to show how to keep the recursion
closed while still preserving the same computational structure as in
[the main algorithm](algorithms.md).
:::

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
\mathcal{H}=H_0+\mathcal{H}'_S+\mathcal{H}'_R,
:::

where $S$ denotes the selected part, $R$ the remainder to eliminate, and $H_0$
is entirely selected.
The selected/remaining split is exactly the one introduced on
[the main algorithm page](algorithms.md).

## A parameterization that closes the recursion

A direct treatment of $\mathcal{U}$ alone is inconvenient because
$\mathcal{U}^{-1}$ is an independent series.
To recover a closed recurrence, we introduce sum and difference combinations of
$\mathcal{U}$ and $\mathcal{U}^{-1}$:

:::{math}
:label: nh:PM_def
\mathcal{P}\equiv \mathcal{U}+\mathcal{U}^{-1},
\qquad
\mathcal{M}\equiv \mathcal{U}-\mathcal{U}^{-1}.
:::

Then

:::{math}
:label: nh:U_from_PM
\mathcal{U}=\frac{\mathcal{P}+\mathcal{M}}{2},
\qquad
\mathcal{U}^{-1}=\frac{\mathcal{P}-\mathcal{M}}{2}.
:::

Since $\mathcal{U}_0=\mathcal{U}^{-1}_0=1$, it is convenient to write

:::{math}
:label: nh:Pprime_def
\mathcal{P}=2+\mathcal{P}',
\qquad
\mathcal{M}_0=\mathcal{P}'_0=0.
:::

To fix the gauge, we impose

:::{math}
:label: nh:min_diff
\mathcal{M}_S=0
:::

at each perturbative order.
This is the non-Hermitian counterpart of the minimal-generator gauge used in
the Hermitian derivation.

### Inverse constraint

The condition $\mathcal{U}^{-1}\mathcal{U}=1$ becomes

:::{math}
:label: nh:inv_constraint_1
(\mathcal{P}-\mathcal{M})(\mathcal{P}+\mathcal{M})=4,
:::

that is,

:::{math}
:label: nh:inv_constraint_2
\mathcal{P}^2-\mathcal{M}^2+[\mathcal{P},\mathcal{M}]=4.
:::

With $\mathcal{P}=2+\mathcal{P}'$, this gives

:::{math}
:label: nh:Pprime_rec
\mathcal{P}'=-\frac{1}{4}\Big(\mathcal{P}'^2-\mathcal{M}^2+[\mathcal{P}',\mathcal{M}]\Big).
:::

Because $\mathcal{P}'_0=\mathcal{M}_0=0$, the right-hand side at order
$\mathbf{n}$ depends only on lower orders.
So the inverse constraint already gives a closed recursion for $\mathcal{P}'$
once $\mathcal{M}$ is known.

### Elimination condition

Projecting the condition $\tilde{\mathcal{H}}_R=0$ onto the remaining sector and
isolating the terms linear in $\mathcal{M}$ gives

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

This is the non-Hermitian analogue of the Sylvester/Lyapunov step in the
Hermitian derivation.
Again, the right-hand side is recursive because every product contains at least
one primed series with vanishing zero order.

## Relation to the Hermitian algorithm

The construction above is designed so that it reduces back to the Hermitian
algorithm when $\mathcal{H}$ is Hermitian and
$\mathcal{U}^{-1}=\mathcal{U}^{\dagger}$.

Introduce

:::{math}
:label: nh:UG_from_PM
\mathcal{U}=1+\mathcal{U}',
\qquad
\mathcal{U}^{-1}=1+\mathcal{G},
\qquad
\mathcal{U}'=\frac{\mathcal{P}'+\mathcal{M}}{2},
\qquad
\mathcal{G}=\frac{\mathcal{P}'-\mathcal{M}}{2}.
:::

Then $\mathcal{U}^{-1}\mathcal{U}=1$ gives

:::{math}
:label: nh:UG_inverse_rec
\mathcal{G}+\mathcal{U}'+\mathcal{G}\mathcal{U}'=0.
:::

If $\mathcal{H}=\mathcal{H}^{\dagger}$ and the similarity transform is
unitary, then

:::{math}
:label: nh:herm_limit_assumption
\mathcal{U}^{-1}=\mathcal{U}^{\dagger}
\quad\Longrightarrow\quad
\mathcal{G}=\mathcal{U}'^{\dagger}.
:::

Substituting this into Eq. {eq}`nh:UG_inverse_rec` yields

:::{math}
:label: nh:herm_limit_unitarity
\mathcal{U}'^{\dagger}+\mathcal{U}'+\mathcal{U}'^{\dagger}\mathcal{U}'=0,
:::

which is exactly the Hermitian unitarity recursion.
Moreover,

:::{math}
:label: nh:herm_limit_W
\mathcal{W}=-\frac{1}{2}\mathcal{U}'^{\dagger}\mathcal{U}'
:::

recovers the Hermitian $W$ recursion from [the main algorithm](algorithms.md),
while $\mathcal{M}=\mathcal{U}'-\mathcal{G}=2\mathcal{V}$ turns the gauge
condition $\mathcal{M}_S=0$ into the familiar $\mathcal{V}_S=0$.

The Sylvester step also reduces correctly.
If the right-hand side of Eq. {eq}`nh:M_rec` is Hermitian, then in the
eigenbasis of $H_0$

:::{math}
:label: nh:herm_lyap
\mathcal{M}_{ij}=\frac{(\mathcal{R}_R)_{ij}}{E_i-E_j},
\qquad
\mathcal{M}_{ji}=
\frac{(\mathcal{R}_R)_{ji}}{E_j-E_i}
=-\mathcal{M}_{ij}^{*},
:::

so $\mathcal{M}$ is anti-Hermitian and $\mathcal{V}=\mathcal{M}/2$ is the same
anti-Hermitian generator as in the Hermitian path.

## Optimized form used in the implementation

The derivation above is useful conceptually, but the code is cleaner when
written in terms of the correction series $\mathcal{U}'$ and the inverse
correction $\mathcal{G}$.
As in [the main algorithm](algorithms.md), the goal is to arrange the formulas
so that $H_0$ appears only inside the Sylvester solve and not in reusable
Cauchy-product terms.

We therefore work with

:::{math}
\mathcal{U}=1+\mathcal{U}',
\qquad
\mathcal{U}^{-1}=1+\mathcal{G},
:::

for which Eq. {eq}`nh:UG_inverse_rec` becomes

:::{math}
:label: nh:G_and_gauge
\mathcal{G}=-\mathcal{U}'-\mathcal{G}\mathcal{U}',
\qquad
(\mathcal{U}'-\mathcal{G})_S=0.
:::

The second relation is simply the gauge condition $\mathcal{M}_S=0$ rewritten
in terms of $\mathcal{U}'$ and $\mathcal{G}$.

Next define three auxiliaries

:::{math}
:label: nh:XAB_defs
\mathcal{X}\equiv[\mathcal{H}_S,\mathcal{U}'],
\qquad
\mathcal{A}\equiv\mathcal{H}'_R\mathcal{U}',
\qquad
\mathcal{B}\equiv\mathcal{X}+\mathcal{H}'_R+\mathcal{A}.
:::

With these definitions the transformed Hamiltonian takes the compact form

:::{math}
:label: nh:Htilde_B
\tilde{\mathcal{H}}=\mathcal{H}_S+\mathcal{B}+\mathcal{G}\mathcal{B}.
:::

This is the non-Hermitian analogue of the optimized Hermitian formula:
once $\mathcal{X}$, $\mathcal{A}$, and $\mathcal{B}$ are known, the effective
Hamiltonian can be assembled without extra reusable products by $H_0$.

The remaining-part condition $\tilde{\mathcal{H}}_R=0$ then gives

:::{math}
:label: nh:XR_rec
\mathcal{X}_R=-(\mathcal{H}'_R+\mathcal{A}+\mathcal{G}\mathcal{B})_R,
:::

while the selected part of $\mathcal{X}$ is fixed directly by the commutator
definition

:::{math}
:label: nh:XS_def
\mathcal{X}_S=[\mathcal{H}'_S,\mathcal{U}']_S.
:::

Finally, the off-selected part of $\mathcal{U}'$ is obtained from a Sylvester
equation

:::{math}
:label: nh:Sylvester_Uprime
[H_0,\mathcal{U}']_R
=\mathcal{X}_R-[\mathcal{H}'_S,\mathcal{U}']_R.
:::

## Order-by-order evaluation

At perturbative order $\mathbf{n}$, the implementation proceeds as follows:

1. Use Eq. {eq}`nh:G_and_gauge` to obtain $\mathcal{U}'_{\mathbf{n},S}$.
2. Compute $\mathcal{A}_{\mathbf{n}}=(\mathcal{H}'_R\mathcal{U}')_{\mathbf{n}}$.
3. Compute $\mathcal{X}_{\mathbf{n},R}$ from Eq. {eq}`nh:XR_rec`.
4. Compute $\mathcal{X}_{\mathbf{n},S}$ from Eq. {eq}`nh:XS_def`.
5. Solve Eq. {eq}`nh:Sylvester_Uprime` for $\mathcal{U}'_{\mathbf{n},R}$.
6. Compute $\mathcal{G}_{\mathbf{n}}$ from Eq. {eq}`nh:G_and_gauge` and then
   $\mathcal{B}_{\mathbf{n}}$ from Eq. {eq}`nh:XAB_defs`.
7. Evaluate $\tilde{\mathcal{H}}_{\mathbf{n},S}$ from Eq. {eq}`nh:Htilde_B`.

All right-hand sides are closed in lower orders except the single Sylvester
solve in step 5.
So, just as in the Hermitian algorithm, each order consists of one linear solve
plus a fixed amount of Cauchy-product work.

## Compact reference

Collecting the implementation formulas in one place:

:::{math}
:label: nh:closed_defs
\begin{aligned}
\mathcal{H} &\equiv H_0 + \mathcal{H}'_S + \mathcal{H}'_R, \\
\mathcal{M} &\equiv \mathcal{U}'-\mathcal{G}, \\
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
\mathcal{U}'_0 &= 0,\qquad \mathcal{G}_0 = 0,\qquad \mathcal{X}_0=0,\qquad \mathcal{M}_S=0, \\
\mathcal{G} &= -\mathcal{U}'-\mathcal{G}\mathcal{U}', \\
(\mathcal{U}'-\mathcal{G})_S &= 0, \\
\mathcal{A} &= \mathcal{H}'_R\mathcal{U}', \\
\mathcal{X}_R &= -(\mathcal{H}'_R+\mathcal{A}+\mathcal{G}\mathcal{B})_R,\qquad
\mathcal{X}_S = [\mathcal{H}'_S,\mathcal{U}']_S, \\
[H_0,\mathcal{U}']_R &= \mathcal{X}_R-[\mathcal{H}'_S,\mathcal{U}']_R.
\end{aligned}
:::

At each perturbative order, Eq. {eq}`nh:closed_recs` is closed in
$\{\mathcal{U}',\mathcal{G},\mathcal{A},\mathcal{B},\mathcal{X}\}$
and uses one Sylvester solve (last line); Eq. {eq}`nh:closed_defs` then yields
$\tilde{\mathcal{H}}_{\mathbf{n},S}$.
