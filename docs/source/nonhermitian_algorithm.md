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

This page summarizes a non-Hermitian generalization of perturbative block-diagonalization, where
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

To fix the gauge we impose

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

Because $\mathcal{P}'_0=\mathcal{M}_0=0$, the right-hand side at order $\mathbf{n}$ uses only lower orders.

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

### Hermitian Limit (Consistency Check)

Introduce auxiliary series in the same style as the Hermitian derivation:

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

If $\mathcal{H}=\mathcal{H}^{\dagger}$ and we impose unitarity,

:::{math}
:label: nh:herm_limit_assumption
\mathcal{U}^{-1}=\mathcal{U}^{\dagger}
\quad\Longrightarrow\quad
\mathcal{G}=\mathcal{U}'^{\dagger}.
:::

Substituting into Eq. {eq}`nh:UG_inverse_rec`:

:::{math}
:label: nh:herm_limit_unitarity
\mathcal{U}'^{\dagger}+\mathcal{U}'+\mathcal{U}'^{\dagger}\mathcal{U}'=0,
:::

which is the Hermitian unitarity recursion. Defining
$\mathcal{W}=(\mathcal{U}'+\mathcal{U}'^{\dagger})/2$ and
$\mathcal{V}=(\mathcal{U}'-\mathcal{U}'^{\dagger})/2$, we recover

:::{math}
:label: nh:herm_limit_W
\mathcal{W}=-\frac{1}{2}\mathcal{U}'^{\dagger}\mathcal{U}'.
:::

Also $\mathcal{M}=\mathcal{U}'-\mathcal{G}=2\mathcal{V}$, so $\mathcal{M}_S=0$ is exactly $\mathcal{V}_S=0$.
Thus the non-Hermitian setup reduces to the Hermitian one.

To close consistency, the Sylvester/Lyapunov step must also return an anti-Hermitian
generator in this limit. Write

:::{math}
:label: nh:herm_lyap
[H_0,\mathcal{M}]_R=\mathcal{R}_R,
\qquad
\mathcal{M}=2\mathcal{V},
:::

where Eq. {eq}`nh:M_rec` defines $\mathcal{R}_R$.
Using induction over perturbation order:

1. Assume lower-order terms satisfy
   $\mathcal{P}'^{\dagger}=\mathcal{P}'$ and
   $\mathcal{M}^{\dagger}=-\mathcal{M}$.
2. Then the right-hand side of Eq. {eq}`nh:M_rec` at the new order is Hermitian,
   so $\mathcal{R}_R^{\dagger}=\mathcal{R}_R$.
3. In the eigenbasis of $H_0$ (for $i,j$ in different eigensubspaces),

:::{math}
\mathcal{M}_{ij}=\frac{(\mathcal{R}_R)_{ij}}{E_i-E_j},
\qquad
\mathcal{M}_{ji}=\frac{(\mathcal{R}_R)_{ji}}{E_j-E_i}
=-\mathcal{M}_{ij}^{*},
:::

so $\mathcal{M}^{\dagger}=-\mathcal{M}$ and therefore
$\mathcal{V}^{\dagger}=-\mathcal{V}$.

Hence the Lyapunov/Sylvester solve is consistent with the anti-Hermitian
generator required by the Hermitian algorithm.

## Toward an Optimized Algorithm

To avoid multiplications by $H_0$ in reusable Cauchy products, use

:::{math}
:label: nh:Htilde_B
\tilde{\mathcal{H}}=\mathcal{H}_S+\mathcal{B}+\mathcal{G}\mathcal{B},
:::

with auxiliaries

:::{math}
:label: nh:XAB_defs
\mathcal{X}\equiv[\mathcal{H}_S,\mathcal{U}'],
\qquad
\mathcal{A}\equiv\mathcal{H}'_R\mathcal{U}',
\qquad
\mathcal{B}\equiv\mathcal{X}+\mathcal{H}'_R+\mathcal{A}.
:::

This follows from
$\mathcal{U}=1+\mathcal{U}'$, $\mathcal{U}^{-1}=1+\mathcal{G}$ and

a) $\mathcal{U}^{-1}\mathcal{H}_S\mathcal{U}=\mathcal{H}_S+\mathcal{U}^{-1}[\mathcal{H}_S,\mathcal{U}']$,

b) $\mathcal{U}^{-1}\mathcal{H}'_R\mathcal{U}=\mathcal{H}'_R+\mathcal{H}'_R\mathcal{U}'+\mathcal{G}(\mathcal{X}+\mathcal{H}'_R+\mathcal{A})$.

From $\tilde{\mathcal{H}}_R=0$:

:::{math}
:label: nh:XR_rec
\mathcal{B}_R=-(\mathcal{G}\mathcal{B})_R
\quad\Longleftrightarrow\quad
\mathcal{X}_R=-(\mathcal{H}'_R+\mathcal{A}+\mathcal{G}\mathcal{B})_R.
:::

The selected part is

:::{math}
:label: nh:XS_def
\mathcal{X}_S=[\mathcal{H}'_S,\mathcal{U}']_S.
:::

The Sylvester step is

:::{math}
:label: nh:Sylvester_Uprime
[H_0,\mathcal{U}']_R
=\mathcal{X}_R-[\mathcal{H}'_S,\mathcal{U}']_R.
:::

Inverse recursion and gauge:

:::{math}
:label: nh:G_and_gauge
\mathcal{G}=-\mathcal{U}'-\mathcal{G}\mathcal{U}',
\qquad
(\mathcal{U}'-\mathcal{G})_S=0.
:::

### Order-by-order Evaluation

At order $\mathbf{n}$:

1. Use Eq. {eq}`nh:G_and_gauge` to obtain $\mathcal{U}'_{\mathbf{n},S}$.
2. Compute $\mathcal{A}_{\mathbf{n}}=(\mathcal{H}'_R\mathcal{U}')_{\mathbf{n}}$.
3. Compute $\mathcal{X}_{\mathbf{n},R}$ from Eq. {eq}`nh:XR_rec`.
4. Compute $\mathcal{X}_{\mathbf{n},S}$ from Eq. {eq}`nh:XS_def`.
5. Solve Eq. {eq}`nh:Sylvester_Uprime` for $\mathcal{U}'_{\mathbf{n},R}$.
6. Compute $\mathcal{G}_{\mathbf{n}}$ from Eq. {eq}`nh:G_and_gauge` and then $\mathcal{B}_{\mathbf{n}}$ from Eq. {eq}`nh:XAB_defs`.
7. Evaluate $\tilde{\mathcal{H}}_{\mathbf{n},S}$ from Eq. {eq}`nh:Htilde_B`.

This has the same key structural property as the optimized Hermitian algorithm: $H_0$ appears only in the Sylvester solve and not in Cauchy-product building blocks.

## Closed Recursive System

### Definitions

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

### Recursive Relations

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

## Structure-Preserving Constraints (Trace and Hermiticity)

For Lindbladians (or general Liouville-space generators), we often want the
transformed generator to keep:

- trace preservation, and
- Hermiticity preservation.

This is not automatic for an arbitrary selective split; it follows from extra
constraints on $S/R$ and the Sylvester solver.

### Liouville-Space Conventions

Use column-major vectorization:

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
