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

# Lindbladian perturbation theory

This page collects the Liouvillian-specific material that sits on top of
Pymablock's non-Hermitian block-diagonalization algorithm. It assumes the
notation and recursion from
[Non-Hermitian Algorithm](nonhermitian_algorithm.md).

The goal is to answer two questions:

1. When does selective elimination preserve trace and Hermiticity?
2. How do we recover Hamiltonian and jump-operator series from an effective
   Liouvillian?

For a worked example, see
[Adiabatic elimination for a Lindblad equation](tutorial/lindblad_adiabatic_elimination.md).

## Preserving trace and Hermiticity

For open quantum systems, the object being block-diagonalized is often a
Liouvillian superoperator

:::{math}
\mathcal{L}(\rho)= -i[H,\rho]
+ \sum_k \Big(J_k\rho J_k^\dagger
- \tfrac12\{J_k^\dagger J_k,\rho\}\Big),
:::

or a perturbative generalization of it. After the similarity transformation,

:::{math}
\tilde{\mathcal{L}}=\mathcal{U}^{-1}\mathcal{L}\mathcal{U},
:::

we usually want the reduced generator to keep two structural properties:

- trace preservation,
- Hermiticity preservation.

This does not follow from an arbitrary selective split. It requires extra
compatibility conditions on the selected and eliminated parts, and on the
Sylvester solver used in the non-Hermitian recursion.

### Liouville-space conventions

We use row-major vectorization, meaning that we stack the rows of $\rho$ into
$\mathrm{vec}(\rho)$. In that convention,

:::{math}
A\rho B \;\mapsto\; (A\otimes B^T)\,\mathrm{vec}(\rho).
:::

This is the same convention used in the
[Lindblad adiabatic elimination tutorial](tutorial/lindblad_adiabatic_elimination.md).

Let $\tau$ denote the trace row functional:

:::{math}
\mathrm{tr}(\rho)=\tau\,\mathrm{vec}(\rho).
:::

Let $K$ be the permutation matrix that implements Hermitian conjugation in
vectorized form,

:::{math}
\mathrm{vec}(\rho^\dagger)=K\,\mathrm{vec}(\rho)^*.
:::

We then define

:::{math}
X^\sharp \equiv K X^* K.
:::

A superoperator preserves Hermiticity exactly when $X^\sharp=X$.

### Assumptions

In addition to the non-Hermitian recursion, assume the following.

1. The selected and eliminated parts are complementary projectors:
   $S+R=I$, $S^2=S$, $R^2=R$, and $SR=RS=0$.
2. The split respects trace:
   :::{math}
   \tau R(X)=0 \;\;\forall X
   \quad\Longleftrightarrow\quad
   \tau S(X)=\tau X.
   :::
3. The split respects Hermiticity:
   :::{math}
   S(X^\sharp)=S(X)^\sharp,\qquad R(X^\sharp)=R(X)^\sharp.
   :::
4. The Sylvester solver respects the same structure on the eliminated part:
   whenever $Y=R(Y)$,
   :::{math}
   S(\mathrm{Sylv}(Y))=0,\quad
   \tau\,\mathrm{Sylv}(Y)=0,\quad
   \mathrm{Sylv}(Y^\sharp)=\mathrm{Sylv}(Y)^\sharp.
   :::
5. The input perturbative coefficients already preserve trace and Hermiticity:
   :::{math}
   \tau \mathcal{H}_{\mathbf{n}}=0,\qquad
   \mathcal{H}_{\mathbf{n}}^\sharp=\mathcal{H}_{\mathbf{n}}
   \quad\text{for all orders }\mathbf{n}.
   :::

### Proposition 1: trace preservation

Assume 1, 2, 4, and 5. Then

:::{math}
\tau\mathcal{U}=\tau,\qquad
\tau\mathcal{U}^{-1}=\tau,\qquad
\tau\tilde{\mathcal{L}}=0.
:::

**Proof.** At zeroth order, $\mathcal{U}'_0=\mathcal{G}_0=0$, so
$\tau\mathcal{U}_0=\tau\mathcal{U}^{-1}_0=\tau$.

Now fix a higher order $\mathbf{n}$. From
$\mathcal{G}=-\mathcal{U}'-\mathcal{G}\mathcal{U}'$ we get

:::{math}
\tau\mathcal{G}_{\mathbf{n}}
=-\tau\mathcal{U}'_{\mathbf{n}}
-\tau(\mathcal{G}\mathcal{U}')_{\mathbf{n}}.
:::

The product term depends only on lower orders, so the induction hypothesis
gives $\tau(\mathcal{G}\mathcal{U}')_{\mathbf{n}}=0$. Hence
$\tau\mathcal{G}_{\mathbf{n}}=-\tau\mathcal{U}'_{\mathbf{n}}$.

The gauge condition $(\mathcal{U}'-\mathcal{G})_S=0$ and the trace-compatible
split imply

:::{math}
\tau(\mathcal{U}'-\mathcal{G})_{\mathbf{n}}=0,
:::

so $\tau\mathcal{U}'_{\mathbf{n}}=\tau\mathcal{G}_{\mathbf{n}}$. Combining the
two identities gives

:::{math}
\tau\mathcal{U}'_{\mathbf{n}}=\tau\mathcal{G}_{\mathbf{n}}=0.
:::

Therefore $\tau\mathcal{U}=\tau$ and $\tau\mathcal{U}^{-1}=\tau$ order by
order. Finally,

:::{math}
\tau\tilde{\mathcal{L}}
=\tau\,\mathcal{U}^{-1}\mathcal{L}\mathcal{U}
=\tau\,\mathcal{L}\mathcal{U}
=0,
:::

because $\tau\mathcal{L}=0$ at every perturbative order. $\square$

### Proposition 2: Hermiticity preservation

Assume 1, 3, 4, and 5. Then

:::{math}
\mathcal{U}^\sharp=\mathcal{U},\qquad
(\mathcal{U}^{-1})^\sharp=\mathcal{U}^{-1},\qquad
\tilde{\mathcal{L}}^\sharp=\tilde{\mathcal{L}}.
:::

**Proof.** At zeroth order the claim is immediate because
$\mathcal{U}'_0=\mathcal{G}_0=0$.

At order $\mathbf{n}$, every right-hand side in the non-Hermitian recursion is
built from lower-order terms by products, commutators, and the projectors
$S,R$. If the lower-order terms satisfy the sharp symmetry, then so do those
products and commutators, and assumptions 3 and 4 preserve that symmetry under
projection and Sylvester solving. Therefore $\mathcal{U}'_{\mathbf{n}}$ and
$\mathcal{G}_{\mathbf{n}}$ also satisfy the sharp symmetry.

By induction, $\mathcal{U}^\sharp=\mathcal{U}$ and
$(\mathcal{U}^{-1})^\sharp=\mathcal{U}^{-1}$ at every order. It then follows
that

:::{math}
\tilde{\mathcal{L}}^\sharp
=(\mathcal{U}^{-1}\mathcal{L}\mathcal{U})^\sharp
=(\mathcal{U}^{-1})^\sharp \mathcal{L}^\sharp \mathcal{U}^\sharp
=\mathcal{U}^{-1}\mathcal{L}\mathcal{U}
=\tilde{\mathcal{L}}.
:::

So the transformed generator still preserves Hermiticity. $\square$

### Practical mask rules

For a boolean elimination mask inside one Liouville block, the abstract
conditions above become concrete:

- do not eliminate entries in rows that carry trace support,
- close the mask under the dagger-index map $(i,j)\leftrightarrow(j,i)$ in
  Liouville indexing.

These are the structural conditions used in the non-Hermitian tests.

:::{note}
These conditions preserve trace and Hermiticity, not complete positivity.
Outside the perturbative regime, positivity can still fail.
:::

## Recovering Hamiltonian and jump operators

Suppose the transformed Liouvillian has the perturbative expansion

:::{math}
\tilde{\mathcal{L}}(\lambda)=\sum_{n\ge 0}\lambda^n\tilde{\mathcal{L}}_n.
:::

We want to rewrite each coefficient in Lindblad form, order by order.

### Step 1: recover the Hamiltonian part and Kossakowski matrix

Choose an operator basis $\{F_a\}_{a=1}^{d^2-1}$ on the traceless subspace,
orthonormal in the Hilbert-Schmidt inner product. Define the corresponding
superoperator basis

:::{math}
\mathcal{K}_a(\rho)\equiv -i[F_a,\rho],\qquad
\mathcal{D}_{ab}(\rho)\equiv F_a\rho F_b^\dagger
-\frac12\{F_b^\dagger F_a,\rho\}.
:::

Then each perturbative coefficient can be written as

:::{math}
\tilde{\mathcal{L}}_n
=\sum_a h_{a,n}\mathcal{K}_a
+\sum_{a,b}(C_n)_{ab}\mathcal{D}_{ab}.
:::

This is a linear decomposition in the unknown coefficients
$(h_{a,n},(C_n)_{ab})$. After vectorization, collect the basis operators into
the matrix

:::{math}
A\equiv
\big[\mathrm{vec}(\mathcal{K}_1)\;\cdots\;\mathrm{vec}(\mathcal{K}_{d^2-1})\;
\mathrm{vec}(\mathcal{D}_{11})\;\cdots\;\mathrm{vec}(\mathcal{D}_{d^2-1,d^2-1})\big].
:::

For each order, solve

:::{math}
x_n \equiv
\begin{bmatrix}
h_n\\
\mathrm{vec}(C_n)
\end{bmatrix}
=A^+\,\mathrm{vec}(\tilde{\mathcal{L}}_n),
:::

where $A^+$ is a pseudoinverse, or an inverse if the basis was chosen so that
the map is square. This gives

:::{math}
H_n=\sum_a h_{a,n}F_a.
:::

So the map $\tilde{\mathcal{L}}_n \mapsto (H_n,C_n)$ is linear and local in
perturbation order.

If $\tilde{\mathcal{L}}_n$ preserves trace and Hermiticity, then $H_n$ is
Hermitian and $C_n$ is Hermitian. Positivity is a separate question: $C_n$ need
not be positive semidefinite order by order.

### Step 2: recover jump-operator series

To stay on the completely positive branch near $\lambda=0$, factorize

:::{math}
C(\lambda)=B(\lambda)B(\lambda)^\dagger.
:::

Expand both sides as

:::{math}
B(\lambda)=\sum_{n\ge0}\lambda^n B_n,
\qquad
C(\lambda)=\sum_{n\ge0}\lambda^n C_n.
:::

At order $\mathbf{n}$ this gives

:::{math}
C_n
=B_0 B_n^\dagger + B_n B_0^\dagger
+\sum_{k=1}^{n-1} B_k B_{n-k}^\dagger.
:::

The sum on the right is already known from lower orders, so once a gauge is
fixed, this equation is linear in $B_n$. A triangular gauge or an orthogonality
condition on $B_0^\dagger B_n$ both work.

Now write the columns of $B$ as

:::{math}
B(\lambda)=\big[b_1(\lambda)\;\cdots\;b_r(\lambda)\big],\qquad
b_\mu(\lambda)=\sum_{n\ge0}\lambda^n b_{\mu,n},
:::

and define the jump operators by

:::{math}
J_\mu(\lambda)\equiv \sum_a b_{a\mu}(\lambda)\,F_a
=\sum_{n\ge0}\lambda^n J_{\mu,n}.
:::

Then the transformed generator takes the form

:::{math}
\tilde{\mathcal{L}}(\rho)= -i[H,\rho]
+\sum_{\mu=1}^r \Big(
J_\mu\rho J_\mu^\dagger
-\frac12\{J_\mu^\dagger J_\mu,\rho\}
\Big),
:::

and each coefficient $J_{\mu,n}$ is obtained linearly from $b_{\mu,n}$.

:::{note}
This construction stays on a positive branch, meaning
$C(\lambda)\succeq 0$ wherever the series converges. If the target Liouvillian
is not completely positive, that branch must fail at some finite $\lambda$,
typically when an eigenvalue of $C$ crosses zero. A more general signed
factorization uses $C=B\Sigma B^\dagger$ with
$\Sigma=\mathrm{diag}(\pm1,0)$.
:::
