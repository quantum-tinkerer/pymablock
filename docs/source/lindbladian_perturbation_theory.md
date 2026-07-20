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

A density matrix $\rho$ is a positive semidefinite Hermitian operator with
unit trace. Its Markovian evolution obeys a master equation

:::{math}
\frac{d\rho}{dt}=\mathcal L(\rho),
:::

where the linear map $\mathcal L$ is called a Liouvillian. A
Gorini--Kossakowski--Sudarshan--Lindblad (GKSL) generator has the form

:::{math}
:label: lindblad:gksl
\mathcal L(\rho)
=-i[H,\rho]
+\sum_k\left(
J_k\rho J_k^\dagger
-\frac12\{J_k^\dagger J_k,\rho\}
\right).
:::

The Hermitian operator $H$ generates coherent evolution. Each jump operator
$J_k$ describes one dissipative channel. The first term in the parentheses
adds the state produced by a jump, while the anticommutator subtracts the
corresponding probability from the state before the jump. These contributions
cancel under the trace. Equation {eq}`lindblad:gksl` therefore preserves the
trace and Hermiticity of $\rho$, and its finite-time evolution
$e^{t\mathcal L}$ is completely positive.

Pymablock does not derive Eq. {eq}`lindblad:gksl` from a microscopic bath.
It starts from a perturbative series for $\mathcal L$ and applies a similarity
transformation,

:::{math}
:label: lindblad:similarity
\widetilde{\mathcal L}
=\mathcal U^{-1}\mathcal L\mathcal U.
:::

If $x=\operatorname{vec}(\rho)$ and
$\widetilde x=\mathcal U^{-1}x$, then
$d\widetilde x/dt=\widetilde{\mathcal L}\widetilde x$. Thus
$\mathcal U$ embeds the transformed coordinates into the original operator
space. Restricting Eq. {eq}`lindblad:similarity` to a block gives an effective
generator and an embedding of its states into the full problem.

An arbitrary similarity transformation need not map density matrices to
density matrices. It may preserve the spectrum of $\mathcal L$ while
destroying trace preservation, Hermiticity preservation, complete positivity,
or the purely Hamiltonian character of a closed-system problem. This page
therefore develops three increasingly strong mask rules:

1. preserve trace;
2. preserve Hermiticity;
3. preserve unitary evolution when the input has no jumps.

The first two rules apply to general Liouvillians. The third uses the extra
Hilbert-space structure of a commutator. The final section explains how to
extract Hamiltonian and jump-operator series from an effective Liouvillian.
For a complete calculation, see
[Adiabatic elimination for a Lindblad equation](tutorial/lindblad_adiabatic_elimination.md).

## From the master equation to a matrix

We use row-major vectorization. The basis operator
$|i\rangle\langle j|$ becomes the vector basis element labelled by the pair
$(i,j)$, with $j$ varying fastest. Consequently,

:::{math}
:label: lindblad:vectorization
A\rho B \;\mapsto\; (A\otimes B^T)\,\operatorname{vec}(\rho).
:::

Indeed,

:::{math}
(A\rho B)_{ij}
= \sum_{k,l} A_{ik}\rho_{kl}B_{lj}
= \sum_{k,l}(A\otimes B^T)_{ij,kl}\rho_{kl}.
:::

The two factors in $A\otimes B^T$ are the ordinary matrices $A$ and $B^T$.
They are not vectorized separately. The Kronecker product already has the
correct $d^2\times d^2$ shape to act on the $d^2$ components of
$\operatorname{vec}(\rho)$.

Applying this identity to Eq. {eq}`lindblad:gksl` gives the Liouvillian matrix

:::{math}
:label: lindblad:matrix
\begin{aligned}
\mathcal L={}&-i\left(H\otimes I-I\otimes H^T\right)\\
&+\sum_k\left[
J_k\otimes J_k^*
-\frac12\left(
J_k^\dagger J_k\otimes I
+I\otimes(J_k^\dagger J_k)^T
\right)
\right].
\end{aligned}
:::

Here and below, the same symbol $\mathcal L$ denotes the abstract
superoperator and its matrix in the stated operator basis. Equation
{eq}`lindblad:matrix` turns the master equation into an ordinary linear system,

:::{math}
\frac{d}{dt}\operatorname{vec}(\rho)
=\mathcal L\operatorname{vec}(\rho),
:::

which is the form used by the non-Hermitian block-diagonalization algorithm.

Two structures on operator space will matter below. The trace is a row vector
$\tau$ satisfying

:::{math}
\operatorname{tr}(\rho)=\tau\operatorname{vec}(\rho),
:::

and Hermitian conjugation is the antilinear map

:::{math}
\operatorname{vec}(\rho^\dagger)=K\operatorname{vec}(\rho)^*,
:::

where $K$ exchanges the two operator indices. A superoperator $X$ preserves
Hermiticity precisely when

:::{math}
:label: lindblad:sharp
X^\sharp=X,
\qquad
X^\sharp=KX^*K.
:::

Trace preservation is the independent condition $\tau X=0$.

The sharp operation tests only Hermiticity preservation. A matrix satisfying
$\tau X=0$ and $X^\sharp=X$ need not generate completely positive evolution.
For a time-independent generator, complete positivity additionally requires
that its dissipative part admit the positive Kossakowski factorization
discussed at the end of this page.

## What the recursion must preserve

Pymablock splits every matrix into a retained part $S(X)$ and an eliminated
part $R(X)$, with $S+R=I$. The non-Hermitian recursion preserves trace and
Hermiticity when the split and the Sylvester solve respect the corresponding
operator-space structures. Concretely, we require

:::{math}
\tau R(X)=0,
\qquad
R(X^\sharp)=R(X)^\sharp,
\qquad
S(X^\sharp)=S(X)^\sharp,
:::

and, for every eliminated right-hand side $Y=R(Y)$,

:::{math}
\tau\,\operatorname{Sylv}(Y)=0,
\qquad
\operatorname{Sylv}(Y^\sharp)
=\operatorname{Sylv}(Y)^\sharp.
:::

Each input coefficient must satisfy the same physical conditions,

:::{math}
\tau\mathcal L_{\boldsymbol n}=0,
\qquad
\mathcal L_{\boldsymbol n}^\sharp=\mathcal L_{\boldsymbol n}.
:::

Under these assumptions, the perturbative transformation and transformed
generator obey

:::{math}
\tau\mathcal U=\tau,
\qquad
\tau\mathcal U^{-1}=\tau,
\qquad
\tau\widetilde{\mathcal L}=0,
:::

and

:::{math}
\mathcal U^\sharp=\mathcal U,
\qquad
(\mathcal U^{-1})^\sharp=\mathcal U^{-1},
\qquad
\widetilde{\mathcal L}^\sharp=\widetilde{\mathcal L}.
:::

These are abstract conditions on linear projectors. The next section turns
them into entrywise rules for the boolean masks accepted by Pymablock.

## Masks act in the working basis

The entries of a boolean mask refer to the basis in which Pymablock performs
the recursion. For Lindbladian perturbation theory this is normally an
eigenoperator basis of $\mathcal L_0$, rather than the matrix-unit basis used
in Eq. {eq}`lindblad:vectorization`.

We write the right and left eigenoperators as

:::{math}
\mathcal L_0 R_\alpha=\lambda_\alpha R_\alpha,
\qquad
(L^\alpha|\mathcal L_0=\lambda_\alpha(L^\alpha|,
\qquad
(L^\alpha|R_\beta)=\delta^\alpha_\beta.
:::

The normalization is not fixed by biorthogonality: each pair may be rescaled
as $R_\alpha\mapsto c_\alpha R_\alpha$ and
$L^\alpha\mapsto L^\alpha/c_\alpha^*$. This freedom matters because a
boolean mask selects individual matrix entries. The simple rules below assume
a **structure-adapted eigenbasis** with the following choices:

- one left zero mode equals the trace functional;
- every other right eigenoperator is traceless;
- Hermitian conjugation acts on the right eigenoperators as an involution
  without extra scale factors, $R_{\bar\alpha}=R_\alpha^\dagger$.

This involution does not pair every eigenoperator with a distinct partner. If
$\lambda_\alpha$ is real, its eigenspace is closed under Hermitian
conjugation, and we choose a Hermitian basis in it. These basis operators are
fixed points, $\bar\alpha=\alpha$. Only non-real eigenvalues produce distinct
pairs: we choose a basis of the $\lambda$ eigenspace and obtain a basis of the
$\lambda^*$ eigenspace by Hermitian conjugation. The index set therefore
decomposes into one-element and two-element orbits,

:::{math}
\{\alpha\},\quad R_\alpha^\dagger=R_\alpha,
\qquad\text{or}\qquad
\{\alpha,\bar\alpha\},\quad
R_{\bar\alpha}=R_\alpha^\dagger.
:::

Degenerate real eigenspaces leave freedom to rotate their Hermitian basis,
but do not require two-element orbits. If $\mathcal L_0$ is defective, an
eigenbasis does not exist and these pixelwise rules must instead be formulated
in a structure-adapted Jordan basis.

The same structure appears without complex eigenoperators if we regard the
Hermitian operators as a real vector space. A real eigenvalue gives a
$1\times1$ block. For a non-real pair $\lambda=a+ib$ and $\lambda^*=a-ib$,
we form the Hermitian operators

:::{math}
X=\frac{R+R^\dagger}{2},
\qquad
Y=\frac{R-R^\dagger}{2i}.
:::

Writing a Hermitian operator in their real span as $\rho=xX+yY$, the
Liouvillian acts on its coordinates through the $2\times2$ block

:::{math}
\mathcal L_0\rho
\quad\longleftrightarrow\quad
\begin{pmatrix}a&b\\-b&a\end{pmatrix}
\begin{pmatrix}x\\y\end{pmatrix}.
:::

Thus the structure-adapted decomposition consists of $1\times1$ real blocks
for real modes and $2\times2$ real blocks for non-real conjugate pairs.

:::{figure} _static/lindblad_mask_pairing.svg
:alt: A structure-adapted eigenoperator basis turns trace and Hermiticity constraints into simple boolean mask rules.
:width: 620px

In a structure-adapted basis, real modes are fixed by Hermitian conjugation,
while non-real modes occur in distinct conjugate pairs. Trace preservation
protects one output row, and Hermiticity requires the mask to be constant on
the resulting entry orbits.
:::

### Trace preservation

Let $R(X)$ denote the part selected for elimination and $S(X)=X-R(X)$ the
part retained in the effective model. The split preserves trace if

:::{math}
:label: lindblad:trace-split
\tau R(X)=0
\quad\text{for every }X.
:::

Then $\tau S(X)=\tau X$: the split assigns all trace-changing components to
the retained part. In the structure-adapted eigenbasis, $\tau$ is one
coordinate row, so Eq. {eq}`lindblad:trace-split` has a direct boolean rule:

:::{important}
The elimination mask must be false on the trace row.
:::

For a trace-preserving input, both $R(X)$ and $S(X)$ are then separately
trace preserving. The Sylvester solve divides matrix entries by eigenvalue
differences and does not change their positions, so it preserves the same
protected row.

### Hermiticity preservation

Because $\mathcal L_0$ preserves Hermiticity,

:::{math}
\mathcal L_0R_\alpha^\dagger
=\lambda_\alpha^*R_\alpha^\dagger.
:::

Non-real eigenspaces therefore occur in conjugate pairs, while real
eigenspaces map to themselves. In the structure-adapted basis, the sharp
operation maps a superoperator entry as

:::{math}
(\alpha,\beta)\longleftrightarrow
(\bar\alpha,\bar\beta).
:::

The boolean rule is consequently

:::{important}
The mask must assign the same value to $(\alpha,\beta)$ and
$(\bar\alpha,\bar\beta)$.
:::

This simultaneous action also decomposes the matrix entries into one- and
two-element orbits. An entry $(\alpha,\beta)$ is fixed only when both indices
are fixed points, so that $R_\alpha$ and $R_\beta$ are Hermitian eigenoperators
with real eigenvalues. If either index belongs to a non-real pair, the entry
has a distinct partner $(\bar\alpha,\bar\beta)$, and the mask must select both
or neither.

Equivalently, the mask projector commutes with the sharp operation:
$R(X^\sharp)=R(X)^\sharp$. The diagonal Sylvester solve respects this pairing
because the paired denominator is the complex conjugate of the original one.
Together with the protected trace row, this makes the perturbative
transformation and the effective Liouvillian trace and Hermiticity preserving
at every order.

In a generic eigenbasis, Hermitian conjugation has a matrix representation
$D$ rather than a permutation $\alpha\mapsto\bar\alpha$. The correct
condition is then that the mask projector commute with the antilinear map
$x\mapsto Dx^*$. A boolean entry mask generally fails this condition. This is
why choosing and normalizing the eigenoperators is part of choosing a mask,
not a harmless preprocessing detail.

### What a mask does not choose

The block decomposition identifies the unperturbed subspaces. A mask may
select couplings between those subspaces, or refine a block that Pymablock is
asked to diagonalize, but it cannot perturbatively eliminate a resonant entry
inside an unperturbed eigenspace. Such an entry has a zero Sylvester
denominator and belongs in the retained problem. Changing it requires a
different choice of unperturbed blocks.

## Preserving unitary evolution

Trace and Hermiticity preservation do not ensure that a Hamiltonian problem
remains Hamiltonian. A general similarity transform of
$-i[H,\mathord\cdot]$ can produce a trace- and Hermiticity-preserving
generator with a nonzero dissipative part. To exclude this, the Liouville-space
transformation must itself come from a Hilbert-space unitary.

Suppose

:::{math}
\mathcal L_0=-i[H_0,\mathord\cdot],
\qquad
H_0|a\rangle=E_a|a\rangle.
:::

The matrix units form a structure-adapted eigenbasis,

:::{math}
R_{ab}=|a\rangle\langle b|,
\qquad
\mathcal L_0R_{ab}=-i(E_a-E_b)R_{ab},
\qquad
R_{ab}^\dagger=R_{ba}.
:::

Now choose a symmetric Hamiltonian-space mask $m$. The entry $m_{ac}=1$
means that the Hermitian calculation eliminates $H_{ac}$ and $H_{ca}$. A
Hilbert-space unitary acts in Liouville space by

:::{math}
\operatorname{Ad}_U(\rho)=U\rho U^\dagger,
\qquad
\operatorname{Ad}_U=U\otimes U^*.
:::

The mask compatible with this transformation is the **OR lift**

:::{math}
:label: lindblad:or-lift
(\widehat R_m)_{ij,kl}=m_{ik}\mathbin{\lor}m_{jl}.
:::

It marks a Liouville-space entry whenever the underlying unitary may change
the ket index, the bra index, or both.

:::{figure} _static/lindblad_unitary_lift.svg
:alt: A Hamiltonian mask lifted to Liouville space, compared with the smaller support of a commutator generator.
:width: 900px

Eliminating the $0\leftrightarrow2$ Hamiltonian coupling produces two index
stripes in Liouville space. The OR lift includes their intersections because
the finite transformation is a product $U\otimes U^*$; a coherent
Liouvillian itself occupies only the one-index-at-a-time commutator support.
:::

The larger pattern in Eq. {eq}`lindblad:or-lift` is essential. Infinitesimally,
a generator entry acts on one index at a time,

:::{math}
[|a\rangle\langle c|,R_{kl}]
=\delta_{ck}R_{al}-\delta_{la}R_{kc},
:::

and a coherent Liouvillian therefore has support

:::{math}
(\delta_{jl}m_{ik})\mathbin{\lor}(\delta_{ik}m_{lj}).
:::

At higher orders, however, $U\otimes U^*$ also contains terms in which both
indices change. Pymablock uses one mask both for the effective generator and
for the gauge of the transformation and its inverse. The OR lift includes
the full gauge support needed to reproduce the Hilbert-space unitary
calculation order by order.

This construction works for every admissible symmetric Hamiltonian mask,
including nontransitive patterns. Degenerate energy gaps require care:
entries with equal Liouvillian eigenvalues have zero Sylvester denominator and
must be removed from the elimination mask.

The OR lifts form a broad, robust family of masks that preserve unitary
evolution. They are not known to exhaust all admissible Liouville-space
masks. One may add entries that the transformation never populates for a
particular problem, and dissipative problems may admit masks that eliminate
some jump terms while preserving complete positivity. Such extensions depend
on the support and positivity structure of the specific series; trace and
Hermiticity rules alone do not classify them.

:::{warning}
Trace preservation and Hermiticity preservation do not imply complete
positivity. A truncated effective series can also lose positivity outside its
perturbative regime even when the exact transformed generator is physical.
:::

## Recovering Hamiltonian and jump operators

After block diagonalization, the coefficients of

:::{math}
\widetilde{\mathcal L}(\lambda)
=\sum_{n\geq0}\lambda^n\widetilde{\mathcal L}_n
:::

are matrices in Liouville space. We first recover a Hamiltonian and a
Kossakowski matrix at each order, then factor the complete Kossakowski series
to obtain jump operators.

### Linear reconstruction

Choose a Hermitian, Hilbert--Schmidt orthonormal basis
$\{F_a\}_{a=1}^{d^2-1}$ for the traceless operators. Define

:::{math}
\mathcal K_a(\rho)=-i[F_a,\rho],
\qquad
\mathcal D_{ab}(\rho)=F_a\rho F_b
-\frac12\{F_bF_a,\rho\}.
:::

Every trace- and Hermiticity-preserving coefficient has a unique
decomposition, up to the irrelevant identity part of the Hamiltonian,

:::{math}
:label: lindblad:linear-decomposition
\widetilde{\mathcal L}_n
=\sum_a h_{a,n}\mathcal K_a
+\sum_{a,b}(C_n)_{ab}\mathcal D_{ab},
:::

with real $h_{a,n}$ and Hermitian $C_n$. Vectorizing the superoperators turns
Eq. {eq}`lindblad:linear-decomposition` into one fixed linear system,

:::{math}
\begin{bmatrix}
\operatorname{vec}(\mathcal K_1)&\cdots&
\operatorname{vec}(\mathcal D_{d^2-1,d^2-1})
\end{bmatrix}
\begin{bmatrix}h_n\\\operatorname{vec}(C_n)\end{bmatrix}
=\operatorname{vec}(\widetilde{\mathcal L}_n).
:::

The same inverse or pseudoinverse applies at every perturbative order, and
$H_n=\sum_a h_{a,n}F_a$. Complete positivity is not an order-by-order
condition: the individual matrices $C_n$ need not be positive semidefinite.

### Factoring the Kossakowski series

The linear solve gives a Kossakowski series. We first use one perturbation
parameter,

:::{math}
C(\lambda)=\sum_{n\geq0}\lambda^nC_n,
\qquad
B(\lambda)=\sum_{n\geq0}\lambda^nB_n,
:::

and seek a factor such that

:::{math}
C(\lambda)=B(\lambda)B(\lambda)^\dagger.
:::

After choosing a zeroth-order factor $C_0=B_0B_0^\dagger$, matching order $n$
gives

:::{math}
:label: lindblad:factor-recurrence
C_n=B_0B_n^\dagger+B_nB_0^\dagger
+\sum_{k=1}^{n-1}B_kB_{n-k}^\dagger.
:::

The behavior depends on whether $B_0$ is regular.

#### Regular zeroth-order factor

Suppose the rank of $C(\lambda)$ is constant near the expansion point and its
nonzero eigenvalues remain separated from zero. We may then choose $B_0$ with
full rank on this fixed-dimensional support. The sum in
Eq. {eq}`lindblad:factor-recurrence` contains only known lower-order terms.
After fixing the unitary freedom of the factor, the remaining equation is
linear in $B_n$. A triangular gauge or an orthogonality condition on
$B_0^\dagger B_n$ provides such a gauge.

Several perturbation parameters do not change this conclusion. For
$\boldsymbol\lambda=(\lambda_1,\ldots,\lambda_N)$, we write

:::{math}
C(\boldsymbol\lambda)
=\sum_{\mathbf n\geq0}C_{\mathbf n}\boldsymbol\lambda^{\mathbf n},
\qquad
B(\boldsymbol\lambda)
=\sum_{\mathbf n\geq0}B_{\mathbf n}\boldsymbol\lambda^{\mathbf n}.
:::

Matching a multi-index gives

:::{math}
:label: lindblad:multivariate-factor-recurrence
C_{\mathbf n}
=\sum_{\mathbf k\leq\mathbf n}
B_{\mathbf k}B_{\mathbf n-\mathbf k}^\dagger.
:::

Ordering the calculation by total degree $|\mathbf n|$ leaves only the two
terms containing $B_{\mathbf n}$ unknown. Thus a regular $B_0$ again gives
the same linear problem at each multi-index.

#### Vanishing or rank-deficient zeroth order

If $B_0=0$, Eq. {eq}`lindblad:factor-recurrence` has no linear term and cannot
start the recursion. Suppose the first nonzero homogeneous part of $B$ has
total degree $r$,

:::{math}
B^{(r)}(\boldsymbol\lambda)
=\sum_{|\mathbf n|=r}
B_{\mathbf n}\boldsymbol\lambda^{\mathbf n}.
:::

Then all terms of $C$ below degree $2r$ must vanish, and its first nonzero
homogeneous part must satisfy the nonlinear Gram-factorization condition

:::{math}
:label: lindblad:leading-gram-factor
C^{(2r)}(\boldsymbol\lambda)
=B^{(r)}(\boldsymbol\lambda)
B^{(r)}(\boldsymbol\lambda)^\dagger.
:::

This is the main obstruction. In one variable, an ordinary power series for
$B$ requires the first nonzero order of $C$ to be even, with a positive
semidefinite leading coefficient. If $C$ starts at an odd order, a one-sided
expansion may instead involve fractional powers such as
$B\sim\lambda^{1/2}$.

In several variables, $C^{(2r)}$ must be a matrix-valued sum of squares of
homogeneous matrix polynomials. Pointwise positivity of
$C^{(2r)}(\boldsymbol\lambda)$ is necessary but does not in general guarantee
such a polynomial Gram factor; analytic factorization at a rank-changing
point is an additional condition. Allowing additional columns of $B$, hence
additional jump channels, helps. For example,

:::{math}
C(x,y,z)=x^2+y^2+z^2
=B(x,y,z)B(x,y,z)^\dagger,
\qquad
B=(x+iy\;\;z).
:::

No single complex linear jump amplitude can produce this rank-three real
quadratic form: the real and imaginary parts of one amplitude span at most
two squares. A second jump channel supplies the missing direction.

Equation {eq}`lindblad:leading-gram-factor` is therefore underdetermined in a
useful way. It determines only the sum of products of the leading jump
amplitudes, not the amplitudes themselves or even their minimal number. We
should retain this freedom while attempting the next orders, rather than
diagonalizing $C^{(2r)}$ pointwise and fixing a jump basis immediately.

This freedom has a convex formulation. Let $z_r(\boldsymbol\lambda)$ be the
column vector of all monomials of total degree $r$, and define

:::{math}
Z_r(\boldsymbol\lambda)
=I\otimes z_r(\boldsymbol\lambda)^T.
:::

Every homogeneous polynomial jump factor can be written as
$B^{(r)}=Z_rV$ for a constant coefficient matrix $V$. Hence

:::{math}
:label: lindblad:gram-spectrahedron
C^{(2r)}(\boldsymbol\lambda)
=Z_r(\boldsymbol\lambda)GZ_r(\boldsymbol\lambda)^\dagger,
\qquad
G=VV^\dagger\succeq0.
:::

Equating polynomial coefficients in
Eq. {eq}`lindblad:gram-spectrahedron` imposes linear constraints on $G$.
The feasible matrices therefore form a convex set, the Gram spectrahedron of
$C^{(2r)}$. Factoring one feasible $G$ produces jump amplitudes. Right-unitary
jump-basis rotations change $V$ but leave $G$ fixed, whereas different points
of the Gram spectrahedron are the genuinely different sum-of-squares
decompositions discussed above. The number of jump channels equals
$\operatorname{rank}G$; minimizing it is a separate, nonconvex rank problem
and is not required for existence.

For a chosen leading factor $B^{(r)}$, higher homogeneous orders again enter
linearly. Unlike the regular-$B_0$ case, multiplication by
$B^{(r)}(\boldsymbol\lambda)$ couples all monomials of a given degree and need
not be surjective. One should therefore regard the leading Gram factor and
the first continuation equations as a coupled problem. A different
sum-of-squares decomposition, or a nonminimal number of jump channels, may
make a continuation available that a prematurely fixed factor obscures.

We can avoid that premature choice at any finite perturbative order. Let
$z_{r:R}$ contain all monomials of degrees $r$ through $R$, set
$Z_{r:R}=I\otimes z_{r:R}^T$, and solve

:::{math}
:label: lindblad:truncated-gram-problem
G\succeq0,
\qquad
\left[
Z_{r:R}(\boldsymbol\lambda)
G
Z_{r:R}(\boldsymbol\lambda)^\dagger
\right]_{\mathbf n}
=C_{\mathbf n}
\quad\text{for all matched }\mathbf n.
:::

This is a semidefinite feasibility problem: every coefficient constraint is
linear in $G$. Coefficients beyond the available truncation remain
unconstrained. If the problem is feasible, a factorization $G=VV^\dagger$
gives $B=Z_{r:R}V$ whose product matches the known Kossakowski series. Thus a
finite-order implementation can solve for all jump amplitudes jointly and
postpone both the jump-basis gauge and the choice of jump count until after
feasibility is established.

Feasibility of Eq. {eq}`lindblad:truncated-gram-problem` is stronger than
pointwise positivity of a truncated polynomial. It certifies a polynomial
Gram factor within the chosen degree bound. Failure may mean that no such
factor exists, that the degree bound is too small, or that fractional or
nonanalytic dependence is unavoidable. Extending the truncation requires a
compatible sequence of Gram matrices, so one finite feasible problem does not
by itself prove the existence of an all-orders analytic factor.

If $B_0$ is rank deficient rather than zero, these two regimes coexist. The
recursion is regular on the range of $B_0$, while jump directions in its
kernel must satisfy the same leading Gram-factorization problem. In practice,
the prospects are good when analytic jump operators are already known from
the underlying model. Starting only from a Kossakowski series that changes
rank, there is no general guarantee of an analytic jump-operator series.

For one real coupling parameter, the situation is favorable. An analytic
Hermitian matrix family can be diagonalized analytically locally, and
positivity on a two-sided neighborhood forces every vanishing eigenvalue to
start at even order. Signed analytic square roots of those eigenvalues then
give an analytic, possibly nonminimal factor. The genuinely difficult case is
a rank-changing family in several parameters, where no simultaneous analytic
eigenbasis or canonical square root need exist.

For univariate polynomial families, the global polynomial analogue is the
[matrix Fejer--Riesz factorization](https://arxiv.org/abs/1707.08261): a
Hermitian matrix polynomial that is positive semidefinite on the real line
admits a polynomial sum-of-squares factorization. There is no corresponding
unconditional multivariate statement, which is why the Gram feasibility
problem becomes substantive when several perturbation parameters create new
jump directions.

Pure jump-basis gauge is more limited. Replacing
$B(\boldsymbol\lambda)$ by
$B(\boldsymbol\lambda)U(\boldsymbol\lambda)$ with a unitary matrix $U$ leaves
$C$ unchanged. Leading factors related by a constant right unitary have
equivalent linearized continuation equations, so this gauge alone cannot turn
an incompatible next order into a compatible one. The useful freedom above
is broader: different polynomial Gram decompositions and additional jump
channels need not amount to a fixed unitary rotation within a minimal factor.

None of these freedoms changes the lowest order of $C=BB^\dagger$. The
derivative of the map $B\mapsto BB^\dagger$ vanishes at $B=0$, so the leading
Gram problem remains nonlinear. Once a suitable leading decomposition has
been selected, its residual unitary freedom may be fixed and the higher-order
equations solved linearly.

For a system that becomes dissipative by coupling to a lossy auxiliary
system, nonpolynomial jumps are therefore often an artifact of the chosen
perturbation parameter. If $g$ is the microscopic coupling amplitude, the
effective dissipator commonly starts as

:::{math}
C(g)=g^2C_2+O(g^3),
\qquad
B(g)=gB_1+O(g^2),
\qquad
C_2=B_1B_1^\dagger.
:::

The jump operators are analytic in $g$. If we instead use the induced rate
$\gamma=g^2$ as the perturbation parameter, the same factor reads
$B(\gamma)=\sqrt{\gamma}B_1+\cdots$. The square root reflects the
reparameterization from an amplitude to a rate, not singular physics. Genuine
nonanalyticity remains possible when the Kossakowski rank changes in a way
that admits no analytic Gram factor in the chosen microscopic parameters.

If $b_\mu(\lambda)$ is column $\mu$ of $B(\lambda)$, then

:::{math}
J_\mu(\lambda)=\sum_a b_{a\mu}(\lambda)F_a
:::

is the corresponding jump-operator series. Together with
$H(\lambda)=\sum_n\lambda^nH_n$, these operators reconstruct

:::{math}
\widetilde{\mathcal L}(\rho)
=-i[H,\rho]
+\sum_\mu\left(
J_\mu\rho J_\mu^\dagger
-\frac12\{J_\mu^\dagger J_\mu,\rho\}
\right).
:::

All cases also require $C(\boldsymbol\lambda)\succeq0$ on the branch being
expanded. A rank change can force fractional powers or a different expansion
point. If the effective generator is not completely positive, one may instead
use a signed factorization $C=B\Sigma B^\dagger$, but its columns are not
Lindblad jump operators with positive rates.
