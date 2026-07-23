# Matrix product operators

Matrix product operators (MPOs) allow Pymablock to construct effective Hamiltonians without storing exponentially large dense matrices.
Pymablock only needs an algebra for individual perturbative coefficients and a solver for Sylvester's equation, so the tensor-network implementation can remain independent of the perturbative algorithm.

## When is MPO perturbation theory advantageous?

For a chain of $L$ sites with local dimension $d$, a dense operator has dimension $D\times D$, with $D=d^L$, and therefore requires $\mathcal{O}(d^{2L})$ storage.
An MPO with representative bond dimension $\chi$ instead requires $\mathcal{O}(L d^2\chi^2)$ storage.
The exponential saving is real only while the required $\chi$ grows moderately: the relevant quantity is the operator entanglement of the perturbative coefficients, not merely the locality of the starting Hamiltonian.

MPO perturbation theory is most useful when separated energy scales give a low-order, compressible effective Hamiltonian that will be reused for several states or observables.
If only one ground state is needed, direct DMRG may be cheaper because it avoids constructing the full transformation in operator space.

A practical pilot should increase system size and perturbative order while monitoring runtime, bond dimension, truncation error, and Sylvester residual.
Rapid growth of $\chi$ or GMRES iterations means that the MPO advantage is disappearing.

## Backend-neutral operators

We introduce {autolink}`~pymablock.mpo.BackendMPO` as a thin wrapper around a backend-native MPO.
The wrapper delegates addition, scalar multiplication, operator multiplication, and adjoints to an implementation of {autolink}`~pymablock.mpo.MPOBackend`.
Pymablock then uses the wrapped objects in the same lazy Cauchy products as dense matrices.

The backend owns all approximation choices.
In particular, it decides how and when to compress an MPO and must return new operators rather than mutate cached perturbative coefficients.
Two wrapped MPOs can only be combined when they share the same backend instance, which prevents accidental mixing of incompatible tensor conventions or truncation policies.

## Adding and multiplying MPOs

We first consider an MPO tensor with output index $s$, input index $t$, and virtual indices $a$ and $b$.
Adding two MPOs takes a direct sum of their virtual spaces, so the uncompressed bond dimension is

$$
\chi_{A+B} = \chi_A + \chi_B.
$$

The boundary tensors are concatenated, while every interior tensor is block diagonal in its virtual indices.
The TeNPy example compresses the result immediately after constructing this exact direct sum.

We then multiply two MPOs by contracting their intermediate physical index at every site.
For $C=AB$, the local tensor is

$$
C_i^{s,t}\big((a,c),(b,d)\big)
  = \sum_u A_i^{s,u}(a,b) B_i^{u,t}(c,d).
$$

The two left virtual indices and the two right virtual indices are fused.
The uncompressed bond dimension is therefore

$$
\chi_{AB} = \chi_A\chi_B.
$$

This product rule is exact before compression and fixes the order of operator multiplication.
Because repeated perturbative products would otherwise multiply bond dimensions rapidly, the TeNPy backend converts every result to an MPS on the local operator space, performs an SVD compression, and converts it back to an MPO.
Exact cancellations are returned as a bond-dimension-one zero MPO because a zero tensor cannot be put into normalized MPS canonical form.

## Solving Sylvester's equation

Pymablock computes the off-diagonal transformation at every new perturbative order by solving the linear operator equation

$$
A X-X B=Y.
$$

Here, $A$ and $B$ are the unperturbed Hamiltonian blocks, $Y$ is known from lower perturbative orders, and $X$ is the unknown transformation coefficient.
This is not an eigenvalue problem.
It asks us to invert the linear map

$$
\mathcal{L}(X) = AX-XB
$$

on one particular right-hand side $Y$.

### Turning operators into vectors

GMRES is formulated for linear systems on vectors, so we regard an operator as a vector in operator space.
With row-major vectorization,

$$
\operatorname{vec}(A X)
  = (A\otimes I)\operatorname{vec}(X)
$$

and

$$
\operatorname{vec}(X B)
  = (I\otimes B^T)\operatorname{vec}(X).
$$

Writing $x=\operatorname{vec}(X)$ and $b=\operatorname{vec}(Y)$ therefore turns the Sylvester equation into

$$
\mathsf{L}x=b,
\qquad
\mathsf{L}=A\otimes I-I\otimes B^T.
$$

The transpose in the right-action term is essential.
It follows from the vectorization convention and is not a Hermitian adjoint.

If the original Hilbert space has dimension $D$, then $\mathsf{L}$ is a $D^2\times D^2$ matrix.
The point of the tensor-network solver is to apply this matrix without constructing it.
The TeNPy example stores $x$, $b$, and all temporary operator-space vectors as MPSs with local dimension $d^2$, while it stores $\mathsf{L}$ as an MPO.
These MPSs encode vectorized operators; they are not physical many-body wavefunctions.
Their MPS overlap is the Frobenius inner product $\langle U,V\rangle=\operatorname{Tr}(U^\dagger V)$ of the corresponding operators.

### What one GMRES cycle does

Suppose that $x_0$ is the current approximation to the solution.
The first cycle uses $x_0=0$, while later cycles start from the result of the preceding cycle.
GMRES first computes the actual residual

$$
r_0=b-\mathsf{L}x_0,
\qquad
\beta=\lVert r_0\rVert,
\qquad
v_1=r_0/\beta.
$$

It then looks for a correction in the Krylov space

$$
\mathcal{K}_m(\mathsf{L},r_0)
=\operatorname{span}\left\{
r_0,\mathsf{L}r_0,\ldots,\mathsf{L}^{m-1}r_0
\right\}.
$$

This space contains correction directions obtained by repeatedly applying the Sylvester map to the current residual.
GMRES does not store those raw powers because they quickly become nearly linearly dependent.
Instead, the Arnoldi iteration constructs an orthonormal basis $v_1,\ldots,v_m$:

1. Apply the Sylvester superoperator to the newest basis vector, $w=\mathsf{L}v_j$.
2. Remove every component already represented in the basis, using $h_{ij}=\langle v_i,w\rangle$ and $w\leftarrow w-h_{ij}v_i$.
3. Normalize the remaining component to obtain $v_{j+1}$.

The coefficients $h_{ij}$ form a small upper-Hessenberg matrix $\overline{H}_m$.
In exact arithmetic, Arnoldi gives

$$
\mathsf{L}V_m=V_{m+1}\overline{H}_m,
$$

where the columns of $V_m$ are the Krylov basis vectors.
The approximation after this cycle has the form

$$
x_m=x_0+V_m y.
$$

GMRES chooses the coefficients $y$ that minimize the residual norm.
Using the Arnoldi relation and $r_0=\beta v_1$, the large minimization becomes the small dense least-squares problem

$$
y=\operatorname*{arg\,min}_z
\left\lVert\beta e_1-\overline{H}_m z\right\rVert_2.
$$

Only this $(m+1)\times m$ problem is solved with dense linear algebra.
The many-body dimension appears only in MPO--MPS applications, MPS overlaps, and compressed MPS sums.

The correspondence with the TeNPy operations is:

| GMRES object or operation | TeNPy representation |
| --- | --- |
| $x$, $b$, $r$, and each $v_j$ | an MPS on the local operator space |
| $\mathsf{L}=A\otimes I-I\otimes B^T$ | an MPO |
| $\mathsf{L}v_j$ | apply the MPO to the MPS, then compress |
| $\langle v_i,w\rangle$ | an MPS overlap |
| $w-h_{ij}v_i$ and $x_0+V_my$ | compressed finite-MPS additions |

The example performs modified Gram--Schmidt twice in every Arnoldi step.
The second pass repairs much of the loss of orthogonality caused by floating-point arithmetic and by compressing every MPO application and MPS sum.

### Why GMRES is restarted

Keeping more Krylov vectors improves the minimization but also increases memory use and exposes every vector to further tensor-network additions.
The example therefore keeps at most `krylov_dimension` basis vectors, uses them to update $x_0$, discards the basis, recomputes $r_0=b-\mathsf{L}x_0$, and starts another cycle.
This is restarted GMRES; the default Krylov dimension is eight.

Compression means that the Arnoldi relation is only approximate, so the residual predicted by the small Hessenberg problem is not sufficient as a convergence test.
After every restart, and once more before returning the result, the implementation reapplies the compressed superoperator and measures the actual residual.
That independently recomputed residual is the quantity reported to Pymablock.

## Residual and truncation control

Approximate MPO algebra changes the meaning of an otherwise exact formal perturbation series.
Compression errors enter additions and products, while the iterative solver adds an independent error to every solution of Sylvester's equation.
The resulting series coefficients therefore only approximate the coefficients that an exact operator algebra would produce.

We require the backend solver to return a {autolink}`~pymablock.mpo.SylvesterResult`.
The result contains a convergence flag and the relative residual measured after reconstructing the compressed solution,

$$
r =
\frac{\lVert AX-XB-Y\rVert_F}{\lVert Y\rVert_F}.
$$

The adapter created by {autolink}`~pymablock.mpo.make_mpo_sylvester_solver` rejects an unconverged solve, a non-finite residual, or a residual above its acceptance threshold.
This check prevents a failed approximate solve from silently entering higher perturbative orders.

Users should record three diagnostics for every requested order:

- the largest retained bond dimension,
- the accumulated or maximum reported truncation error,
- the true relative Sylvester residual.

Convergence requires repeating the calculation with a larger `chi_max`, a smaller `svd_min`, and a tighter solver tolerance.
The effective Hamiltonian terms relevant to the physical conclusion must remain stable under these changes.
A stable ground-state energy or a small internal Krylov residual alone does not establish convergence of every effective operator.

## Applicability and limitations

The initial example targets finite open chains with square local spaces and `conserve=None`.
It assumes that the spectra of the unperturbed blocks are separated, so the Sylvester operator is invertible on the requested off-diagonal block.
It does not introduce a pseudoinverse cutoff, spectral broadening, preconditioner, or null-space projection.

Custom Sylvester solvers cannot currently be combined with `fully_diagonalize`.
The example also does not support infinite or extensive MPOs, charge-conserving tensor legs, or a guarantee that the required bond dimension stays bounded at high perturbative order.
These restrictions are explicit so that later backends can extend them without changing Pymablock's core algebra interface.

The [minimal TeNPy MPO tutorial](tutorial/tenpy_mpo.md) compares a complete two-site calculation with dense Pymablock.
The [Ising-chain benchmark](tutorial/tenpy_mpo_ising.md) applies the method to a six-site MPO problem with an analytical effective Hamiltonian.
