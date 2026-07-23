# Matrix product operators

Matrix product operators (MPOs) allow Pymablock to construct effective Hamiltonians without storing exponentially large dense matrices.
Pymablock only needs an algebra for individual perturbative coefficients and a solver for Sylvester's equation, so the tensor-network implementation can remain independent of the perturbative algorithm.

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

Pymablock computes the off-diagonal transformation at every new perturbative order from Sylvester's equation

$$
A X-X B=Y.
$$

We use row-major vectorization, for which

$$
\operatorname{vec}(A X)
  = (A\otimes I)\operatorname{vec}(X)
$$

and

$$
\operatorname{vec}(X B)
  = (I\otimes B^T)\operatorname{vec}(X).
$$

The equation consequently becomes

$$
\left(A\otimes I-I\otimes B^T\right)\operatorname{vec}(X)
  = \operatorname{vec}(Y).
$$

The transpose in the right-action term is essential.
It follows from the vectorization convention and is not a Hermitian adjoint.

The TeNPy example represents the vectorized MPO as an MPS with local dimension $d^2$.
It constructs $A\otimes I-I\otimes B^T$ as another MPO and applies restarted GMRES without forming a dense matrix.
Each superoperator application and every Krylov-vector sum is compressed.
Two-pass modified Gram--Schmidt reduces the loss of orthogonality caused by finite precision and tensor truncation.

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

The [TeNPy MPO tutorial](tutorial/tenpy_mpo.md) implements the complete two-site calculation and compares the first- and second-order results with dense Pymablock.
