# Matrix product operators

Matrix product operators (MPOs) allow Pymablock to construct effective Hamiltonians without storing exponentially large dense matrices.
Pymablock only needs an algebra for individual perturbative coefficients and a solver for Sylvester's equation, so the tensor-network implementation can remain independent of the perturbative algorithm.

## When is MPO perturbation theory advantageous?

For a chain of $L$ sites with local dimension $d$, a dense operator has dimension $D\times D$, with $D=d^L$, and therefore requires $\mathcal{O}(d^{2L})$ storage.
An MPO with representative bond dimension $\chi$ instead requires $\mathcal{O}(L d^2\chi^2)$ storage.
The exponential saving is real only while the required $\chi$ grows moderately: the relevant quantity is the operator entanglement of the perturbative coefficients, not merely the locality of the starting Hamiltonian.

MPO perturbation theory is most useful when separated energy scales give a low-order, compressible effective Hamiltonian that will be reused for several states or observables.
If only a few states are needed, the implicit MPS formulation below avoids constructing the full transformation in operator space; direct DMRG may still be cheaper if no perturbative effective model is required.

A practical pilot should increase system size and perturbative order while monitoring runtime, bond dimension, truncation error, and Sylvester residual.
Rapid growth of $\chi$, GMRES iterations, or variational sweeps means that the tensor-network advantage is disappearing.

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

## Choosing between full MPO and implicit MPS

The two formulations differ in what they represent, not in whether they assemble a dense matrix.
Both are matrix-free tensor-network calculations.
The full formulation stores the complete transformation coefficient $X$ as an MPO, whereas the implicit formulation stores only its action on selected states.

| | Full MPO | Implicit MPS |
| --- | --- | --- |
| Unknown | the complete operator $X$ | the columns $QX|\phi_a\rangle$ |
| Local dimension | $d^2$ | $d$ |
| Linear solver in the examples | compressed GMRES | DMRG-style variational sweeps |
| Result | an effective MPO | a small effective matrix and corrected states |

Choose the implicit formulation when the retained space contains a few known MPS eigenstates.
Only the columns that act on those states can enter their effective Hamiltonian, so constructing the rest of $X$ would do unnecessary work.
The cost is one corrected MPS per retained state, which makes this approach unattractive for an extensive retained space.

Choose the full-MPO formulation when the operator itself is the result.
This is appropriate when the retained space is extensive, or when the same effective Hamiltonian or transformation will be reused for many states, observables, dynamics, or later DMRG calculations.
Its cost is instead controlled by the operator entanglement of the perturbative coefficients.

As a practical rule: use implicit MPS for a few target states, full MPO for a reusable effective operator, and direct DMRG when only one ground state at one parameter value is needed.

## Full-MPO Sylvester solve

At each perturbative order Pymablock solves

$$
AX-XB=Y.
$$

With row-major vectorization this becomes

$$
\left(A\otimes I-I\otimes B^T\right)\operatorname{vec}(X)
=\operatorname{vec}(Y).
$$

The transpose is fixed by the vectorization convention; it is not an adjoint.
The TeNPy full-MPO example stores the vectorized operators as MPSs on local dimension $d^2$ and applies the Sylvester superoperator as an MPO.
Thus, “full MPO” does not mean that the exponentially large superoperator matrix is constructed.

One restarted GMRES cycle starts from the true residual, repeatedly applies the superoperator, and orthogonalizes the resulting MPSs to build a short Krylov basis.
It then solves only the small Hessenberg least-squares problem and adds the resulting Krylov correction to $X$.
Restarting limits the number of stored MPSs.
Because compression spoils the exact Arnoldi relation, the implementation recomputes $\|AX-XB-Y\|_F$ after every restart instead of trusting the Hessenberg estimate.

GMRES was chosen for this path because it requires only MPO application, MPS addition, overlaps, and compression, and it also handles invertible indefinite or non-normal maps.
Its drawback is that several operator-space MPSs must be stored and repeatedly combined.

## Implicit MPS solve

Let $\{|\phi_a\rangle\}_{a=1}^m$ be orthonormal eigenstates spanning the retained space $P$, and let $Q=1-P$.
The implicit method never constructs a basis for $Q$ or the full operator $X$.
It stores only complement-space columns and solves

$$
Q(H_0-E_a)Q|\eta_a\rangle=|S_a\rangle,
\qquad
\langle\phi_b|\eta_a\rangle=0.
$$

{autolink}`~pymablock.implicit.ImplicitBlock` represents $P$--$P$ blocks as small dense matrices, $Q$--$P$ blocks as MPS bundles, and $Q$--$Q$ blocks as lazy projected MPO applications.
Products such as a row times a column become ordinary overlap matrices.
Applying $Q$ only subtracts overlaps with the reference MPSs, so no projector MPO is built.
Pymablock's perturbative recursion itself is unchanged.

For a Hermitian, sign-definite shifted operator, the equation is the stationarity condition of the Hylleraas functional

$$
\mathcal{F}[\eta]
=\frac12\langle\eta|(H_0-E_a)|\eta\rangle
-\operatorname{Re}\langle\eta|S_a\rangle.
$$

The TeNPy example optimizes two neighboring MPS tensors at a time.
Contracting all other sites gives a local linear equation.
Orthogonality is imposed in the same solve with Lagrange multipliers,

$$
\begin{pmatrix}
K_\mathrm{loc} & C^\dagger\\
C & 0
\end{pmatrix}
\begin{pmatrix}\theta\\ \lambda\end{pmatrix}
=
\begin{pmatrix}s_\mathrm{loc}\\0\end{pmatrix}.
$$

After solving, an SVD moves the optimization center and truncates to `chi_max`.
A left-to-right and right-to-left pass form one sweep.
After every sweep the code projects again, applies the original MPO, and measures the true global residual and all reference-state overlaps.
For clarity, this documentation backend rebuilds its left and right environments at every local update; a production backend should cache them to reduce the sweep cost from quadratic to linear in chain length.

The example deliberately does not square $H_0-E_a$, because that squares the condition number and increases the operator bond dimension.
It also does not silently use a pseudoinverse, broadening, or penalty projector: a singular local constrained equation is reported as a failure.
For general indefinite or non-Hermitian projected equations, a dedicated MINRES/GMRES-like local strategy is still required.

## Residual and truncation control

Approximate MPO algebra changes the meaning of an otherwise exact formal perturbation series.
Compression errors enter additions and products, while the iterative solver adds an independent error to every solution of Sylvester's equation.
The resulting series coefficients therefore only approximate the coefficients that an exact operator algebra would produce.

The full-MPO solver returns a {autolink}`~pymablock.mpo.SylvesterResult` with the compressed-operator residual

$$
r_\mathrm{MPO} =
\frac{\lVert AX-XB-Y\rVert_F}{\lVert Y\rVert_F}.
$$

The implicit solver returns a {autolink}`~pymablock.implicit.StateSolveResult` with

$$
r_\mathrm{MPS} =
\frac{\lVert Q(H_0-E_a)Q|\eta_a\rangle-|S_a\rangle\rVert}
{\lVert S_a\rVert}.
$$

Both adapters reject an unconverged solve, a non-finite residual, or a residual above their acceptance threshold.
This check prevents a failed approximate solve from silently entering higher perturbative orders.

Users should record the relevant diagnostics for every requested order:

- the largest retained bond dimension,
- the accumulated or maximum reported truncation error,
- the true relative Sylvester residual,
- for implicit solves, the largest overlap with a retained reference state.

Convergence requires repeating the calculation with a larger `chi_max`, a smaller `svd_min`, and a tighter solver tolerance.
The effective Hamiltonian terms relevant to the physical conclusion must remain stable under these changes.
A stable energy or a small local/Krylov residual alone does not establish convergence of every requested coefficient.

## Applicability and limitations

The TeNPy examples target finite open chains with square local spaces and `conserve=None`.
The full-MPO path assumes separated block spectra.
The implicit path assumes a small set of orthonormal MPS eigenstates and a Hermitian shifted problem on their complement; the demonstrated Hylleraas sweep is intended for sign-definite shifts.
Neither path introduces spectral broadening or a pseudoinverse.

Custom Sylvester solvers cannot currently be combined with `fully_diagonalize`.
The examples also do not support infinite tensor networks, charge-conserving tensor legs, or a guarantee that the required bond dimension stays bounded at high perturbative order.
These restrictions are explicit so that later backends can extend them without changing Pymablock's core algebra interface.

The [minimal TeNPy MPO tutorial](tutorial/tenpy_mpo.md) compares a complete two-site calculation with dense Pymablock.
The [Ising-chain benchmark](tutorial/tenpy_mpo_ising.md) applies the method to a six-site MPO problem with an analytical effective Hamiltonian.
The [implicit MPS tutorial](tutorial/tenpy_implicit_ising.md) treats the exactly solvable transverse-field Ising chain using two retained product-state MPSs.
