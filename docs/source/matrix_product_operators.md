# Matrix product operators

Matrix product operators (MPOs) allow Pymablock to construct effective Hamiltonians without storing exponentially large dense matrices.
The perturbative recursion is unchanged: it combines lower-order coefficients and solves a Sylvester equation for each new off-diagonal coefficient.
A tensor-network backend supplies the required operator algebra and linear solver.

## When MPO perturbation theory is useful

For a chain of $L$ sites with local dimension $d$, a dense operator has dimension $D\times D$, with $D=d^L$, and therefore requires $\mathcal{O}(d^{2L})$ storage.
An MPO with representative bond dimension $\chi$ instead requires $\mathcal{O}(L d^2\chi^2)$ storage.
The exponential saving persists only while $\chi$ grows moderately.
This bond dimension reflects the operator entanglement of each perturbative coefficient—its complexity across a spatial cut—not merely the locality of the starting Hamiltonian.

MPO perturbation theory is most useful when separated energy scales produce a low-order, compressible effective Hamiltonian that will be reused for several states or observables.

A practical pilot should increase system size and perturbative order while monitoring runtime, bond dimension, truncation error, and Sylvester residual.
Rapid growth of $\chi$, GMRES iterations, or variational sweeps means that the tensor-network advantage is disappearing.

## Operator algebra and compression

{autolink}`~pymablock.mpo.BackendMPO` wraps a backend-native MPO and delegates addition, scaling, multiplication, and adjoints to {autolink}`~pymablock.mpo.MPOBackend`.
Pymablock can therefore use MPOs in the same lazy perturbative products as dense matrices.
The backend chooses the compression method and returns new operators so that cached series coefficients remain unchanged.

MPO addition and multiplication increase the virtual bond dimensions before compression:

$$
\chi_{A+B}=\chi_A+\chi_B,
\qquad
\chi_{AB}=\chi_A\chi_B.
$$

Multiplication contracts the intermediate physical index locally.
For MPO tensors with output index $s$, input index $t$, and virtual indices $a,b,c,d$,

$$
C_i^{s,t}\big((a,c),(b,d)\big)
  = \sum_u A_i^{s,u}(a,b) B_i^{u,t}(c,d).
$$

The contraction is exact before compression and preserves the order $AB$.
The TeNPy example compresses after every sum and product because repeated perturbative operations would otherwise make the bond dimensions grow rapidly.
Compression makes the algebra approximate, so the final coefficients require numerical convergence checks.

## Choosing between full MPO and implicit MPS

The two formulations differ in what they represent, not in whether they assemble a dense matrix.
Both are matrix-free tensor-network calculations.
The full formulation stores the complete transformation coefficient $X$ as an MPO, whereas the implicit formulation stores only its action on selected states.

| | Full MPO | Implicit MPS |
| --- | --- | --- |
| Unknown | the complete operator $X$ | the columns $QX\lvert\phi_a\rangle$ |
| Local dimension | $d^2$ | $d$ |
| Linear solver in the examples | compressed GMRES | DMRG-style variational sweeps |
| Result | an effective MPO | a small effective matrix and corrected states |

Choose the implicit formulation when the retained space contains a few known MPS eigenstates.
Only the columns that act on those states can enter their effective Hamiltonian, so constructing the rest of $X$ would do unnecessary work.
The cost is one corrected MPS per retained state, which makes this approach unattractive for an extensive retained space.

Choose the full-MPO formulation when the operator itself is the result.
This is appropriate when the retained space is extensive, or when the same effective Hamiltonian or transformation will be reused for many states, observables, dynamics, or later DMRG calculations.
Its cost is instead controlled by the operator entanglement of the perturbative coefficients.

As a practical rule, use implicit MPS for a few target states, full MPO for a reusable effective operator, and direct DMRG when only one ground state at one parameter value is needed.

## Sylvester equation and energy denominators

At each perturbative order, Pymablock solves

$$
AX-XB=Y.
$$

Here $A$ and $B$ are the unperturbed Hamiltonian blocks, $Y$ is the known right-hand side at the requested order, and $X$ is the unknown off-diagonal correction.
In eigenbases of $A$ and $B$, each matrix element obeys

$$
(a_i-b_j)X_{ij}=Y_{ij}.
$$

The Sylvester equation therefore applies the usual perturbative energy denominators without constructing either eigenbasis.
It has a unique solution when the spectra of $A$ and $B$ do not overlap, and it becomes ill-conditioned when the two spectra approach each other.

The full-MPO and implicit-MPS formulations solve this same equation for different representations of $X$.

### Full-MPO solve with GMRES

Row-major vectorization rewrites the equation as one linear system:

$$
\left(A\otimes I-I\otimes B^T\right)\operatorname{vec}(X)
=\operatorname{vec}(Y).
$$

The transpose follows from right multiplication under this vectorization convention; it is not an adjoint.
The TeNPy example represents $\operatorname{vec}(X)$ as an MPS with local dimension $d^2$ and applies the linear map

$$
\mathcal{L}(X)=AX-XB
$$

as an MPO.
It never constructs the exponentially large matrix in the vectorized equation.

GMRES improves an estimate $X_0$ by minimizing the residual over the Krylov space

$$
R_0=Y-\mathcal{L}(X_0),
\qquad
\mathcal{K}_m
=\operatorname{span}\{R_0,\mathcal{L}(R_0),\ldots,
\mathcal{L}^{m-1}(R_0)\}.
$$

Repeated applications of $\mathcal{L}$ build an orthonormal basis of this space.
GMRES then chooses $\Delta X\in\mathcal{K}_m$ that minimizes

$$
\left\|R_0-\mathcal{L}(\Delta X)\right\|_F.
$$

Only the small projected least-squares problem is solved as a dense matrix.
After applying the correction, GMRES restarts from the new residual to limit the number of stored operator-space MPSs.
This method needs only applications of $\mathcal{L}$, MPS linear combinations, and inner products.
It also applies when the Sylvester map is indefinite or non-normal.

Compression after MPO application and MPS addition breaks the exact Krylov relations assumed by ordinary GMRES.
The implementation therefore recomputes the full residual $\|AX-XB-Y\|_F$ after every restart and uses that value, rather than the small least-squares estimate, to decide convergence.

### Implicit MPS solve

When the retained space contains only a few eigenstates, we need only the columns of $X$ that act on those states.
The Sylvester equation then reduces to

$$
Q(H_0-E_a)Q\lvert\eta_a\rangle=\lvert S_a\rangle,
\qquad
\langle\phi_b\vert\eta_a\rangle=0.
$$

The orthonormal eigenstates $\{\lvert\phi_a\rangle\}$ span the retained space $P$, $E_a$ is the corresponding unperturbed energy, $Q=1-P$, and $\lvert\eta_a\rangle=QX\lvert\phi_a\rangle$.
Each source $\lvert S_a\rangle$ comes from the corresponding column of $Y$.
The implicit formulation solves one response state $\lvert\eta_a\rangle$ per retained state instead of representing the full operator $X$.

{autolink}`~pymablock.implicit.ImplicitBlock` stores $P$--$P$ blocks as small dense matrices, $Q$--$P$ blocks as MPS bundles, and $Q$--$Q$ blocks as lazy projected operator applications.
Applying $Q$ subtracts overlaps with the reference MPSs, so the code never constructs a projector MPO or a basis for the complement.

For a Hermitian, sign-definite projected shifted Hamiltonian, the response equation is the stationary condition of the Hylleraas functional

$$
\mathcal{F}[\eta]
=\frac12\langle\eta|(H_0-E_a)|\eta\rangle
-\operatorname{Re}\langle\eta|S_a\rangle.
$$

The TeNPy solver varies two neighboring MPS tensors while holding the others fixed, as in a two-site DMRG sweep.
It is a linear-response solve, not a ground-state search: each local update solves the response equation with explicit constraints that enforce orthogonality to every reference state.
An SVD then moves the optimization center and truncates the MPS.
After each sweep, the code projects the state again and measures the global residual and reference-state overlaps.

The solver acts directly with $H_0-E_a$ rather than squaring it, which avoids squaring the condition number and increasing the MPO bond dimension.
It reports singular local equations instead of adding a pseudoinverse, broadening, or penalty projector.
General indefinite or non-Hermitian projected equations require a different local solver.

## Residual and truncation control

MPO compression and iterative linear solves turn an exact formal series into a numerical approximation.
We must therefore test both the tensor representation and the Sylvester solve.

The full-MPO solver returns a {autolink}`~pymablock.mpo.SylvesterResult` with the compressed-operator residual

$$
r_\mathrm{MPO} =
\frac{\lVert AX-XB-Y\rVert_F}{\lVert Y\rVert_F}.
$$

The implicit solver returns a {autolink}`~pymablock.implicit.StateSolveResult` with

$$
r_\mathrm{MPS} =
\frac{\lVert Q(H_0-E_a)Q\lvert\eta_a\rangle-\lvert S_a\rangle\rVert}
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
The requested effective-Hamiltonian coefficients must remain stable under these changes.
A small global residual is necessary but does not by itself test the MPO or MPS truncation error.

## Applicability and limitations

The TeNPy examples target finite open chains with square local spaces and `conserve=None`.
The full-MPO path requires separated block spectra.
The implicit path requires a small set of orthonormal MPS eigenstates; the demonstrated Hylleraas sweep targets Hermitian, sign-definite shifted problems.
Neither path introduces spectral broadening or a pseudoinverse.

Custom Sylvester solvers cannot currently be combined with `fully_diagonalize`.
The examples also do not support infinite tensor networks, charge-conserving tensor legs, or a guarantee that the required bond dimension stays bounded at high perturbative order.
These restrictions are explicit so that later backends can extend them without changing Pymablock's core algebra interface.

The [minimal TeNPy MPO tutorial](tutorial/tenpy_mpo.md) compares a complete two-site calculation with dense Pymablock.
The [Ising-chain benchmark](tutorial/tenpy_mpo_ising.md) applies the method to a six-site MPO problem with an analytical effective Hamiltonian.
The [implicit MPS tutorial](tutorial/tenpy_implicit_ising.md) treats the exactly solvable transverse-field Ising chain using two retained product-state MPSs.
