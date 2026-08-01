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
Following the notation of [the algorithm](algorithms.md), the full formulation stores each complete anti-Hermitian transformation coefficient $V_{\mathbf n}^{AB}$ as an MPO.
The implicit formulation stores only the columns of $V_{\mathbf n}^{IE}$ that act on eigenstates in the explicit subspace $E$.

| | Full MPO | Implicit MPS |
| --- | --- | --- |
| Unknown | the complete operator $V_{\mathbf n}^{AB}$ | the states $P_IV_{\mathbf n}^{IE}\lvert\phi_a\rangle$ |
| Local dimension | $d^2$ | $d$ |
| Linear solver in the examples | compressed GMRES | DMRG-style variational sweeps |
| Result | an effective MPO | a small effective matrix and corrected states |

Here $\mathbf n$ is a perturbative multi-index, the columns of $\Psi_E$ are the known eigenstates $\lvert\phi_a\rangle$, and $P_I=1-\Psi_E\Psi_E^\dagger$ projects onto the implicit subspace $I$.

Choose the implicit formulation when the explicit subspace $E$ contains a few known MPS eigenstates.
Only the columns that act on those states can enter their effective Hamiltonian, so constructing the rest of $V_{\mathbf n}^{IE}$ would do unnecessary work.
The cost is one corrected MPS per explicit state, which makes this approach unattractive for an extensive explicit subspace.

Choose the full-MPO formulation when the operator itself is the result.
This is appropriate when the selected subspace is extensive, or when the same effective Hamiltonian or transformation will be reused for many states, observables, dynamics, or later DMRG calculations.
Its cost is instead controlled by the operator entanglement of the perturbative coefficients.

As a practical rule, use implicit MPS for a few target states, full MPO for a reusable effective operator, and direct DMRG when only one ground state at one parameter value is needed.

## Sylvester equation and energy denominators

At each perturbative order $\mathbf n$, Pymablock determines the $AB$ block of $\mathcal V$ from

$$
H_0^{AA}V_{\mathbf n}^{AB}
-V_{\mathbf n}^{AB}H_0^{BB}
=F_{\mathbf n}^{AB}.
$$

The algorithm defines $\mathcal V$ as the anti-Hermitian part of $\mathcal U'$.
We introduce $F_{\mathbf n}^{AB}$ for the known right-hand side passed to `solve_sylvester`; in the notation of the algorithm,

$$
F_{\mathbf n}^{AB}
=-\left(\mathcal Y-[\mathcal V,\mathcal H'_S]\right)_{\mathbf n}^{AB}.
$$

In eigenbases of $H_0^{AA}$ and $H_0^{BB}$, each matrix element obeys

$$
\left(E_i^A-E_j^B\right)(V_{\mathbf n}^{AB})_{ij}
=(F_{\mathbf n}^{AB})_{ij}.
$$

The Sylvester equation therefore applies the usual perturbative energy denominators without constructing either eigenbasis.
It has a unique solution when the spectra of $H_0^{AA}$ and $H_0^{BB}$ do not overlap, and it becomes ill-conditioned when the two spectra approach each other.

The full-MPO and implicit-MPS formulations solve this same equation for different representations of $V_{\mathbf n}$.

### Full-MPO solve with GMRES

Row-major vectorization rewrites the equation as one linear system:

$$
\left(H_0^{AA}\otimes I-I\otimes (H_0^{BB})^T\right)
\operatorname{vec}(V_{\mathbf n}^{AB})
=\operatorname{vec}(F_{\mathbf n}^{AB}).
$$

The transpose follows from right multiplication under this vectorization convention; it is not an adjoint.
The TeNPy example represents $\operatorname{vec}(V_{\mathbf n}^{AB})$ as an MPS with local dimension $d^2$ and applies the linear map

$$
\mathcal{L}^{AB}(V_{\mathbf n}^{AB})
=H_0^{AA}V_{\mathbf n}^{AB}-V_{\mathbf n}^{AB}H_0^{BB}
$$

as an MPO.
It never constructs the exponentially large matrix in the vectorized equation.

GMRES improves an estimate $V_{\mathbf n,0}^{AB}$ by minimizing the residual over the Krylov space

$$
r_0=F_{\mathbf n}^{AB}-\mathcal{L}^{AB}(V_{\mathbf n,0}^{AB}),
\qquad
\mathcal{K}_m
=\operatorname{span}\{r_0,\mathcal{L}^{AB}(r_0),\ldots,
(\mathcal{L}^{AB})^{m-1}(r_0)\}.
$$

Repeated applications of $\mathcal{L}$ build an orthonormal basis of this space.
GMRES then chooses $\Delta V_{\mathbf n}^{AB}\in\mathcal{K}_m$ that minimizes

$$
\left\|r_0-\mathcal{L}^{AB}(\Delta V_{\mathbf n}^{AB})\right\|_F.
$$

Only the small projected least-squares problem is solved as a dense matrix.
After applying the correction, GMRES restarts from the new residual to limit the number of stored operator-space MPSs.
This method needs only applications of $\mathcal{L}$, MPS linear combinations, and inner products.
It also applies when the Sylvester map is indefinite or non-normal.

Compression after MPO application and MPS addition breaks the exact Krylov relations assumed by ordinary GMRES.
The implementation therefore recomputes the full residual
$\|H_0^{AA}V_{\mathbf n}^{AB}-V_{\mathbf n}^{AB}H_0^{BB}-F_{\mathbf n}^{AB}\|_F$
after every restart and uses that value, rather than the small least-squares estimate, to decide convergence.

### Implicit MPS solve

When the explicit subspace $E$ contains only a few eigenstates, we need only the corresponding columns of $V_{\mathbf n}^{IE}$.
The Sylvester equation then reduces to

$$
P_I(H_0-E_a)P_I\lvert v_{\mathbf n,a}\rangle
=\lvert f_{\mathbf n,a}\rangle,
\qquad
\langle\phi_b\vert v_{\mathbf n,a}\rangle=0.
$$

The orthonormal eigenstates $\{\lvert\phi_a\rangle\}$ are the columns of $\Psi_E$, and $E_a$ is the corresponding unperturbed energy.
The response and source states are
$\lvert v_{\mathbf n,a}\rangle=P_IV_{\mathbf n}^{IE}\lvert\phi_a\rangle$
and
$\lvert f_{\mathbf n,a}\rangle=P_IF_{\mathbf n}^{IE}\lvert\phi_a\rangle$.
The implicit formulation solves one response state $\lvert v_{\mathbf n,a}\rangle$ per explicit state instead of representing the full operator $V_{\mathbf n}^{IE}$.

{autolink}`~pymablock.implicit.ImplicitBlock` stores $EE$ blocks as small dense matrices, $IE$ blocks as MPS bundles, and $II$ blocks as lazy projected operator applications.
Applying $P_I$ subtracts overlaps with the reference MPSs, so the code never constructs a projector MPO or a basis for $I$.

For a Hermitian, sign-definite projected shifted Hamiltonian, the response equation is the stationary condition of the Hylleraas functional

$$
\mathcal{F}[v_{\mathbf n,a}]
=\frac12\langle v_{\mathbf n,a}|(H_0-E_a)|v_{\mathbf n,a}\rangle
-\operatorname{Re}\langle v_{\mathbf n,a}|f_{\mathbf n,a}\rangle.
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
\frac{
\lVert H_0^{AA}V_{\mathbf n}^{AB}
-V_{\mathbf n}^{AB}H_0^{BB}
-F_{\mathbf n}^{AB}\rVert_F
}{
\lVert F_{\mathbf n}^{AB}\rVert_F
}.
$$

The implicit solver returns a {autolink}`~pymablock.implicit.StateSolveResult` with

$$
r_\mathrm{MPS} =
\frac{
\lVert P_I(H_0-E_a)P_I\lvert v_{\mathbf n,a}\rangle
-\lvert f_{\mathbf n,a}\rangle\rVert
}{
\lVert f_{\mathbf n,a}\rVert
}.
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

The [full-MPO tutorial](tutorial/tenpy_mpo.md) first compares a four-site Ising-chain calculation with dense Pymablock and an analytical expansion, then obtains the same effective interactions for a 24-site system beyond dense operator storage.
The [implicit MPS tutorial](tutorial/tenpy_implicit_ising.md) treats the exactly solvable transverse-field Ising chain using two retained product-state MPSs.
