# Lean formalization of the Pymablock algorithm

This directory proves correctness of the Hermitian and non-Hermitian selective-diagonalization recurrences in Lean 4 and mathlib.
The development covers any finite number of blocks, arbitrary block sizes, any finite number of perturbation parameters, and every perturbative order.
In fact, the series results allow an arbitrary parameter index type: each monomial has finite support.

For a Hermitian matrix-valued formal series `H`, a block-diagonal constant coefficient, and a solvable inter-block Sylvester equation, `blockDiagonalize_correct` proves that the constructed outputs satisfy

```text
U(0) = I
U† U = U U† = I
H_tilde = U† H U
off(H_tilde) = 0
H_tilde† = H_tilde
diag(skew(U - I)) = 0
```

`matrix_block_diagonalization` supplies a concrete complex-matrix Sylvester solver from real eigenvalues of `H₀`, assuming only separation between different blocks.
Degeneracy within a block is permitted.
`solution_unique` proves uniqueness of the recursive coefficient construction.
`truncated_correct` proves the identities through any total degree `N`, including mixed terms, when the series are truncated.
These algebraic statements concern formal series. The separate convergence theorem below proves a positive radius for analytic finite-matrix inputs.

## Selective diagonalization

`matrix_selective_diagonalization` permits any symmetric, reflexive retained-entry relation `keep i j`, without transitivity.
The input is a coefficientwise Hermitian complex-matrix formal series with `H₀ = diag(E_i)` in a finite basis.
The only energy-separation requirement is `¬ keep i j → E_i ≠ E_j`; degeneracies on retained entries are allowed.
The constructed output has zero entries wherever `keep` is false, at every multi-index, and satisfies both unitary identities, Hermiticity, and the retained-entry gauge `diag(skew(U - I)) = 0`.
The same unique causal construction and finite-order theorem apply.
An arbitrary number of blocks can be combined with selective diagonalization inside them by retaining only same-block entries allowed by each block's mask.

`Selection` assumes only rational linearity, idempotence, and compatibility with the adjoint.
It imposes no multiplication or block-partition laws.
The concrete masked Sylvester solver proves the remaining solver contract from the entrywise energy gaps.
`Tests/Selective.lean` checks a nontransitive three-state mask with two perturbation parameters and retained degeneracy, and verifies that the extra retained commutator correction is nonzero in a concrete example.

## Non-Hermitian algorithm

`NonHermitian.matrix_nonhermitian_diagonalization` implements the [documented non-Hermitian recurrence](../docs/source/nonhermitian_algorithm.md).
It assumes no Hermiticity and no relation between the inverse and adjoint.
In a finite eigenbasis, `H₀ = diag(E_i)` may have complex eigenvalues.
The retained-entry mask must contain the diagonal but may be asymmetric and nontransitive.
Only eliminated entries require `E_i ≠ E_j`.
`NonHermitian.matrix_block_diagonalization` specializes this to any finite block partition.

The constructed outputs satisfy, as multivariate formal series at every order,

```text
U(0) = U_inv(0) = I
U_inv U = U U_inv = I
H_tilde = U_inv H U
remaining(H_tilde) = 0
selected((U - U_inv)/2) = 0
```

The abstract theorem `NonHermitian.diagonalize_correct` needs only a rational linear idempotent projection, `selected(H₀) = H₀`, and the explicit commutator-solver contract.
No star structure is needed in that theorem.
The matrix theorem constructs the solver from the complex energy denominators, discharging that contract.
The concrete eigenbasis theorem does not cover defective `H₀`; the abstract theorem can be used whenever an appropriate Sylvester solver is separately supplied and verified.
Construction of biorthogonal bases and the documentation's implicit oblique-projector solver remain outside the proof.

The state contains `q = U'`, `g = U_inv'`, and `B`, with `V = (q-g)/2` derived.
One update follows the documented equations:

```text
W = -g q / 2
A = H'_R q
B_plus = selected(B + g B)
Z = (A - g H'_R - g B - B_plus g) / 2
B_new = selected([V,H'_S] + Z - A) - remaining(g B)
Y = B_new + H'_R + A - Z
V_new = solve(Y - [V,H'_S])
q_new = W + V_new
g_new = W - V_new
H_tilde = H₀ + H'_S - B_plus
```

The proof establishes strict causality and a unique fixed point, then derives both inverse identities and the commutator identity `B + H'_R + H'_R q = [q,H_S]`.
The defect satisfies `2D = -(qD + Dg)` and therefore vanishes by total-degree induction.
This avoids assuming the correctness identities used to motivate the recurrence.
`NonHermitian.truncated_correct` proves both inverse identities and similarity through any finite total degree.
`Tests/NonHermitian.lean` checks complex energies, a one-sided mask, multiple parameters, and actual first-order coefficients for `H = [[0,λ],[2λ,1]]`, whose inverse correction differs from the adjoint correction.

## Reproduce

From the repository root:

```console
pixi install -e lean --locked
pixi run lean-setup
pixi run lean
```

The setup task installs the official elan manager if needed, without changing shell profiles.
Its Linux x86-64 archive has a pinned version and SHA-256 digest.
`lean-toolchain` pins Lean and `lake-manifest.json` locks mathlib and its dependencies.
Using the official Lean toolchain makes mathlib's matching compiled cache available.
Pixi isolates the report tooling from the numerical package environment.
All downloaded toolchains and `.lake/` build artifacts remain untracked.

`pixi run lean` builds the proof library, compiles the examples, audits the theorem dependencies, and writes the report.
It rejects proof dependencies on `sorryAx`, custom axioms, or native compiler-oracle axioms.
The only allowed foundational axioms are `propext`, `Classical.choice`, and `Quot.sound`.

The report is available as [HTML](.lake/reports/correspondence.html), [Markdown](.lake/reports/correspondence.md), and [JSON](.lake/reports/correspondence.json).
Regenerate it with `pixi run lean-correspondence`.

## Proof organization

The organization follows [qt/rmt_nlin](https://gitlab.kwant-project.org/qt/rmt_nlin/-/tree/t3code/assess-lean-formalization/formalization): small proof modules, a separate correspondence registry, and an exported assumptions ledger.
`Pymablock.lean` imports the mathematical development.

| Modules | Role |
| --- | --- |
| `Library/Recursion`, `Library/Series`, `Library/SeriesBlocks` | Total-degree filtration, multivariate Cauchy products, and finite coefficient construction. |
| `Library/Blocks`, `Library/Parts`, `Library/Algebra` | Matrix block projections, Hermitian parts, and noncommutative identities. |
| `Construction`, `Recurrence` | The actual optimized update, its unique fixed point, and the equations it satisfies. |
| `Optimized`, `Library/Vanishing`, `Invariants` | Unitarity, gauge, and the identity `X = [U', H_S]`. |
| `Correctness`, `Hamiltonian`, `FiniteOrder` | End-to-end theorems for the constructed outputs and their truncations. |
| `Sylvester`, `SpectralSolver`, `MatrixTheorem`, `Selective` | Solver contract and its concrete realization from separated energies. |
| `LeastAction/*` | Positive-block Frobenius minimality and local positivity. |
| `Convergence/*` | Absolute coefficient bounds, positive output radius, summation identities, and realized least action. |
| `NonHermitian/*` | Explicit inverse construction, non-Hermitian correctness, complex spectral solver, and truncations. |
| `Manuscript/Registry`, `Manuscript/Exports` | Links to manuscript labels and exports of checked types, hypotheses, definitions, and axioms. |
| `FirstOrder`, `Tests/TwoLevel` | Leading coefficient of the constructed unitary, with an explicit two-level Hamiltonian. |
| `Tests/Selective` | Nontransitive mask, retained degeneracy, and nonzero selective correction. |
| `Tests/Examples` | One parameter/two blocks, two parameters/three blocks, and three degenerate two-dimensional blocks; a solver-sign calculation. |

The export follows both theorem types and proofs through project declarations.
It includes structure constructors so that assumptions inside `Selection`, `SylvesterSolver`, and the non-Hermitian `Projection` and `Solver` remain visible.
The registry associates equations with checked declarations; it does not establish a formal equivalence between Lean and the text of the manuscript or Python implementation.

## Correspondence to the implementation

The construction uses the general branch of `pymablock/algorithms.py:main` with `two_block_optimized` false, including the selective correction used when `commuting_blocks` is false.
The names `diag` and `off` denote retained and eliminated entries; the retained mask need not be a block partition.
We write `q = U'`, `herm(a) = (a + a†)/2`, and `skew(a) = (a - a†)/2`.
The update is

```text
A = H'_R q
K = [skew(q), H'_S]
B_new = -diag(skew(q† B) + herm(A)) - off(q† B) + diag(herm(K))
X = B_new + H'_R + A
W = -q† q / 2
V = solve(herm(X) - [skew(q), H'_S])
q_new = W + V
H_tilde = H₀ + H'_S + diag(herm(A) - herm(q† B) - herm(K))
```

The solver convention is `[solve(Y), H₀] = off(Y)`.
In an eigenbasis this divides `Y_ij` by `E_j - E_i`.
Python's Sylvester solver uses the opposite commutator convention, so the minus sign in its `V` definition produces this same equation.
For Hermitian `H'_S`, `K` is Hermitian, so `herm(K) = K`; this is precisely Python's `V @ H'_diag + (V @ H'_diag).adj`.
For ordinary block partitions its retained part vanishes.
The new `B` is used inside the `q` update to resolve the same-degree dependency before advancing the total degree.
The proof establishes strict causality of this update and constructs each coefficient by finitely many iterations.

The crucial step is proving that `X` really is the commutator used in the algorithm's derivation.
Unitarity and the two adjoint parts of `X` imply a homogeneous recurrence for its defect.
The defect has only the zero formal-series solution.
Substitution then gives `H_tilde = H_S - (B + q† B)`, whose off-diagonal part vanishes by the `B` recurrence.

The formalization does not verify Python execution, lazy caching, numerical Sylvester solvers, floating-point errors, or the optional two-block fast path.
It also does not prove the complexity claims or uniqueness among every possible selective diagonalizer beyond the stated recurrence.

## Manuscript finding

At the base revision `59d9994f`, `paper/algorithm.tex` equation `eq:sylvester_optimized` prints `B - H' - A` in the Sylvester source.
This does not match the implementation or the preceding definition of `B`.
Already at first order it gives `-H'_R`, whereas the required source is `+H'_R`.
The checked equation is

```text
[V, H₀] = off(herm(B + H'_R + H'_R U') - [V, H'_S]).
```

The report records this as an unverified manuscript equation and explains the discrepancy.
This MR leaves the manuscript text unchanged.

## Convergence and actual sums

`Convergence.matrix_convergent` proves that both returned series `U` and `H_tilde` converge absolutely on a polydisc of some positive radius `r <= R`.
It uses the same finite complex-matrix basis, arbitrary block partition, Hermitian input, diagonal `H₀`, and cross-block energy separation as `matrix_block_diagonalization`.
The additional **input** assumption is

```text
R > 0,  sum_n ||H_n||_F R^|n| < infinity.
```

Here `n` is a multi-index and `|n|` its total degree; all mixed terms are included.
Polynomial inputs satisfy this automatically, as checked by `absolute_C`, `absolute_monomial`, and the example with two perturbations and three blocks.
An arbitrary formal series need not satisfy it.
Convergence of the output is a conclusion, not an assumption.

The proof uses absolute coefficient sums and the existing finite-iteration construction.
Let `K >= 1` bound the retained projection, its complement, and the Sylvester solver in Frobenius norm.
`matrix_controls` constructs such a finite bound for the concrete entrywise maps; it does not assume boundedness of the output.
If the positive-degree input parts each have weighted mass at most `epsilon`, and the iterated `q = U-I` and `B` each have mass at most `t`, one update obeys

```text
mass(B_new) <= 2 K t^2 + 3 K epsilon t
mass(q_new) <= t^2/2 + K (2 K t^2 + 3 K epsilon t + epsilon + 3 epsilon t).
```

Choosing `t = 1/(16 K^2)` and `epsilon = t^2` makes these bounds invariant.
Shrinking the input radius makes its positive-degree mass sufficiently small; this follows by dominated convergence.
Every finite set of output coefficients agrees with a sufficiently advanced finite iteration, so the uniform bounds pass to the actual constructed formal solution.
The result proves existence of a positive radius, not the optimal radius or convergence at all perturbation strengths.

`evaluate` sums the multivariate series at real parameter values.
The proof checks absolute summability, continuity on the closed polydisc, the Cauchy-product identity, and compatibility with adjoints.
`realized_unitary`, `realized_gauge`, and `realized_hamiltonian` transfer the formal identities to these sums, including both unitary identities, conjugation, and block elimination.
For any finite number of parameters, `matrix_least_action` then proves local closest-to-identity minimality directly from the analytic input assumptions; no separate continuous-realization or realized-unitarity assumption is needed.

The abstract convergence theorem applies to the Hermitian/selective recurrence whenever its linear maps satisfy the explicit norm bounds.
The concrete end-to-end theorem specializes to ordinary block partitions.
The non-Hermitian recurrence has additional terms and is not covered by this convergence proof.

## Least-action minimality

`LeastAction.closest_to_identity_norm` proves a unique global minimum in the **Frobenius norm** for any finite block partition and fixed assignment of invariant subspaces.
The matrices are finite complex matrices; blocks can have arbitrary sizes.
Its assumptions are:

- `U` and the competitor `T` are unitary.
- Each block is assigned the same subspace: `U P_a U† = T P_a T†` for every coordinate block projector `P_a`.
- `A = diag_blocks(U)` is positive definite: each Hermitian diagonal block has strictly positive eigenvalues. This is not entrywise positivity.

The conclusion is `||U - I||_F <= ||T - I||_F`, with equality **if and only if `T = U`**.
The competitor is arbitrary within that assignment, not just perturbatively close to `U`.
No Hamiltonian or spectral-gap premise is needed in this geometric theorem; those enter when constructing `U` and its assigned invariant subspaces.

`assignment_factor` proves that every such competitor has the form `T = U D`, with `D` block-diagonal and unitary.
`distance_certificate` proves the exact full-matrix certificate

```text
||U D - I||_F² - ||U - I||_F² = ||sqrt(A) (D - I)||_F²,  A = diag_blocks(U).
```

This is the block-matrix form of the sum over blocks `sum_a ||sqrt(U_aa) (D_a - I)||_F²`.
The implemented theorem uses the full retained matrix, so no enumeration of block labels is required.
`frobeniusSq_eq_norm_sq` identifies the explicit trace objective with mathlib's Frobenius norm squared.
Positivity makes the square root invertible, forcing `D = I` in the equality case.

### Connection to the perturbative gauge

`constructed_retained_hermitian` derives Hermiticity of every retained coefficient of the **constructed** formal output from `blockDiagonalize_correct`.
`locally_closest_to_identity` proves the local analytic statement for a matrix family `U(x)` continuous at the origin, with `U(0) = I`, satisfying exact unitarity and the realized gauge in a neighborhood.
The parameter space is arbitrary, including any finite number of real perturbations.
The gauge implies Hermitian retained blocks; continuity puts them on the positive branch near the origin.
The proof uses the sufficient condition `||A - I||_op < 1`, where this auxiliary norm is the operator norm, not the minimization objective.

For the finite Hermitian block problem with the analytic input assumption above, `Convergence.matrix_least_action` now discharges the realization assumptions: it proves convergence, continuity, and passage of the formal unitary and gauge identities through summation.
The original conditional local theorem remains available for other independently constructed families.
The exact finite-matrix certificate avoids the manuscript's order-by-order norm argument, whose cross-order terms require separate justification.
This proves minimality for block partitions on the positive branch; it does not assert it for arbitrary selective entry masks, non-Hermitian transformations, or every matrix norm.

`Tests/LeastAction.lean` checks three unequal blocks, arbitrarily many real parameters, a nontrivial rational rotation, and the necessity of the positive branch: `-I` satisfies the gauge and has the same subspaces as `I`, yet is farther from `I`.
