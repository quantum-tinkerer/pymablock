# Lean formalization of the Pymablock algorithm

This directory proves correctness of the Hermitian block-partition recurrence in Lean 4 and mathlib.
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
These statements concern formal series and do not assume or establish analytic convergence.

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
| `Sylvester`, `SpectralSolver`, `MatrixTheorem` | Solver contract and its concrete realization from separated energies. |
| `Manuscript/Registry`, `Manuscript/Exports` | Links to manuscript labels and exports of checked types, hypotheses, definitions, and axioms. |
| `FirstOrder`, `Tests/TwoLevel` | Leading coefficient of the constructed unitary, with an explicit two-level Hamiltonian. |
| `Tests/Examples` | One parameter/two blocks, two parameters/three blocks, and three degenerate two-dimensional blocks; a solver-sign calculation. |

The export follows both theorem types and proofs through project declarations.
It includes structure constructors so that assumptions inside `BlockStructure` and `SylvesterSolver` remain visible.
The registry associates equations with checked declarations; it does not establish a formal equivalence between Lean and the text of the manuscript or Python implementation.

## Correspondence to the implementation

The construction uses the general branch of `pymablock/algorithms.py:main`, for ordinary block partitions (`commuting_blocks` true, `two_block_optimized` false).
We write `q = U'`, `herm(a) = (a + a†)/2`, and `skew(a) = (a - a†)/2`.
The update is

```text
A = H'_R q
B_new = -diag(skew(q† B) + herm(A)) - off(q† B)
X = B_new + H'_R + A
W = -q† q / 2
V = solve(herm(X) - [skew(q), H'_S])
q_new = W + V
H_tilde = H₀ + H'_S + diag(herm(A) - herm(q† B))
```

The solver convention is `[solve(Y), H₀] = off(Y)`.
In an eigenbasis this divides `Y_ij` by `E_j - E_i`.
Python's Sylvester solver uses the opposite commutator convention, so the minus sign in its `V` definition produces this same equation.
The new `B` is used inside the `q` update to resolve the same-degree dependency before advancing the total degree.
The proof establishes strict causality of this update and constructs each coefficient by finitely many iterations.

The crucial step is proving that `X` really is the commutator used in the algorithm's derivation.
Unitarity and the two adjoint parts of `X` imply a homogeneous recurrence for its defect.
The defect has only the zero formal-series solution.
Substitution then gives `H_tilde = H_S - (B + q† B)`, whose off-diagonal part vanishes by the `B` recurrence.

The formalization does not verify Python execution, lazy caching, numerical Sylvester solvers, floating-point errors, arbitrary element masks, the non-Hermitian algorithm, or the optional two-block fast path.
It also does not prove the complexity claims or uniqueness among every possible block-diagonalizer beyond the stated recurrence.

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

## Additional goal: least-action minimality

Prove that the gauge selects the unique unitary closest to the identity in Frobenius norm, for any finite block partition and a fixed invariant-subspace assignment.
This is an explicit additional goal, not a completed result of this formalization.

For an actual unitary matrix with positive-definite diagonal blocks, a block-diagonal unitary `D` gives the certificate

```text
||U D - I||_F² - ||U - I||_F²
  = sum_a ||sqrt(U_aa) (D_a - I)||_F² >= 0.
```

The remaining proof must connect the formal gauge to positivity on a continuous or convergent branch near the identity and establish the scope of the competing transformations.
The manuscript's order-by-order norm argument needs a separate justification because the squared norm contains cross-order terms.
No claim about arbitrary element masks or every matrix norm is included.
See the [least-action construction](https://arxiv.org/html/2505.11167v1#S2.SS1) and [the distinction from an off-block generator](https://arxiv.org/html/2408.14637v1).
