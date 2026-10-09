import Pymablock

/-! Correspondence metadata is separate from the mathematical definitions.
Labels locate manuscript equations; the declaration names are checked by Lean.
The registry records relations to those equations, not replacement formulas. -/
namespace Pymablock.Manuscript
open Lean

structure Occurrence where
  label : String
  declaration : Name
  relation : String

/-- The proof covers ordinary block partitions. The manuscript additionally
allows arbitrary masks, which are recorded as a boundary in the report. -/
def occurrences : Array Occurrence := #[
  ⟨"eq:problem_definition", ``Pymablock.matrix_block_diagonalization,
    "Correctness for arbitrary block partitions and arbitrary parameter index types."⟩,
  ⟨"eq:U", ``Pymablock.herm_add_skew, "Hermitian and anti-Hermitian decomposition."⟩,
  ⟨"eq:unitarity", ``Pymablock.blockDiagonalize_correct,
    "Both unitary identities for the constructed formal-series output."⟩,
  ⟨"eq:W", ``Pymablock.wSeries, "Definition of the W update."⟩,
  ⟨"eq:XYZ", ``Pymablock.recurrence_x_commutator,
    "The optimized auxiliary X equals the commutator, derived from the recurrence."⟩,
  ⟨"eq:H_tilde", ``Pymablock.conjugation_identity,
    "Algebraic identity under the stated unitarity premise."⟩,
  ⟨"eq:Z", ``Pymablock.optimized_x_skew,
    "Twice the anti-Hermitian part of X, derived from the B recurrence."⟩,
  ⟨"eq:sylvester", ``Pymablock.SylvesterSolver.series_equation,
    "Coefficientwise Sylvester contract; solved concretely by spectralSolver."⟩,
  ⟨"eq:B_offdiag", ``Pymablock.off_bUpdate,
    "Off-diagonal B update for ordinary block partitions."⟩,
  ⟨"eq:B_diag", ``Pymablock.bUpdate,
    "Specialization to block partitions, where the diagonal commutator term vanishes."⟩,
  ⟨"eq:H_tilde_optimized", ``Pymablock.effective,
    "Specialization to block partitions; equality to U-adjoint H U is proved."⟩]

/-- Terminal results are exported with their actual checked theorem types,
proposition binders, transitive project dependencies, and kernel axioms. -/
def roots : Array Name := #[
  ``Pymablock.matrix_block_diagonalization,
  ``Pymablock.blockDiagonalize_correct,
  ``Pymablock.solution_unique,
  ``Pymablock.truncated_correct,
  ``Pymablock.solution_first_order,
  ``Pymablock.recurrence_x_commutator]

end Pymablock.Manuscript
