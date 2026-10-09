import Pymablock

/-! Correspondence metadata is separate from the mathematical definitions.
Labels locate manuscript equations; the declaration names are checked by Lean.
The registry records relations to those equations, not replacement formulas. -/
namespace Pymablock.Manuscript
open Lean

structure Occurrence where
  source : String
  label : String
  declaration : Name
  relation : String

/-- Correspondence for block partitions and arbitrary symmetric entry masks. -/
def occurrences : Array Occurrence := #[
  ⟨"paper/algorithm.tex", "eq:problem_definition", ``Pymablock.matrix_block_diagonalization,
    "Correctness for arbitrary block partitions and arbitrary parameter index types."⟩,
  ⟨"paper/algorithm.tex", "eq:U", ``Pymablock.herm_add_skew, "Hermitian and anti-Hermitian decomposition."⟩,
  ⟨"paper/algorithm.tex", "eq:unitarity", ``Pymablock.blockDiagonalize_correct,
    "Both unitary identities for the constructed formal-series output."⟩,
  ⟨"paper/algorithm.tex", "eq:W", ``Pymablock.wSeries, "Definition of the W update."⟩,
  ⟨"paper/algorithm.tex", "eq:XYZ", ``Pymablock.recurrence_x_commutator,
    "The optimized auxiliary X equals the commutator, derived from the recurrence."⟩,
  ⟨"paper/algorithm.tex", "eq:H_tilde", ``Pymablock.conjugation_identity,
    "Algebraic identity under the stated unitarity premise."⟩,
  ⟨"paper/algorithm.tex", "eq:Z", ``Pymablock.optimized_x_skew,
    "Twice the anti-Hermitian part of X, derived from the B recurrence."⟩,
  ⟨"paper/algorithm.tex", "eq:sylvester", ``Pymablock.SylvesterSolver.series_equation,
    "Coefficientwise Sylvester contract; solved concretely by spectralSolver."⟩,
  ⟨"paper/algorithm.tex", "eq:B_offdiag", ``Pymablock.off_bUpdate,
    "Eliminated-entry B update for arbitrary symmetric masks."⟩,
  ⟨"paper/algorithm.tex", "eq:B_diag", ``Pymablock.bUpdate,
    "General B update including the retained commutator correction."⟩,
  ⟨"paper/algorithm.tex", "eq:H_tilde_optimized", ``Pymablock.effective,
    "General selective formula; equality to U-adjoint H U is proved."⟩,
  ⟨"docs/source/nonhermitian_algorithm.md", "nh:setup",
    ``Pymablock.NonHermitian.diagonalize_correct,
    "Constructed two-sided inverse, similarity transform, elimination, and gauge."⟩,
  ⟨"docs/source/nonhermitian_algorithm.md", "nh:W_rec",
    ``Pymablock.NonHermitian.wSeries, "The quadratic W update."⟩,
  ⟨"docs/source/nonhermitian_algorithm.md", "nh:Z_rec",
    ``Pymablock.NonHermitian.zSeries, "The optimized Z update, without products by H0."⟩,
  ⟨"docs/source/nonhermitian_algorithm.md", "nh:Htilde_B",
    ``Pymablock.NonHermitian.similarity, "The algebraic similarity identity."⟩,
  ⟨"docs/source/nonhermitian_algorithm.md", "nh:Y_S",
    ``Pymablock.NonHermitian.recurrence_y, "Y is the V commutator, derived from the recurrence."⟩,
  ⟨"docs/source/nonhermitian_algorithm.md", "nh:XAB_defs",
    ``Pymablock.NonHermitian.recurrence_x, "X is the U-prime commutator; the defect vanishes."⟩,
  ⟨"docs/source/nonhermitian_algorithm.md", "nh:closed_recs",
    ``Pymablock.NonHermitian.solution_recurrence, "The constructed fixed point satisfies the closed recurrence."⟩]

/-- Terminal results are exported with their actual checked theorem types,
proposition binders, transitive project dependencies, and kernel axioms. -/
def roots : Array Name := #[
  ``Pymablock.LeastAction.closest_to_identity_norm,
  ``Pymablock.LeastAction.distance_certificate,
  ``Pymablock.LeastAction.locally_closest_to_identity,
  ``Pymablock.LeastAction.constructed_retained_hermitian,
  ``Pymablock.NonHermitian.matrix_nonhermitian_diagonalization,
  ``Pymablock.NonHermitian.diagonalize_correct,
  ``Pymablock.NonHermitian.solution_unique,
  ``Pymablock.NonHermitian.truncated_correct,
  ``Pymablock.NonHermitian.recurrence_x,
  ``Pymablock.NonHermitian.solution_first_order,
  ``Pymablock.matrix_selective_diagonalization,
  ``Pymablock.MatrixSelection.eliminated_entries_iff,
  ``Pymablock.matrix_block_diagonalization,
  ``Pymablock.blockDiagonalize_correct,
  ``Pymablock.solution_unique,
  ``Pymablock.truncated_correct,
  ``Pymablock.solution_first_order,
  ``Pymablock.recurrence_x_commutator]

end Pymablock.Manuscript
