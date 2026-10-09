import Pymablock.LeastAction.Norm
import Pymablock.LeastAction.Formal

noncomputable section
set_option backward.isDefEq.respectTransparency false
open scoped ComplexOrder Matrix.Norms.Frobenius Topology

namespace Pymablock.LeastAction.Tests

/-- A six-state example with three blocks of unequal sizes. -/
def threeBlocks (i : Fin 6) : Fin 3 := if i < 1 then 0 else if i < 3 then 1 else 2

example (U T : Matrix (Fin 6) (Fin 6) ℂ)
    (hU : star U * U = 1) (hU' : U * star U = 1)
    (hT : star T * T = 1) (hT' : T * star T = 1)
    (ha : SameAssignment threeBlocks U T) (hp : (MatrixBlocks.diag threeBlocks U).PosDef) :
    ‖U - 1‖ ≤ ‖T - 1‖ ∧ (‖T - 1‖ = ‖U - 1‖ ↔ T = U) :=
  closest_to_identity_norm threeBlocks U T hU hU' hT hT' ha hp

/-- Any finite number of independent real perturbations is admitted by the
local theorem; continuity, exact unitarity, and the realized gauge are explicit. -/
example (k : ℕ) (U : (Fin k → ℝ) → Matrix (Fin 6) (Fin 6) ℂ)
    (hc : ContinuousAt U 0) (h0 : U 0 = 1)
    (hg : ∀ᶠ x in nhds 0, MatrixBlocks.diag threeBlocks (skew (U x - 1)) = 0)
    (hu : ∀ᶠ x in nhds 0, star (U x) * U x = 1 ∧ U x * star (U x) = 1) :
    ∀ᶠ x in nhds 0, ∀ T : Matrix (Fin 6) (Fin 6) ℂ,
      star T * T = 1 → T * star T = 1 → SameAssignment threeBlocks (U x) T →
      frobeniusSq (U x - 1) ≤ frobeniusSq (T - 1) ∧
        (frobeniusSq (T - 1) = frobeniusSq (U x - 1) ↔ T = U x) :=
  locally_closest_to_identity 0 threeBlocks U hc h0 hg hu

/-- A nontrivial rational rotation on the positive branch. -/
def rotation : Matrix (Fin 2) (Fin 2) ℂ := !![3/5, 4/5; -4/5, 3/5]

theorem rotation_unitary : star rotation * rotation = 1 ∧ rotation * star rotation = 1 := by
  constructor <;> ext i j <;> fin_cases i <;> fin_cases j <;>
    norm_num [rotation, Matrix.mul_apply, Matrix.star_apply, Fin.sum_univ_two, map_ofNat]

theorem rotation_positive : (MatrixBlocks.diag (fun i : Fin 2 => i) rotation).PosDef := by
  have he : MatrixBlocks.diag (fun i : Fin 2 => i) rotation =
      Matrix.diagonal (fun _ => (3/5 : ℂ)) := by
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [rotation]
  rw [he, Matrix.posDef_diagonal_iff]
  intro i
  norm_num

example (T : Matrix (Fin 2) (Fin 2) ℂ)
    (ht : star T * T = 1) (ht' : T * star T = 1)
    (ha : SameAssignment (fun i : Fin 2 => i) rotation T) :
    ‖rotation - 1‖ ≤ ‖T - 1‖ ∧ (‖T - 1‖ = ‖rotation - 1‖ ↔ T = rotation) :=
  closest_to_identity_norm _ _ _ rotation_unitary.1 rotation_unitary.2 ht ht' ha rotation_positive

/-- The gauge alone does not select the minimum: -I satisfies it and has the
same subspaces as I, but is farther away. This is why positivity is essential. -/
example :
    let U : Matrix (Fin 2) (Fin 2) ℂ := -1
    star U * U = 1 ∧
      MatrixBlocks.diag (fun i : Fin 2 => i) (skew (U - 1)) = 0 ∧
      SameAssignment (fun i : Fin 2 => i) U 1 ∧
      frobeniusSq ((1 : Matrix (Fin 2) (Fin 2) ℂ) - 1) < frobeniusSq (U - 1) := by
  dsimp only
  have hn : star (-1 : Matrix (Fin 2) (Fin 2) ℂ) * -1 = 1 := by
    rw [star_neg, star_one, neg_mul_neg, one_mul]
  refine ⟨hn, ?_, ?_, ?_⟩
  · rw [skew_of_selfadjoint _ (by rw [star_sub, star_neg, star_one])]
    exact map_zero _
  · intro a
    rw [star_neg, star_one, one_mul, mul_one, neg_mul, one_mul, mul_neg,
      mul_one, neg_neg]
  · rw [frobeniusSq_sub_one _ hn]
    norm_num [frobeniusSq, Matrix.trace_neg, Matrix.trace_one]

end Pymablock.LeastAction.Tests
