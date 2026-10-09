import Pymablock.LeastAction.Local

noncomputable section
open scoped ComplexOrder Matrix.Norms.Frobenius

namespace Pymablock.LeastAction

variable {ι β : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq β]

/-- The same unique-minimum statement in mathlib's Frobenius norm itself,
rather than its square. -/
 theorem closest_to_identity_norm (block : ι → β) (U T : Matrix ι ι ℂ)
    (hU : star U * U = 1) (hU' : U * star U = 1)
    (hT : star T * T = 1) (hT' : T * star T = 1)
    (hassign : SameAssignment block U T) (hpos : (MatrixBlocks.diag block U).PosDef) :
    ‖U - 1‖ ≤ ‖T - 1‖ ∧ (‖T - 1‖ = ‖U - 1‖ ↔ T = U) := by
  obtain ⟨hm, he⟩ := closest_to_identity block U T hU hU' hT hT' hassign hpos
  simp only [frobeniusSq_eq_norm_sq] at hm he
  constructor
  · nlinarith [norm_nonneg (U - 1), norm_nonneg (T - 1)]
  · constructor
    · intro h
      exact he.mp (by rw [h])
    · rintro rfl
      rfl

end Pymablock.LeastAction
