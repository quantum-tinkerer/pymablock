import Pymablock

noncomputable section
open Pymablock MvPowerSeries
namespace Pymablock.Tests

/-- Eliminate only the 0-1 pair; retain 0-2 and 1-2.
The retained relation is not transitive, so this is not a block partition. -/
def keepPair (i j : Fin 3) : Prop := ¬ ((i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 0))
instance : DecidableRel keepPair := fun _ _ => inferInstanceAs (Decidable (¬ _))

theorem keepPair_symm (i j : Fin 3) : keepPair i j ↔ keepPair j i := by
  unfold keepPair
  tauto

theorem keepPair_refl (i : Fin 3) : keepPair i i := by
  fin_cases i <;> decide

example : ¬ (∀ i j k, keepPair i j → keepPair j k → keepPair i k) := by
  intro h
  have he := h 0 2 1 (by decide) (by decide)
  exact (by decide : ¬ keepPair 0 1) he

/-- Retained degeneracy between states 0 and 2 is allowed. -/
example (H : MvPowerSeries (Fin 2) (Matrix (Fin 3) (Fin 3) ℂ))
    (hH : star H = H)
    (h0 : H 0 = Matrix.diagonal (fun i => ((if i = 1 then 1 else 0 : ℝ) : ℂ))) :
    ∃ ht u : MvPowerSeries (Fin 2) (Matrix (Fin 3) (Fin 3) ℂ),
      u 0 = 1 ∧ star u * u = 1 ∧ u * star u = 1 ∧ ht = star u * H * u ∧
      (MatrixSelection.selection keepPair keepPair_symm).series.off ht = 0 ∧
      star ht = ht ∧
      (MatrixSelection.selection keepPair keepPair_symm).series.diag (skew (u - 1)) = 0 := by
  apply matrix_selective_diagonalization keepPair keepPair_symm keepPair_refl
    (fun i => if i = 1 then 1 else 0) H hH h0
  intro i j h
  fin_cases i <;> fin_cases j <;> simp_all [keepPair]

/-- The selective correction is nonzero on a retained entry. Omitting it,
as in the block-only recurrence, would make this equality false. -/
example :
    bUpdate (MatrixSelection.selection (R := ℚ) keepPair keepPair_symm) 0 0
      (comm (!![0, 1, 0; -1, 0, 0; 0, 0, 0])
        (!![0, 0, 1; 0, 0, 1; 1, 1, 0])) 0 2 = 1 := by
  norm_num [bUpdate, MatrixSelection.selection, Selection.off, herm, skew,
    comm, Matrix.mul_apply, Matrix.star_apply, Fin.sum_univ_succ, keepPair, Matrix.cons_val_two, Matrix.vecHead, Matrix.vecTail]
  decide

end Pymablock.Tests
