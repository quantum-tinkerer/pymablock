import Pymablock

noncomputable section
namespace Pymablock.NonHermitian.Tests
open MvPowerSeries

/-- One-sided elimination is allowed: this mask is not symmetric. -/
def keepLower (i j : Fin 2) : Prop := i = j ∨ (i = 1 ∧ j = 0)
instance : DecidableRel keepLower := fun _ _ => inferInstanceAs (Decidable (_ ∨ _))

/-- Complex energies, multiple perturbations, and no Hermiticity premise. -/
example (H : MvPowerSeries (Fin 2) (Matrix (Fin 2) (Fin 2) ℂ))
    (h0 : H 0 = Matrix.diagonal (fun i => (i.val : ℂ)*Complex.I)) :
    ∃ ht u ui : MvPowerSeries (Fin 2) (Matrix (Fin 2) (Fin 2) ℂ),
      u 0 = 1 ∧ ui 0 = 1 ∧ ui*u = 1 ∧ u*ui = 1 ∧ ht = ui*H*u ∧
      (MatrixModel.projection keepLower).series.remaining ht = 0 ∧
      (MatrixModel.projection keepLower).series.selected ((1/2 : ℚ) • (u-ui)) = 0 := by
  apply matrix_nonhermitian_diagonalization keepLower (fun i => Or.inl rfl)
    (fun i => (i.val : ℂ)*Complex.I) H h0
  intro i j h
  fin_cases i <;> fin_cases j <;> simp_all [keepLower, Complex.ext_iff]

abbrev Two := Matrix (Fin 2) (Fin 2) ℚ
def perturbation : MvPowerSeries Unit Two :=
  monomial (Finsupp.single () 1) !![0, 1; 2, 0]
def pairSolution : State Unit Two :=
  solution (MatrixModel.projection (fun i j : Fin 2 => i = j))
    (MatrixModel.spectralMap (fun i j : Fin 2 => i = j) (fun i => (i.val : ℚ))) 0 perturbation

/-- Actual constructed coefficients for H = [[0, λ], [2λ, 1]]. -/
theorem first_order_values :
    qSeries pairSolution (Finsupp.single () 1) = !![0, 1; -2, 0] ∧
      gSeries pairSolution (Finsupp.single () 1) = !![0, -1; 2, 0] := by
  have hzero : perturbation 0 = 0 := by
    change coeff 0 (monomial (Finsupp.single () 1) (!![0, 1; 2, 0] : Two)) = 0
    rw [coeff_monomial]
    have hn : (0 : Unit →₀ ℕ) ≠ Finsupp.single () 1 := by
      intro h
      have he := congrArg (fun m : Unit →₀ ℕ => m ()) h
      simp at he
    simp [hn]
  have hc : perturbation (Finsupp.single () 1) = !![0, 1; 2, 0] := by
    change coeff (Finsupp.single () 1) (monomial (Finsupp.single () 1) (!![0, 1; 2, 0] : Two)) = _
    simp
  have he := solution_first_order (MatrixModel.projection (fun i j : Fin 2 => i = j))
    (MatrixModel.spectralMap (fun i j : Fin 2 => i = j) (fun i => (i.val : ℚ)))
    0 perturbation (by simp) hzero (Finsupp.single () 1) (by simp)
  change qSeries pairSolution (Finsupp.single () 1) = _ ∧
    gSeries pairSolution (Finsupp.single () 1) = _ at he
  rw [hc] at he
  constructor
  · rw [he.1]
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [MatrixModel.spectralMap_apply]
  · rw [he.2]
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [MatrixModel.spectralMap_apply]

/-- The inverse produced by the recurrence is demonstrably not the adjoint. -/
example : gSeries pairSolution (Finsupp.single () 1) ≠
    star (qSeries pairSolution (Finsupp.single () 1)) := by
  rw [first_order_values.1, first_order_values.2]
  intro h
  have he := congrArg (fun m : Two => m 0 1) h
  norm_num [Matrix.star_apply] at he

end Pymablock.NonHermitian.Tests
