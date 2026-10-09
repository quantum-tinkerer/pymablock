import Pymablock

noncomputable section
open Pymablock MvPowerSeries

namespace Pymablock.Tests

abbrev TwoLevel := Matrix (Fin 2) (Fin 2) ℚ

def coupling : TwoLevel := !![0, 1; 1, 0]
def perturbation : MvPowerSeries Unit TwoLevel := monomial (Finsupp.single () 1) coupling

/-- For H = [[0, λ], [λ, 1]], the constructed first-order unitary correction
is [[0, 1], [-1, 0]]. This tests the actual infinite-series construction. -/
theorem two_level_first_order :
    qSeries (solution (MatrixBlocks.blockStructure (id : Fin 2 → Fin 2))
      (MatrixBlocks.spectralMap id (fun i => (i.val : ℚ))) 0 perturbation)
      (Finsupp.single () 1) = !![0, 1; -1, 0] := by
  have hzero : perturbation 0 = 0 := by
    change coeff 0 (monomial (Finsupp.single () 1) coupling) = 0
    rw [coeff_monomial]
    have hn : (0 : Unit →₀ ℕ) ≠ Finsupp.single () 1 := by
      intro h
      have he := congrArg (fun m : Unit →₀ ℕ => m ()) h
      simp at he
    simp [hn]
  have hstar : star perturbation = perturbation := by
    ext n i j
    simp only [coeff_star, perturbation, coeff_monomial]
    split_ifs
    · fin_cases i <;> fin_cases j <;> norm_num [coupling, Matrix.star_apply]
    · simp
  rw [solution_first_order (MatrixBlocks.blockStructure (id : Fin 2 → Fin 2))
    (MatrixBlocks.spectralMap id (fun i => (i.val : ℚ))) 0 perturbation
    (by simp) hzero hstar _ (by simp)]
  have hc : perturbation (Finsupp.single () 1) = coupling := by
    change coeff (Finsupp.single () 1) (monomial (Finsupp.single () 1) coupling) = coupling
    simp
  rw [hc]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [MatrixBlocks.spectralMap_apply, coupling]

end Pymablock.Tests
