import Pymablock

/-! Regression examples exercise the public theorem at different dimensions.
They use kernel-checked proofs, without native_decide. -/
noncomputable section
open MvPowerSeries Pymablock

namespace Pymablock.Tests

/-- The usual two-level, single-parameter problem is an instance. -/
example (H : MvPowerSeries (Fin 1) (Matrix (Fin 2) (Fin 2) ℂ))
    (hH : star H = H) (h0 : H 0 = Matrix.diagonal (fun i => (i.val : ℂ))) :
    ∃ ht u : MvPowerSeries (Fin 1) (Matrix (Fin 2) (Fin 2) ℂ),
      u 0 = 1 ∧ star u * u = 1 ∧ u * star u = 1 ∧ ht = star u * H * u ∧
      (MatrixBlocks.blockStructure id).series.off ht = 0 ∧ star ht = ht ∧
      (MatrixBlocks.blockStructure id).series.diag (skew (u - 1)) = 0 := by
  apply matrix_block_diagonalization id (fun i => (i.val : ℝ)) H hH
  · simpa using h0
  · intro i j hij he
    exact hij (Fin.ext (by exact_mod_cast he))

/-- Three blocks and two perturbations, including all mixed orders. -/
example (H : MvPowerSeries (Fin 2) (Matrix (Fin 3) (Fin 3) ℂ))
    (hH : star H = H) (h0 : H 0 = Matrix.diagonal (fun i => (i.val : ℂ))) :
    ∃ ht u : MvPowerSeries (Fin 2) (Matrix (Fin 3) (Fin 3) ℂ),
      u 0 = 1 ∧ star u * u = 1 ∧ u * star u = 1 ∧ ht = star u * H * u ∧
      (MatrixBlocks.blockStructure id).series.off ht = 0 ∧ star ht = ht ∧
      (MatrixBlocks.blockStructure id).series.diag (skew (u - 1)) = 0 := by
  apply matrix_block_diagonalization id (fun i => (i.val : ℝ)) H hH
  · simpa using h0
  · intro i j hij he
    exact hij (Fin.ext (by exact_mod_cast he))

/-- Three two-dimensional blocks, with degenerate energies in each block. -/
example (H : MvPowerSeries (Fin 3) (Matrix (Fin 3 × Fin 2) (Fin 3 × Fin 2) ℂ))
    (hH : star H = H)
    (h0 : H 0 = Matrix.diagonal (fun i => (i.1.val : ℂ))) :
    ∃ ht u : MvPowerSeries (Fin 3) (Matrix (Fin 3 × Fin 2) (Fin 3 × Fin 2) ℂ),
      u 0 = 1 ∧ star u * u = 1 ∧ u * star u = 1 ∧ ht = star u * H * u ∧
      (MatrixBlocks.blockStructure Prod.fst).series.off ht = 0 ∧ star ht = ht ∧
      (MatrixBlocks.blockStructure Prod.fst).series.diag (skew (u - 1)) = 0 := by
  apply matrix_block_diagonalization Prod.fst (fun i => (i.1.val : ℝ)) H hH
  · simpa using h0
  · intro i j hij he
    exact hij (Fin.ext (by exact_mod_cast he))

/-- The concrete solver has the sign required by [V,H₀] = source. -/
example : MatrixBlocks.spectralMap (id : Fin 2 → Fin 2) (fun i => (i.val : ℚ))
    (!![0, 1; 1, 0] : Matrix (Fin 2) (Fin 2) ℚ) = !![0, 1; -1, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [MatrixBlocks.spectralMap_apply]

end Pymablock.Tests
