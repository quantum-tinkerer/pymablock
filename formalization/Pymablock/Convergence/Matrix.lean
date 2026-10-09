import Pymablock.Convergence.Radius
import Pymablock.SpectralSolver
import Mathlib.Analysis.Matrix.Normed
import Mathlib.Analysis.Normed.Module.FiniteDimension

noncomputable section
open scoped Matrix.Norms.Frobenius
namespace Pymablock.Convergence

variable {ι β σ : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq β]

/-- Entry multiplication as a complex linear map on a finite matrix space. -/
def entryLinear (d : Matrix ι ι ℂ) : Matrix ι ι ℂ →ₗ[ℂ] Matrix ι ι ℂ where
  toFun a i j := d i j * a i j
  map_add' a b := by ext i j; simp [mul_add]
  map_smul' c a := by ext i j; simp [mul_left_comm]

 theorem entry_bound (d : Matrix ι ι ℂ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ a : Matrix ι ι ℂ, ‖entryLinear d a‖ ≤ C * ‖a‖ := by
  let L := (entryLinear d).toContinuousLinearMap
  exact ⟨‖L‖, norm_nonneg _, fun a => L.le_opNorm a⟩

/-- Finiteness supplies all norm controls for the concrete block projection and
energy-denominator solver. The energy gap is needed for correctness, not this bound. -/
 theorem matrix_controls (block : ι → β) (e : ι → ℂ) :
    ∃ K, Controls (MatrixBlocks.blockStructure block) (MatrixBlocks.spectralMap block e) K := by
  classical
  obtain ⟨Cd,hCd,hd⟩ := entry_bound (fun i j => if block i = block j then 1 else 0)
  obtain ⟨Co,hCo,ho⟩ := entry_bound (fun i j => if block i = block j then 0 else 1)
  obtain ⟨Cs,hCs,hs⟩ := entry_bound (fun i j => if block i = block j then 0 else (e j-e i)⁻¹)
  let K := 1+Cd+Co+Cs
  have hdK : Cd ≤ K := by dsimp [K]; linarith
  have hoK : Co ≤ K := by dsimp [K]; linarith
  have hsK : Cs ≤ K := by dsimp [K]; linarith
  refine ⟨K,⟨by dsimp [K]; linarith, ?_, ?_, ?_⟩⟩
  · intro a
    have he : (MatrixBlocks.blockStructure block).diag a =
        entryLinear (fun i j => if block i = block j then 1 else 0) a := by
      ext i j
      change (if block i = block j then a i j else 0) =
        (if block i = block j then 1 else 0)*a i j
      split_ifs <;> simp
    rw [he]
    exact (hd a).trans (mul_le_mul_of_nonneg_right hdK (norm_nonneg _))
  · intro a
    have he : (MatrixBlocks.blockStructure block).off a =
        entryLinear (fun i j => if block i = block j then 0 else 1) a := by
      ext i j
      change a i j - (if block i = block j then a i j else 0) =
        (if block i = block j then 0 else 1)*a i j
      split_ifs <;> simp
    rw [he]
    exact (ho a).trans (mul_le_mul_of_nonneg_right hoK (norm_nonneg _))
  · intro a
    have he : MatrixBlocks.spectralMap block e a =
        entryLinear (fun i j => if block i = block j then 0 else (e j-e i)⁻¹) a := by
      ext i j
      simp only [MatrixBlocks.spectralMap_apply, entryLinear, LinearMap.coe_mk, AddHom.coe_mk]
      split_ifs <;> simp [div_eq_mul_inv, mul_comm]
    rw [he]
    exact (hs a).trans (mul_le_mul_of_nonneg_right hsK (norm_nonneg _))

/-- Convergence for the concrete finite-matrix algorithm, with any number of
blocks and perturbation indices. No output bounds or convergence premises occur. -/
 theorem matrix_solution_positive_radius (block : ι → β) (e : ι → ℂ)
    (hs hr : MvPowerSeries σ (Matrix ι ι ℂ)) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0)
    {R : ℝ} (hR : 0 < R) (hhs : Absolute hs R) (hhr : Absolute hr R) :
    ∃ r, 0 < r ∧ r ≤ R ∧
      Absolute (qSeries (solution (MatrixBlocks.blockStructure block)
        (MatrixBlocks.spectralMap block e) hs hr)) r ∧
      Absolute (bSeries (solution (MatrixBlocks.blockStructure block)
        (MatrixBlocks.spectralMap block e) hs hr)) r := by
  obtain ⟨K,hK⟩ := matrix_controls block e
  exact solution_positive_radius _ _ hK hs hr hs0 hr0 hR hhs hhr

end Pymablock.Convergence
