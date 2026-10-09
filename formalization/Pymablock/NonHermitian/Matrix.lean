import Pymablock.NonHermitian.Correctness
import Mathlib.Data.Complex.Basic

noncomputable section
namespace Pymablock.NonHermitian
namespace MatrixModel
variable {ι R : Type*} [Fintype ι] [DecidableEq ι] [Field R] [CharZero R]

/-- Arbitrary retained-entry masks; no symmetry condition is needed. -/
def projection (keep : ι → ι → Prop) [DecidableRel keep] : Projection (Matrix ι ι R) where
  selected := {
    toFun := fun a i j => if keep i j then a i j else 0
    map_add' := by intro a b; ext i j; by_cases h : keep i j <;> simp [h]
    map_smul' := by intro c a; ext i j; by_cases h : keep i j <;> simp [h] }
  idempotent a := by
    ext i j
    change (if keep i j then (if keep i j then a i j else 0) else 0) =
      (if keep i j then a i j else 0)
    split_ifs <;> rfl

/-- Matrix elements are divided by E_j - E_i, matching [V,H₀] = source. -/
def spectralMap (keep : ι → ι → Prop) [DecidableRel keep] (e : ι → R) : Matrix ι ι R →ₗ[ℚ] Matrix ι ι R where
  toFun y i j := if keep i j then 0 else y i j / (e j - e i)
  map_add' a b := by
    ext i j
    by_cases h : keep i j <;> simp [h, add_div]
  map_smul' c a := by
    ext i j
    by_cases h : keep i j
    · simp only [h, ite_true, Matrix.smul_apply]
      change (0 : R) = c • (0 : R)
      exact (smul_zero c : c • (0 : R) = 0).symm
    · simp [h, div_eq_mul_inv]

omit [Fintype ι] [DecidableEq ι] in
@[simp] theorem spectralMap_apply (keep : ι → ι → Prop) [DecidableRel keep] (e : ι → R)
    (y : Matrix ι ι R) (i j : ι) :
    spectralMap keep e y i j = if keep i j then 0 else y i j / (e j - e i) := rfl

/-- Energy separation is required only at the entries being eliminated. -/
def spectralSolver (keep : ι → ι → Prop) [DecidableRel keep]
    (e : ι → R)
    (gap : ∀ i j, ¬ keep i j → e i ≠ e j) :
    Solver (projection keep) (Matrix.diagonal e) where
  solve := spectralMap keep e
  selected_zero y := by
    ext i j
    change (if keep i j then spectralMap keep e y i j else 0) = 0
    simp only [spectralMap_apply]
    split_ifs <;> rfl
  equation y := by
    ext i j
    simp only [comm, Matrix.sub_apply, Matrix.mul_diagonal, Matrix.diagonal_mul,
      spectralMap_apply]
    change (if keep i j then 0 else y i j / (e j - e i)) * e j -
      e i * (if keep i j then 0 else y i j / (e j - e i)) =
        y i j - (if keep i j then y i j else 0)
    by_cases h : keep i j
    · simp only [if_pos h, zero_mul, mul_zero, sub_self]
    · simp only [if_neg h, sub_zero]
      have hg : e j - e i ≠ 0 := sub_ne_zero.mpr (Ne.symm (gap i j h))
      field_simp


end MatrixModel
open MvPowerSeries
variable {σ ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Complex energies, arbitrary (possibly asymmetric) masks, and no
Hermiticity assumption. Diagonal entries must be retained. -/
theorem matrix_nonhermitian_diagonalization
    (keep : ι → ι → Prop) [DecidableRel keep] (reflexive : ∀ i, keep i i)
    (e : ι → ℂ) (H : MvPowerSeries σ (Matrix ι ι ℂ))
    (h0 : H 0 = Matrix.diagonal e)
    (gap : ∀ i j, ¬ keep i j → e i ≠ e j) :
    ∃ ht u ui : MvPowerSeries σ (Matrix ι ι ℂ),
      u 0 = 1 ∧ ui 0 = 1 ∧ ui*u = 1 ∧ u*ui = 1 ∧ ht = ui*H*u ∧
      (MatrixModel.projection keep).series.remaining ht = 0 ∧
      (MatrixModel.projection keep).series.selected ((1/2 : ℚ) • (u-ui)) = 0 := by
  classical
  let P : Projection (Matrix ι ι ℂ) := MatrixModel.projection keep
  let S : Solver P (H 0) := by
    rw [h0]
    exact MatrixModel.spectralSolver keep e gap
  have hd : P.selected (H 0) = H 0 := by
    rw [h0]
    ext i j
    change (if keep i j then Matrix.diagonal e i j else 0) = Matrix.diagonal e i j
    by_cases h : i = j
    · subst j; simp [reflexive]
    · simp [h]
  exact ⟨(diagonalize P H S).1, (diagonalize P H S).2.1,
    (diagonalize P H S).2.2, diagonalize_correct P H hd S⟩

/-- Non-Hermitian block diagonalization is a specialization of entry selection. -/
theorem matrix_block_diagonalization {β : Type*} [DecidableEq β]
    (block : ι → β) (e : ι → ℂ) (H : MvPowerSeries σ (Matrix ι ι ℂ))
    (h0 : H 0 = Matrix.diagonal e)
    (gap : ∀ i j, block i ≠ block j → e i ≠ e j) :
    ∃ ht u ui : MvPowerSeries σ (Matrix ι ι ℂ),
      u 0 = 1 ∧ ui 0 = 1 ∧ ui*u = 1 ∧ u*ui = 1 ∧ ht = ui*H*u ∧
      (MatrixModel.projection (fun i j => block i = block j)).series.remaining ht = 0 ∧
      (MatrixModel.projection (fun i j => block i = block j)).series.selected
        ((1/2 : ℚ) • (u-ui)) = 0 :=
  matrix_nonhermitian_diagonalization (fun i j => block i = block j)
    (fun _ => rfl) e H h0 gap

end Pymablock.NonHermitian
