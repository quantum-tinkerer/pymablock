import Pymablock.Hamiltonian
import Mathlib.Data.Complex.Basic

/-! Concrete projection and Sylvester solver for arbitrary symmetric entry masks. -/
noncomputable section
namespace Pymablock.MatrixSelection

variable {ι R : Type*} [Fintype ι] [DecidableEq ι]
    [Field R] [CharZero R] [StarRing R]

/-- Keep any symmetric set of entries, without a transitivity requirement. -/
def selection (keep : ι → ι → Prop) [DecidableRel keep]
    (symmetric : ∀ i j, keep i j ↔ keep j i) : Selection (Matrix ι ι R) where
  diag := {
    toFun := fun a i j => if keep i j then a i j else 0
    map_add' := by intro a b; ext i j; by_cases h : keep i j <;> simp [h]
    map_smul' := by intro c a; ext i j; by_cases h : keep i j <;> simp [h] }
  idempotent a := by
    ext i j
    change (if keep i j then (if keep i j then a i j else 0) else 0) = (if keep i j then a i j else 0)
    split_ifs <;> rfl
  star_diag a := by
    ext i j
    change (if keep i j then star (a j i) else 0) = star (if keep j i then a j i else 0)
    by_cases h : keep i j
    · simp [h, (symmetric i j).mp h]
    · simp [h, show ¬ keep j i from fun hj => h ((symmetric i j).mpr hj)]

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

omit [Fintype ι] [DecidableEq ι] [StarRing R] in
@[simp] theorem spectralMap_apply (keep : ι → ι → Prop) [DecidableRel keep] (e : ι → R)
    (y : Matrix ι ι R) (i j : ι) :
    spectralMap keep e y i j = if keep i j then 0 else y i j / (e j - e i) := rfl

/-- Energy separation is required only at the entries being eliminated. -/
def spectralSolver (keep : ι → ι → Prop) [DecidableRel keep]
    (symmetric : ∀ i j, keep i j ↔ keep j i) (e : ι → R)
    (he : ∀ i, star (e i) = e i)
    (gap : ∀ i j, ¬ keep i j → e i ≠ e j) :
    SylvesterSolver (selection keep symmetric) (Matrix.diagonal e) where
  solve := spectralMap keep e
  diagonal_zero y := by
    ext i j
    change (if keep i j then spectralMap keep e y i j else 0) = 0
    simp only [spectralMap_apply]
    split_ifs <;> rfl
  adjoint y := by
    ext i j
    simp only [Matrix.star_apply, spectralMap_apply, Matrix.neg_apply]
    by_cases h : keep i j
    · simp only [if_pos h, if_pos ((symmetric i j).mp h), star_zero, neg_zero]
    · simp only [if_neg h, if_neg (fun hji => h ((symmetric i j).mpr hji)), star_div₀, star_sub, he]
      rw [show e i - e j = -(e j - e i) by ring, div_neg]
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

/-- Vanishing of the eliminated projection means every eliminated matrix
entry vanishes coefficientwise, with no restrictions on retained entries. -/
theorem eliminated_entries_iff {σ : Type*}
    (keep : ι → ι → Prop) [DecidableRel keep]
    (symmetric : ∀ i j, keep i j ↔ keep j i)
    (H : MvPowerSeries σ (Matrix ι ι R)) :
    (selection keep symmetric).series.off H = 0 ↔
      ∀ n i j, ¬ keep i j → H n i j = 0 := by
  constructor
  · intro h n i j hij
    have he := congrArg (fun f : MvPowerSeries σ (Matrix ι ι R) => f n i j) h
    change H n i j - (if keep i j then H n i j else 0) = 0 at he
    simpa only [if_neg hij, sub_zero] using he
  · intro h
    funext n i j
    change H n i j - (if keep i j then H n i j else 0) = 0
    by_cases hij : keep i j
    · simp only [if_pos hij, sub_self]
    · simp only [if_neg hij, sub_zero, h n i j hij]

end Pymablock.MatrixSelection

namespace Pymablock
open MvPowerSeries
variable {σ ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Arbitrary symmetric retained-entry masks, including nontransitive masks.
Only removed entries require separated unperturbed energies. -/
theorem matrix_selective_diagonalization
    (keep : ι → ι → Prop) [DecidableRel keep]
    (symmetric : ∀ i j, keep i j ↔ keep j i) (reflexive : ∀ i, keep i i)
    (e : ι → ℝ) (H : MvPowerSeries σ (Matrix ι ι ℂ))
    (hH : star H = H) (h0 : H 0 = Matrix.diagonal (fun i => (e i : ℂ)))
    (gap : ∀ i j, ¬ keep i j → e i ≠ e j) :
    ∃ ht u : MvPowerSeries σ (Matrix ι ι ℂ),
      u 0 = 1 ∧ star u * u = 1 ∧ u * star u = 1 ∧ ht = star u * H * u ∧
      (MatrixSelection.selection keep symmetric).series.off ht = 0 ∧ star ht = ht ∧
      (MatrixSelection.selection keep symmetric).series.diag (skew (u - 1)) = 0 := by
  classical
  let P : Selection (Matrix ι ι ℂ) := MatrixSelection.selection keep symmetric
  let S : SylvesterSolver P (H 0) := by
    rw [h0]
    exact MatrixSelection.spectralSolver keep symmetric (fun i => (e i : ℂ)) (by intro i; simp)
      (by intro i j hij he; exact gap i j hij (Complex.ofReal_injective he))
  have hd : P.diag (H 0) = H 0 := by
    rw [h0]
    ext i j
    change (if keep i j then Matrix.diagonal (fun i => (e i : ℂ)) i j else 0) =
      Matrix.diagonal (fun i => (e i : ℂ)) i j
    by_cases h : i = j
    · subst j; simp [reflexive]
    · simp [h]
  exact ⟨(blockDiagonalize P H S).1, (blockDiagonalize P H S).2,
    blockDiagonalize_correct P H hH hd S⟩

end Pymablock
