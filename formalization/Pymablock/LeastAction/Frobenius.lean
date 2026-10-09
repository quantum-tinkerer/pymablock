import Mathlib.Analysis.Matrix.Order
import Mathlib.Analysis.Matrix.Normed
import Mathlib.Tactic

noncomputable section

namespace Pymablock.LeastAction

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Squared Frobenius distance from zero, with no ambient operator-norm convention. -/
def frobeniusSq (A : Matrix ι ι ℂ) : ℝ := (Matrix.trace (star A * A)).re

omit [DecidableEq ι] in
 theorem frobeniusSq_eq_sum (A : Matrix ι ι ℂ) :
    frobeniusSq A = ∑ i, ∑ j, Complex.normSq (A i j) := by
  simp only [frobeniusSq, Matrix.trace, Matrix.diag, Matrix.mul_apply,
    Matrix.star_apply, Complex.re_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  exact congrArg Complex.re (Complex.normSq_eq_conj_mul_self (z := A i j)).symm

omit [DecidableEq ι] in
 theorem frobeniusSq_nonneg (A : Matrix ι ι ℂ) : 0 ≤ frobeniusSq A := by
  rw [frobeniusSq_eq_sum]
  exact Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => Complex.normSq_nonneg _

omit [DecidableEq ι] in
@[simp] theorem frobeniusSq_eq_zero (A : Matrix ι ι ℂ) : frobeniusSq A = 0 ↔ A = 0 := by
  rw [frobeniusSq_eq_sum]
  simp only [Finset.sum_eq_zero_iff_of_nonneg (fun i _ =>
    Finset.sum_nonneg fun j _ => Complex.normSq_nonneg (A i j)), Finset.mem_univ, true_implies,
    Finset.sum_eq_zero_iff_of_nonneg (fun j _ => Complex.normSq_nonneg (A _ j)),
    Complex.normSq_eq_zero, Matrix.ext_iff]
  rfl

/-- Unitarity turns least distance into greatest real trace. -/
 theorem frobeniusSq_sub_one (U : Matrix ι ι ℂ) (hU : star U * U = 1) :
    frobeniusSq (U - 1) = 2 * Fintype.card ι - 2 * (Matrix.trace U).re := by
  have h : star (U - 1) * (U - 1) = (1 : Matrix ι ι ℂ) + 1 - star U - U := by
    rw [star_sub, star_one]
    linear_combination (norm := noncomm_ring) hU
  rw [frobeniusSq, h]
  simp [Matrix.trace_sub, Matrix.trace_add, Matrix.trace_one,
    show Matrix.trace (star U) = star (Matrix.trace U) from Matrix.trace_conjTranspose U]
  ring

open scoped Matrix.Norms.Frobenius in
/-- Agreement with mathlib's Frobenius norm, explicitly distinguished from its
operator-norm instance. -/
 theorem frobeniusSq_eq_norm_sq (A : Matrix ι ι ℂ) : frobeniusSq A = ‖A‖ ^ 2 := by
  rw [Matrix.frobenius_norm_def, ← Real.sqrt_eq_rpow]
  rw [frobeniusSq_eq_sum]
  simp_rw [Complex.normSq_eq_norm_sq, Real.rpow_two]
  exact (Real.sq_sqrt (Finset.sum_nonneg fun i _ =>
    Finset.sum_nonneg fun j _ => sq_nonneg ‖A i j‖)).symm

end Pymablock.LeastAction
