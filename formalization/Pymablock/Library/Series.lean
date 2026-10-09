import Mathlib.RingTheory.MvPowerSeries.Basic
import Mathlib.Algebra.Star.Basic
import Mathlib.Algebra.Star.BigOperators
import Pymablock.Library.Recursion

/-! Coefficientwise adjoints and the total-degree filtration of formal series.
Formal parameters are real: the adjoint acts only on coefficients. -/

noncomputable section

namespace Pymablock

open MvPowerSeries
open Finset

variable {σ A : Type*} [Ring A] [StarRing A]

instance seriesStar : Star (MvPowerSeries σ A) := ⟨fun f n => star (f n)⟩

@[simp] theorem coeff_star (f : MvPowerSeries σ A) (n : σ →₀ ℕ) :
    coeff n (star f) = star (coeff n f) := rfl

instance seriesStarRing : StarRing (MvPowerSeries σ A) where
  star_involutive f := by ext n; simp
  star_mul f g := by
    classical
    ext n
    simp only [coeff_star, coeff_mul, star_sum, star_mul]
    exact Finsupp.sum_antidiagonal_swap n (fun i j => star (coeff j g) * star (coeff i f))
  star_add f g := by ext n; simp

/-- Equality modulo terms of total degree at least d. -/
def JetEq (d : ℕ) (f g : MvPowerSeries σ A) : Prop :=
  AgreesBelow Finsupp.degree d f g

/-- Remove the constant coefficient. This is used inside the fixed-point map
so every product there is a strictly positive-degree Cauchy product. -/
def positive (f : MvPowerSeries σ A) : MvPowerSeries σ A :=
  fun n => @ite _ (n = 0) (Classical.propDecidable _) 0 (f n)

omit [StarRing A] in
@[simp] theorem positive_zero (f : MvPowerSeries σ A) : positive f 0 = 0 := by
  simp [positive]

omit [StarRing A] in
theorem positive_eq_self (f : MvPowerSeries σ A) (h : f 0 = 0) : positive f = f := by
  funext n
  by_cases hn : n = 0 <;> simp [positive, hn, h]

omit [StarRing A] in
theorem jetEq_positive {d : ℕ} {f g : MvPowerSeries σ A} (h : JetEq d f g) :
    JetEq d (positive f) (positive g) := by
  intro n hn
  by_cases hz : n = 0 <;> simp [positive, hz, h n hn]

omit [StarRing A] in
/-- In a Cauchy product of two series with zero constant term, each nonzero
contribution uses strictly lower total degrees on both sides. -/
theorem jetEq_mul_positive {d : ℕ} {f f' g g' : MvPowerSeries σ A}
    (hf : JetEq d f f') (hg : JetEq d g g')
    (hf0 : f 0 = 0) (hf'0 : f' 0 = 0) (hg0 : g 0 = 0) (hg'0 : g' 0 = 0) :
    JetEq (d + 1) (f * g) (f' * g') := by
  classical
  intro n hn
  change coeff n (f * g) = coeff n (f' * g')
  simp only [coeff_mul]
  apply Finset.sum_congr rfl
  rintro ⟨i,j⟩ hij
  have hij' : i + j = n := Finset.mem_antidiagonal.mp hij
  by_cases hi : i = 0
  · subst i; simp [coeff_apply, hf0, hf'0]
  by_cases hj : j = 0
  · subst j; simp [coeff_apply, hg0, hg'0]
  have hi' : 0 < Finsupp.degree i := Nat.pos_of_ne_zero (by simpa [Finsupp.degree_eq_zero_iff] using hi)
  have hj' : 0 < Finsupp.degree j := Nat.pos_of_ne_zero (by simpa [Finsupp.degree_eq_zero_iff] using hj)
  have hd : Finsupp.degree i + Finsupp.degree j = Finsupp.degree n := by
    rw [← map_add, hij']
  change f i * g j = f' i * g' j
  rw [hf i (by omega), hg j (by omega)]

end Pymablock
