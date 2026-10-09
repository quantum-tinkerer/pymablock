import Pymablock.Library.Series
import Pymablock.Library.Algebra
import Mathlib.Algebra.Group.Torsion

noncomputable section

namespace Pymablock

open MvPowerSeries Finset

variable {σ A : Type*} [Ring A] [StarRing A] [IsAddTorsionFree A]

omit [StarRing A] in
/-- A homogeneous recurrence with positive-degree multipliers has only the
zero solution. No bound on the number of formal parameters is needed. -/
theorem homogeneous_eq_zero (a b d : MvPowerSeries σ A)
    (ha : a 0 = 0) (hb : b 0 = 0)
    (hd : d + d = -(a * d + d * b)) : d = 0 := by
  classical
  have h : ∀ k n, Finsupp.degree n = k → d n = 0 := by
    intro k
    induction k using Nat.strong_induction_on with
    | h k ih =>
      intro n hn
      have had : coeff n (a * d) = 0 := by
        rw [coeff_mul]
        apply Finset.sum_eq_zero
        rintro ⟨i,j⟩ hij
        have hij' : i + j = n := Finset.mem_antidiagonal.mp hij
        by_cases hi : i = 0
        · subst i; simp [coeff_apply, ha]
        have hi' : 0 < Finsupp.degree i := Nat.pos_of_ne_zero (by simpa [Finsupp.degree_eq_zero_iff] using hi)
        have he : Finsupp.degree i + Finsupp.degree j = k := by rw [← map_add, hij', hn]
        change a i * d j = 0
        rw [ih (Finsupp.degree j) (by omega) j rfl, mul_zero]
      have hdb : coeff n (d * b) = 0 := by
        rw [coeff_mul]
        apply Finset.sum_eq_zero
        rintro ⟨i,j⟩ hij
        have hij' : i + j = n := Finset.mem_antidiagonal.mp hij
        by_cases hj : j = 0
        · subst j; simp [coeff_apply, hb]
        have hj' : 0 < Finsupp.degree j := Nat.pos_of_ne_zero (by simpa [Finsupp.degree_eq_zero_iff] using hj)
        have he : Finsupp.degree i + Finsupp.degree j = k := by rw [← map_add, hij', hn]
        change d i * b j = 0
        rw [ih (Finsupp.degree i) (by omega) i rfl, zero_mul]
      have he := congrArg (coeff n) hd
      simp only [map_add, map_neg, had, hdb, add_zero, neg_zero, coeff_apply] at he
      exact two_nsmul_eq_zero.mp (by simpa [two_nsmul, two_mul] using he)
  ext n
  exact h _ n rfl

/-- The two algebraic recurrences for X force it to be the desired commutator.
This theorem closes the apparent circularity in the derivation of X. -/
theorem commutator_of_recurrences (q hs x : MvPowerSeries σ A)
    (hq : q 0 = 0) (hu : star (1 + q) * (1 + q) = 1) (hh : star hs = hs)
    (hherm : x + star x = comm q hs + star (comm q hs))
    (hskew : x - star x = -(star q * x) + star x * q) :
    x = comm q hs := by
  have h := homogeneous_eq_zero (star q) q (x - comm q hs)
    (by change star (q 0) = 0; simp [hq]) hq
    (comm_defect_equation q hs x hu hh hherm hskew)
  exact sub_eq_zero.mp h

end Pymablock
