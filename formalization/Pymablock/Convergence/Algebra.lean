import Pymablock.Convergence.Radius

noncomputable section
namespace Pymablock.Convergence
open MvPowerSeries
variable {σ A : Type*} [NormedRing A]

 theorem Absolute.bound {f : MvPowerSeries σ A} {r : ℝ} (hf : Absolute f r) (hr : 0 ≤ r) :
    Bound (fun n => r ^ Finsupp.degree n) f (∑' n, ‖f n‖ * r ^ Finsupp.degree n) :=
  fun s => hf.sum_le_tsum s (fun _ _ => mul_nonneg (norm_nonneg _) (pow_nonneg hr _))

 theorem Absolute.radius_mono {f : MvPowerSeries σ A} {r R : ℝ}
    (hf : Absolute f R) (hr : 0 ≤ r) (hrR : r ≤ R) : Absolute f r :=
  ((hf.bound (hr.trans hrR)).radius_mono hr hrR).summable (fun _ => pow_nonneg hr _)

 theorem Absolute.add {f g : MvPowerSeries σ A} {r : ℝ} (hf : Absolute f r)
    (hg : Absolute g r) (hr : 0 ≤ r) : Absolute (f+g) r :=
  ((hf.bound hr).add (fun _ => pow_nonneg hr _) (hg.bound hr)).summable (fun _ => pow_nonneg hr _)

 theorem Absolute.mul {f g : MvPowerSeries σ A} {r : ℝ} (hf : Absolute f r)
    (hg : Absolute g r) (hr : 0 ≤ r) : Absolute (f*g) r :=
  ((hf.bound hr).mul (fun _ => pow_nonneg hr _) (fun i j => by rw [map_add, pow_add])
    (hg.bound hr)).summable (fun _ => pow_nonneg hr _)

 theorem absolute_one (r : ℝ) : Absolute (1 : MvPowerSeries σ A) r := by
  classical
  apply summable_of_ne_finset_zero (s := {0})
  intro n hn
  have hn0 : n ≠ 0 := by simpa using hn
  change ‖coeff n (1 : MvPowerSeries σ A)‖ * _ = 0
  simp [coeff_one, hn0]

/-- Polynomial inputs automatically satisfy the analytic input hypothesis. -/
 theorem absolute_monomial (n : σ →₀ ℕ) (a : A) (r : ℝ) : Absolute (monomial n a) r := by
  classical
  apply summable_of_ne_finset_zero (s := {n})
  intro k hk
  have hkn : k ≠ n := by simpa using hk
  change ‖coeff k (monomial n a)‖ * _ = 0
  simp [coeff_monomial, hkn]

 theorem absolute_C (a : A) (r : ℝ) : Absolute (C a : MvPowerSeries σ A) r := by
  simpa only [monomial_zero_eq_C_apply] using absolute_monomial (0 : σ →₀ ℕ) a r

section Linear
variable [NormedAlgebra ℚ A]
 theorem Absolute.map {f : MvPowerSeries σ A} {r K : ℝ} (hf : Absolute f r)
    (hr : 0 ≤ r) (L : A →ₗ[ℚ] A) (hK : 0 ≤ K) (hL : ∀ a, ‖L a‖ ≤ K*‖a‖) :
    Absolute (coefficientMap L f) r :=
  ((hf.bound hr).map (fun _ => pow_nonneg hr _) L hK hL).summable (fun _ => pow_nonneg hr _)
end Linear

 theorem Absolute.positive {f : MvPowerSeries σ A} {r : ℝ} (hf : Absolute f r)
    (hr : 0 ≤ r) : Absolute (Pymablock.positive f) r :=
  ((hf.bound hr).positive (fun _ => pow_nonneg hr _)).summable (fun _ => pow_nonneg hr _)

section Star
variable [StarRing A] [NormedStarGroup A]
 theorem Absolute.star {f : MvPowerSeries σ A} {r : ℝ} (hf : Absolute f r)
    (hr : 0 ≤ r) : Absolute (Star.star f) r :=
  (hf.bound hr).star.summable (fun _ => pow_nonneg hr _)
end Star

end Pymablock.Convergence
