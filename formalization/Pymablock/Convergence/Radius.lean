import Pymablock.Convergence.Recurrence
import Mathlib.Analysis.Normed.Group.Tannery
import Mathlib.Analysis.SpecificLimits.Basic

noncomputable section
open scoped Topology
open Filter
namespace Pymablock.Convergence

variable {σ A : Type*} [NormedRing A]

/-- Absolute convergence on a common polydisc of radius r. -/
def Absolute (f : MvPowerSeries σ A) (r : ℝ) : Prop :=
  Summable (fun n => ‖f n‖ * r ^ Finsupp.degree n)

/-- A convergent series with zero constant term has arbitrarily small absolute
coefficient mass after shrinking its radius. -/
 theorem small_radius (f : MvPowerSeries σ A) (f0 : f 0 = 0)
    {R ε : ℝ} (hR : 0 < R) (hε : 0 < ε) (hf : Absolute f R) :
    ∃ r, 0 < r ∧ r ≤ R ∧ Bound (fun n => r ^ Finsupp.degree n) f ε := by
  let r : ℕ → ℝ := fun k => R * (1 / ((k : ℝ)+1))
  have hr0 : ∀ k, 0 < r k := fun k => mul_pos hR (by positivity)
  have hrR : ∀ k, r k ≤ R := by
    intro k
    dsimp [r]
    apply mul_le_of_le_one_right hR.le
    apply (div_le_one (by positivity : (0:ℝ) < k+1)).mpr
    linarith [Nat.cast_nonneg (α := ℝ) k]
  have hr : Tendsto r atTop (nhds 0) := by
    simpa [r] using tendsto_const_nhds.mul
      (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ))
  have hpoint (n : σ →₀ ℕ) : Tendsto (fun k => ‖f n‖ * r k ^ Finsupp.degree n)
      atTop (nhds 0) := by
    by_cases hn : n = 0
    · subst n; simp [f0]
    · have hd : Finsupp.degree n ≠ 0 := by intro h; exact hn ((Finsupp.degree_eq_zero_iff n).mp h)
      simpa [zero_pow hd] using tendsto_const_nhds.mul (hr.pow (Finsupp.degree n))
  have hdom : ∀ k n, ‖‖f n‖ * r k ^ Finsupp.degree n‖ ≤ ‖f n‖ * R ^ Finsupp.degree n := by
    intro k n
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (norm_nonneg _) (pow_nonneg (hr0 k).le _))]
    exact mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (hr0 k).le (hrR k) _) (norm_nonneg _)
  have hlim : Tendsto (fun k => ∑' n, ‖f n‖ * r k ^ Finsupp.degree n) atTop (nhds 0) := by
    simpa using tendsto_tsum_of_dominated_convergence hf hpoint (Eventually.of_forall hdom)
  obtain ⟨k,hk⟩ := (hlim.eventually_lt_const hε).exists
  refine ⟨r k, hr0 k, hrR k, ?_⟩
  have hs : Absolute f (r k) := hf.of_nonneg_of_le
    (fun n => mul_nonneg (norm_nonneg _) (pow_nonneg (hr0 k).le _))
    (fun n => by simpa only [Real.norm_eq_abs,
      abs_of_nonneg (mul_nonneg (norm_nonneg _) (pow_nonneg (hr0 k).le _))] using hdom k n)
  intro s
  exact (hs.sum_le_tsum s (fun n _ => mul_nonneg (norm_nonneg _) (pow_nonneg (hr0 k).le _))).trans hk.le

 theorem Bound.radius_mono {f : MvPowerSeries σ A} {r R C : ℝ}
    (h : Bound (fun n => R ^ Finsupp.degree n) f C) (hr : 0 ≤ r) (hrR : r ≤ R) :
    Bound (fun n => r ^ Finsupp.degree n) f C := by
  intro s
  exact (Finset.sum_le_sum fun n _ => mul_le_mul_of_nonneg_left
    (pow_le_pow_left₀ hr hrR _) (norm_nonneg _)).trans (h s)

/-- Positive, explicit invariant-ball parameters exist for every finite control constant. -/
 theorem small_parameters {K : ℝ} (hK : 1 ≤ K) : ∃ t ε : ℝ, 0 < t ∧ 0 < ε ∧
    t*t/2+K*(2*K*t*t+3*K*ε*t+ε+3*ε*t) ≤ t ∧ 2*K*t*t+3*K*ε*t ≤ t := by
  have hK0 : 0 < K := lt_of_lt_of_le (by norm_num) hK
  let t := 1/(16*K*K)
  have ht : 0 < t := by dsimp [t]; positivity
  have hKt : 16*K*K*t = 1 := by dsimp [t]; field_simp
  have ht1 : t ≤ 1 := by
    dsimp [t]
    apply (div_le_one (by positivity : (0:ℝ) < 16*K*K)).mpr
    nlinarith
  have hbn : 2*K*t*t+3*K*(t*t)*t ≤ 5*K*t*t := by
    nlinarith [mul_nonneg (mul_nonneg hK0.le (sq_nonneg t)) (sub_nonneg.mpr ht1)]
  have hqq : t*t/2+K*(2*K*t*t+3*K*(t*t)*t+t*t+3*(t*t)*t) ≤ 10*K*K*t*t := by
    have h3 : 3*(t*t)*t ≤ 3*t*t := by nlinarith [mul_nonneg (sq_nonneg t) (sub_nonneg.mpr ht1)]
    have hinner := mul_le_mul_of_nonneg_left (add_le_add (add_le_add_right hbn (t*t)) h3) hK0.le
    have hc : (1/2:ℝ)+5*K*K+4*K ≤ 10*K*K := by nlinarith
    have hh := mul_le_mul_of_nonneg_right hc (sq_nonneg t)
    nlinarith
  refine ⟨t,t*t,ht,mul_pos ht ht,?_,?_⟩
  · have hh := congrArg (fun x : ℝ => x*t) hKt
    nlinarith [sq_nonneg (K*t)]
  · have hh := congrArg (fun x : ℝ => x*t) hKt
    have hh2 := mul_nonneg (sub_nonneg.mpr hK) (mul_nonneg hK0.le (sq_nonneg t))
    nlinarith

section Solution
variable [NormedAlgebra ℚ A] [StarRing A] [NormedStarGroup A]

/-- Analytic input and bounded linear operations imply a positive radius of
absolute convergence for the series actually constructed by Pymablock. -/
 theorem solution_positive_radius (P : Selection A) (solve : A →ₗ[ℚ] A) {K : ℝ}
    (ctl : Controls P solve K) (hs hr : MvPowerSeries σ A)
    (hs0 : hs 0 = 0) (hr0 : hr 0 = 0) {R : ℝ} (hR : 0 < R)
    (hhs : Absolute hs R) (hhr : Absolute hr R) :
    ∃ r, 0 < r ∧ r ≤ R ∧
      Absolute (qSeries (solution P solve hs hr)) r ∧
      Absolute (bSeries (solution P solve hs hr)) r := by
  obtain ⟨t,ε,ht,hε,hq,hb⟩ := small_parameters ctl.one_le
  obtain ⟨rs,hrs,hrsR,hbs⟩ := small_radius hs hs0 hR hε hhs
  obtain ⟨rr,hrr,hrrR,hbr⟩ := small_radius hr hr0 hR hε hhr
  let r := min rs rr
  have hrp : 0 < r := lt_min hrs hrr
  refine ⟨r,hrp,(min_le_left rs rr).trans hrsR,?_⟩
  exact solution_summable P solve ctl ht.le hq hb _
    (fun n => pow_nonneg hrp.le _) (fun i j => by rw [map_add, pow_add])
    hs hr hs0 hr0 (hbs.radius_mono hrp.le (min_le_left _ _))
    (hbr.radius_mono hrp.le (min_le_right _ _))

end Solution
end Pymablock.Convergence
