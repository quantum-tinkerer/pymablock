import Pymablock.Convergence.Bounds
import Pymablock.Construction

noncomputable section
namespace Pymablock.Convergence
open MvPowerSeries

variable {σ A : Type*} [NormedRing A] [NormedAlgebra ℚ A] [StarRing A] [NormedStarGroup A]

/-- Norm contracts for the three linear operations. Concrete finite matrix maps
have finite constants; these are quantitative bounds, not convergence assumptions. -/
structure Controls (P : Selection A) (solve : A →ₗ[ℚ] A) (K : ℝ) : Prop where
  one_le : 1 ≤ K
  diag : ∀ a, ‖P.diag a‖ ≤ K * ‖a‖
  off : ∀ a, ‖P.off a‖ ≤ K * ‖a‖
  solver : ∀ a, ‖solve a‖ ≤ K * ‖a‖

 theorem step_bound (P : Selection A) (solve : A →ₗ[ℚ] A) {K : ℝ}
    (ctl : Controls P solve K) (w : (σ →₀ ℕ) → ℝ)
    (hw : ∀ n, 0 ≤ w n) (hm : ∀ i j, w (i+j) = w i * w j)
    (hs hr : MvPowerSeries σ A) (s : State σ A) {ε t : ℝ}
    (hhs : Bound w hs ε) (hhr : Bound w hr ε)
    (hq : Bound w (qSeries s) t) (hb : Bound w (bSeries s) t) :
    Bound w (qSeries (step P solve hs hr s))
      (t*t/2 + K*(2*K*t*t + 3*K*ε*t + ε + 3*ε*t)) ∧
    Bound w (bSeries (step P solve hs hr s)) (2*K*t*t + 3*K*ε*t) := by
  have hK : 0 ≤ K := le_trans (by norm_num) ctl.one_le
  let q := positive (qSeries s)
  let b := positive (bSeries s)
  have hq' : Bound w q t := hq.positive hw
  have hb' : Bound w b t := hb.positive hw
  have hqb := hq'.star.mul hw hm hb'
  have hrq := hhr.mul hw hm hq'
  have hk : Bound w (comm (skew q) hs) (2*ε*t) := by
    have h := ((hq'.skew hw).mul hw hm hhs).sub hw (hhs.mul hw hm (hq'.skew hw))
    convert h using 1
    ring
  let bn := bUpdate P.series (star q*b) (hr*q) (comm (skew q) hs)
  have hbn : Bound w bn (2*K*t*t+3*K*ε*t) := by
    have h := ((((hqb.skew hw).add hw (hrq.herm hw)).map hw P.diag hK ctl.diag).neg.sub hw
      (hqb.map hw P.off hK ctl.off)).add hw ((hk.herm hw).map hw P.diag hK ctl.diag)
    convert h using 1
    ring
  have hww : Bound w ((-1/2 : ℚ) • (star q*q)) (t*t/2) := by
    have h := (hq'.star.mul hw hm hq').smul (-1/2 : ℚ)
    convert h using 1
    norm_num [← Rat.norm_cast_real]; ring
  have hv := ((((hbn.add hw hhr).add hw hrq).herm hw).sub hw hk).map hw solve hK ctl.solver
  constructor
  · have h := (hww.add hw hv).positive hw
    convert h using 1
    ring
  · exact hbn.positive hw

/-- Explicit smallness conditions make the finite recurrence iterations uniformly
bounded. They can always be met by shrinking the input radius. -/
 theorem invariant_bound (P : Selection A) (solve : A →ₗ[ℚ] A) {K ε t : ℝ}
    (ctl : Controls P solve K) (ht : 0 ≤ t)
    (hqbound : t*t/2+K*(2*K*t*t+3*K*ε*t+ε+3*ε*t) ≤ t)
    (hbbound : 2*K*t*t+3*K*ε*t ≤ t)
    (w : (σ →₀ ℕ) → ℝ) (hw : ∀ n, 0 ≤ w n)
    (hm : ∀ i j, w (i+j) = w i*w j)
    (hs hr : MvPowerSeries σ A) (hhs : Bound w hs ε) (hhr : Bound w hr ε) :
    ∀ k, Bound w (qSeries ((step P solve hs hr)^[k] (fun _ => (0,0)))) t ∧
      Bound w (bSeries ((step P solve hs hr)^[k] (fun _ => (0,0)))) t := by
  intro k
  induction k with
  | zero => exact ⟨(bound_zero w).mono ht, (bound_zero w).mono ht⟩
  | succ k ih =>
    rw [Function.iterate_succ_apply']
    obtain ⟨hq,hb⟩ := step_bound P solve ctl w hw hm hs hr _ hhs hhr ih.1 ih.2
    exact ⟨hq.mono hqbound, hb.mono hbbound⟩

/-- Absolute convergence of the actual multivariate causal solution under explicit
input-size bounds. The index type of perturbations is unrestricted. -/
 theorem solution_summable (P : Selection A) (solve : A →ₗ[ℚ] A) {K ε t : ℝ}
    (ctl : Controls P solve K) (ht : 0 ≤ t)
    (hqbound : t*t/2+K*(2*K*t*t+3*K*ε*t+ε+3*ε*t) ≤ t)
    (hbbound : 2*K*t*t+3*K*ε*t ≤ t)
    (w : (σ →₀ ℕ) → ℝ) (hw : ∀ n, 0 ≤ w n)
    (hm : ∀ i j, w (i+j) = w i*w j)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0)
    (hhs : Bound w hs ε) (hhr : Bound w hr ε) :
    Summable (fun n => ‖qSeries (solution P solve hs hr) n‖ * w n) ∧
    Summable (fun n => ‖bSeries (solution P solve hs hr) n‖ * w n) := by
  have hi := invariant_bound P solve ctl ht hqbound hbbound w hw hm hs hr hhs hhr
  have hq : Bound w (qSeries (solution P solve hs hr)) t :=
    causal_bound Finsupp.degree _ (step_causal P solve hs hr hs0 hr0) _
      (fun n a => ‖a.1‖ * w n) t (fun k => (hi k).1)
  have hb : Bound w (bSeries (solution P solve hs hr)) t :=
    causal_bound Finsupp.degree _ (step_causal P solve hs hr hs0 hr0) _
      (fun n a => ‖a.2‖ * w n) t (fun k => (hi k).2)
  exact ⟨hq.summable hw, hb.summable hw⟩

end Pymablock.Convergence
