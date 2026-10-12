import Pymablock.Convergence.Output

noncomputable section
namespace Pymablock.Convergence
open MvPowerSeries
variable {σ A : Type*} [NormedRing A] [NormedAlgebra ℚ A] [StarRing A] [NormedStarGroup A]

/-- Projection bounds and the inverse-energy solver bound have different units. -/
structure GapControls (P : Selection A) (solve : A →ₗ[ℚ] A) (L : ℝ) : Prop where
  nonneg : 0 ≤ L
  diag : ∀ a, ‖P.diag a‖ ≤ ‖a‖
  off : ∀ a, ‖P.off a‖ ≤ ‖a‖
  solver : ∀ a, ‖solve a‖ ≤ L * ‖a‖

/-- A two-variable majorant with separate retained and eliminated input masses. -/
theorem step_majorant (P : Selection A) (solve : A →ₗ[ℚ] A) {L : ℝ}
    (ctl : GapControls P solve L) (w : (σ →₀ ℕ) → ℝ)
    (hw : ∀ n, 0 ≤ w n) (hm : ∀ i j, w (i+j) = w i*w j)
    (hs hr : MvPowerSeries σ A) (s : State σ A) {d o t b : ℝ}
    (hhs : Bound w hs d) (hhr : Bound w hr o)
    (hq : Bound w (qSeries s) t) (hb : Bound w (bSeries s) b) :
    Bound w (qSeries (step P solve hs hr s)) (t*t/2+L*(2*t*b+o+(2*o+4*d)*t)) ∧
    Bound w (bSeries (step P solve hs hr s)) (2*t*b+(o+2*d)*t) := by
  let q := positive (qSeries s)
  let bb := positive (bSeries s)
  have hq' : Bound w q t := hq.positive hw
  have hb' : Bound w bb b := hb.positive hw
  have hqb := hq'.star.mul hw hm hb'
  have hrq := hhr.mul hw hm hq'
  have hk : Bound w (comm (skew q) hs) (2*d*t) := by
    convert ((hq'.skew hw).mul hw hm hhs).sub hw (hhs.mul hw hm (hq'.skew hw)) using 1; ring
  have hd : ∀ a, ‖P.diag a‖ ≤ 1*‖a‖ := by simpa using ctl.diag
  have ho : ∀ a, ‖P.off a‖ ≤ 1*‖a‖ := by simpa using ctl.off
  have hbn : Bound w (bUpdate P.series (star q*bb) (hr*q) (comm (skew q) hs))
      (2*t*b+(o+2*d)*t) := by
    have h := ((((hqb.skew hw).add hw (hrq.herm hw)).map hw P.diag (by norm_num) hd).neg.sub hw
      (hqb.map hw P.off (by norm_num) ho)).add hw ((hk.herm hw).map hw P.diag (by norm_num) hd)
    convert h using 1; ring
  have hww : Bound w ((-1/2 : ℚ) • (star q*q)) (t*t/2) := by
    convert (hq'.star.mul hw hm hq').smul (-1/2 : ℚ) using 1
    norm_num [← Rat.norm_cast_real]; ring
  have hv := ((((hbn.add hw hhr).add hw hrq).herm hw).sub hw hk).map hw solve ctl.nonneg ctl.solver
  constructor
  · convert (hww.add hw hv).positive hw using 1; ring
  · exact hbn.positive hw

/-- An invariant rectangle certifies bounds on the actual constructed solution. -/
theorem solution_majorant (P : Selection A) (solve : A →ₗ[ℚ] A) {L d o t b : ℝ}
    (ctl : GapControls P solve L) (ht : 0 ≤ t) (hb : 0 ≤ b)
    (hqbound : t*t/2+L*(2*t*b+o+(2*o+4*d)*t) ≤ t)
    (hbbound : 2*t*b+(o+2*d)*t ≤ b)
    (w : (σ →₀ ℕ) → ℝ) (hw : ∀ n, 0 ≤ w n)
    (hm : ∀ i j, w (i+j) = w i*w j)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0)
    (hhs : Bound w hs d) (hhr : Bound w hr o) :
    Bound w (qSeries (solution P solve hs hr)) t ∧
    Bound w (bSeries (solution P solve hs hr)) b := by
  have hi : ∀ k, Bound w (qSeries ((step P solve hs hr)^[k] (fun _ => (0,0)))) t ∧
      Bound w (bSeries ((step P solve hs hr)^[k] (fun _ => (0,0)))) b := by
    intro k
    induction k with
    | zero => exact ⟨(bound_zero w).mono ht, (bound_zero w).mono hb⟩
    | succ k ih =>
      rw [Function.iterate_succ_apply']
      obtain ⟨h1,h2⟩ := step_majorant P solve ctl w hw hm hs hr _ hhs hhr ih.1 ih.2
      exact ⟨h1.mono hqbound,h2.mono hbbound⟩
  exact ⟨causal_bound Finsupp.degree _ (step_causal P solve hs hr hs0 hr0) _
    (fun n a => ‖a.1‖*w n) t (fun k => (hi k).1),
    causal_bound Finsupp.degree _ (step_causal P solve hs hr hs0 hr0) _
    (fun n a => ‖a.2‖*w n) b (fun k => (hi k).2)⟩

/-- An explicit certificate without an artificial lower bound on the solver norm. -/
theorem solution_nine_over_130 (P : Selection A) (solve : A →ₗ[ℚ] A) {L ε r : ℝ}
    (ctl : GapControls P solve L) (hr : 0 ≤ r) (hε : 0 ≤ ε) (hsmall : L*ε ≤ 9/130)
    (hs ho : MvPowerSeries σ A) (hs0 : hs 0 = 0) (ho0 : ho 0 = 0)
    (hhs : Bound (fun n => r^Finsupp.degree n) hs ε)
    (hho : Bound (fun n => r^Finsupp.degree n) ho ε) :
    Absolute (qSeries (solution P solve hs ho)) r ∧
    Absolute (bSeries (solution P solve hs ho)) r := by
  have hh := solution_majorant P solve ctl (t := 1/5) (b := ε) (by norm_num) hε
    (by nlinarith) (by linarith) _ (fun n => pow_nonneg hr _)
    (fun i j => by rw [map_add,pow_add]) hs ho hs0 ho0 hhs hho
  exact ⟨hh.1.summable (fun n => pow_nonneg hr _),hh.2.summable (fun n => pow_nonneg hr _)⟩

/-- A more useful certificate when retained and eliminated inputs differ.
Taking t=1/4 gives L*(7 o + 6 d) <= 7/8. -/
theorem solution_split_bound (P : Selection A) (solve : A →ₗ[ℚ] A) {L d o r : ℝ}
    (ctl : GapControls P solve L) (hr : 0 ≤ r) (hd : 0 ≤ d) (ho : 0 ≤ o)
    (hsmall : L*(7*o+6*d) ≤ 7/8)
    (hs hh : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hh0 : hh 0 = 0)
    (hhs : Bound (fun n => r^Finsupp.degree n) hs d)
    (hhh : Bound (fun n => r^Finsupp.degree n) hh o) :
    Absolute (qSeries (solution P solve hs hh)) r ∧
    Absolute (bSeries (solution P solve hs hh)) r := by
  have h := solution_majorant P solve ctl (t := 1/4) (b := (o+2*d)/2)
    (by norm_num) (by positivity) (by nlinarith) (by linarith) _
    (fun n => pow_nonneg hr _) (fun i j => by rw [map_add,pow_add]) hs hh hs0 hh0 hhs hhh
  exact ⟨h.1.summable (fun n => pow_nonneg hr _),h.2.summable (fun n => pow_nonneg hr _)⟩

/-- A one-dimensional search over 0 < t < 1/2 improves the explicit constants.
The inequality is polynomial, so any rational certificate can be checked exactly. -/
theorem solution_optimized_bound (P : Selection A) (solve : A →ₗ[ℚ] A) {L d o t r : ℝ}
    (ctl : GapControls P solve L) (hr : 0 ≤ r) (hd : 0 ≤ d) (ho : 0 ≤ o)
    (ht : 0 < t) (ht2 : t < 1/2)
    (hsmall : L*(o*(1-2*t*t)+4*d*t*(1-t)) ≤ t*(1-t/2)*(1-2*t))
    (hs hh : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hh0 : hh 0 = 0)
    (hhs : Bound (fun n => r^Finsupp.degree n) hs d)
    (hhh : Bound (fun n => r^Finsupp.degree n) hh o) :
    Absolute (qSeries (solution P solve hs hh)) r ∧
    Absolute (bSeries (solution P solve hs hh)) r := by
  let b := t*(o+2*d)/(1-2*t)
  have hden : 0 < 1-2*t := by linarith
  have hb : 0 ≤ b := by dsimp [b]; positivity
  have he : b*(1-2*t) = t*(o+2*d) := by
    dsimp [b]
    exact div_mul_cancel₀ _ (ne_of_gt hden)
  have hbb : 2*t*b+(o+2*d)*t ≤ b := by nlinarith
  have hqq : t*t/2+L*(2*t*b+o+(2*o+4*d)*t) ≤ t := by
    apply (mul_le_mul_iff_left₀ hden).mp
    have hscaled := congrArg (fun z : ℝ => L*(2*t)*z) he
    nlinarith
  have h := solution_majorant P solve ctl ht.le hb hqq hbb _
    (fun n => pow_nonneg hr _) (fun i j => by rw [map_add,pow_add]) hs hh hs0 hh0 hhs hhh
  exact ⟨h.1.summable (fun n => pow_nonneg hr _),h.2.summable (fun n => pow_nonneg hr _)⟩

/-- Transfer an optimized certificate to both returned series. -/
theorem blockDiagonalize_optimized_bound (P : Selection A) (H : MvPowerSeries σ A)
    (hH : star H = H) (hd0 : P.diag (H 0) = H 0) (S : SylvesterSolver P (H 0))
    {L d o t r : ℝ} (ctl : GapControls P S.solve L) (hr : 0 ≤ r) (ha : Absolute H r)
    (hd : 0 ≤ d) (ho : 0 ≤ o) (ht : 0 < t) (ht2 : t < 1/2)
    (hsmall : L*(o*(1-2*t*t)+4*d*t*(1-t)) ≤ t*(1-t/2)*(1-2*t))
    (hhs : Bound (fun n => r^Finsupp.degree n) (selected P H) d)
    (hho : Bound (fun n => r^Finsupp.degree n) (remaining P H) o) :
    Absolute (blockDiagonalize P H S).1 r ∧ Absolute (blockDiagonalize P H S).2 r := by
  have hq := (solution_optimized_bound P S.solve ctl hr hd ho ht ht2 hsmall
    (selected P H) (remaining P H) (by simp [selected]) (by simp [remaining]) hhs hho).1
  have hu : Absolute (blockDiagonalize P H S).2 r := (absolute_one r).add hq hr
  refine ⟨?_,hu⟩
  rw [(blockDiagonalize_correct P H hH hd0 S).2.2.2.1]
  exact ((hu.star hr).mul ha hr).mul hu hr

/-- An explicit coefficient-mass certificate for both returned series. -/
theorem blockDiagonalize_at_radius (P : Selection A) (H : MvPowerSeries σ A)
    (hH : star H = H) (hd : P.diag (H 0) = H 0) (S : SylvesterSolver P (H 0))
    {L ε r : ℝ} (ctl : GapControls P S.solve L) (hr : 0 ≤ r) (hε : 0 ≤ ε)
    (hsmall : L*ε ≤ 9/130) (hin : Bound (fun n => r^Finsupp.degree n) (positive H) ε) :
    Absolute (blockDiagonalize P H S).1 r ∧ Absolute (blockDiagonalize P H S).2 r := by
  have hs : Bound (fun n => r^Finsupp.degree n) (selected P H) ε :=
    hin.of_coeff_le (fun n => pow_nonneg hr _) (fun n => ctl.diag _)
  have ho : Bound (fun n => r^Finsupp.degree n) (remaining P H) ε :=
    hin.of_coeff_le (fun n => pow_nonneg hr _) (fun n => ctl.off _)
  have hq := (solution_nine_over_130 P S.solve ctl hr hε hsmall (selected P H) (remaining P H)
    (by simp [selected]) (by simp [remaining]) hs ho).1
  have hu : Absolute (blockDiagonalize P H S).2 r := (absolute_one r).add hq hr
  have ha : Absolute H r := by
    rw [← constant_add_positive H]
    exact (absolute_C _ _).add (hin.summable (fun n => pow_nonneg hr _)) hr
  refine ⟨?_,hu⟩
  rw [(blockDiagonalize_correct P H hH hd S).2.2.2.1]
  exact ((hu.star hr).mul ha hr).mul hu hr

end Pymablock.Convergence
