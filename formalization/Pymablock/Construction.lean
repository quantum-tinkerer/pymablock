import Pymablock.Sylvester

/-! The optimized Pymablock recursion is a strictly causal fixed-point map.
The state stores the coefficients of (U', B). All other series are derived. -/
noncomputable section
namespace Pymablock
open MvPowerSeries

variable {σ A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

/-- A coefficient family containing U' and the optimized auxiliary B. -/
abbrev State (σ A : Type*) := (σ →₀ ℕ) → A × A

def qSeries (s : State σ A) : MvPowerSeries σ A := fun n => (s n).1
def bSeries (s : State σ A) : MvPowerSeries σ A := fun n => (s n).2

/-- One causal evaluation. Constant coefficients are explicitly fixed to zero.
The q update uses the newly evaluated B, just as lazy evaluation resolves
same-degree dependencies in the Python algorithm. -/
def step (P : Selection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (s : State σ A) : State σ A :=
  let q := positive (qSeries s)
  let b := positive (bSeries s)
  let bn := bUpdate P.series (star q * b) (hr * q) (comm (skew q) hs)
  let x := bn + hr + hr * q
  let w := (-1 / 2 : ℚ) • (star q * q)
  let v := coefficientMap solve (herm x - comm (skew q) hs)
  fun n => (positive (w + v) n, positive bn n)

 theorem step_zero (P : Selection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (s : State σ A) : step P solve hs hr s 0 = (0, 0) := by
  simp [step]

 theorem step_causal (P : Selection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0) :
    StrictlyCausal Finsupp.degree (step P solve hs hr) := by
  intro d s t h
  have hq : JetEq d (positive (qSeries s)) (positive (qSeries t)) :=
    jetEq_positive (fun n hn => congrArg Prod.fst (h n hn))
  have hb : JetEq d (positive (bSeries s)) (positive (bSeries t)) :=
    jetEq_positive (fun n hn => congrArg Prod.snd (h n hn))
  let q := positive (qSeries s)
  let q' := positive (qSeries t)
  let b := positive (bSeries s)
  let b' := positive (bSeries t)
  have hq0 : q 0 = 0 := positive_zero _
  have hq'0 : q' 0 = 0 := positive_zero _
  have hb0 : b 0 = 0 := positive_zero _
  have hb'0 : b' 0 = 0 := positive_zero _
  have hqs0 : (star q) 0 = 0 := by change star (q 0) = 0; simp [hq0]
  have hqs'0 : (star q') 0 = 0 := by change star (q' 0) = 0; simp [hq'0]
  have hc := jetEq_mul_positive hq.star hb hqs0 hqs'0 hb0 hb'0
  have ha := jetEq_mul_positive (JetEq.refl d hr) hq hr0 hr0 hq0 hq'0
  have hw := (jetEq_mul_positive hq.star hq hqs0 hqs'0 hq0 hq'0).smul (-1 / 2 : ℚ)
  have hv0 : (skew q) 0 = 0 := by change (1/2 : ℚ) • (q 0 - star (q 0)) = 0; simp [hq0]
  have hv'0 : (skew q') 0 = 0 := by change (1/2 : ℚ) • (q' 0 - star (q' 0)) = 0; simp [hq'0]
  have hcomm : JetEq (d+1) (comm (skew q) hs) (comm (skew q') hs) :=
    (jetEq_mul_positive hq.skew (JetEq.refl d hs) hv0 hv'0 hs0 hs0).sub
      (jetEq_mul_positive (JetEq.refl d hs) hq.skew hs0 hs0 hv0 hv'0)
  have hbn : JetEq (d+1) (bUpdate P.series (star q * b) (hr * q) (comm (skew q) hs))
      (bUpdate P.series (star q' * b') (hr * q') (comm (skew q') hs)) := by
    unfold bUpdate
    exact (((hc.skew.add ha.herm).map P.diag).neg.sub (hc.map P.off)).add
      (hcomm.herm.map P.diag)
  have hx := (hbn.add (JetEq.refl (d+1) hr)).add ha
  have hv := (hx.herm.sub hcomm).map solve
  intro n hn
  exact Prod.ext (jetEq_positive (hw.add hv) n hn) (jetEq_positive hbn n hn)

/-- The unique formal solution is obtained by a finite number of evaluations
for each multi-index. -/
def solution (P : Selection A) (solve : A →ₗ[ℚ] A) (hs hr : MvPowerSeries σ A) : State σ A :=
  causalSolution Finsupp.degree (step P solve hs hr) (fun _ => (0, 0))

 theorem solution_fixed (P : Selection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0) :
    step P solve hs hr (solution P solve hs hr) = solution P solve hs hr :=
  causalSolution_fixed _ _ (step_causal P solve hs hr hs0 hr0) _

 theorem solution_unique (P : Selection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0)
    (s : State σ A) (h : step P solve hs hr s = s) : s = solution P solve hs hr :=
  causal_fixed_unique _ _ (step_causal P solve hs hr hs0 hr0) _ _ h
    (solution_fixed P solve hs hr hs0 hr0)

end Pymablock
