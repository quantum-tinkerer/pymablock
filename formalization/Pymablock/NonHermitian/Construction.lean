import Pymablock.NonHermitian.Projection

noncomputable section
namespace Pymablock.NonHermitian
open MvPowerSeries
variable {σ A : Type*} [Ring A] [Algebra ℚ A]

/-- Coefficients of U', G, and B. V = (U' - G)/2 is derived. -/
abbrev State (σ A : Type*) := (σ →₀ ℕ) → A × A × A
def qSeries (s : State σ A) : MvPowerSeries σ A := fun n => (s n).1
def gSeries (s : State σ A) : MvPowerSeries σ A := fun n => (s n).2.1
def bSeries (s : State σ A) : MvPowerSeries σ A := fun n => (s n).2.2
def vSeries (q g : MvPowerSeries σ A) : MvPowerSeries σ A := (1/2 : ℚ) • (q-g)
def wSeries (q g : MvPowerSeries σ A) : MvPowerSeries σ A := (-1/2 : ℚ) • (g*q)
def plusSeries (P : Projection A) (g b : MvPowerSeries σ A) : MvPowerSeries σ A :=
  P.series.selected (b+g*b)
def zSeries (P : Projection A) (hr q g b : MvPowerSeries σ A) : MvPowerSeries σ A :=
  (1/2 : ℚ) • (hr*q-g*hr-g*b-plusSeries P g b*g)
def bUpdate (P : Projection A) (hs hr q g b : MvPowerSeries σ A) : MvPowerSeries σ A :=
  P.series.selected (comm (vSeries q g) hs + zSeries P hr q g b - hr*q) -
    P.series.remaining (g*b)
def vUpdate (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr q g b : MvPowerSeries σ A) : MvPowerSeries σ A :=
  coefficientMap solve (bUpdate P hs hr q g b + hr + hr*q -
    zSeries P hr q g b - comm (vSeries q g) hs)

/-- Same-degree B is evaluated before V; all products use lower degrees. -/
def step (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (s : State σ A) : State σ A :=
  let q := positive (qSeries s)
  let g := positive (gSeries s)
  let b := positive (bSeries s)
  let w := wSeries q g
  let v := vUpdate P solve hs hr q g b
  fun n => (positive (w+v) n, positive (w-v) n, positive (bUpdate P hs hr q g b) n)

 theorem step_zero (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (s : State σ A) : step P solve hs hr s 0 = (0,0,0) := by
  simp [step]

 theorem step_causal (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0) :
    StrictlyCausal Finsupp.degree (step P solve hs hr) := by
  intro d s t h
  have hq : JetEq d (positive (qSeries s)) (positive (qSeries t)) :=
    jetEq_positive (fun n hn => congrArg Prod.fst (h n hn))
  have hg : JetEq d (positive (gSeries s)) (positive (gSeries t)) :=
    jetEq_positive (fun n hn => congrArg (fun a : A × A × A => a.2.1) (h n hn))
  have hb : JetEq d (positive (bSeries s)) (positive (bSeries t)) :=
    jetEq_positive (fun n hn => congrArg (fun a : A × A × A => a.2.2) (h n hn))
  let q := positive (qSeries s); let q' := positive (qSeries t)
  let g := positive (gSeries s); let g' := positive (gSeries t)
  let b := positive (bSeries s); let b' := positive (bSeries t)
  have hq0 : q 0 = 0 := positive_zero _
  have hq'0 : q' 0 = 0 := positive_zero _
  have hg0 : g 0 = 0 := positive_zero _
  have hg'0 : g' 0 = 0 := positive_zero _
  have hb0 : b 0 = 0 := positive_zero _
  have hb'0 : b' 0 = 0 := positive_zero _
  have hc := jetEq_mul_positive hg hb hg0 hg'0 hb0 hb'0
  have ha := jetEq_mul_positive (JetEq.refl d hr) hq hr0 hr0 hq0 hq'0
  have hgr := jetEq_mul_positive hg (JetEq.refl d hr) hg0 hg'0 hr0 hr0
  have hw := (jetEq_mul_positive hg hq hg0 hg'0 hq0 hq'0).smul (-1/2 : ℚ)
  have hp : JetEq d (plusSeries P g b) (plusSeries P g' b') :=
    (hb.add (hg.mul hb)).map P.selected
  have hp0 : plusSeries P g b 0 = 0 := by simp [plusSeries, hb0, hg0]
  have hp'0 : plusSeries P g' b' 0 = 0 := by simp [plusSeries, hb'0, hg'0]
  have hpg := jetEq_mul_positive hp hg hp0 hp'0 hg0 hg'0
  have hz : JetEq (d+1) (zSeries P hr q g b) (zSeries P hr q' g' b') :=
    (((ha.sub hgr).sub hc).sub hpg).smul _
  have hv : JetEq d (vSeries q g) (vSeries q' g') := (hq.sub hg).smul _
  have hv0 : vSeries q g 0 = 0 := by simp [vSeries, hq0, hg0]
  have hv'0 : vSeries q' g' 0 = 0 := by simp [vSeries, hq'0, hg'0]
  have hk : JetEq (d+1) (comm (vSeries q g) hs) (comm (vSeries q' g') hs) :=
    (jetEq_mul_positive hv (JetEq.refl d hs) hv0 hv'0 hs0 hs0).sub
      (jetEq_mul_positive (JetEq.refl d hs) hv hs0 hs0 hv0 hv'0)
  have hbn : JetEq (d+1) (bUpdate P hs hr q g b) (bUpdate P hs hr q' g' b') :=
    (((hk.add hz).sub ha).map P.selected).sub (hc.map P.remaining)
  have hvn : JetEq (d+1) (vUpdate P solve hs hr q g b) (vUpdate P solve hs hr q' g' b') :=
    (((hbn.add (JetEq.refl (d+1) hr)).add ha).sub hz |>.sub hk).map solve
  intro n hn
  exact Prod.ext (jetEq_positive (hw.add hvn) n hn)
    (Prod.ext (jetEq_positive (hw.sub hvn) n hn) (jetEq_positive hbn n hn))

def solution (P : Projection A) (solve : A →ₗ[ℚ] A) (hs hr : MvPowerSeries σ A) : State σ A :=
  causalSolution Finsupp.degree (step P solve hs hr) (fun _ => (0,0,0))

 theorem solution_fixed (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0) :
    step P solve hs hr (solution P solve hs hr) = solution P solve hs hr :=
  causalSolution_fixed _ _ (step_causal P solve hs hr hs0 hr0) _

 theorem solution_unique (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0)
    (s : State σ A) (h : step P solve hs hr s = s) : s = solution P solve hs hr :=
  causal_fixed_unique _ _ (step_causal P solve hs hr hs0 hr0) _ _ h
    (solution_fixed P solve hs hr hs0 hr0)

end Pymablock.NonHermitian
