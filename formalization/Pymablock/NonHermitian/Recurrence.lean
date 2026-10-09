import Pymablock.NonHermitian.Construction

noncomputable section
namespace Pymablock.NonHermitian
open MvPowerSeries
variable {σ A : Type*} [Ring A] [Algebra ℚ A]

/-- These equations are proved from the fixed point, not assumed by the
public correctness theorem. -/
structure Recurrence (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr q g b : MvPowerSeries σ A) : Prop where
  q_zero : q 0 = 0
  g_zero : g 0 = 0
  b_zero : b 0 = 0
  b_eq : b = bUpdate P hs hr q g b
  q_eq : q = wSeries q g + vUpdate P solve hs hr q g b
  g_eq : g = wSeries q g - vUpdate P solve hs hr q g b

 theorem recurrence_of_fixed (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hr0 : hr 0 = 0)
    (s : State σ A) (h : step P solve hs hr s = s) :
    Recurrence P solve hs hr (qSeries s) (gSeries s) (bSeries s) := by
  have hz : s 0 = (0,0,0) := by rw [← h, step_zero]
  have hq0 : qSeries s 0 = 0 := congrArg Prod.fst hz
  have hg0 : gSeries s 0 = 0 := congrArg (fun a : A × A × A => a.2.1) hz
  have hb0 : bSeries s 0 = 0 := congrArg (fun a : A × A × A => a.2.2) hz
  have hqp := positive_eq_self (qSeries s) hq0
  have hgp := positive_eq_self (gSeries s) hg0
  have hbp := positive_eq_self (bSeries s) hb0
  let q := qSeries s; let g := gSeries s; let b := bSeries s
  have hbz : bUpdate P hs hr q g b 0 = 0 := by
    simp [bUpdate, zSeries, plusSeries, vSeries, comm, q, g, b, hq0, hg0, hb0, hr0]
  have hvz : vUpdate P solve hs hr q g b 0 = 0 := by
    change solve ((bUpdate P hs hr q g b + hr + hr*q - zSeries P hr q g b - comm (vSeries q g) hs) 0) = 0
    simp [hbz, zSeries, plusSeries, vSeries, comm, q, g, b, hq0, hg0, hb0, hr0]
  have hwz : wSeries q g 0 = 0 := by simp [wSeries, q, g, hq0, hg0]
  have hqz : (wSeries q g + vUpdate P solve hs hr q g b) 0 = 0 := by
    simp only [series_add_apply, hwz, hvz, add_zero]
  have hgz : (wSeries q g - vUpdate P solve hs hr q g b) 0 = 0 := by
    simp only [series_sub_apply, hwz, hvz, sub_self]
  refine ⟨hq0, hg0, hb0, ?_, ?_, ?_⟩
  · have he := congrArg bSeries h
    dsimp only [step] at he
    rw [hqp, hgp, hbp] at he
    change positive (bUpdate P hs hr q g b) = b at he
    rw [positive_eq_self _ hbz] at he
    exact he.symm
  · have he := congrArg qSeries h
    dsimp only [step] at he
    rw [hqp, hgp, hbp] at he
    change positive (wSeries q g + vUpdate P solve hs hr q g b) = q at he
    rw [positive_eq_self _ hqz] at he
    exact he.symm
  · have he := congrArg gSeries h
    dsimp only [step] at he
    rw [hqp, hgp, hbp] at he
    change positive (wSeries q g - vUpdate P solve hs hr q g b) = g at he
    rw [positive_eq_self _ hgz] at he
    exact he.symm

 theorem solution_recurrence (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0) :
    Recurrence P solve hs hr (qSeries (solution P solve hs hr))
      (gSeries (solution P solve hs hr)) (bSeries (solution P solve hs hr)) :=
  recurrence_of_fixed P solve hs hr hr0 _ (solution_fixed P solve hs hr hs0 hr0)

end Pymablock.NonHermitian
