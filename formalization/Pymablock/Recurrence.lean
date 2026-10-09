import Pymablock.Construction

noncomputable section
namespace Pymablock
open MvPowerSeries

variable {σ A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

/-- The equations satisfied by the constructed coefficient families.
These are output properties, proved below, not additional input assumptions. -/
structure Recurrence (P : BlockStructure A) (solve : A →ₗ[ℚ] A)
    (hs hr q b : MvPowerSeries σ A) : Prop where
  q_zero : q 0 = 0
  b_zero : b 0 = 0
  b_eq : b = bUpdate P.series (star q * b) (hr * q)
  q_eq : q = (-1 / 2 : ℚ) • (star q * q) +
    coefficientMap solve (herm (b + hr + hr * q) - comm (skew q) hs)

 theorem recurrence_of_fixed (P : BlockStructure A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hr0 : hr 0 = 0)
    (s : State σ A) (h : step P solve hs hr s = s) :
    Recurrence P solve hs hr (qSeries s) (bSeries s) := by
  have hz : s 0 = (0,0) := by rw [← h, step_zero]
  have hq0 : qSeries s 0 = 0 := congrArg Prod.fst hz
  have hb0 : bSeries s 0 = 0 := congrArg Prod.snd hz
  have hqpos := positive_eq_self (qSeries s) hq0
  have hbpos := positive_eq_self (bSeries s) hb0
  let q := qSeries s
  let b := bSeries s
  let bn := bUpdate P.series (star q * b) (hr * q)
  have hbn0 : bn 0 = 0 := by
    simp [bn, bUpdate, q, b, hq0, hb0, hr0, herm, skew]
  have hbEq : b = bn := by
    have he := congrArg bSeries h
    dsimp only [step] at he
    rw [hqpos, hbpos] at he
    change positive bn = b at he
    rw [positive_eq_self bn hbn0] at he
    exact he.symm
  have hv0 : (coefficientMap solve (herm (bn + hr + hr * q) - comm (skew q) hs)) 0 = 0 := by
    simp [comm, q, hq0, hbn0, hr0, herm, skew]
  have hn0 : ((-1/2 : ℚ) • (star q * q) +
      coefficientMap solve (herm (bn + hr + hr * q) - comm (skew q) hs)) 0 = 0 := by
    rw [series_add_apply, hv0, add_zero]
    simp [q, hq0]
  refine ⟨hq0, hb0, hbEq, ?_⟩
  have he := congrArg qSeries h
  dsimp only [step] at he
  rw [hqpos, hbpos] at he
  change positive ((-1/2 : ℚ) • (star q * q) +
    coefficientMap solve (herm (bn + hr + hr * q) - comm (skew q) hs)) = q at he
  rw [positive_eq_self _ hn0, ← hbEq] at he
  exact he.symm

 theorem solution_recurrence (P : BlockStructure A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0) :
    Recurrence P solve hs hr (qSeries (solution P solve hs hr)) (bSeries (solution P solve hs hr)) :=
  recurrence_of_fixed P solve hs hr hr0 _ (solution_fixed P solve hs hr hs0 hr0)

end Pymablock
