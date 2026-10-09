import Pymablock.NonHermitian.Recurrence
import Pymablock.FirstOrder

noncomputable section
namespace Pymablock.NonHermitian
open MvPowerSeries
variable {σ A : Type*} [Ring A] [Algebra ℚ A]

/-- The constructed forward and inverse corrections have opposite first-order
coefficients; no adjoint relation is imposed. -/
theorem solution_first_order (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0)
    (n : σ →₀ ℕ) (hn : Finsupp.degree n = 1) :
    qSeries (solution P solve hs hr) n = solve (hr n) ∧
      gSeries (solution P solve hs hr) n = -solve (hr n) := by
  let q := qSeries (solution P solve hs hr)
  let g := gSeries (solution P solve hs hr)
  let b := bSeries (solution P solve hs hr)
  have hrec : Recurrence P solve hs hr q g b := solution_recurrence P solve hs hr hs0 hr0
  have hc := product_first_order g b hrec.g_zero hrec.b_zero n hn
  have ha := product_first_order hr q hr0 hrec.q_zero n hn
  have hw := product_first_order g q hrec.g_zero hrec.q_zero n hn
  have hgr := product_first_order g hr hrec.g_zero hr0 n hn
  have hp0 : plusSeries P g b 0 = 0 := by simp [plusSeries, hrec.b_zero, hrec.g_zero]
  have hpg := product_first_order (plusSeries P g b) g hp0 hrec.g_zero n hn
  have hv0 : vSeries q g 0 = 0 := by simp [vSeries, hrec.q_zero, hrec.g_zero]
  have hvh := product_first_order (vSeries q g) hs hv0 hs0 n hn
  have hhv := product_first_order hs (vSeries q g) hs0 hv0 n hn
  have hk : comm (vSeries q g) hs n = 0 := by
    simp only [comm, series_sub_apply, hvh, hhv, sub_self]
  have hz : zSeries P hr q g b n = 0 := by
    simp only [zSeries, series_smul_apply, series_sub_apply, ha, hgr, hc, hpg,
      sub_self, smul_zero]
  have hb : bUpdate P hs hr q g b n = 0 := by
    simp only [bUpdate, series_sub_apply, Projection.selected_apply,
      Projection.remaining_series_apply, series_add_apply, hk, hz, ha, hc,
      zero_add, sub_self, map_zero]
  have hv : vUpdate P solve hs hr q g b n = solve (hr n) := by
    simp only [vUpdate, coefficientMap_apply, series_sub_apply, series_add_apply,
      hb, ha, hz, hk, zero_add, add_zero, sub_zero]
  constructor
  · have he := congrFun hrec.q_eq n
    simpa only [series_add_apply, wSeries, series_smul_apply, hw, smul_zero,
      zero_add, hv] using he
  · have he := congrFun hrec.g_eq n
    simpa only [series_sub_apply, wSeries, series_smul_apply, hw, smul_zero,
      zero_sub, hv] using he

end Pymablock.NonHermitian
