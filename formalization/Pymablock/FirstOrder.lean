import Pymablock.Recurrence

/-! A concrete coefficient consequence used by the worked two-level example
and by the audit of the printed optimized Sylvester equation. -/
noncomputable section
namespace Pymablock
open MvPowerSeries

variable {σ A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

omit [Algebra ℚ A] [StarRing A] in
theorem product_first_order (f g : MvPowerSeries σ A) (hf : f 0 = 0) (hg : g 0 = 0)
    (n : σ →₀ ℕ) (hn : Finsupp.degree n = 1) : (f * g) n = 0 := by
  have hf' : JetEq 1 f 0 := by
    intro m hm
    have hm0 : m = 0 := (Finsupp.degree_eq_zero_iff m).mp (by omega)
    subst m
    exact hf
  have hg' : JetEq 1 g 0 := by
    intro m hm
    have hm0 : m = 0 := (Finsupp.degree_eq_zero_iff m).mp (by omega)
    subst m
    exact hg
  have he := jetEq_mul_positive hf' hg' hf rfl hg rfl n (by omega)
  simpa using he

/-- The leading coefficient has the positive Sylvester source H'_R. -/
theorem solution_first_order (P : BlockStructure A) (solve : A →ₗ[ℚ] A)
    (hs hr : MvPowerSeries σ A) (hs0 : hs 0 = 0) (hr0 : hr 0 = 0)
    (hhr : star hr = hr) (n : σ →₀ ℕ) (hn : Finsupp.degree n = 1) :
    qSeries (solution P solve hs hr) n = solve (hr n) := by
  let q := qSeries (solution P solve hs hr)
  let b := bSeries (solution P solve hs hr)
  have hrec : Recurrence P solve hs hr q b := solution_recurrence P solve hs hr hs0 hr0
  have hqs0 : (star q) 0 = 0 := by simp [hrec.q_zero]
  have hc := product_first_order (star q) b hqs0 hrec.b_zero n hn
  have ha := product_first_order hr q hr0 hrec.q_zero n hn
  have hw := product_first_order (star q) q hqs0 hrec.q_zero n hn
  have hv0 : skew q 0 = 0 := by simp [skew, hrec.q_zero]
  have hvh := product_first_order (skew q) hs hv0 hs0 n hn
  have hhv := product_first_order hs (skew q) hs0 hv0 n hn
  have hb : b n = 0 := by
    have he := congrFun hrec.b_eq n
    simp only [bUpdate, series_sub_apply, series_neg_apply, series_diag_apply,
      series_off_apply, series_add_apply, series_herm_apply, series_skew_apply, hc, ha] at he
    simpa [herm, skew] using he
  have hhn : star (hr n) = hr n := congrFun hhr n
  have he := congrFun hrec.q_eq n
  simpa [hw, hb, ha, comm, hvh, hhv, herm_of_selfadjoint _ hhn] using he

end Pymablock
