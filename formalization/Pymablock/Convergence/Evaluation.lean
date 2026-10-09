import Pymablock.Convergence.Algebra
import Mathlib.Analysis.Normed.Group.FunctionSeries

noncomputable section
set_option backward.isDefEq.respectTransparency false
open scoped Topology BigOperators
namespace Pymablock.Convergence

variable {σ A : Type*} [NormedRing A] [NormedAlgebra ℝ A] [CompleteSpace A]

/-- A real multivariate monomial. Parameters are not conjugated. -/
def monomialValue (x : σ → ℝ) (n : σ →₀ ℕ) : ℝ := n.prod fun i k => x i ^ k

def evaluate (f : MvPowerSeries σ A) (x : σ → ℝ) : A :=
  ∑' n, monomialValue x n • f n

 theorem monomial_bound (x : σ → ℝ) (n : σ →₀ ℕ) {r : ℝ}
    (_hr : 0 ≤ r) (hx : ∀ i, ‖x i‖ ≤ r) : ‖monomialValue x n‖ ≤ r ^ Finsupp.degree n := by
  classical
  unfold monomialValue Finsupp.prod
  rw [norm_prod, Finsupp.degree_apply, ← Finset.prod_pow_eq_pow_sum]
  apply Finset.prod_le_prod
  · intro i _; exact norm_nonneg _
  · intro i _; rw [norm_pow]; exact pow_le_pow_left₀ (norm_nonneg _) (hx i) _

 theorem monomial_add (x : σ → ℝ) (i j : σ →₀ ℕ) :
    monomialValue x (i+j) = monomialValue x i * monomialValue x j := by
  classical
  exact Finsupp.prod_add_index (fun _ _ => pow_zero _) (fun _ _ _ _ => pow_add _ _ _)

omit [CompleteSpace A] in
 theorem Absolute.summable_eval_norm {f : MvPowerSeries σ A} {r : ℝ}
    (hf : Absolute f r) (hr : 0 ≤ r) (x : σ → ℝ) (hx : ∀ i, ‖x i‖ ≤ r) :
    Summable (fun n => ‖monomialValue x n • f n‖) :=
  hf.of_nonneg_of_le (fun n => norm_nonneg _) (fun n => by
    rw [norm_smul]
    exact (mul_le_mul_of_nonneg_right (monomial_bound x n hr hx) (norm_nonneg _)).trans_eq (mul_comm _ _))

 theorem Absolute.summable_eval {f : MvPowerSeries σ A} {r : ℝ}
    (hf : Absolute f r) (hr : 0 ≤ r) (x : σ → ℝ) (hx : ∀ i, ‖x i‖ ≤ r) :
    Summable (fun n => monomialValue x n • f n) := (hf.summable_eval_norm hr x hx).of_norm

 theorem continuous_monomial (n : σ →₀ ℕ) : Continuous (fun x : σ → ℝ => monomialValue x n) := by
  classical
  unfold monomialValue Finsupp.prod
  exact continuous_finset_prod _ fun i _ => (continuous_apply i).pow _

/-- The convergent series defines a continuous family throughout the closed polydisc. -/
 theorem Absolute.continuousOn_evaluate {f : MvPowerSeries σ A} {r : ℝ}
    (hf : Absolute f r) (hr : 0 ≤ r) :
    ContinuousOn (evaluate f) {x : σ → ℝ | ∀ i, ‖x i‖ ≤ r} := by
  apply continuousOn_tsum
  · intro n; exact ((continuous_monomial n).smul continuous_const).continuousOn
  · exact hf
  · intro n x hx
    rw [norm_smul]
    exact (mul_le_mul_of_nonneg_right (monomial_bound x n hr hx) (norm_nonneg _)).trans_eq (mul_comm _ _)

 theorem evaluate_add {f g : MvPowerSeries σ A} {r : ℝ}
    (hf : Absolute f r) (hg : Absolute g r) (hr : 0 ≤ r)
    (x : σ → ℝ) (hx : ∀ i, ‖x i‖ ≤ r) : evaluate (f+g) x = evaluate f x + evaluate g x := by
  simp only [evaluate, series_add_apply, smul_add]
  exact (hf.summable_eval hr x hx).tsum_add (hg.summable_eval hr x hx)

omit [CompleteSpace A] in
 theorem evaluate_one (x : σ → ℝ) : evaluate (1 : MvPowerSeries σ A) x = 1 := by
  classical
  unfold evaluate
  rw [tsum_eq_single (0 : σ →₀ ℕ)]
  · change monomialValue x 0 • MvPowerSeries.coeff 0 (1 : MvPowerSeries σ A) = 1
    simp [monomialValue]
  · intro n hn
    change monomialValue x n • MvPowerSeries.coeff n 1 = 0
    simp [MvPowerSeries.coeff_one, hn]

 theorem monomial_zero [DecidableEq σ] (n : σ →₀ ℕ) :
    monomialValue (0 : σ → ℝ) n = if n = 0 then 1 else 0 := by
  classical
  by_cases hn : n = 0
  · subst n; simp [monomialValue]
  · rw [if_neg hn]
    obtain ⟨i,hi⟩ := Finsupp.support_nonempty_iff.mpr hn
    unfold monomialValue Finsupp.prod
    apply Finset.prod_eq_zero hi
    exact zero_pow (Finsupp.mem_support_iff.mp hi)

omit [CompleteSpace A] in
 theorem evaluate_zero (f : MvPowerSeries σ A) : evaluate f (0 : σ → ℝ) = f 0 := by
  classical
  unfold evaluate
  rw [tsum_eq_single (0 : σ →₀ ℕ)]
  · simp [monomial_zero]
  · intro n hn; simp [monomial_zero, hn]

 theorem polydisc_mem_nhds [Finite σ] {r : ℝ} (hr : 0 < r) :
    {x : σ → ℝ | ∀ i, ‖x i‖ ≤ r} ∈ nhds 0 := by
  have hb : ∀ᶠ x : σ → ℝ in nhds 0, ∀ i, ‖x i‖ < r := by
    rw [Filter.eventually_all]
    intro i
    exact ((continuous_apply i).norm.continuousAt.eventually_lt continuousAt_const (by simpa using hr))
  exact hb.mono (fun x hx i => (hx i).le)

 theorem Absolute.continuousAt_evaluate_zero [Finite σ] {f : MvPowerSeries σ A} {r : ℝ}
    (hf : Absolute f r) (hr : 0 < r) : ContinuousAt (evaluate f) 0 :=
  (hf.continuousOn_evaluate hr.le).continuousAt (polydisc_mem_nhds hr)

 theorem evaluate_linear (L : A →L[ℝ] A) {f : MvPowerSeries σ A} {r : ℝ}
    (hf : Absolute f r) (hr : 0 ≤ r) (x : σ → ℝ) (hx : ∀ i, ‖x i‖ ≤ r) :
    evaluate (fun n => L (f n)) x = L (evaluate f x) := by
  rw [evaluate, evaluate, L.map_tsum (hf.summable_eval hr x hx)]
  apply tsum_congr
  intro n
  exact (L.map_smul (monomialValue x n) (f n)).symm

omit [NormedAlgebra ℝ A] [CompleteSpace A] in
/-- Regrouping by total multi-index is legitimate because each fiber is finite. -/
 theorem fiber_sum [DecidableEq σ] (n : σ →₀ ℕ) (F : ((σ →₀ ℕ) × (σ →₀ ℕ)) → A) :
    (∑' p : {p : (σ →₀ ℕ) × (σ →₀ ℕ) // p.1+p.2=n}, F p) =
      ∑ p ∈ Finset.antidiagonal n, F p := by
  classical
  calc
    _ = ∑' p, ({p : (σ →₀ ℕ) × (σ →₀ ℕ) | p.1+p.2=n} : Set _).indicator F p :=
      tsum_subtype _ _
    _ = ∑ p ∈ Finset.antidiagonal n, F p := by
      rw [tsum_eq_sum (s := Finset.antidiagonal n) (by
        intro p hp
        exact Set.indicator_of_notMem (by simpa only [Set.mem_setOf_eq, Finset.mem_antidiagonal] using hp) F)]
      apply Finset.sum_congr rfl
      intro p hp
      exact Set.indicator_of_mem (s := {p : (σ →₀ ℕ) × (σ →₀ ℕ) | p.1+p.2=n})
        (a := p) (Finset.mem_antidiagonal.mp hp) F

/-- Absolute convergence justifies the Cauchy product; evaluation is used only
on series for which convergence has been proved. -/
 theorem evaluate_mul {f g : MvPowerSeries σ A} {r : ℝ}
    (hf : Absolute f r) (hg : Absolute g r) (hr : 0 ≤ r)
    (x : σ → ℝ) (hx : ∀ i, ‖x i‖ ≤ r) : evaluate (f*g) x = evaluate f x * evaluate g x := by
  classical
  let F := fun n => monomialValue x n • f n
  let G := fun n => monomialValue x n • g n
  have hs := summable_mul_of_summable_norm (hf.summable_eval_norm hr x hx)
    (hg.summable_eval_norm hr x hx)
  have hfib := hs.hasSum.tsum_fiberwise (fun p : (σ →₀ ℕ) × (σ →₀ ℕ) => p.1+p.2)
  have hid (n : σ →₀ ℕ) : monomialValue x n • (f*g) n =
      ∑' p : {p : (σ →₀ ℕ) × (σ →₀ ℕ) // p.1+p.2=n}, F p.1.1 * G p.1.2 := by
    rw [fiber_sum n (fun p => F p.1 * G p.2)]
    change monomialValue x n • MvPowerSeries.coeff n (f*g) = _
    rw [MvPowerSeries.coeff_mul, Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro p hp
    rw [← Finset.mem_antidiagonal.mp hp, monomial_add]
    dsimp [F,G]
    rw [smul_mul_smul]
    rfl
  calc
    evaluate (f*g) x = ∑' n, ∑' p : {p : (σ →₀ ℕ) × (σ →₀ ℕ) // p.1+p.2=n},
        F p.1.1 * G p.1.2 := tsum_congr hid
    _ = ∑' p : (σ →₀ ℕ) × (σ →₀ ℕ), F p.1 * G p.2 := hfib.tsum_eq
    _ = _ := (tsum_mul_tsum_of_summable_norm (hf.summable_eval_norm hr x hx)
      (hg.summable_eval_norm hr x hx)).symm

section Star
variable [StarRing A] [NormedStarGroup A] [StarModule ℝ A]

omit [CompleteSpace A] in
 theorem evaluate_star (f : MvPowerSeries σ A) (x : σ → ℝ) :
    evaluate (star f) x = star (evaluate f x) := by
  simp only [evaluate, tsum_star, series_star_apply, star_smul, star_trivial]

end Star
end Pymablock.Convergence
