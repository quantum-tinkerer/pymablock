import Pymablock.Convergence.Gap

noncomputable section
namespace Pymablock.Convergence
open MvPowerSeries
variable {σ A : Type*} [NormedRing A]

/-- Positive-degree mass decreases at least linearly when shrinking a radius. -/
theorem Bound.shrink {f : MvPowerSeries σ A} (hf0 : f 0 = 0) {R c M : ℝ}
    (hR : 0 ≤ R) (hc : 0 ≤ c) (hc1 : c ≤ 1)
    (h : Bound (fun n => R^Finsupp.degree n) f M) :
    Bound (fun n => (c*R)^Finsupp.degree n) f (c*M) := by
  intro s
  calc
    _ ≤ ∑ n ∈ s, c*(‖f n‖*R^Finsupp.degree n) := by
      apply Finset.sum_le_sum
      intro n _
      by_cases hn : n = 0
      · subst n; simp [hf0]
      · have hd : Finsupp.degree n ≠ 0 := fun he => hn ((Finsupp.degree_eq_zero_iff n).mp he)
        have hp : c^Finsupp.degree n ≤ c := by
          obtain ⟨k,hk⟩ := Nat.exists_eq_succ_of_ne_zero hd
          rw [hk,pow_succ]
          simpa using mul_le_mul_of_nonneg_right (pow_le_one₀ hc hc1) hc
        dsimp only
        rw [mul_pow]
        nlinarith [mul_le_mul_of_nonneg_right hp (mul_nonneg (norm_nonneg (f n)) (pow_nonneg hR (Finsupp.degree n)))]
    _ = c*(∑ n ∈ s, ‖f n‖*R^Finsupp.degree n) := (Finset.mul_sum _ _ _).symm
    _ ≤ _ := mul_le_mul_of_nonneg_left (h s) hc

/-- A finite monomial has exactly its expected weighted coefficient bound. -/
theorem bound_monomial (w : (σ →₀ ℕ) → ℝ) (hw : ∀ n, 0 ≤ w n) (n : σ →₀ ℕ) (a : A) :
    Bound w (monomial n a) (‖a‖*w n) := by
  classical
  intro s
  by_cases hn : n ∈ s
  · rw [Finset.sum_eq_single n]
    · change ‖coeff n (monomial n a)‖*w n ≤ _
      simp
    · intro k _ hkn
      change ‖coeff k (monomial n a)‖*w k = 0
      simp [coeff_monomial,hkn]
    · exact fun hh => False.elim (hh hn)
  · have he : ∑ k ∈ s, ‖(monomial n a : MvPowerSeries σ A) k‖*w k = 0 := by
      apply Finset.sum_eq_zero
      intro k hk
      have hkn : k ≠ n := by intro he; subst k; exact hn hk
      change ‖coeff k (monomial n a)‖*w k = 0
      simp [coeff_monomial,hkn]
    rw [he]
    exact mul_nonneg (norm_nonneg _) (hw n)

theorem bound_sum {ι : Type*} (s : Finset ι) (w : (σ →₀ ℕ) → ℝ) (hw : ∀ n, 0 ≤ w n)
    (f : ι → MvPowerSeries σ A) (C : ι → ℝ) (h : ∀ i ∈ s, Bound w (f i) (C i)) :
    Bound w (∑ i ∈ s, f i) (∑ i ∈ s, C i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simpa using bound_zero (A := A) w
  | @insert i s hi ih =>
    simp only [Finset.sum_insert hi]
    exact (h i (Finset.mem_insert_self _ _)).add hw (ih (fun j hj => h j (Finset.mem_insert_of_mem hj)))

/-- Affine input in any finite number of perturbation parameters. -/
def affine [Fintype σ] [DecidableEq σ] (H0 : A) (V : σ → A) : MvPowerSeries σ A :=
  C H0 + ∑ i, monomial (Finsupp.single i 1) (V i)

section Affine
variable [Fintype σ] [DecidableEq σ]

theorem affine_positive (H0 : A) (V : σ → A) :
    positive (affine H0 V) = ∑ i, monomial (Finsupp.single i 1) (V i) := by
  have hn (i : σ) : (0 : σ →₀ ℕ) ≠ Finsupp.single i 1 := by
    intro h
    have he := congrArg (fun f : σ →₀ ℕ => f i) h
    simp at he
  funext n
  by_cases hn0 : n = 0
  · subst n
    rw [positive_zero]
    change 0 = coeff 0 (∑ i, monomial (Finsupp.single i 1) (V i))
    rw [map_sum]
    simp only [coeff_monomial,hn,ite_false,Finset.sum_const_zero]
  · rw [positive, if_neg hn0]
    change coeff n (C H0 + ∑ i, monomial (Finsupp.single i 1) (V i)) = coeff n _
    rw [map_add,coeff_C,if_neg hn0,zero_add]
    rfl

theorem affine_zero (H0 : A) (V : σ → A) : affine H0 V 0 = H0 := by
  have hn (i : σ) : (0 : σ →₀ ℕ) ≠ Finsupp.single i 1 := by
    intro h
    have he := congrArg (fun f : σ →₀ ℕ => f i) h
    simp at he
  change coeff 0 (C H0 + ∑ i, monomial (Finsupp.single i 1) (V i)) = H0
  rw [map_add,coeff_C,if_pos rfl,map_sum]
  simp only [coeff_monomial,hn,ite_false,Finset.sum_const_zero,add_zero]

theorem affine_hermitian [StarRing A] (H0 : A) (V : σ → A)
    (h0 : star H0 = H0) (hV : ∀ i, star (V i) = V i) : star (affine H0 V) = affine H0 V := by
  funext n
  change star (coeff n (C H0 + ∑ i, monomial (Finsupp.single i 1) (V i))) =
    coeff n (C H0 + ∑ i, monomial (Finsupp.single i 1) (V i))
  simp only [map_add,map_sum,coeff_C,coeff_monomial,star_add,star_sum]
  congr 1
  · split_ifs <;> simp [h0]
  · apply Finset.sum_congr rfl
    intro i _
    split_ifs <;> simp [hV]

/-- Anisotropic weights give the familiar sum of radius times perturbation norm. -/
theorem affine_mass (H0 : A) (V : σ → A) (w : (σ →₀ ℕ) → ℝ) (hw : ∀ n, 0 ≤ w n) :
    Bound w (positive (affine H0 V)) (∑ i, ‖V i‖*w (Finsupp.single i 1)) := by
  rw [affine_positive]
  exact bound_sum Finset.univ w hw _ _ (fun i _ => bound_monomial w hw _ _)

theorem affine_common_radius_mass (H0 : A) (V : σ → A) {r : ℝ} (hr : 0 ≤ r) :
    Bound (fun n => r^Finsupp.degree n) (positive (affine H0 V)) (r*∑ i, ‖V i‖) := by
  convert affine_mass H0 V (fun n => r^Finsupp.degree n) (fun n => pow_nonneg hr _) using 1
  simp [Finsupp.degree_single,pow_one,Finset.mul_sum,mul_comm]

/-- The familiar gap / total perturbation norm radius, with its numerical constant. -/
theorem affine_radius_certificate (H0 : A) (V : σ → A) {δ : ℝ} (hδ : 0 < δ)
    (hV : 0 < ∑ i, ‖V i‖) :
    let r := 9*δ/(130*∑ i, ‖V i‖)
    0 < r ∧ Bound (fun n => r^Finsupp.degree n) (positive (affine H0 V)) (9*δ/130) := by
  dsimp only
  have hr : 0 < 9*δ/(130*∑ i, ‖V i‖) := by positivity
  refine ⟨hr,?_⟩
  convert affine_common_radius_mass H0 V hr.le using 1
  field_simp

end Affine
/-- A usable radius from one known positive-degree coefficient mass at R.
The zero-mass case is constant input and needs no shrinkage. -/
theorem analytic_radius_certificate (H : MvPowerSeries σ A) {δ R M : ℝ}
    (hδ : 0 < δ) (hR : 0 < R) (hM : 0 < M)
    (hin : Bound (fun n => R^Finsupp.degree n) (positive H) M) :
    let r := min 1 (9*δ/(130*M))*R
    0 < r ∧ r ≤ R ∧ Bound (fun n => r^Finsupp.degree n) (positive H) (9*δ/130) := by
  let c := min 1 (9*δ/(130*M))
  have hc : 0 < c := lt_min (by norm_num) (by positivity)
  have hc1 : c ≤ 1 := min_le_left _ _
  have hcb : c*M ≤ 9*δ/130 := by
    have hh := mul_le_mul_of_nonneg_right (min_le_right 1 (9*δ/(130*M))) hM.le
    have he : (9*δ/(130*M))*M = 9*δ/130 := by field_simp
    rw [he] at hh
    exact hh
  exact ⟨mul_pos hc hR,mul_le_of_le_one_left hR.le hc1,
    (hin.shrink (positive_zero H) hR.le hc.le hc1).mono hcb⟩

section Weighted
variable [NormedAlgebra ℚ A] [StarRing A] [NormedStarGroup A]

/-- Anisotropic polydiscs and other nonnegative multiplicative weights.
This controls both returned series, not just the auxiliary fixed point. -/
theorem blockDiagonalize_weighted (P : Selection A) (H : MvPowerSeries σ A)
    (hH : star H = H) (hd : P.diag (H 0) = H 0) (S : SylvesterSolver P (H 0))
    {L ε : ℝ} (ctl : GapControls P S.solve L) (hε : 0 ≤ ε) (hsmall : L*ε ≤ 9/130)
    (w : (σ →₀ ℕ) → ℝ) (hw : ∀ n, 0 ≤ w n) (hm : ∀ i j, w (i+j)=w i*w j)
    (hin : Bound w (positive H) ε) :
    Summable (fun n => ‖(blockDiagonalize P H S).1 n‖*w n) ∧
    Summable (fun n => ‖(blockDiagonalize P H S).2 n‖*w n) := by
  have hs : Bound w (selected P H) ε := hin.of_coeff_le hw (fun n => ctl.diag _)
  have ho : Bound w (remaining P H) ε := hin.of_coeff_le hw (fun n => ctl.off _)
  have hq := (solution_majorant P S.solve ctl (t := 1/5) (b := ε) (by norm_num) hε
    (by nlinarith) (by linarith) w hw hm (selected P H) (remaining P H)
    (by simp [selected]) (by simp [remaining]) hs ho).1
  have hconst (a : A) : Bound w (C a) (‖a‖*w 0) := by
    simpa only [monomial_zero_eq_C_apply] using bound_monomial w hw 0 a
  have h1 : Bound w (1 : MvPowerSeries σ A) (‖(1 : A)‖*w 0) := by simpa using hconst 1
  have hu : Bound w (blockDiagonalize P H S).2 (‖(1 : A)‖*w 0+1/5) := h1.add hw hq
  have ha : Bound w H (‖H 0‖*w 0+ε) := by
    simpa only [constant_add_positive] using (hconst (H 0)).add hw hin
  refine ⟨?_,hu.summable hw⟩
  rw [(blockDiagonalize_correct P H hH hd S).2.2.2.1]
  exact ((hu.star.mul hw hm ha).mul hw hm hu).summable hw

end Weighted

end Pymablock.Convergence
