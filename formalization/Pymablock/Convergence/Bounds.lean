import Pymablock.Library.SeriesBlocks
import Mathlib.Analysis.Normed.Ring.InfiniteSum
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Analysis.Normed.Group.Rat

noncomputable section
open scoped BigOperators
open MvPowerSeries

namespace Pymablock.Convergence

variable {σ A : Type*} [NormedRing A]

/-- An absolute coefficient bound, expressed by all finite subsums. -/
def Bound (w : (σ →₀ ℕ) → ℝ) (f : MvPowerSeries σ A) (C : ℝ) : Prop :=
  ∀ s : Finset (σ →₀ ℕ), ∑ n ∈ s, ‖f n‖ * w n ≤ C

 theorem Bound.nonneg {w : (σ →₀ ℕ) → ℝ} {f : MvPowerSeries σ A} {C : ℝ}
    (h : Bound w f C) : 0 ≤ C := by simpa using h ∅

 theorem Bound.summable {w : (σ →₀ ℕ) → ℝ} (hw : ∀ n, 0 ≤ w n)
    {f : MvPowerSeries σ A} {C : ℝ} (h : Bound w f C) :
    Summable (fun n => ‖f n‖ * w n) :=
  summable_of_sum_le (fun n => mul_nonneg (norm_nonneg _) (hw n)) h

 theorem Bound.mono {w : (σ →₀ ℕ) → ℝ} {f : MvPowerSeries σ A} {C D : ℝ}
    (h : Bound w f C) (hCD : C ≤ D) : Bound w f D := fun s => (h s).trans hCD

 theorem bound_zero (w : (σ →₀ ℕ) → ℝ) : Bound w (0 : MvPowerSeries σ A) 0 := by
  intro s; simp [series_zero_apply]

 theorem Bound.of_coeff_le {w : (σ →₀ ℕ) → ℝ} (hw : ∀ n, 0 ≤ w n)
    {f g : MvPowerSeries σ A} {C : ℝ} (hg : Bound w g C)
    (h : ∀ n, ‖f n‖ ≤ ‖g n‖) : Bound w f C := by
  intro s
  exact (Finset.sum_le_sum fun n _ => mul_le_mul_of_nonneg_right (h n) (hw n)).trans (hg s)

 theorem Bound.add {w : (σ →₀ ℕ) → ℝ} (hw : ∀ n, 0 ≤ w n)
    {f g : MvPowerSeries σ A} {C D : ℝ} (hf : Bound w f C) (hg : Bound w g D) :
    Bound w (f+g) (C+D) := by
  intro s
  calc
    _ ≤ ∑ n ∈ s, (‖f n‖ + ‖g n‖) * w n := Finset.sum_le_sum fun n _ =>
      mul_le_mul_of_nonneg_right (norm_add_le _ _) (hw n)
    _ = (∑ n ∈ s, ‖f n‖ * w n) + ∑ n ∈ s, ‖g n‖ * w n := by
      simp [add_mul, Finset.sum_add_distrib]
    _ ≤ _ := add_le_add (hf s) (hg s)

 theorem Bound.neg {w : (σ →₀ ℕ) → ℝ} {f : MvPowerSeries σ A} {C : ℝ}
    (hf : Bound w f C) : Bound w (-f) C := by
  intro s; simpa using hf s

 theorem Bound.sub {w : (σ →₀ ℕ) → ℝ} (hw : ∀ n, 0 ≤ w n)
    {f g : MvPowerSeries σ A} {C D : ℝ} (hf : Bound w f C) (hg : Bound w g D) :
    Bound w (f-g) (C+D) := by simpa only [sub_eq_add_neg] using hf.add hw hg.neg

 theorem Bound.positive {w : (σ →₀ ℕ) → ℝ} (hw : ∀ n, 0 ≤ w n)
    {f : MvPowerSeries σ A} {C : ℝ} (hf : Bound w f C) : Bound w (positive f) C := by
  apply hf.of_coeff_le hw
  intro n
  by_cases hn : n = 0 <;> simp [Pymablock.positive, hn]

 theorem Bound.mul {w : (σ →₀ ℕ) → ℝ} (hw : ∀ n, 0 ≤ w n)
    (hm : ∀ i j, w (i+j) = w i * w j)
    {f g : MvPowerSeries σ A} {C D : ℝ} (hf : Bound w f C) (hg : Bound w g D) :
    Bound w (f*g) (C*D) := by
  classical
  intro s
  let t := s.biUnion Finset.antidiagonal
  let a := fun n => ‖f n‖ * w n
  let b := fun n => ‖g n‖ * w n
  have ha : ∀ n, 0 ≤ a n := fun n => mul_nonneg (norm_nonneg _) (hw n)
  have hb : ∀ n, 0 ≤ b n := fun n => mul_nonneg (norm_nonneg _) (hw n)
  have hcoeff (n : σ →₀ ℕ) : ‖(f*g) n‖ * w n ≤
      ∑ ij ∈ Finset.antidiagonal n, a ij.1 * b ij.2 := by
    change ‖coeff n (f*g)‖ * w n ≤ _
    rw [coeff_mul]
    calc
      _ ≤ (∑ ij ∈ Finset.antidiagonal n, ‖f ij.1 * g ij.2‖) * w n :=
        mul_le_mul_of_nonneg_right (norm_sum_le _ _) (hw n)
      _ = ∑ ij ∈ Finset.antidiagonal n, ‖f ij.1 * g ij.2‖ * w n := Finset.sum_mul _ _ _
      _ ≤ _ := by
        apply Finset.sum_le_sum
        intro ij hij
        rw [← Finset.mem_antidiagonal.mp hij, hm]
        dsimp [a,b]
        nlinarith [norm_mul_le (f ij.1) (g ij.2), hw ij.1, hw ij.2,
          mul_le_mul_of_nonneg_right (norm_mul_le (f ij.1) (g ij.2))
            (mul_nonneg (hw ij.1) (hw ij.2))]
  calc
    _ ≤ ∑ n ∈ s, ∑ ij ∈ Finset.antidiagonal n, a ij.1 * b ij.2 :=
      Finset.sum_le_sum fun n _ => hcoeff n
    _ = ∑ ij ∈ t, a ij.1 * b ij.2 := by
      symm
      apply Finset.sum_biUnion
      intro i hi j hj hij
      apply Finset.disjoint_left.mpr
      intro p hp hq
      exact hij ((Finset.mem_antidiagonal.mp hp).symm.trans (Finset.mem_antidiagonal.mp hq))
    _ ≤ ∑ ij ∈ (t.image Prod.fst) ×ˢ (t.image Prod.snd), a ij.1 * b ij.2 := by
      apply Finset.sum_le_sum_of_subset_of_nonneg
      · intro ij hij
        exact Finset.mem_product.mpr ⟨Finset.mem_image.mpr ⟨ij,hij,rfl⟩,
          Finset.mem_image.mpr ⟨ij,hij,rfl⟩⟩
      · intro ij _ _; exact mul_nonneg (ha _) (hb _)
    _ = (∑ i ∈ t.image Prod.fst, a i) * ∑ j ∈ t.image Prod.snd, b j := by
      rw [Finset.sum_mul_sum, Finset.sum_product]
    _ ≤ C*D := mul_le_mul (hf _) (hg _) (Finset.sum_nonneg fun n _ => hb n) hf.nonneg

section Linear
variable [NormedAlgebra ℚ A]

 theorem Bound.map {w : (σ →₀ ℕ) → ℝ} (hw : ∀ n, 0 ≤ w n)
    {f : MvPowerSeries σ A} {C K : ℝ} (hf : Bound w f C)
    (L : A →ₗ[ℚ] A) (hK : 0 ≤ K) (hL : ∀ a, ‖L a‖ ≤ K * ‖a‖) :
    Bound w (coefficientMap L f) (K*C) := by
  intro s
  calc
    _ ≤ ∑ n ∈ s, (K * ‖f n‖) * w n := Finset.sum_le_sum fun n _ =>
      mul_le_mul_of_nonneg_right (hL _) (hw n)
    _ = K * ∑ n ∈ s, ‖f n‖ * w n := by rw [Finset.mul_sum]; simp [mul_assoc]
    _ ≤ _ := mul_le_mul_of_nonneg_left (hf s) hK

 theorem Bound.smul {w : (σ →₀ ℕ) → ℝ}
    {f : MvPowerSeries σ A} {C : ℝ} (hf : Bound w f C) (c : ℚ) :
    Bound w (c • f) (‖c‖*C) := by
  intro s
  simp only [series_smul_apply, norm_smul, mul_assoc, ← Finset.mul_sum]
  exact mul_le_mul_of_nonneg_left (hf s) (norm_nonneg c)

end Linear

section Star
variable [NormedAlgebra ℚ A] [StarRing A] [NormedStarGroup A]

omit [NormedAlgebra ℚ A] in
 theorem Bound.star {w : (σ →₀ ℕ) → ℝ} {f : MvPowerSeries σ A} {C : ℝ}
    (hf : Bound w f C) : Bound w (star f) C := by
  intro s; simpa using hf s

 theorem Bound.herm {w : (σ →₀ ℕ) → ℝ} (hw : ∀ n, 0 ≤ w n)
    {f : MvPowerSeries σ A} {C : ℝ} (hf : Bound w f C) : Bound w (herm f) C := by
  have h := (hf.add hw hf.star).smul (1/2 : ℚ)
  convert h using 1
  norm_num [← Rat.norm_cast_real]; ring

 theorem Bound.skew {w : (σ →₀ ℕ) → ℝ} (hw : ∀ n, 0 ≤ w n)
    {f : MvPowerSeries σ A} {C : ℝ} (hf : Bound w f C) : Bound w (skew f) C := by
  have h := (hf.sub hw hf.star).smul (1/2 : ℚ)
  convert h using 1
  norm_num [← Rat.norm_cast_real]; ring

end Star

/-- Uniform absolute bounds pass to the coefficientwise stabilized causal solution. -/
 theorem causal_bound {I C : Type*} (rank : I → ℕ) (F : (I → C) → I → C)
    (hF : StrictlyCausal rank F) (seed : I → C) (p : I → C → ℝ) (M : ℝ)
    (hb : ∀ k, ∀ s : Finset I, ∑ i ∈ s, p i ((F^[k] seed) i) ≤ M) :
    ∀ s : Finset I, ∑ i ∈ s, p i (causalSolution rank F seed i) ≤ M := by
  classical
  intro s
  let d := s.sup rank + 1
  calc
    _ = ∑ i ∈ s, p i ((F^[d] seed) i) := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [causalSolution_agrees rank F hF seed d i (by
        have := Finset.le_sup (f := rank) hi
        dsimp [d]; omega)]
    _ ≤ _ := hb d s

end Pymablock.Convergence
