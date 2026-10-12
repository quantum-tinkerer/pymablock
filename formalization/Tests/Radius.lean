import Pymablock.Convergence.Inputs

noncomputable section
open scoped Matrix.Norms.Frobenius
open Pymablock Pymablock.Convergence MvPowerSeries
namespace Pymablock.Convergence.Tests

abbrev RadiusMatrix := Matrix (Fin 3) (Fin 3) ℂ

/-- Two parameters and three blocks: an explicit radius from the input norms,
with no output convergence premise. -/
example (V : Fin 2 → RadiusMatrix) (hV : ∀ i, star (V i) = V i)
    (hv : 0 < ∑ i, ‖V i‖) :
    let H := affine (Matrix.diagonal (fun i : Fin 3 => (i.val : ℂ))) V
    let r := 9/(130*∑ i, ‖V i‖)
    ∃ ht u : MvPowerSeries (Fin 2) RadiusMatrix,
      Absolute ht r ∧ Absolute u r ∧ star u*u=1 ∧
      ht=star u*H*u ∧ (MatrixBlocks.blockStructure (fun i : Fin 3 => i)).series.off ht=0 := by
  let H0 : RadiusMatrix := Matrix.diagonal (fun i : Fin 3 => (i.val : ℂ))
  have hh0 : star H0 = H0 := by
    ext i j
    by_cases hij : i = j
    · subst j; simp [H0,Matrix.star_apply]
    · simp [H0,Matrix.star_apply,hij,Ne.symm hij]
  have hc := affine_radius_certificate H0 V (δ := 1) (by norm_num) hv
  dsimp only at hc
  simp only [mul_one] at hc
  have hg : ∀ i j : Fin 3, i ≠ j → (1 : ℝ) ≤ ‖((j.val : ℝ) : ℂ)-((i.val : ℝ) : ℂ)‖ := by
    intro i j hij
    fin_cases i <;> fin_cases j <;> norm_num [Complex.norm_def] at *
  obtain ⟨ht,u,ha,hb,_,hl,_,he,ho,_,_⟩ := matrix_radius_bound (fun i : Fin 3 => i)
    (fun i => (i.val : ℝ)) (affine H0 V) (affine_hermitian H0 V hh0 hV)
    (by simpa [H0] using affine_zero H0 V) (δ := 1) (ε := 9/130)
    (by norm_num) hc.1.le (by norm_num) hg (by norm_num) hc.2
  exact ⟨ht,u,ha,hb,hl,he,ho⟩

/-- The stronger 1/8 threshold for purely off-block perturbation mass. -/
example {σ A : Type*} [NormedRing A] [NormedAlgebra ℚ A] [StarRing A] [NormedStarGroup A]
    (P : Selection A) (solve : A →ₗ[ℚ] A) (ctl : GapControls P solve 1)
    (hs ho : MvPowerSeries σ A) (hs0 : hs 0 = 0) (ho0 : ho 0 = 0)
    (hd : Bound (fun n => (1 : ℝ)^Finsupp.degree n) hs 0)
    (hh : Bound (fun n => (1 : ℝ)^Finsupp.degree n) ho (1/8)) :
    Absolute (qSeries (solution P solve hs ho)) 1 ∧
    Absolute (bSeries (solution P solve hs ho)) 1 :=
  solution_split_bound P solve ctl (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hs ho hs0 ho0 hd hh

/-- The polynomial optimized inequality recovers the simple universal constant. -/
example : (9/130 : ℝ)*(1-2*(1/5)*(1/5)) + 4*(9/130)*(1/5)*(1-1/5) =
    (1/5)*(1-(1/5)/2)*(1-2*(1/5)) := by norm_num

end Pymablock.Convergence.Tests
