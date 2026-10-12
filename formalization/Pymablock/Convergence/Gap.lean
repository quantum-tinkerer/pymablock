import Pymablock.Convergence.Quantitative

noncomputable section
open scoped Matrix.Norms.Frobenius
namespace Pymablock.Convergence
variable {ι β σ : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq β]

private theorem norm_sq_entries (a : Matrix ι ι ℂ) : ‖a‖^2 = ∑ i, ∑ j, ‖a i j‖^2 := by
  rw [Matrix.frobenius_norm_def, ← Real.sqrt_eq_rpow]
  rw [Real.sq_sqrt (by positivity)]
  simp

/-- Entrywise multipliers are bounded by their largest modulus in Frobenius norm. -/
theorem norm_le_of_entries (a b : Matrix ι ι ℂ) {L : ℝ} (hL : 0 ≤ L)
    (h : ∀ i j, ‖a i j‖ ≤ L*‖b i j‖) : ‖a‖ ≤ L*‖b‖ := by
  have hs : ‖a‖^2 ≤ (L*‖b‖)^2 := by
    rw [mul_pow, norm_sq_entries, norm_sq_entries, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro j _
    simpa [mul_pow] using sq_le_sq₀ (norm_nonneg (a i j)) (mul_nonneg hL (norm_nonneg (b i j))) |>.mpr (h i j)
  nlinarith [norm_nonneg a, norm_nonneg b, mul_nonneg hL (norm_nonneg b)]

/-- Every retained/eliminated entry projection is a Frobenius contraction. -/
theorem block_contractions (block : ι → β) :
    (∀ a : Matrix ι ι ℂ, ‖(MatrixBlocks.blockStructure block).diag a‖ ≤ ‖a‖) ∧
    (∀ a : Matrix ι ι ℂ, ‖(MatrixBlocks.blockStructure block).off a‖ ≤ ‖a‖) := by
  constructor
  · intro a
    simpa using norm_le_of_entries ((MatrixBlocks.blockStructure block).diag a) a
      (L := 1) (by norm_num) (by
        intro i j
        change ‖if block i = block j then a i j else 0‖ ≤ _
        split_ifs <;> simp)
  · intro a
    simpa using norm_le_of_entries ((MatrixBlocks.blockStructure block).off a) a
      (L := 1) (by norm_num) (by
        intro i j
        change ‖a i j - (if block i = block j then a i j else 0)‖ ≤ _
        split_ifs <;> simp)

/-- The inverse Sylvester bound is 1/gap, with no matrix-size or block-count factor. -/
theorem gap_controls (block : ι → β) (e : ι → ℂ) {δ : ℝ} (hδ : 0 < δ)
    (gap : ∀ i j, block i ≠ block j → δ ≤ ‖e j-e i‖) :
    GapControls (MatrixBlocks.blockStructure block) (MatrixBlocks.spectralMap block e) (1/δ) := by
  obtain ⟨hd,ho⟩ := block_contractions block
  refine ⟨by positivity,hd,ho,?_⟩
  intro a
  apply norm_le_of_entries _ a (by positivity)
  intro i j
  rw [MatrixBlocks.spectralMap_apply]
  split_ifs with hij
  · simp only [norm_zero]; positivity
  · rw [norm_div]
    calc
      _ ≤ ‖a i j‖/δ := div_le_div_of_nonneg_left (norm_nonneg _) hδ (gap i j hij)
      _ = _ := by ring

/-- A known input majorant gives convergence at a specified radius, not an
unspecified smaller one. Retained degeneracies are allowed. -/
theorem matrix_radius_bound (block : ι → β) (e : ι → ℝ)
    (H : MvPowerSeries σ (Matrix ι ι ℂ)) (hH : star H = H)
    (h0 : H 0 = Matrix.diagonal (fun i => (e i : ℂ)))
    {δ ε r : ℝ} (hδ : 0 < δ) (hr : 0 ≤ r) (hε : 0 ≤ ε)
    (gap : ∀ i j, block i ≠ block j → δ ≤ ‖(e j : ℂ)-(e i : ℂ)‖)
    (hsmall : ε ≤ 9*δ/130)
    (hin : Bound (fun n => r^Finsupp.degree n) (positive H) ε) :
    ∃ ht u : MvPowerSeries σ (Matrix ι ι ℂ),
      Absolute ht r ∧ Absolute u r ∧ u 0 = 1 ∧ star u*u=1 ∧ u*star u=1 ∧
      ht=star u*H*u ∧ (MatrixBlocks.blockStructure block).series.off ht=0 ∧
      star ht=ht ∧ (MatrixBlocks.blockStructure block).series.diag (skew (u-1))=0 := by
  classical
  let P : Selection (Matrix ι ι ℂ) := MatrixBlocks.blockStructure block
  have hg : ∀ i j, block i ≠ block j → (e i : ℂ) ≠ (e j : ℂ) := by
    intro i j hij he
    have hh := gap i j hij
    rw [he,sub_self,norm_zero] at hh
    exact (not_le_of_gt hδ) hh
  let base := MatrixBlocks.spectralSolver block (fun i => (e i : ℂ)) (by intro i; simp) hg
  let S : SylvesterSolver P (H 0) := h0.symm ▸ base
  have heq : S.solve = MatrixBlocks.spectralMap block (fun i => (e i : ℂ)) :=
    transport_solver P h0.symm base
  have hc : GapControls P S.solve (1/δ) := by
    rw [heq]; exact gap_controls block _ hδ gap
  have hd : P.diag (H 0) = H 0 := by
    rw [h0]
    ext i j
    change (if block i = block j then Matrix.diagonal (fun i => (e i : ℂ)) i j else 0) = _
    by_cases hij : i = j
    · subst j; simp
    · simp [hij]
  have hsm : (1/δ)*ε ≤ 9/130 := by
    rw [one_div,mul_comm,← div_eq_mul_inv]
    apply (div_le_iff₀ hδ).mpr
    nlinarith
  obtain ⟨ht,hu⟩ := blockDiagonalize_at_radius P H hH hd S hc hr hε hsm hin
  exact ⟨(blockDiagonalize P H S).1,(blockDiagonalize P H S).2,ht,hu,
    blockDiagonalize_correct P H hH hd S⟩

end Pymablock.Convergence
