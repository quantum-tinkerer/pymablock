import Pymablock.Convergence.Matrix
import Pymablock.Convergence.Algebra

noncomputable section
namespace Pymablock.Convergence
open MvPowerSeries

variable {σ A : Type*} [NormedRing A] [NormedAlgebra ℚ A] [StarRing A] [NormedStarGroup A]

/-- Both returned formal series have a positive absolute convergence radius.
The input is analytic; output convergence is a conclusion. -/
 theorem blockDiagonalize_positive_radius (P : Selection A) (H : MvPowerSeries σ A)
    (hH : star H = H) (hd : P.diag (H 0) = H 0) (S : SylvesterSolver P (H 0))
    {K R : ℝ} (ctl : Controls P S.solve K) (hR : 0 < R) (ha : Absolute H R) :
    ∃ r, 0 < r ∧ r ≤ R ∧ Absolute (blockDiagonalize P H S).1 r ∧
      Absolute (blockDiagonalize P H S).2 r := by
  have hK : 0 ≤ K := le_trans (by norm_num) ctl.one_le
  have hp := ha.positive hR.le
  have hs : Absolute (selected P H) R := hp.map hR.le P.diag hK ctl.diag
  have hr : Absolute (remaining P H) R := hp.map hR.le P.off hK ctl.off
  obtain ⟨r,hrp,hrR,hq,_⟩ := solution_positive_radius P S.solve ctl (selected P H) (remaining P H)
    (by simp [selected]) (by simp [remaining]) hR hs hr
  have hu : Absolute (blockDiagonalize P H S).2 r := (absolute_one r).add hq hrp.le
  have hc := blockDiagonalize_correct P H hH hd S
  refine ⟨r,hrp,hrR,?_,hu⟩
  rw [hc.2.2.2.1]
  exact ((hu.star hrp.le).mul (ha.radius_mono hrp.le hrR) hrp.le).mul hu hrp.le

omit [NormedStarGroup A] in
 theorem transport_solver (P : Selection A) {a b : A} (h : a = b)
    (S : SylvesterSolver P a) : (h ▸ S).solve = S.solve := by
  cases h
  rfl

section Concrete
open scoped Matrix.Norms.Frobenius
variable {ι β : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq β]

/-- End-to-end convergence and formal correctness from analytic Hermitian input,
a diagonal unperturbed Hamiltonian, and cross-block spectral separation. -/
 theorem matrix_convergent (block : ι → β) (e : ι → ℝ)
    (H : MvPowerSeries σ (Matrix ι ι ℂ)) (hH : star H = H)
    (h0 : H 0 = Matrix.diagonal (fun i => (e i : ℂ)))
    (gap : ∀ i j, block i ≠ block j → e i ≠ e j)
    {R : ℝ} (hR : 0 < R) (ha : Absolute H R) :
    ∃ ht u : MvPowerSeries σ (Matrix ι ι ℂ), ∃ r : ℝ,
      0 < r ∧ r ≤ R ∧ Absolute ht r ∧ Absolute u r ∧
      u 0 = 1 ∧ star u * u = 1 ∧ u * star u = 1 ∧ ht = star u * H * u ∧
      (MatrixBlocks.blockStructure block).series.off ht = 0 ∧ star ht = ht ∧
      (MatrixBlocks.blockStructure block).series.diag (skew (u-1)) = 0 := by
  classical
  let P : Selection (Matrix ι ι ℂ) := MatrixBlocks.blockStructure block
  let base := MatrixBlocks.spectralSolver block (fun i => (e i : ℂ))
    (by intro i; simp) (by intro i j hij he; exact gap i j hij (Complex.ofReal_injective he))
  let S : SylvesterSolver P (H 0) := h0.symm ▸ base
  have heq : S.solve = MatrixBlocks.spectralMap block (fun i => (e i : ℂ)) := by
    exact transport_solver P h0.symm base
  obtain ⟨K,hK⟩ := matrix_controls block (fun i => (e i : ℂ))
  have hc : Controls P S.solve K := by rw [heq]; exact hK
  have hd : P.diag (H 0) = H 0 := by
    rw [h0]
    ext i j
    change (if block i = block j then Matrix.diagonal (fun i => (e i : ℂ)) i j else 0) = _
    by_cases hij : i = j
    · subst j; simp
    · simp [hij]
  obtain ⟨r,hr,hrR,ht,hu⟩ := blockDiagonalize_positive_radius P H hH hd S hc hR ha
  exact ⟨(blockDiagonalize P H S).1,(blockDiagonalize P H S).2,r,hr,hrR,ht,hu,
    blockDiagonalize_correct P H hH hd S⟩

end Concrete
end Pymablock.Convergence
