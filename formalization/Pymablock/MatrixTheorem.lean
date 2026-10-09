import Pymablock.SpectralSolver
import Pymablock.FiniteOrder

/-! A matrix theorem whose only mathematical inputs are the Hermitian
Hamiltonian, an eigenbasis for H₀, a block partition, and separated energies. -/
noncomputable section
namespace Pymablock
open MvPowerSeries

variable {σ ι β : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq β]

/-- Arbitrarily many perturbations and blocks, with a concrete complex-matrix
Sylvester solver. Degeneracies inside each block are permitted. -/
theorem matrix_block_diagonalization
    (block : ι → β) (e : ι → ℝ) (H : MvPowerSeries σ (Matrix ι ι ℂ))
    (hH : star H = H) (h0 : H 0 = Matrix.diagonal (fun i => (e i : ℂ)))
    (gap : ∀ i j, block i ≠ block j → e i ≠ e j) :
    ∃ ht u : MvPowerSeries σ (Matrix ι ι ℂ),
      u 0 = 1 ∧ star u * u = 1 ∧ u * star u = 1 ∧ ht = star u * H * u ∧
      (MatrixBlocks.blockStructure block).series.off ht = 0 ∧ star ht = ht ∧
      (MatrixBlocks.blockStructure block).series.diag (skew (u - 1)) = 0 := by
  classical
  let P : BlockStructure (Matrix ι ι ℂ) := MatrixBlocks.blockStructure block
  let S : SylvesterSolver P (H 0) := by
    rw [h0]
    exact MatrixBlocks.spectralSolver block (fun i => (e i : ℂ)) (by intro i; simp)
      (by intro i j hij he; exact gap i j hij (Complex.ofReal_injective he))
  have hd : P.diag (H 0) = H 0 := by
    rw [h0]
    ext i j
    change (if block i = block j then Matrix.diagonal (fun i => (e i : ℂ)) i j else 0) =
      Matrix.diagonal (fun i => (e i : ℂ)) i j
    by_cases h : i = j
    · subst j; simp
    · simp [h]
  exact ⟨(blockDiagonalize P H S).1, (blockDiagonalize P H S).2,
    blockDiagonalize_correct P H hH hd S⟩

end Pymablock
