import Pymablock.NonHermitian.Correctness

noncomputable section
namespace Pymablock.NonHermitian
open MvPowerSeries
variable {σ A : Type*} [Ring A] [Algebra ℚ A]

/-- Truncating all three outputs preserves both inverse identities and the
similarity transformation through every chosen total degree, including mixed terms. -/
theorem truncated_correct (P : Projection A) (H : MvPowerSeries σ A)
    (h0 : P.selected (H 0) = H 0) (S : Solver P (H 0)) (N : ℕ) :
    let result := diagonalize P H S
    let ht := truncate N result.1
    let u := truncate N result.2.1
    let ui := truncate N result.2.2
    JetEq (N+1) (ui*u) 1 ∧ JetEq (N+1) (u*ui) 1 ∧
      JetEq (N+1) ht (ui * truncate N H * u) ∧ P.series.remaining ht = 0 := by
  let result := diagonalize P H S
  have hc := diagonalize_correct P H h0 S
  dsimp only at hc ⊢
  obtain ⟨_, _, hl, hr, he, hoff, _⟩ := hc
  have hu := truncate_agrees N result.2.1
  have hi := truncate_agrees N result.2.2
  have hh := truncate_agrees N result.1
  refine ⟨?_, ?_, ?_, ?_⟩
  · simpa only [result, hl] using hi.mul hu
  · simpa only [result, hr] using hu.mul hi
  · have ht : JetEq (N+1) (truncate N result.1)
        ((diagonalize P H S).2.2 * H * (diagonalize P H S).2.1) := by
      simpa only [result, he] using hh
    exact ht.trans ((hi.mul (truncate_agrees N H)).mul hu).symm
  · ext n
    change P.remaining (truncate N result.1 n) = 0
    by_cases hn : Finsupp.degree n ≤ N
    · simp only [truncate, if_pos hn]
      exact congrFun hoff n
    · simp only [truncate, if_neg hn, map_zero]

end Pymablock.NonHermitian
