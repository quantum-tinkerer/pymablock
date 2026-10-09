import Pymablock.Hamiltonian

/-! Finite total-degree truncations of the formal solution. -/
noncomputable section
namespace Pymablock
open MvPowerSeries

variable {σ A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

namespace JetEq
omit [Ring A] [Algebra ℚ A] [StarRing A] in
 theorem symm {d : ℕ} {f g : MvPowerSeries σ A} (h : JetEq d f g) : JetEq d g f :=
  fun n hn => (h n hn).symm
omit [Ring A] [Algebra ℚ A] [StarRing A] in
 theorem trans {d : ℕ} {f g h : MvPowerSeries σ A} (hfg : JetEq d f g) (hgh : JetEq d g h) :
    JetEq d f h := fun n hn => (hfg n hn).trans (hgh n hn)
omit [Algebra ℚ A] [StarRing A] in
 theorem mul {d : ℕ} {f f' g g' : MvPowerSeries σ A}
    (hf : JetEq d f f') (hg : JetEq d g g') : JetEq d (f * g) (f' * g') := by
  classical
  intro n hn
  change coeff n (f * g) = coeff n (f' * g')
  rw [coeff_mul, coeff_mul]
  apply Finset.sum_congr rfl
  rintro ⟨i,j⟩ hij
  have he : Finsupp.degree i + Finsupp.degree j = Finsupp.degree n := by
    rw [← map_add, Finset.mem_antidiagonal.mp hij]
  change f i * g j = f' i * g' j
  rw [hf i (by omega), hg j (by omega)]
end JetEq

/-- Keep all coefficients of total degree at most N. For finitely many
parameters this is a polynomial; multiplication remains the Cauchy product. -/
def truncate (N : ℕ) (f : MvPowerSeries σ A) : MvPowerSeries σ A :=
  fun n => if Finsupp.degree n ≤ N then f n else 0

omit [Algebra ℚ A] [StarRing A] in
 theorem truncate_agrees (N : ℕ) (f : MvPowerSeries σ A) : JetEq (N+1) (truncate N f) f := by
  intro n hn
  simp [truncate, show Finsupp.degree n ≤ N by omega]

/-- Products of the truncated outputs obey the defining identities modulo
total degree N+1. This includes all mixed perturbative terms. -/
theorem truncated_correct (P : BlockStructure A) (H : MvPowerSeries σ A)
    (hH : star H = H) (h0diag : P.diag (H 0) = H 0)
    (S : SylvesterSolver P (H 0)) (N : ℕ) :
    let result := blockDiagonalize P H S
    let ht := truncate N result.1
    let u := truncate N result.2
    JetEq (N+1) (star u * u) 1 ∧
      JetEq (N+1) (u * star u) 1 ∧
      JetEq (N+1) ht (star u * truncate N H * u) ∧
      P.series.off ht = 0 := by
  let result := blockDiagonalize P H S
  have hc := blockDiagonalize_correct P H hH h0diag S
  dsimp only at hc ⊢
  have hu := truncate_agrees N result.2
  have hh := truncate_agrees N result.1
  refine ⟨?_, ?_, ?_, ?_⟩
  · have h := hu.star.mul hu
    simpa only [result, hc.2.1] using h
  · have h := hu.mul hu.star
    simpa only [result, hc.2.2.1] using h
  · have h := (hu.star.mul (truncate_agrees N H)).mul hu
    have he := hc.2.2.2.1
    have hh' : JetEq (N+1) (truncate N result.1) (star result.2 * H * result.2) := by
      simpa only [result, he] using hh
    exact hh'.trans h.symm
  · have hoff := hc.2.2.2.2.1
    ext n
    change P.off (truncate N result.1 n) = 0
    by_cases hn : Finsupp.degree n ≤ N
    · simp only [truncate, if_pos hn]
      exact congrFun hoff n
    · simp only [truncate, if_neg hn, map_zero]

end Pymablock
