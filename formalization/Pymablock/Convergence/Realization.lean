import Pymablock.Convergence.Output
import Pymablock.Convergence.Evaluation
import Pymablock.LeastAction.Formal

noncomputable section
set_option backward.isDefEq.respectTransparency false
open scoped Matrix.Norms.Frobenius ComplexOrder Topology
namespace Pymablock.Convergence

variable {σ ι β : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq β]

def diagCL (block : ι → β) : Matrix ι ι ℂ →L[ℝ] Matrix ι ι ℂ :=
  (entryLinear (fun i j => if block i = block j then 1 else 0)).toContinuousLinearMap.restrictScalars ℝ

omit [DecidableEq ι] in
 theorem diagCL_apply (block : ι → β) (a : Matrix ι ι ℂ) :
    diagCL block a = MatrixBlocks.diag block a := by
  ext i j
  change (if block i = block j then 1 else 0)*a i j = if block i = block j then a i j else 0
  split_ifs <;> simp

 theorem evaluate_diag (block : ι → β) {f : MvPowerSeries σ (Matrix ι ι ℂ)}
    {r : ℝ} (hf : Absolute f r) (hr : 0 ≤ r) (x : σ → ℝ) (hx : ∀ i, ‖x i‖ ≤ r) :
    evaluate ((MatrixBlocks.blockStructure block).series.diag f) x =
      MatrixBlocks.diag block (evaluate f x) := by
  have he : (fun n => diagCL block (f n)) = (MatrixBlocks.blockStructure block).series.diag f := by
    funext n
    exact diagCL_apply block (f n)
  rw [← he, evaluate_linear (diagCL block) hf hr x hx, diagCL_apply]

/-- The formal two-sided unitary equations hold for the convergent sum. -/
 theorem realized_unitary {u : MvPowerSeries σ (Matrix ι ι ℂ)} {r : ℝ}
    (ha : Absolute u r) (hr : 0 ≤ r) (hl : star u*u=1) (hh : u*star u=1)
    (x : σ → ℝ) (hx : ∀ i, ‖x i‖ ≤ r) :
    star (evaluate u x)*evaluate u x=1 ∧ evaluate u x*star (evaluate u x)=1 := by
  have h1 := congrArg (fun f => evaluate f x) hl
  have h2 := congrArg (fun f => evaluate f x) hh
  dsimp only at h1 h2
  rw [evaluate_mul (ha.star hr) ha hr x hx, evaluate_star, evaluate_one] at h1
  rw [evaluate_mul ha (ha.star hr) hr x hx, evaluate_star, evaluate_one] at h2
  exact ⟨h1,h2⟩

/-- The formal gauge passes through summation, without being postulated for the realization. -/
 theorem realized_gauge (block : ι → β) {u : MvPowerSeries σ (Matrix ι ι ℂ)} {r : ℝ}
    (ha : Absolute u r) (hr : 0 ≤ r)
    (hg : (MatrixBlocks.blockStructure block).series.diag (skew (u-1)) = 0)
    (x : σ → ℝ) (hx : ∀ i, ‖x i‖ ≤ r) :
    MatrixBlocks.diag block (skew (evaluate u x-1)) = 0 := by
  let P := MatrixBlocks.blockStructure (R := ℂ) block
  have hh := LeastAction.selected_selfadjoint_of_gauge P.series u hg
  have he := congrArg (fun f => evaluate f x) hh
  dsimp only at he
  rw [evaluate_star, evaluate_diag block ha hr x hx] at he
  change P.diag (skew (evaluate u x-1)) = 0
  rw [P.diag_skew]
  apply skew_of_selfadjoint
  simp only [map_sub, star_sub]
  rw [show star (P.diag (evaluate u x)) = P.diag (evaluate u x) from he]
  have h1 : star (P.diag 1) = P.diag 1 := by rw [← P.star_diag, star_one]
  rw [h1]

/-- Similarity and block elimination also survive evaluation of the convergent series. -/
 theorem realized_hamiltonian (block : ι → β)
    {H ht u : MvPowerSeries σ (Matrix ι ι ℂ)} {r : ℝ}
    (hH : Absolute H r) (hu : Absolute u r) (hht : Absolute ht r) (hr : 0 ≤ r)
    (he : ht = star u*H*u) (hoff : (MatrixBlocks.blockStructure block).series.off ht = 0)
    (x : σ → ℝ) (hx : ∀ i, ‖x i‖ ≤ r) :
    evaluate ht x = star (evaluate u x)*evaluate H x*evaluate u x ∧
      (MatrixBlocks.blockStructure block).off (evaluate ht x) = 0 := by
  constructor
  · rw [he, evaluate_mul ((hu.star hr).mul hH hr) hu hr x hx,
      evaluate_mul (hu.star hr) hH hr x hx, evaluate_star]
  · have hd : (MatrixBlocks.blockStructure block).series.diag ht = ht := by
      exact (sub_eq_zero.mp hoff).symm
    have hh := congrArg (fun f => evaluate f x) hd
    dsimp only at hh
    rw [evaluate_diag block hht hr x hx] at hh
    change evaluate ht x - MatrixBlocks.diag block (evaluate ht x) = 0
    rw [hh, sub_self]

/-- The convergence proof discharges the previous realization assumptions in the
local least-action theorem. Only properties of the formal series are inputs. -/
 theorem convergent_locally_closest [Finite σ] (block : ι → β)
    (u : MvPowerSeries σ (Matrix ι ι ℂ)) {r : ℝ} (hr : 0 < r) (ha : Absolute u r)
    (h0 : u 0 = 1) (hl : star u*u=1) (hh : u*star u=1)
    (hg : (MatrixBlocks.blockStructure block).series.diag (skew (u-1)) = 0) :
    ∀ᶠ x : σ → ℝ in nhds 0, ∀ T : Matrix ι ι ℂ,
      star T*T=1 → T*star T=1 → LeastAction.SameAssignment block (evaluate u x) T →
      LeastAction.frobeniusSq (evaluate u x-1) ≤ LeastAction.frobeniusSq (T-1) ∧
        (LeastAction.frobeniusSq (T-1) = LeastAction.frobeniusSq (evaluate u x-1) ↔ T = evaluate u x) := by
  apply LeastAction.locally_closest_to_identity 0 block (evaluate u)
    (ha.continuousAt_evaluate_zero hr) (by rw [evaluate_zero,h0])
  · filter_upwards [polydisc_mem_nhds (σ := σ) hr] with x hx
    exact realized_gauge block ha hr.le hg x hx
  · filter_upwards [polydisc_mem_nhds (σ := σ) hr] with x hx
    exact realized_unitary ha hr.le hl hh x hx

/-- End-to-end least action: analytic Hermitian input and separated blocks imply
convergent output whose sum is locally the unique closest unitary. -/
 theorem matrix_least_action [Finite σ] (block : ι → β) (e : ι → ℝ)
    (H : MvPowerSeries σ (Matrix ι ι ℂ)) (hH : star H = H)
    (h0 : H 0 = Matrix.diagonal (fun i => (e i : ℂ)))
    (gap : ∀ i j, block i ≠ block j → e i ≠ e j)
    {R : ℝ} (hR : 0 < R) (ha : Absolute H R) :
    ∃ u : MvPowerSeries σ (Matrix ι ι ℂ), ∃ r : ℝ, 0 < r ∧ Absolute u r ∧
    u 0 = 1 ∧ star u*u=1 ∧ u*star u=1 ∧
    (MatrixBlocks.blockStructure block).series.off (star u*H*u)=0 ∧
    ∀ᶠ x : σ → ℝ in nhds 0, ∀ T : Matrix ι ι ℂ,
      star T*T=1 → T*star T=1 → LeastAction.SameAssignment block (evaluate u x) T →
      LeastAction.frobeniusSq (evaluate u x-1) ≤ LeastAction.frobeniusSq (T-1) ∧
        (LeastAction.frobeniusSq (T-1) = LeastAction.frobeniusSq (evaluate u x-1) ↔ T = evaluate u x) := by
  obtain ⟨ht,u,r,hr,_,_,hu,hzero,hl,hh,he,hoff,_,hg⟩ := matrix_convergent block e H hH h0 gap hR ha
  refine ⟨u,r,hr,hu,hzero,hl,hh,?_,convergent_locally_closest block u hr hu hzero hl hh hg⟩
  rw [← he]
  exact hoff

end Pymablock.Convergence
