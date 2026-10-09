import Pymablock.LeastAction.Minimality
import Pymablock.Library.Parts
import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.CStarAlgebra.ContinuousFunctionalCalculus.Order

noncomputable section
set_option backward.isDefEq.respectTransparency false
open scoped ComplexOrder MatrixOrder Matrix.Norms.L2Operator Topology

namespace Pymablock.LeastAction

/-- The formal gauge already proved for the recurrence is precisely a
self-adjointness statement about the selected part of U. -/
 theorem selected_selfadjoint_of_gauge {A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]
    (P : Selection A) (u : A) (hg : P.diag (skew (u - 1)) = 0) :
    star (P.diag u) = P.diag u := by
  have h := congrArg P.diag (two_skew (u - 1))
  simp only [map_add, hg, zero_add, star_sub, star_one, sub_sub_sub_cancel_right,
    map_sub, P.star_diag] at h
  exact (sub_eq_zero.mp h.symm).symm

variable {ι β : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq β]

/-- The exact Pymablock gauge implies Hermitian retained blocks. -/
 theorem gauge_hermitian (block : ι → β) (U : Matrix ι ι ℂ)
    (hg : MatrixBlocks.diag block (skew (U - 1)) = 0) :
    (MatrixBlocks.diag block U).IsHermitian := by
  exact selected_selfadjoint_of_gauge (MatrixBlocks.blockStructure block) U hg

/-- Bundle mathlib's operator-norm structures without adding mathematical assumptions. -/
local instance : CStarAlgebra (Matrix ι ι ℂ) where

/-- A quantitative sufficient condition for the positive branch. The norm here
is the operator norm; the objective in the minimality theorem is Frobenius. -/
 theorem posDef_of_norm_sub_one_lt (A : Matrix ι ι ℂ) (hA : A.IsHermitian)
    (hn : ‖A - 1‖ < 1) : A.PosDef := by
  have hb := IsSelfAdjoint.neg_algebraMap_norm_le_self (a := A - 1)
    (hA.isSelfAdjoint.sub (IsSelfAdjoint.one _))
  have hp : IsStrictlyPositive (algebraMap ℝ (Matrix ι ι ℂ) (1 - ‖A - 1‖)) :=
    isStrictlyPositive_algebraMap (sub_pos.mpr hn)
  apply Matrix.IsStrictlyPositive.posDef
  apply hp.of_le
  rw [map_sub, map_one]
  have hh := add_le_add_right hb (1 : Matrix ι ι ℂ)
  convert hh using 1
  abel

omit [Fintype ι] [DecidableEq ι] in
 theorem continuous_diag (block : ι → β) :
    Continuous (fun U : Matrix ι ι ℂ => MatrixBlocks.diag block U) := by
  apply continuous_pi
  intro i
  apply continuous_pi
  intro j
  by_cases h : block i = block j
  · simpa only [MatrixBlocks.diag_apply, if_pos h] using
      (continuous_apply j).comp (continuous_apply i)
  · simp only [MatrixBlocks.diag_apply, if_neg h]
    exact continuous_const

omit [Fintype ι] in
@[simp] theorem diag_one (block : ι → β) :
    MatrixBlocks.diag block (1 : Matrix ι ι ℂ) = 1 := by
  ext i j
  by_cases hij : i = j
  · subst j; simp
  · simp [hij]

/-- Any continuous realization of the gauge through the identity has positive
diagonal blocks in a neighborhood. No convergence of a formal series is asserted. -/
 theorem eventually_posDef {X : Type*} [TopologicalSpace X] (x₀ : X)
    (block : ι → β) (U : X → Matrix ι ι ℂ) (hc : ContinuousAt U x₀)
    (h0 : U x₀ = 1)
    (hg : ∀ᶠ x in nhds x₀, MatrixBlocks.diag block (skew (U x - 1)) = 0) :
    ∀ᶠ x in nhds x₀, (MatrixBlocks.diag block (U x)).PosDef := by
  have hcont : ContinuousAt (fun x => MatrixBlocks.diag block (U x) - 1) x₀ :=
    ((continuous_diag block).continuousAt.comp hc).sub continuousAt_const
  have hn : ∀ᶠ x in nhds x₀, ‖MatrixBlocks.diag block (U x) - 1‖ < 1 :=
    hcont.norm.eventually_lt continuousAt_const (by simp [h0])
  filter_upwards [hg, hn] with x hx hnx
  exact posDef_of_norm_sub_one_lt _ (gauge_hermitian block (U x) hx) hnx

/-- Conditional local least-action theorem. The parameter space is arbitrary,
so in particular it covers any finite number of real perturbation parameters. -/
 theorem locally_closest_to_identity {X : Type*} [TopologicalSpace X] (x₀ : X)
    (block : ι → β) (U : X → Matrix ι ι ℂ) (hc : ContinuousAt U x₀)
    (h0 : U x₀ = 1)
    (hg : ∀ᶠ x in nhds x₀, MatrixBlocks.diag block (skew (U x - 1)) = 0)
    (hu : ∀ᶠ x in nhds x₀, star (U x) * U x = 1 ∧ U x * star (U x) = 1) :
    ∀ᶠ x in nhds x₀, ∀ T : Matrix ι ι ℂ,
      star T * T = 1 → T * star T = 1 → SameAssignment block (U x) T →
      frobeniusSq (U x - 1) ≤ frobeniusSq (T - 1) ∧
        (frobeniusSq (T - 1) = frobeniusSq (U x - 1) ↔ T = U x) := by
  filter_upwards [eventually_posDef x₀ block U hc h0 hg, hu] with x hp hx
  intro T ht ht' ha
  exact closest_to_identity block (U x) T hx.1 hx.2 ht ht' ha hp

end Pymablock.LeastAction
