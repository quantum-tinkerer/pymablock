import Mathlib.Analysis.CStarAlgebra.Matrix
import Pymablock.LeastAction.Frobenius
import Pymablock.LeastAction.Blocks

noncomputable section
set_option backward.isDefEq.respectTransparency false
open scoped MatrixOrder ComplexOrder Matrix.Norms.L2Operator

namespace Pymablock.LeastAction

variable {ι β : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq β]

/-- Exact nonnegative certificate for the cost of a within-block rotation.
The square root is the positive matrix square root of the retained blocks. -/
 theorem distance_certificate (block : ι → β) (U D : Matrix ι ι ℂ)
    (hU : star U * U = 1) (hD : star D * D = 1) (hD' : D * star D = 1)
    (hblock : BlockDiagonal block D) (hpos : (MatrixBlocks.diag block U).PosDef) :
    frobeniusSq (U * D - 1) - frobeniusSq (U - 1) =
      frobeniusSq (CFC.sqrt (MatrixBlocks.diag block U) * (D - 1)) := by
  let A := MatrixBlocks.diag block U
  let R := CFC.sqrt A
  have hA : IsStrictlyPositive A := hpos.isStrictlyPositive
  have hR : star R = R := (IsStrictlyPositive.sqrt A hA).isSelfAdjoint
  have hRR : R * R = A := CFC.sqrt_mul_sqrt_self A hpos.posSemidef.nonneg
  have hUD : star (U * D) * (U * D) = 1 := by
    simp only [star_mul]
    calc
      star D * star U * (U * D) = star D * (star U * U) * D := by noncomm_ring
      _ = 1 := by rw [hU, mul_one, hD]
  have hexpand : star (R * (D - 1)) * (R * (D - 1)) =
      star D * A * D - star D * A - A * D + A := by
    rw [star_mul, star_sub, star_one, hR]
    calc
      _ = star D * (R * R) * D - star D * (R * R) - (R * R) * D + R * R := by
        noncomm_ring
      _ = _ := by rw [hRR]
  have hcycle : Matrix.trace (star D * A * D) = Matrix.trace A := by
    rw [Matrix.trace_mul_cycle, hD', one_mul]
  have hconj : (Matrix.trace (star D * A)).re = (Matrix.trace (A * D)).re := by
    have ha : star A = A := hpos.isHermitian
    have ht : Matrix.trace (star D * A) = star (Matrix.trace (A * D)) := by
      rw [← Matrix.trace_conjTranspose]
      change Matrix.trace (star D * A) = Matrix.trace (star (A * D))
      rw [star_mul, ha]
    rw [ht]
    rfl
  rw [frobeniusSq_sub_one _ hUD, frobeniusSq_sub_one _ hU]
  change _ = frobeniusSq (R * (D - 1))
  rw [frobeniusSq, hexpand, Matrix.trace_add, Matrix.trace_sub, Matrix.trace_sub,
    hcycle, Complex.add_re, Complex.sub_re, Complex.sub_re, hconj]
  rw [trace_mul_diag block U D hblock]
  change _ = (Matrix.trace A).re - (Matrix.trace (A * D)).re -
    (Matrix.trace (A * D)).re + (Matrix.trace A).re
  dsimp [A]
  rw [trace_diag]
  ring

/-- Positive diagonal blocks uniquely minimize the squared Frobenius distance
within the complete block-unitary freedom of a fixed subspace assignment. -/
 theorem block_rotation_minimality (block : ι → β) (U D : Matrix ι ι ℂ)
    (hU : star U * U = 1) (hD : star D * D = 1) (hD' : D * star D = 1)
    (hblock : BlockDiagonal block D) (hpos : (MatrixBlocks.diag block U).PosDef) :
    frobeniusSq (U - 1) ≤ frobeniusSq (U * D - 1) ∧
      (frobeniusSq (U * D - 1) = frobeniusSq (U - 1) ↔ D = 1) := by
  have hc := distance_certificate block U D hU hD hD' hblock hpos
  constructor
  · have hn := frobeniusSq_nonneg (CFC.sqrt (MatrixBlocks.diag block U) * (D - 1))
    linarith
  · constructor
    · intro he
      have hz : CFC.sqrt (MatrixBlocks.diag block U) * (D - 1) = 0 := by
        apply (frobeniusSq_eq_zero _).mp
        linarith
      have hu := (IsStrictlyPositive.sqrt _ hpos.isStrictlyPositive).isUnit
      have hd : D - 1 = 0 := hu.mul_left_cancel (by simpa using hz)
      exact sub_eq_zero.mp hd
    · rintro rfl
      simp

/-- Global unique minimum among all unitaries with the same labelled invariant
subspaces. There is no restriction on the number or sizes of the blocks. -/
 theorem closest_to_identity (block : ι → β) (U T : Matrix ι ι ℂ)
    (hU : star U * U = 1) (hU' : U * star U = 1)
    (hT : star T * T = 1) (hT' : T * star T = 1)
    (hassign : SameAssignment block U T) (hpos : (MatrixBlocks.diag block U).PosDef) :
    frobeniusSq (U - 1) ≤ frobeniusSq (T - 1) ∧
      (frobeniusSq (T - 1) = frobeniusSq (U - 1) ↔ T = U) := by
  obtain ⟨he, hd, hd', hb⟩ := assignment_factor block U T hU hU' hT hT' hassign
  obtain ⟨hmin, heq⟩ := block_rotation_minimality block U (star U * T) hU hd hd' hb hpos
  rw [← he] at hmin heq
  refine ⟨hmin, ?_⟩
  constructor
  · intro hh
    rw [he, heq.mp hh, mul_one]
  · rintro rfl
    rfl

end Pymablock.LeastAction
