import Pymablock.Library.Blocks
import Mathlib.LinearAlgebra.Matrix.Trace

noncomputable section

namespace Pymablock.LeastAction

variable {ι β : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq β]

/-- A competitor can rotate bases inside blocks, but cannot mix assigned subspaces. -/
def BlockDiagonal (block : ι → β) (D : Matrix ι ι ℂ) : Prop :=
  ∀ i j, block i ≠ block j → D i j = 0

omit [DecidableEq ι] in
 theorem trace_diag (block : ι → β) (U : Matrix ι ι ℂ) :
    Matrix.trace (MatrixBlocks.diag block U) = Matrix.trace U := by
  simp [Matrix.trace, Matrix.diag]

omit [DecidableEq ι] in
 theorem trace_mul_diag (block : ι → β) (U D : Matrix ι ι ℂ)
    (hD : BlockDiagonal block D) :
    Matrix.trace (U * D) = Matrix.trace (MatrixBlocks.diag block U * D) := by
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  by_cases h : block i = block j
  · simp [MatrixBlocks.diag_apply, h]
  · simp [MatrixBlocks.diag_apply, h, hD j i (Ne.symm h)]

/-- Coordinate projector onto one block; empty labels are harmless. -/
def projector (block : ι → β) (a : β) : Matrix ι ι ℂ :=
  Matrix.diagonal fun i => if block i = a then 1 else 0

/-- Two transformations realize the same labelled subspaces when their conjugated
coordinate projectors agree. This fixes the assignment, including at degeneracies. -/
def SameAssignment (block : ι → β) (U T : Matrix ι ι ℂ) : Prop :=
  ∀ a, U * projector block a * star U = T * projector block a * star T

 theorem blockDiagonal_of_commute_projectors (block : ι → β) (D : Matrix ι ι ℂ)
    (h : ∀ a, D * projector block a = projector block a * D) :
    BlockDiagonal block D := by
  intro i j hij
  have he := congrArg (fun M : Matrix ι ι ℂ => M i j) (h (block j))
  simpa [projector, Matrix.mul_diagonal, Matrix.diagonal_mul, hij] using he

 theorem assignment_factor (block : ι → β) (U T : Matrix ι ι ℂ)
    (hU : star U * U = 1) (hU' : U * star U = 1)
    (hT : star T * T = 1) (hT' : T * star T = 1)
    (h : SameAssignment block U T) :
    T = U * (star U * T) ∧
      star (star U * T) * (star U * T) = 1 ∧
      (star U * T) * star (star U * T) = 1 ∧
      BlockDiagonal block (star U * T) := by
  have he : T = U * (star U * T) := by rw [← mul_assoc, hU', one_mul]
  refine ⟨he, ?_, ?_, ?_⟩
  · simp only [star_mul, star_star]
    calc
      star T * U * (star U * T) = star T * (U * star U) * T := by noncomm_ring
      _ = 1 := by rw [hU', mul_one, hT]
  · simp only [star_mul, star_star]
    calc
      star U * T * (star T * U) = star U * (T * star T) * U := by noncomm_ring
      _ = 1 := by rw [hT', mul_one, hU]
  · apply blockDiagonal_of_commute_projectors
    intro a
    have ha := congrArg (fun M => star U * M * T) (h a)
    have hl : star U * (U * projector block a * star U) * T =
        projector block a * (star U * T) := by
      calc
        _ = (star U * U) * projector block a * (star U * T) := by noncomm_ring
        _ = _ := by rw [hU, one_mul]
    have hr : star U * (T * projector block a * star T) * T =
        (star U * T) * projector block a := by
      calc
        _ = (star U * T) * projector block a * (star T * T) := by noncomm_ring
        _ = _ := by rw [hT, mul_one]
    dsimp only at ha
    rw [hl, hr] at ha
    exact ha.symm

end Pymablock.LeastAction
