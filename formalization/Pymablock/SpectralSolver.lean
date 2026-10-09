import Pymablock.Hamiltonian
import Mathlib.Data.Complex.Basic

/-! A concrete Sylvester solver in an eigenbasis. The generic theorem also
allows block-level solvers; this construction discharges the solver contract
using explicit separated energies, with arbitrary degeneracy within a block. -/
noncomputable section
namespace Pymablock.MatrixBlocks

variable {ι β R : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq β]
    [Field R] [CharZero R] [StarRing R]

/-- Matrix elements are divided by E_j - E_i, matching [V,H₀] = source. -/
def spectralMap (block : ι → β) (e : ι → R) : Matrix ι ι R →ₗ[ℚ] Matrix ι ι R where
  toFun y i j := if block i = block j then 0 else y i j / (e j - e i)
  map_add' a b := by
    ext i j
    by_cases h : block i = block j <;> simp [h, add_div]
  map_smul' c a := by
    ext i j
    by_cases h : block i = block j
    · simp only [h, ite_true, Matrix.smul_apply]
      change (0 : R) = c • (0 : R)
      exact (smul_zero c : c • (0 : R) = 0).symm
    · simp [h, div_eq_mul_inv]

omit [Fintype ι] [DecidableEq ι] [StarRing R] in
@[simp] theorem spectralMap_apply (block : ι → β) (e : ι → R)
    (y : Matrix ι ι R) (i j : ι) :
    spectralMap block e y i j = if block i = block j then 0 else y i j / (e j - e i) := rfl

/-- No cross-block energy degeneracy is allowed; within-block degeneracy is
unrestricted. All projection, adjoint and commutator laws are proved. -/
def spectralSolver (block : ι → β) (e : ι → R)
    (he : ∀ i, star (e i) = e i)
    (gap : ∀ i j, block i ≠ block j → e i ≠ e j) :
    SylvesterSolver (blockStructure block) (Matrix.diagonal e) where
  solve := spectralMap block e
  diagonal_zero y := by
    ext i j
    change (if block i = block j then spectralMap block e y i j else 0) = 0
    simp only [spectralMap_apply]
    split_ifs <;> rfl
  adjoint y := by
    ext i j
    simp only [Matrix.star_apply, spectralMap_apply, Matrix.neg_apply]
    by_cases h : block i = block j
    · simp only [if_pos h, if_pos h.symm, star_zero, neg_zero]
    · simp only [if_neg h, if_neg (Ne.symm h), star_div₀, star_sub, he]
      rw [show e i - e j = -(e j - e i) by ring, div_neg]
  equation y := by
    ext i j
    simp only [comm, Matrix.sub_apply, Matrix.mul_diagonal, Matrix.diagonal_mul,
      spectralMap_apply]
    change (if block i = block j then 0 else y i j / (e j - e i)) * e j -
      e i * (if block i = block j then 0 else y i j / (e j - e i)) =
        y i j - (if block i = block j then y i j else 0)
    by_cases h : block i = block j
    · simp only [if_pos h, zero_mul, mul_zero, sub_self]
    · simp only [if_neg h, sub_zero]
      have hg : e j - e i ≠ 0 := sub_ne_zero.mpr (Ne.symm (gap i j h))
      field_simp

end Pymablock.MatrixBlocks
