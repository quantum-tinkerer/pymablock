import Mathlib.LinearAlgebra.Matrix.ConjTranspose
import Mathlib.Algebra.Algebra.Rat
import Mathlib.Tactic

noncomputable section

namespace Pymablock

/-- The algebraic laws of projection onto diagonal blocks. They do not fix
how many blocks there are or their sizes. -/
structure BlockStructure (A : Type*) [Ring A] [Algebra ℚ A] [StarRing A] where
  diag : A →ₗ[ℚ] A
  idempotent : ∀ a, diag (diag a) = diag a
  star_diag : ∀ a, diag (star a) = star (diag a)
  mul_left : ∀ a b, diag (diag a * b) = diag a * diag b
  mul_right : ∀ a b, diag (a * diag b) = diag a * diag b

namespace BlockStructure

variable {A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

/-- Projection onto all off-diagonal blocks. -/
def off (P : BlockStructure A) : A →ₗ[ℚ] A := LinearMap.id - P.diag

@[simp] theorem off_apply (P : BlockStructure A) (a : A) : P.off a = a - P.diag a := rfl

@[simp] theorem diag_off (P : BlockStructure A) (a : A) : P.diag (P.off a) = 0 := by
  simp [off, P.idempotent]

@[simp] theorem off_diag (P : BlockStructure A) (a : A) : P.off (P.diag a) = 0 := by
  simp [P.idempotent]

@[simp] theorem off_off (P : BlockStructure A) (a : A) : P.off (P.off a) = P.off a := by
  rw [off_apply P (P.off a), diag_off, sub_zero]

 theorem star_off (P : BlockStructure A) (a : A) : P.off (star a) = star (P.off a) := by
  simp [P.star_diag]

 theorem ext (P : BlockStructure A) {a b : A}
    (hd : P.diag a = P.diag b) (ho : P.off a = P.off b) : a = b := by
  simp only [off_apply, hd] at ho
  exact (sub_left_inj).mp ho

 theorem diag_comm (P : BlockStructure A) (v hs : A)
    (hv : P.diag v = 0) (hh : P.diag hs = hs) :
    P.diag (v * hs - hs * v) = 0 := by
  rw [map_sub, ← hh, P.mul_right, P.mul_left, hv, zero_mul, mul_zero, sub_self]

end BlockStructure

namespace MatrixBlocks

variable {ι β R : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq β]
    [Ring R] [Algebra ℚ R] [StarRing R]

/-- A partition of basis indices specifies arbitrarily many blocks. -/
def diag (block : ι → β) : Matrix ι ι R →ₗ[ℚ] Matrix ι ι R where
  toFun a i j := if block i = block j then a i j else 0
  map_add' a b := by ext i j; by_cases h : block i = block j <;> simp [h]
  map_smul' c a := by ext i j; by_cases h : block i = block j <;> simp [h]

omit [Fintype ι] [DecidableEq ι] [StarRing R] in
@[simp] theorem diag_apply (block : ι → β) (a : Matrix ι ι R) (i j : ι) :
    diag block a i j = if block i = block j then a i j else 0 := rfl

/-- The block projection laws are proved for concrete matrices, rather than
assumed as properties of an uninterpreted projection. -/
def blockStructure (block : ι → β) : BlockStructure (Matrix ι ι R) where
  diag := diag block
  idempotent a := by
    ext i j
    simp only [diag_apply]
    split_ifs <;> rfl
  star_diag a := by
    ext i j
    simp only [diag_apply, Matrix.star_apply]
    by_cases h : block i = block j
    · simp only [if_pos h, if_pos h.symm]
    · simp only [if_neg h, if_neg (Ne.symm h), star_zero]
  mul_left a b := by
    ext i j
    simp only [diag_apply, Matrix.mul_apply]
    by_cases hij : block i = block j
    · simp only [if_pos hij]
      apply Finset.sum_congr rfl
      intro k hk
      by_cases hik : block i = block k
      · have hkj := hik.symm.trans hij
        simp only [if_pos hik, if_pos hkj]
      · simp only [if_neg hik, zero_mul]
    · simp only [if_neg hij]
      symm
      apply Finset.sum_eq_zero
      intro k hk
      by_cases hik : block i = block k
      · have hkj : block k ≠ block j := fun h => hij (hik.trans h)
        simp only [if_pos hik, if_neg hkj, mul_zero]
      · simp only [if_neg hik, zero_mul]
  mul_right a b := by
    ext i j
    simp only [diag_apply, Matrix.mul_apply]
    by_cases hij : block i = block j
    · simp only [if_pos hij]
      apply Finset.sum_congr rfl
      intro k hk
      by_cases hkj : block k = block j
      · have hik := hij.trans hkj.symm
        simp only [if_pos hkj, if_pos hik]
      · simp only [if_neg hkj, mul_zero]
    · simp only [if_neg hij]
      symm
      apply Finset.sum_eq_zero
      intro k hk
      by_cases hkj : block k = block j
      · have hik : block i ≠ block k := fun h => hij (h.trans hkj)
        simp only [if_neg hik, if_pos hkj, zero_mul]
      · simp only [if_neg hkj, mul_zero]

end MatrixBlocks
end Pymablock
