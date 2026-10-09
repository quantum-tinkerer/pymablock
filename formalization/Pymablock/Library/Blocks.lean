import Mathlib.LinearAlgebra.Matrix.ConjTranspose
import Mathlib.Algebra.Algebra.Rat
import Mathlib.Tactic

noncomputable section

namespace Pymablock

/-- A linear, idempotent, star-preserving selection of retained entries.
No compatibility with multiplication is assumed. -/
structure Selection (A : Type*) [Ring A] [Algebra ℚ A] [StarRing A] where
  diag : A →ₗ[ℚ] A
  idempotent : ∀ a, diag (diag a) = diag a
  star_diag : ∀ a, diag (star a) = star (diag a)

namespace Selection

variable {A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

/-- Complementary projection onto entries to eliminate. -/
def off (P : Selection A) : A →ₗ[ℚ] A := LinearMap.id - P.diag

@[simp] theorem off_apply (P : Selection A) (a : A) : P.off a = a - P.diag a := rfl

@[simp] theorem diag_off (P : Selection A) (a : A) : P.diag (P.off a) = 0 := by
  simp [off, P.idempotent]

@[simp] theorem off_diag (P : Selection A) (a : A) : P.off (P.diag a) = 0 := by
  simp [P.idempotent]

@[simp] theorem off_off (P : Selection A) (a : A) : P.off (P.off a) = P.off a := by
  rw [off_apply P (P.off a), diag_off, sub_zero]

 theorem star_off (P : Selection A) (a : A) : P.off (star a) = star (P.off a) := by
  simp [P.star_diag]

 theorem ext (P : Selection A) {a b : A}
    (hd : P.diag a = P.diag b) (ho : P.off a = P.off b) : a = b := by
  simp only [off_apply, hd] at ho
  exact (sub_left_inj).mp ho

end Selection

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
def blockStructure (block : ι → β) : Selection (Matrix ι ι R) where
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

end MatrixBlocks
end Pymablock
