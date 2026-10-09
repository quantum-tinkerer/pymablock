import Pymablock.Library.Blocks
import Pymablock.Library.Parts

noncomputable section
namespace Pymablock

variable {A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

@[simp] theorem herm_neg (a : A) : herm (-a) = -herm a := by
  simp only [herm, star_neg]
  module
@[simp] theorem skew_neg (a : A) : skew (-a) = -skew a := by
  simp only [skew, star_neg]
  module
@[simp] theorem herm_herm (a : A) : herm (herm a) = herm a := herm_of_selfadjoint _ (star_herm _)
@[simp] theorem skew_herm (a : A) : skew (herm a) = 0 := skew_of_selfadjoint _ (star_herm _)
@[simp] theorem herm_skew (a : A) : herm (skew a) = 0 := herm_of_skewadjoint _ (star_skew _)
@[simp] theorem skew_skew (a : A) : skew (skew a) = skew a := skew_of_skewadjoint _ (star_skew _)

namespace BlockStructure

 theorem diag_herm (P : BlockStructure A) (a : A) : P.diag (herm a) = herm (P.diag a) := by
  simp [herm, P.star_diag]
 theorem diag_skew (P : BlockStructure A) (a : A) : P.diag (skew a) = skew (P.diag a) := by
  simp [skew, P.star_diag]
 theorem off_herm (P : BlockStructure A) (a : A) : P.off (herm a) = herm (P.off a) := by
  simp [herm_sub, ← P.diag_herm]
 theorem off_skew (P : BlockStructure A) (a : A) : P.off (skew a) = skew (P.off a) := by
  simp [skew_sub, ← P.diag_skew]

end BlockStructure

/-- The B recurrence in `pymablock/algorithms.py` for ordinary block partitions.
Here c is U'† B and a is H'_offdiag U'. -/
def bUpdate (P : BlockStructure A) (c a : A) : A :=
  -P.diag (skew c + herm a) - P.off c

 theorem off_bUpdate (P : BlockStructure A) (c a : A) :
    P.off (bUpdate P c a) = -P.off c := by
  simp only [bUpdate, map_sub, map_neg, P.off_diag, P.off_off, neg_zero, zero_sub]

 theorem skew_bUpdate (P : BlockStructure A) (c a : A) :
    skew (bUpdate P c a) = -skew c := by
  simp only [bUpdate, skew_sub, skew_neg, ← P.diag_skew,
    skew_add, skew_skew, skew_herm, add_zero, BlockStructure.off_apply]
  abel

 theorem diag_herm_bUpdate (P : BlockStructure A) (c a : A) :
    P.diag (herm (bUpdate P c a)) = -P.diag (herm a) := by
  simp only [bUpdate, herm_sub, herm_neg, ← P.diag_herm, ← P.off_herm,
    herm_add, herm_skew, herm_herm, zero_add, map_sub, map_neg,
    P.idempotent, P.diag_off, sub_zero]

/-- The optimized B recurrence fixes the anti-Hermitian part of X. -/
theorem optimized_x_skew (P : BlockStructure A) (q b hr : A)
    (hh : star hr = hr) (hb : b = bUpdate P (star q * b) (hr * q)) :
    let x := b + hr + hr * q
    x - star x = -(star q * x) + star x * q := by
  dsimp
  have hs : b - star b = -(star q * b - star (star q * b)) := by
    have h := congrArg (fun a : A => a + a) (skew_bUpdate P (star q * b) (hr * q))
    rw [← hb] at h
    dsimp at h
    rw [two_skew] at h
    simpa only [← neg_add, two_skew] using h
  simp only [star_add, star_mul, star_star, hh] at hs ⊢
  linear_combination (norm := noncomm_ring) hs

/-- The diagonal Hermitian part of X vanishes for any block partition. -/
theorem optimized_x_diag (P : BlockStructure A) (q b hr : A)
    (hr0 : P.diag hr = 0) (hb : b = bUpdate P (star q * b) (hr * q)) :
    P.diag (herm (b + hr + hr * q)) = 0 := by
  rw [herm_add, herm_add, map_add, map_add]
  have h := diag_herm_bUpdate P (star q * b) (hr * q)
  rw [← hb] at h
  rw [h, P.diag_herm hr, hr0]
  simp [herm]

end Pymablock

namespace Pymablock
variable {A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

/-- The optimized formula for H_tilde follows from the B update. -/
theorem bUpdate_residual (P : BlockStructure A) (c a : A) :
    bUpdate P c a + c = P.diag (herm c - herm a) := by
  have h := congrArg P.diag (herm_add_skew c)
  simp only [map_add] at h
  simp only [bUpdate, BlockStructure.off_apply, map_add, map_sub]
  linear_combination (norm := module) -h

end Pymablock
