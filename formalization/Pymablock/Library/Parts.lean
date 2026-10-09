import Pymablock.Library.Algebra
import Mathlib.Algebra.Star.Module
import Mathlib.Algebra.Algebra.Rat
import Mathlib.Tactic

noncomputable section
namespace Pymablock
variable {A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

def herm (a : A) : A := (1 / 2 : ℚ) • (a + star a)
def skew (a : A) : A := (1 / 2 : ℚ) • (a - star a)

@[simp] theorem star_herm (a : A) : star (herm a) = herm a := by
  simp [herm, add_comm]

@[simp] theorem star_skew (a : A) : star (skew a) = -skew a := by
  simp only [skew, star_smul, star_sub, star_star, star_trivial]
  module

theorem herm_add_skew (a : A) : herm a + skew a = a := by
  unfold herm skew
  module

theorem two_herm (a : A) : herm a + herm a = a + star a := by
  unfold herm
  module

theorem two_skew (a : A) : skew a + skew a = a - star a := by
  unfold skew
  module

 theorem herm_add (a b : A) : herm (a + b) = herm a + herm b := by
  simp only [herm, star_add]
  module

 theorem herm_sub (a b : A) : herm (a - b) = herm a - herm b := by
  simp only [herm, star_sub]
  module

 theorem skew_add (a b : A) : skew (a + b) = skew a + skew b := by
  simp only [skew, star_add]
  module

 theorem skew_sub (a b : A) : skew (a - b) = skew a - skew b := by
  simp only [skew, star_sub]
  module

@[simp] theorem herm_of_selfadjoint (a : A) (h : star a = a) : herm a = a := by
  simp only [herm, h]
  module

@[simp] theorem skew_of_selfadjoint (a : A) (h : star a = a) : skew a = 0 := by
  simp [skew, h]

@[simp] theorem herm_of_skewadjoint (a : A) (h : star a = -a) : herm a = 0 := by
  simp [herm, h]

@[simp] theorem skew_of_skewadjoint (a : A) (h : star a = -a) : skew a = a := by
  simp only [skew, h]
  module

 theorem herm_comm (q hs : A) (hh : star hs = hs) : herm (comm q hs) = comm (skew q) hs := by
  simp only [herm, skew, comm, star_sub, star_mul, hh, smul_mul_assoc, mul_smul_comm, sub_mul, mul_sub]
  module

end Pymablock
