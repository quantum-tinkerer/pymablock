import Mathlib.Algebra.Star.Basic
import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.LinearCombination

/-!
# Algebraic identities behind Pymablock

These identities hold in any star ring. In the application the ring consists
of matrix-valued formal power series; star fixes the formal parameters.
No assumption about the number of parameters or blocks is used here.
-/

namespace Pymablock

variable {A : Type*} [Ring A] [StarRing A]

/-- The commutator convention used throughout the formalization. -/
def comm (a b : A) : A := a * b - b * a

/-- Pymablock's Hermitian/anti-Hermitian decomposition and its quadratic
recurrence imply unitarity. The doubled equation avoids division by two. -/
theorem unitary_of_recurrence (w v : A)
    (hw : star w = w) (hv : star v = -v)
    (hrec : w + w = -(star (w + v) * (w + v))) :
    star (1 + (w + v)) * (1 + (w + v)) = 1 := by
  have hsum : star (w + v) + (w + v) = w + w := by
    rw [star_add, hw, hv]
    abel
  calc
    star (1 + (w + v)) * (1 + (w + v)) =
        1 + (star (w + v) + (w + v)) + star (w + v) * (w + v) := by
      simp only [star_add, star_one]
      noncomm_ring
    _ = 1 := by rw [hsum, hrec]; noncomm_ring

/-- Moving the selected Hamiltonian to the right eliminates its repeated
products, exactly as in the algorithm's derivation. -/
theorem conjugation_identity (q hs hr : A)
    (hu : star (1 + q) * (1 + q) = 1) :
    star (1 + q) * (hs + hr) * (1 + q) =
      hs - comm q hs - star q * comm q hs +
        star (1 + q) * hr * (1 + q) := by
  have h := congrArg (fun a : A => a * hs) hu
  simp only [star_add, star_one] at h ⊢
  unfold comm
  linear_combination (norm := noncomm_ring) h

/-- The optimized auxiliary B makes the transformed Hamiltonian particularly
simple. This is the bridge between the commutator and optimized recurrences. -/
theorem optimized_conjugation (q hs hr b : A)
    (hu : star (1 + q) * (1 + q) = 1)
    (hx : comm q hs = b + hr + hr * q) :
    star (1 + q) * (hs + hr) * (1 + q) = hs - (b + star q * b) := by
  rw [conjugation_identity q hs hr hu, hx]
  simp only [star_add, star_one]
  noncomm_ring

/-- Off-diagonal cancellation follows from the optimized B recurrence.
The projection is an additive map, allowing any number of blocks. -/
theorem optimized_offdiagonal (off : A →+ A) (q hs hr b : A)
    (hu : star (1 + q) * (1 + q) = 1)
    (hx : comm q hs = b + hr + hr * q)
    (hsdiag : off hs = 0) (hb : off (b + star q * b) = 0) :
    off (star (1 + q) * (hs + hr) * (1 + q)) = 0 := by
  rw [optimized_conjugation q hs hr b hu hx, map_sub, hsdiag, hb, sub_self]

/-- Unitarity fixes the anti-Hermitian part of the commutator. -/
theorem comm_skew_identity (q hs : A)
    (hu : star (1 + q) * (1 + q) = 1) (hh : star hs = hs) :
    comm q hs - star (comm q hs) =
      -(star q * comm q hs) + star (comm q hs) * q := by
  have hleft := congrArg (fun a : A => hs * a) hu
  have hright := congrArg (fun a : A => a * hs) hu
  simp only [star_add, star_one] at hleft hright
  simp only [comm, star_sub, star_mul, hh]
  linear_combination (norm := noncomm_ring) hright - hleft

/-- The commutator defect satisfies a homogeneous, strictly causal equation.
Its unique formal-series solution is zero. -/
theorem comm_defect_equation (q hs x : A)
    (hu : star (1 + q) * (1 + q) = 1) (hh : star hs = hs)
    (hherm : x + star x = comm q hs + star (comm q hs))
    (hskew : x - star x = -(star q * x) + star x * q) :
    (x - comm q hs) + (x - comm q hs) =
      -(star q * (x - comm q hs) + (x - comm q hs) * q) := by
  have hc := comm_skew_identity q hs hu hh
  have hm := congrArg (fun a : A => a * q) hherm
  linear_combination (norm := noncomm_ring) hskew - hc + hherm + hm

end Pymablock
