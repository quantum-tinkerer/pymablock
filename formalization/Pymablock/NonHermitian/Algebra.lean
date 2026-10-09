import Pymablock.Hamiltonian

/-! Algebra of the documented non-Hermitian recurrence. No adjoint identities
are assumed for the Hamiltonian, the forward map, or its inverse. -/
noncomputable section
namespace Pymablock.NonHermitian
open MvPowerSeries
variable {A : Type*} [Ring A]

/-- The W update enforces the left inverse. -/
theorem inverse_left (q g w v : A) (hq : q = w + v) (hg : g = w - v)
    (hw : w + w = -(g * q)) : (1 + g) * (1 + q) = 1 := by
  have he : g + q = w + w := by rw [hq, hg]; abel
  calc
    (1 + g) * (1 + q) = 1 + (g + q) + g * q := by noncomm_ring
    _ = 1 := by rw [he, hw]; abel

/-- A formal series with identity constant coefficient has a two-sided inverse. -/
theorem inverse_right {σ : Type*} (q g : MvPowerSeries σ A) (hq : q 0 = 0)
    (hu : (1 + g) * (1 + q) = 1) : (1 + q) * (1 + g) = 1 := by
  classical
  have hc : constantCoeff (1 + q) = ((1 : Aˣ) : A) := by
    change coeff 0 (1 + q) = 1
    rw [map_add, coeff_zero_one]
    change 1 + q 0 = 1
    rw [hq, add_zero]
  have hi := mul_invOfUnit (1 + q) (1 : Aˣ) hc
  have he : 1 + g = invOfUnit (1 + q) (1 : Aˣ) := by
    calc
      1 + g = (1 + g) * ((1 + q) * invOfUnit (1 + q) (1 : Aˣ)) := by rw [hi, mul_one]
      _ = invOfUnit (1 + q) (1 : Aˣ) := by rw [← mul_assoc, hu, one_mul]
  rw [he]
  exact hi

/-- Differentiating the inverse identity as a commutator. -/
theorem inverse_commutator (q g h : A)
    (hl : (1 + g) * (1 + q) = 1) (hr : (1 + q) * (1 + g) = 1) :
    comm g h = -(1 + g) * comm q h * (1 + g) := by
  have ha := congrArg (fun a : A => a * h * (1 + g)) hl
  have hb := congrArg (fun a : A => (1 + g) * h * a) hr
  unfold comm
  linear_combination (norm := noncomm_ring) ha - hb

/-- The optimized Z recurrence equals a linear expression in X. -/
theorem z_in_x (q g r b z : A) (hr : (1 + q) * (1 + g) = 1)
    (hz : z + z = r*q - g*r - g*b - (b + g*b)*g) :
    let x := b + r + r*q
    z + z = -(g*x + (1+g)*x*g) := by
  dsimp
  have hc := congrArg (fun a : A => (1+g)*r*a) hr
  linear_combination (norm := noncomm_ring) hz + hc

/-- The analogous identity for the actual W commutator. -/
theorem w_commutator (q g w v h : A)
    (hq : q = w + v) (hg : g = w - v)
    (hl : (1+g)*(1+q) = 1) (hr : (1+q)*(1+g) = 1) :
    comm w h + comm w h = -(g * comm q h + (1+g)*comm q h*g) := by
  have hc := inverse_commutator q g h hl hr
  have hs : w + w = q + g := by rw [hq, hg]; abel
  have he := congrArg (fun a : A => comm a h) hs
  simp only [comm] at hc he ⊢
  linear_combination (norm := noncomm_ring) he + hc

/-- The defect obeys a homogeneous recurrence with positive-degree factors. -/
theorem defect_equation (q g w v h r b z : A)
    (hq : q = w+v) (hg : g = w-v)
    (hl : (1+g)*(1+q) = 1) (hr : (1+q)*(1+g) = 1)
    (hz : z+z = r*q-g*r-g*b-(b+g*b)*g)
    (hy : b+r+r*q-z = comm v h) :
    let d := b+r+r*q-comm q h
    d+d = -(q*d+d*g) := by
  let x := b+r+r*q
  let d := x-comm q h
  have hsplit : d = z-comm w h := by
    dsimp [d, x]
    rw [hq] at hy ⊢
    simp only [comm, add_mul, mul_add] at hy ⊢
    linear_combination (norm := noncomm_ring) hy
  have hz' := z_in_x q g r b z hr hz
  have hw' := w_commutator q g w v h hq hg hl hr
  have hd : d+d = -(g*d+(1+g)*d*g) := by
    have he := congrArg (fun a : A => a+a) hsplit
    dsimp [d, x] at he ⊢
    dsimp at hz'
    linear_combination (norm := noncomm_ring) he + hz' - hw'
  have he := congrArg (fun a : A => (1+q)*a) hd
  have ha := congrArg (fun a : A => a*d) hr
  have hb := congrArg (fun a : A => a*d*g) hr
  change d+d = -(q*d+d*g)
  linear_combination (norm := noncomm_ring) he - ha - hb

/-- Once X is proved to be the commutator, B assembles the similarity transform. -/
theorem similarity (q g h r b : A) (hu : (1+g)*(1+q)=1)
    (hx : comm q h = b+r+r*q) :
    (1+g)*(h+r)*(1+q) = h-(b+g*b) := by
  have hh := congrArg (fun a : A => a*h) hu
  have he := congrArg (fun a : A => (1+g)*a) hx
  simp only [comm] at he
  linear_combination (norm := noncomm_ring) hh - he

end Pymablock.NonHermitian
