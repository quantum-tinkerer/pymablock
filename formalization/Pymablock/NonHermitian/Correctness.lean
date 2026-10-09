import Pymablock.NonHermitian.Recurrence
import Pymablock.NonHermitian.Algebra
import Pymablock.Library.Vanishing

noncomputable section
namespace Pymablock.NonHermitian
open MvPowerSeries
variable {σ A : Type*} [Ring A] [Algebra ℚ A]

 theorem recurrence_v (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr q g b : MvPowerSeries σ A) (hrec : Recurrence P solve hs hr q g b) :
    vSeries q g = vUpdate P solve hs hr q g b := by
  unfold vSeries
  linear_combination (norm := module) (1/2 : ℚ) • hrec.q_eq - (1/2 : ℚ) • hrec.g_eq

 theorem recurrence_plus (P : Projection A) (solve : A →ₗ[ℚ] A)
    (hs hr q g b : MvPowerSeries σ A) (hrec : Recurrence P solve hs hr q g b) :
    plusSeries P g b = b+g*b := by
  have he := congrArg P.series.remaining hrec.b_eq
  simp only [bUpdate, map_sub, P.series.remaining_selected, P.series.remaining_remaining,
    zero_sub] at he
  have hz : P.series.remaining (b+g*b) = 0 := by rw [map_add, he]; abel
  change P.series.selected (b+g*b) = b+g*b
  exact (sub_eq_zero.mp hz).symm

 theorem recurrence_y (P : Projection A) (h0 : A) (S : Solver P h0)
    (hs hr q g b : MvPowerSeries σ A) (hoff : P.series.selected hr = 0)
    (hrec : Recurrence P S.solve hs hr q g b) :
    b+hr+hr*q-zSeries P hr q g b = comm (vSeries q g) (C h0+hs) := by
  let y := b+hr+hr*q-zSeries P hr q g b
  let k := comm (vSeries q g) hs
  have hb := congrArg P.series.selected hrec.b_eq
  simp only [bUpdate, map_sub, P.series.idempotent, P.series.selected_remaining, sub_zero] at hb
  have hy : P.series.selected (y-k) = 0 := by
    dsimp [y,k]
    simp only [map_sub, map_add, hb, hoff]
    abel
  have hv : vSeries q g = coefficientMap S.solve (y-k) := by
    rw [recurrence_v P S.solve hs hr q g b hrec]
    unfold vUpdate
    rw [← hrec.b_eq]
  have he := S.series_equation (y-k)
  rw [← hv, Projection.remaining_apply, hy, sub_zero] at he
  change y = comm (vSeries q g) (C h0+hs)
  dsimp [k] at he
  simp only [comm, mul_add, add_mul] at he ⊢
  linear_combination (norm := noncomm_ring) -he

/-- The optimized recurrence really produces the commutator X. -/
theorem recurrence_x (P : Projection A) (h0 : A) (S : Solver P h0)
    (hs hr q g b : MvPowerSeries σ A) (hoff : P.series.selected hr = 0)
    (hrec : Recurrence P S.solve hs hr q g b) :
    b+hr+hr*q = comm q (C h0+hs) := by
  letI : IsAddTorsionFree A := IsAddTorsionFree.of_module_rat A
  let w := wSeries q g
  let v := vUpdate P S.solve hs hr q g b
  have hw : w+w = -(g*q) := by dsimp [w,wSeries]; module
  have hl := inverse_left q g w v hrec.q_eq hrec.g_eq hw
  have hright := inverse_right q g hrec.q_zero hl
  have hz : zSeries P hr q g b + zSeries P hr q g b =
      hr*q-g*hr-g*b-(b+g*b)*g := by
    unfold zSeries
    rw [recurrence_plus P S.solve hs hr q g b hrec]
    module
  have hy := recurrence_y P h0 S hs hr q g b hoff hrec
  rw [recurrence_v P S.solve hs hr q g b hrec] at hy
  have hd := defect_equation q g w v (C h0+hs) hr b (zSeries P hr q g b)
    hrec.q_eq hrec.g_eq hl hright hz hy
  exact sub_eq_zero.mp (homogeneous_eq_zero q g (b+hr+hr*q-comm q (C h0+hs))
    hrec.q_zero hrec.g_zero hd)

/-- Construct (H_tilde, U, U_inv) from the input, following the documented recurrence. -/
def diagonalize (P : Projection A) (H : MvPowerSeries σ A) (S : Solver P (H 0)) :
    MvPowerSeries σ A × MvPowerSeries σ A × MvPowerSeries σ A :=
  let hs := P.series.selected (positive H)
  let hr := P.series.remaining (positive H)
  let s := solution P S.solve hs hr
  (C (H 0)+hs-plusSeries P (gSeries s) (bSeries s), 1+qSeries s, 1+gSeries s)

/-- No Hermiticity or adjoint-inverse hypothesis: the recurrence constructs
both inverses and the similarity transform for a general formal Hamiltonian. -/
theorem diagonalize_correct (P : Projection A) (H : MvPowerSeries σ A)
    (h0 : P.selected (H 0) = H 0) (S : Solver P (H 0)) :
    let result := diagonalize P H S
    let ht := result.1; let u := result.2.1; let ui := result.2.2
    u 0 = 1 ∧ ui 0 = 1 ∧ ui*u = 1 ∧ u*ui = 1 ∧ ht = ui*H*u ∧
      P.series.remaining ht = 0 ∧ P.series.selected ((1/2 : ℚ) • (u-ui)) = 0 := by
  classical
  let hs := P.series.selected (positive H)
  let hr := P.series.remaining (positive H)
  have hs0 : hs 0 = 0 := by simp [hs]
  have hr0 : hr 0 = 0 := by simp [hr]
  let s := solution P S.solve hs hr
  let q := qSeries s; let g := gSeries s; let b := bSeries s
  have hrec : Recurrence P S.solve hs hr q g b := solution_recurrence P S.solve hs hr hs0 hr0
  have hw : wSeries q g + wSeries q g = -(g*q) := by unfold wSeries; module
  have hl := inverse_left q g _ _ hrec.q_eq hrec.g_eq hw
  have hright := inverse_right q g hrec.q_zero hl
  have hoff : P.series.selected hr = 0 := P.series.selected_remaining _
  have hx := recurrence_x P (H 0) S hs hr q g b hoff hrec
  have hsplit : C (H 0)+hs+hr = H := by
    dsimp [hs,hr]
    convert constant_add_positive H using 1; abel
  have he := similarity q g (C (H 0)+hs) hr b hl hx.symm
  rw [hsplit, ← recurrence_plus P S.solve hs hr q g b hrec] at he
  change (1+q) 0 = 1 ∧ (1+g) 0 = 1 ∧ _
  refine ⟨?_, ?_, hl, hright, he.symm, ?_, ?_⟩
  · change coeff 0 (1+q) = 1
    rw [map_add, coeff_zero_one]
    change 1+q 0 = 1
    rw [hrec.q_zero, add_zero]
  · change coeff 0 (1+g) = 1
    rw [map_add, coeff_zero_one]
    change 1+g 0 = 1
    rw [hrec.g_zero, add_zero]
  · have hc : P.series.selected (C (H 0) : MvPowerSeries σ A) = C (H 0) := by
      ext n
      change P.selected (coeff n (C (H 0))) = coeff n (C (H 0))
      simp only [coeff_C]
      split_ifs <;> simp only [h0, map_zero]
    change P.series.remaining (C (H 0)+hs-plusSeries P g b) = 0
    simp only [map_sub, map_add, plusSeries, P.series.remaining_selected,
      Projection.remaining_apply, hc, sub_self, add_zero, sub_zero]
    simpa only [zero_add] using P.series.remaining_selected (positive H)
  · change P.series.selected ((1/2 : ℚ) • ((1+q)-(1+g))) = 0
    rw [show (1+q)-(1+g) = q-g by abel]
    change P.series.selected (vSeries q g) = 0
    rw [recurrence_v P S.solve hs hr q g b hrec]
    exact S.series_selected_zero _

end Pymablock.NonHermitian
