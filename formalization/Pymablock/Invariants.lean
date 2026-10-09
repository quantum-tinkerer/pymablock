import Pymablock.Recurrence
import Pymablock.Library.Vanishing

noncomputable section
namespace Pymablock
open MvPowerSeries

variable {σ A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

/-- The W update. -/
def wSeries (q : MvPowerSeries σ A) : MvPowerSeries σ A := (-1 / 2 : ℚ) • (star q * q)

 theorem star_wSeries (q : MvPowerSeries σ A) : star (wSeries q) = wSeries q := by
  simp [wSeries, star_mul]

omit [Algebra ℚ A] in
 theorem comm_skew_selfadjoint (v h : MvPowerSeries σ A)
    (hv : star v = -v) (hh : star h = h) : star (comm v h) = comm v h := by
  simp only [comm, star_sub, star_mul, hv, hh]
  noncomm_ring

/-- Unitarity and the gauge are invariants of the constructed recurrence. -/
theorem recurrence_parts (P : Selection A) (h0 : A) (S : SylvesterSolver P h0)
    (hs hr q b : MvPowerSeries σ A) (hhs : star hs = hs)
    (rec : Recurrence P S.solve hs hr q b) :
    herm q = wSeries q ∧ P.series.diag (skew q) = 0 ∧
    star (1 + q) * (1 + q) = 1 := by
  let y := herm (b + hr + hr * q) - comm (skew q) hs
  let v := coefficientMap S.solve y
  have hy : star y = y := by
    dsimp [y]
    rw [star_sub, star_herm, comm_skew_selfadjoint _ _ (star_skew _) hhs]
  have hv : star v = -v := by
    dsimp [v]
    rw [S.series_adjoint, hy]
  have hq : q = wSeries q + v := rec.q_eq
  have hw := star_wSeries q
  have hherm : herm q = wSeries q := by
    conv_lhs => rw [hq]
    rw [herm_add, herm_of_selfadjoint _ hw, herm_of_skewadjoint _ hv, add_zero]
  have hskew : skew q = v := by
    conv_lhs => rw [hq]
    rw [skew_add, skew_of_selfadjoint _ hw, skew_of_skewadjoint _ hv, zero_add]
  refine ⟨hherm, ?_, ?_⟩
  · rw [hskew]
    exact S.series_diagonal_zero y
  · have hwrec : wSeries q + wSeries q = -(star (wSeries q + v) * (wSeries q + v)) := by
      rw [← hq]
      unfold wSeries
      module
    simpa only [← hq] using unitary_of_recurrence (wSeries q) v hw hv hwrec

/-- The Sylvester equation fixes the Hermitian part of the commutator. -/
theorem recurrence_x_herm (P : Selection A) (h0 : A) (S : SylvesterSolver P h0)
    (hs hr q b : MvPowerSeries σ A)
    (hh0 : star h0 = h0) (hhs : star hs = hs) (hhr : P.series.diag hr = 0) (rec : Recurrence P S.solve hs hr q b) :
    herm (b + hr + hr * q) = herm (comm q (C h0 + hs)) := by
  classical
  let x := b + hr + hr * q
  let y := herm x - comm (skew q) hs
  let v := coefficientMap S.solve y
  have hy : star y = y := by
    dsimp [y]
    rw [star_sub, star_herm, comm_skew_selfadjoint _ _ (star_skew _) hhs]
  have hv : star v = -v := by dsimp [v]; rw [S.series_adjoint, hy]
  have hq : q = wSeries q + v := rec.q_eq
  have hskew : skew q = v := by
    conv_lhs => rw [hq]
    rw [skew_add, skew_of_selfadjoint _ (star_wSeries q), skew_of_skewadjoint _ hv, zero_add]
  have hk : star (comm (skew q) hs) = comm (skew q) hs :=
    comm_skew_selfadjoint _ _ (star_skew _) hhs
  have hxe : P.series.diag (herm x) = P.series.diag (comm (skew q) hs) := by
    simpa only [herm_of_selfadjoint _ hk] using
      optimized_x_diag P.series q b hr (comm (skew q) hs) hhr rec.b_eq
  have hsyl := S.series_equation y
  change comm v (C h0) = P.series.off (herm x - comm (skew q) hs) at hsyl
  rw [← hskew, Selection.off_apply, map_sub, hxe, sub_self, sub_zero] at hsyl
  have hh : star (C h0 + hs : MvPowerSeries σ A) = C h0 + hs := by
    rw [star_add, hhs]
    congr 1
    ext n
    simp only [coeff_star, coeff_C]
    split_ifs <;> simp only [hh0, star_zero]
  rw [herm_comm q (C h0 + hs) hh]
  change herm x = comm (skew q) (C h0 + hs)
  simp only [comm, mul_add, add_mul] at hsyl ⊢
  linear_combination (norm := noncomm_ring) -hsyl

/-- Correctness of the auxiliary X constructed by the optimized recurrence. -/
theorem recurrence_x_commutator (P : Selection A) (h0 : A) (S : SylvesterSolver P h0)
    (hs hr q b : MvPowerSeries σ A)
    (hh0 : star h0 = h0) (hhs : star hs = hs) (hhr : star hr = hr)
    (hoff : P.series.diag hr = 0)
    (rec : Recurrence P S.solve hs hr q b) :
    b + hr + hr * q = comm q (C h0 + hs) := by
  classical
  letI : IsAddTorsionFree A := IsAddTorsionFree.of_module_rat A
  have hu := (recurrence_parts P h0 S hs hr q b hhs rec).2.2
  have hherm := recurrence_x_herm P h0 S hs hr q b hh0 hhs hoff rec
  have hh : star (C h0 + hs : MvPowerSeries σ A) = C h0 + hs := by
    rw [star_add, hhs]
    congr 1
    ext n
    simp only [coeff_star, coeff_C]
    split_ifs <;> simp only [hh0, star_zero]
  apply commutator_of_recurrences q (C h0 + hs) (b + hr + hr * q) rec.q_zero hu hh
  · have he := congrArg (fun a : MvPowerSeries σ A => a + a) hherm
    dsimp at he
    simpa only [two_herm] using he
  · exact optimized_x_skew P.series q b hr (comm (skew q) hs) hhr rec.b_eq

end Pymablock
