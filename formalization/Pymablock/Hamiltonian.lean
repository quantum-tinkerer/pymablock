import Pymablock.Correctness
import Mathlib.RingTheory.MvPowerSeries.Inverse

/-! Public theorem stated directly for a Hamiltonian formal series. -/
noncomputable section
namespace Pymablock
open MvPowerSeries

variable {σ A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

omit [Algebra ℚ A] in
 theorem star_positive (H : MvPowerSeries σ A) : star (positive H) = positive (star H) := by
  ext n
  change star (positive H n) = positive (star H) n
  by_cases hn : n = 0 <;> simp [positive, hn]

omit [Algebra ℚ A] [StarRing A] in
 theorem constant_add_positive (H : MvPowerSeries σ A) : C (H 0) + positive H = H := by
  classical
  ext n
  rw [map_add, coeff_C]
  by_cases hn : n = 0 <;> simp [positive, coeff_apply, hn]

omit [Algebra ℚ A] in
 theorem unitary_right (q : MvPowerSeries σ A) (hq : q 0 = 0)
    (hu : star (1 + q) * (1 + q) = 1) : (1 + q) * star (1 + q) = 1 := by
  classical
  have hc : constantCoeff (1 + q) = ((1 : Aˣ) : A) := by
    change (1 + q) 0 = 1
    change coeff 0 (1 + q) = 1
    rw [map_add, coeff_zero_one]
    change 1 + q 0 = 1
    rw [hq, add_zero]
  have hi := mul_invOfUnit (1 + q) (1 : Aˣ) hc
  have he : star (1 + q) = invOfUnit (1 + q) (1 : Aˣ) := by
    calc
      star (1 + q) = star (1 + q) * ((1 + q) * invOfUnit (1 + q) (1 : Aˣ)) := by rw [hi, mul_one]
      _ = invOfUnit (1 + q) (1 : Aˣ) := by rw [← mul_assoc, hu, one_mul]
  rw [he]
  exact hi

/-- The selected and remaining positive-order pieces are derived from H. -/
def selected (P : BlockStructure A) (H : MvPowerSeries σ A) : MvPowerSeries σ A :=
  P.series.diag (positive H)
def remaining (P : BlockStructure A) (H : MvPowerSeries σ A) : MvPowerSeries σ A :=
  P.series.off (positive H)

/-- The two series returned by the mathematical algorithm: (H_tilde, U). -/
def blockDiagonalize (P : BlockStructure A) (H : MvPowerSeries σ A)
    (S : SylvesterSolver P (H 0)) : MvPowerSeries σ A × MvPowerSeries σ A :=
  let hs := selected P H
  let hr := remaining P H
  let s := solution P S.solve hs hr
  (effective P (H 0) hs hr (qSeries s) (bSeries s), 1 + qSeries s)

/-- Main theorem. Every coefficient of the constructed transformation is
unitary in both directions, satisfies the Pymablock gauge, and transforms the
input Hamiltonian to the returned block-diagonal Hermitian series. -/
theorem blockDiagonalize_correct (P : BlockStructure A) (H : MvPowerSeries σ A)
    (hH : star H = H) (h0diag : P.diag (H 0) = H 0)
    (S : SylvesterSolver P (H 0)) :
    let result := blockDiagonalize P H S
    let ht := result.1
    let u := result.2
    u 0 = 1 ∧ star u * u = 1 ∧ u * star u = 1 ∧
      ht = star u * H * u ∧ P.series.off ht = 0 ∧ star ht = ht ∧
      P.series.diag (skew (u - 1)) = 0 := by
  classical
  let hs := selected P H
  let hr := remaining P H
  have hpos : star (positive H) = positive H := by rw [star_positive, hH]
  have hh0 : star (H 0) = H 0 := congrFun hH 0
  have hhs : star hs = hs := by
    change star (P.series.diag (positive H)) = P.series.diag (positive H)
    rw [← P.series.star_diag, hpos]
  have hhr : star hr = hr := by
    change star (P.series.off (positive H)) = P.series.off (positive H)
    rw [← P.series.star_off, hpos]
  have hdiag : P.series.diag hs = hs := P.series.idempotent _
  have hoff : P.series.diag hr = 0 := P.series.diag_off _
  have hs0 : hs 0 = 0 := by simp [hs, selected]
  have hr0 : hr 0 = 0 := by simp [hr, remaining]
  have hsplit : C (H 0) + hs + hr = H := by
    dsimp [hs, hr, selected, remaining]
    convert constant_add_positive H using 1; abel
  have hc := solution_correct P (H 0) S hs hr hh0 h0diag hhs hhr hdiag hoff hs0 hr0
  dsimp only at hc
  let s := solution P S.solve hs hr
  let q := qSeries s
  let b := bSeries s
  have hq0 : q 0 = 0 := hc.1
  have hunit : star (1 + q) * (1 + q) = 1 := hc.2.2.1
  have heff : effective P (H 0) hs hr q b = star (1+q) * H * (1+q) := by
    have he := hc.2.2.2.1
    rw [hsplit] at he
    exact he
  change (1 + q) 0 = 1 ∧ _
  refine ⟨?_, hunit, unitary_right q hq0 hunit, heff, hc.2.2.2.2, ?_, ?_⟩
  · change coeff 0 (1 + q) = 1
    rw [map_add, coeff_zero_one]
    change 1 + q 0 = 1
    rw [hq0, add_zero]
  · change star (effective P (H 0) hs hr q b) = effective P (H 0) hs hr q b
    rw [heff]
    simp only [star_mul, star_star, hH, mul_assoc]
  · change P.series.diag (skew ((1 + q) - 1)) = 0
    have hsub : (1 + q) - 1 = q := by abel
    rw [hsub]
    exact hc.2.1

end Pymablock
