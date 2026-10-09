import Pymablock.Invariants

/-! End-to-end correctness of the constructed formal-series algorithm. -/
noncomputable section
namespace Pymablock
open MvPowerSeries

variable {σ A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

/-- The effective Hamiltonian expression evaluated by the optimized algorithm. -/
def effective (P : Selection A) (h0 : A) (hs hr q b : MvPowerSeries σ A) :
    MvPowerSeries σ A := C h0 + hs + P.series.diag (herm (hr * q) - herm (star q * b) - herm (comm (skew q) hs))

/-- Correctness of the actual constructed solution, with no recurrence or
output-correctness assumptions among the theorem's inputs. The parameter
index type σ is arbitrary and P admits arbitrary symmetric matrix masks. -/
theorem solution_correct (P : Selection A) (h0 : A) (S : SylvesterSolver P h0)
    (hs hr : MvPowerSeries σ A)
    (hh0 : star h0 = h0) (hh0diag : P.diag h0 = h0)
    (hhs : star hs = hs) (hhr : star hr = hr)
    (hdiag : P.series.diag hs = hs) (hoff : P.series.diag hr = 0)
    (hs0 : hs 0 = 0) (hr0 : hr 0 = 0) :
    let s := solution P S.solve hs hr
    let q := qSeries s
    let b := bSeries s
    let u := 1 + q
    let ht := effective P h0 hs hr q b
    q 0 = 0 ∧ P.series.diag (skew q) = 0 ∧ star u * u = 1 ∧
      ht = star u * (C h0 + hs + hr) * u ∧ P.series.off ht = 0 := by
  classical
  dsimp only
  let s := solution P S.solve hs hr
  let q := qSeries s
  let b := bSeries s
  have hrec : Recurrence P S.solve hs hr q b := solution_recurrence P S.solve hs hr hs0 hr0
  have hp := recurrence_parts P h0 S hs hr q b hhs hrec
  have hx := recurrence_x_commutator P h0 S hs hr q b hh0 hhs hhr hoff hrec
  have he := optimized_conjugation q (C h0 + hs) hr b hp.2.2 hx.symm
  have hb : b + star q * b = P.series.diag (herm (star q * b) - herm (hr * q) + herm (comm (skew q) hs)) := by
    conv_lhs => lhs; rw [hrec.b_eq]
    exact bUpdate_residual P.series _ _ _
  have ht : effective P h0 hs hr q b = star (1+q) * (C h0 + hs + hr) * (1+q) := by
    rw [he, hb]
    simp only [effective, map_add, map_sub]
    abel
  refine ⟨hrec.q_zero, hp.2.1, hp.2.2, ht, ?_⟩
  have hC : P.series.diag (C h0 : MvPowerSeries σ A) = C h0 := by
    ext n
    change P.diag (coeff n (C h0)) = coeff n (C h0)
    simp only [coeff_C]
    split_ifs <;> simp only [hh0diag, map_zero]
  change P.series.off (effective P h0 hs hr q b) = 0
  simp only [effective, map_add, P.series.off_diag, add_zero,
    Selection.off_apply, hC, hdiag, sub_self, add_zero]

end Pymablock
