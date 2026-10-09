import Pymablock.FiniteOrder

/-! The non-Hermitian algorithm needs a linear idempotent projection, without
any adjoint or multiplication laws. -/
noncomputable section
namespace Pymablock.NonHermitian
open MvPowerSeries
variable {A : Type*} [Ring A] [Algebra ℚ A]

structure Projection (A : Type*) [Ring A] [Algebra ℚ A] where
  selected : A →ₗ[ℚ] A
  idempotent : ∀ a, selected (selected a) = selected a

namespace Projection

def remaining (P : Projection A) : A →ₗ[ℚ] A := LinearMap.id - P.selected
@[simp] theorem remaining_apply (P : Projection A) (a : A) :
    P.remaining a = a - P.selected a := rfl
@[simp] theorem selected_remaining (P : Projection A) (a : A) :
    P.selected (P.remaining a) = 0 := by simp [P.idempotent]
@[simp] theorem remaining_selected (P : Projection A) (a : A) :
    P.remaining (P.selected a) = 0 := by simp [P.idempotent]
@[simp] theorem remaining_remaining (P : Projection A) (a : A) :
    P.remaining (P.remaining a) = P.remaining a := by
  rw [remaining_apply P (P.remaining a), selected_remaining, sub_zero]

def series {σ : Type*} (P : Projection A) : Projection (MvPowerSeries σ A) where
  selected := coefficientMap P.selected
  idempotent f := by ext n; exact P.idempotent _

@[simp] theorem selected_apply {σ : Type*} (P : Projection A)
    (f : MvPowerSeries σ A) (n : σ →₀ ℕ) : P.series.selected f n = P.selected (f n) := rfl
@[simp] theorem remaining_series_apply {σ : Type*} (P : Projection A)
    (f : MvPowerSeries σ A) (n : σ →₀ ℕ) : P.series.remaining f n = P.remaining (f n) := rfl

end Projection

/-- Invert the H₀ commutator only on the eliminated part. No adjoint law. -/
structure Solver (P : Projection A) (h0 : A) where
  solve : A →ₗ[ℚ] A
  selected_zero : ∀ a, P.selected (solve a) = 0
  equation : ∀ a, comm (solve a) h0 = P.remaining a

namespace Solver
variable {P : Projection A} {h0 : A}
 theorem series_equation {σ : Type*} (S : Solver P h0) (y : MvPowerSeries σ A) :
    comm (coefficientMap S.solve y) (C h0) = P.series.remaining y := by
  ext n
  simp only [comm, map_sub, coeff_mul_C, coeff_C_mul]
  exact S.equation (y n)
 theorem series_selected_zero {σ : Type*} (S : Solver P h0) (y : MvPowerSeries σ A) :
    P.series.selected (coefficientMap S.solve y) = 0 := by
  ext n
  exact S.selected_zero (y n)
end Solver
end Pymablock.NonHermitian
