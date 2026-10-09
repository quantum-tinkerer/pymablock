import Pymablock.Library.SeriesBlocks
import Pymablock.Optimized

noncomputable section
namespace Pymablock
open MvPowerSeries

variable {σ A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

/-- An inverse of the commutator on eliminated entries. This is the explicit spectral-gap
input. The sign convention is [solve y, H₀] = off y. -/
structure SylvesterSolver (P : Selection A) (h0 : A) where
  solve : A →ₗ[ℚ] A
  diagonal_zero : ∀ y, P.diag (solve y) = 0
  adjoint : ∀ y, star (solve y) = -solve (star y)
  equation : ∀ y, comm (solve y) h0 = P.off y

namespace SylvesterSolver

variable {P : Selection A} {h0 : A}

 theorem series_equation (S : SylvesterSolver P h0) (y : MvPowerSeries σ A) :
    comm (coefficientMap S.solve y) (C h0) = P.series.off y := by
  ext n
  simp only [comm, map_sub, coeff_mul_C, coeff_C_mul]
  exact S.equation (y n)

 theorem series_diagonal_zero (S : SylvesterSolver P h0) (y : MvPowerSeries σ A) :
    P.series.diag (coefficientMap S.solve y) = 0 := by
  ext n
  exact S.diagonal_zero (y n)

 theorem series_adjoint (S : SylvesterSolver P h0) (y : MvPowerSeries σ A) :
    star (coefficientMap S.solve y) = -coefficientMap S.solve (star y) := by
  ext n
  exact S.adjoint (y n)

end SylvesterSolver
end Pymablock
