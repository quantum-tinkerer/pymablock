import Pymablock.LeastAction.Local
import Pymablock.Hamiltonian

noncomputable section

namespace Pymablock.LeastAction

/-- Every retained coefficient of the actually constructed Hermitian Pymablock
output is Hermitian. Analytic realization and convergence are separate premises. -/
 theorem constructed_retained_hermitian {σ ι β : Type*}
    [Fintype ι] [DecidableEq ι] [DecidableEq β]
    (block : ι → β) (H : MvPowerSeries σ (Matrix ι ι ℂ)) (hH : star H = H)
    (h0diag : MatrixBlocks.diag block (H 0) = H 0)
    (S : SylvesterSolver (MatrixBlocks.blockStructure block) (H 0))
    (n : σ →₀ ℕ) :
    (MatrixBlocks.diag block ((blockDiagonalize (MatrixBlocks.blockStructure block) H S).2 n)).IsHermitian := by
  let P : Selection (Matrix ι ι ℂ) := MatrixBlocks.blockStructure block
  have hc := blockDiagonalize_correct P H hH h0diag S
  have hg := hc.2.2.2.2.2.2
  have hh := selected_selfadjoint_of_gauge P.series _ hg
  exact congrFun hh n

end Pymablock.LeastAction
