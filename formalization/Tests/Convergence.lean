import Pymablock.Convergence.Realization

noncomputable section
open scoped Matrix.Norms.Frobenius Topology
open MvPowerSeries
namespace Pymablock.Convergence.Tests

abbrev M := Matrix (Fin 3) (Fin 3) ℂ
abbrev S := MvPowerSeries (Fin 2) M

/-- Two independent perturbations, three blocks, and all mixed orders. -/
def toy (V W : M) : S := C (Matrix.diagonal fun i : Fin 3 => (i.val : ℂ)) +
  monomial (Finsupp.single 0 1) V + monomial (Finsupp.single 1 1) W

 theorem toy_absolute (V W : M) : Absolute (toy V W) 1 :=
  ((absolute_C _ _).add (absolute_monomial _ _ _) (by norm_num)).add
    (absolute_monomial _ _ _) (by norm_num)

 theorem toy_zero (V W : M) : toy V W 0 = Matrix.diagonal (fun i : Fin 3 => (i.val : ℂ)) := by
  have hn (i : Fin 2) : (0 : Fin 2 →₀ ℕ) ≠ Finsupp.single i 1 := by
    intro h
    have he := congrArg (fun f : Fin 2 →₀ ℕ => f i) h
    simp at he
  change coeff 0 (toy V W) = _
  simp only [toy, map_add, coeff_C, coeff_monomial]
  simp [hn]

 theorem toy_hermitian (V W : M) (hV : star V = V) (hW : star W = W) : star (toy V W) = toy V W := by
  classical
  have hd : star (Matrix.diagonal (fun i : Fin 3 => (i.val : ℂ))) =
      Matrix.diagonal (fun i : Fin 3 => (i.val : ℂ)) := by
    ext i j
    by_cases hij : i = j
    · subst j; simp [Matrix.star_apply]
    · simp [Matrix.star_apply, hij, Ne.symm hij]
  funext n
  change star (coeff n (toy V W)) = coeff n (toy V W)
  simp only [toy, map_add, coeff_C, coeff_monomial]
  split_ifs <;> simp_all

/-- A polynomial input discharges the convergence hypothesis explicitly. -/
example (V W : M) (hV : star V = V) (hW : star W = W) :
    ∃ ht u : S, ∃ r : ℝ, 0 < r ∧ r ≤ 1 ∧ Absolute ht r ∧ Absolute u r ∧
      u 0 = 1 ∧ star u*u=1 ∧ u*star u=1 ∧ ht=star u*toy V W*u ∧
      (MatrixBlocks.blockStructure (fun i : Fin 3 => i)).series.off ht=0 ∧ star ht=ht ∧
      (MatrixBlocks.blockStructure (fun i : Fin 3 => i)).series.diag (skew (u-1))=0 := by
  apply matrix_convergent (fun i : Fin 3 => i) (fun i => (i.val : ℝ)) (toy V W)
    (toy_hermitian V W hV hW) (by simpa using toy_zero V W) _ (by norm_num) (toy_absolute V W)
  intro i j hij he
  dsimp only at he hij
  exact hij (Fin.ext (Nat.cast_injective he))

/-- Boundary evaluation is genuinely summable, not merely a formal assertion. -/
example {u : S} {r : ℝ} (hr : 0 ≤ r) (hu : Absolute u r)
    (hl : star u*u=1) (hh : u*star u=1) (x : Fin 2 → ℝ) (hx : ∀ i, ‖x i‖ ≤ r) :
    Summable (fun n => monomialValue x n • u n) ∧
      star (evaluate u x)*evaluate u x=1 ∧ evaluate u x*star (evaluate u x)=1 :=
  ⟨hu.summable_eval hr x hx, realized_unitary hu hr hl hh x hx⟩

end Pymablock.Convergence.Tests
