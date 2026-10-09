import Mathlib.Logic.Function.Iterate
import Mathlib.Tactic

/-!
# Well-founded coefficient construction

A map on coefficient families is strictly causal if coefficients of rank n
only depend on input coefficients of rank strictly less than n. This gives a
unique fixed point without any analytic convergence assumption. Multi-indices
are ranked by total degree; auxiliary variables can be included in the index.
-/

namespace Pymablock

variable {I C : Type*}

/-- Equality of all coefficients of rank strictly below d. -/
def AgreesBelow (rank : I → ℕ) (d : ℕ) (f g : I → C) : Prop :=
  ∀ i, rank i < d → f i = g i

/-- One evaluation gains one degree of agreement. -/
def StrictlyCausal (rank : I → ℕ) (F : (I → C) → I → C) : Prop :=
  ∀ d f g, AgreesBelow rank d f g → AgreesBelow rank (d + 1) (F f) (F g)

theorem causal_iterate_agrees (rank : I → ℕ) (F : (I → C) → I → C)
    (hF : StrictlyCausal rank F) (seed : I → C) :
    ∀ d n, d ≤ n → AgreesBelow rank d (F^[n] seed) (F^[d] seed) := by
  intro d
  induction d with
  | zero => intro n hn i hi; omega
  | succ d ih =>
    intro n hn
    obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : n ≠ 0)
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
    exact hF d _ _ (ih m (by omega))

/-- A coefficient needs only finitely many evaluations, even when the output
is an infinite formal series. -/
def causalSolution (rank : I → ℕ) (F : (I → C) → I → C) (seed : I → C) : I → C :=
  fun i => (F^[rank i + 1] seed) i

theorem causalSolution_agrees (rank : I → ℕ) (F : (I → C) → I → C)
    (hF : StrictlyCausal rank F) (seed : I → C) (d : ℕ) :
    AgreesBelow rank d (causalSolution rank F seed) (F^[d] seed) := by
  intro i hi
  exact (causal_iterate_agrees rank F hF seed (rank i + 1) d (by omega) i
    (by omega)).symm

/-- Existence is constructive and follows solely from strict causality. -/
theorem causalSolution_fixed (rank : I → ℕ) (F : (I → C) → I → C)
    (hF : StrictlyCausal rank F) (seed : I → C) :
    F (causalSolution rank F seed) = causalSolution rank F seed := by
  funext i
  have h := hF (rank i) _ _ (causalSolution_agrees rank F hF seed (rank i)) i
    (Nat.lt_succ_self _)
  simpa [causalSolution, Function.iterate_succ_apply'] using h

/-- No different coefficient family can satisfy the same recurrence. -/
theorem causal_fixed_unique (rank : I → ℕ) (F : (I → C) → I → C)
    (hF : StrictlyCausal rank F) (f g : I → C) (hf : F f = f) (hg : F g = g) :
    f = g := by
  have h : ∀ d, AgreesBelow rank d f g := by
    intro d
    induction d with
    | zero => intro i hi; omega
    | succ d ih => simpa [hf, hg] using hF d f g ih
  funext i
  exact h (rank i + 1) i (Nat.lt_succ_self _)

end Pymablock
