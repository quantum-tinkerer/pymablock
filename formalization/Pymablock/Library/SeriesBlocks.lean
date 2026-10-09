import Pymablock.Library.Series
import Pymablock.Library.Blocks
import Pymablock.Library.Parts

noncomputable section
namespace Pymablock
open MvPowerSeries

variable {σ A : Type*} [Ring A] [Algebra ℚ A] [StarRing A]

/-- Apply a linear operation to every formal-series coefficient. -/
def coefficientMap (L : A →ₗ[ℚ] A) : MvPowerSeries σ A →ₗ[ℚ] MvPowerSeries σ A where
  toFun f n := L (f n)
  map_add' f g := by ext n; exact L.map_add _ _
  map_smul' c f := by ext n; exact L.map_smul c _

omit [StarRing A] in
@[simp] theorem coefficientMap_apply (L : A →ₗ[ℚ] A) (f : MvPowerSeries σ A) (n : σ →₀ ℕ) :
    coefficientMap L f n = L (f n) := rfl

/-- Block projection commutes with the multivariate Cauchy product in exactly
the same way as the coefficient-level projection. -/
def BlockStructure.series (P : BlockStructure A) : BlockStructure (MvPowerSeries σ A) where
  diag := coefficientMap P.diag
  idempotent f := by ext n; exact P.idempotent _
  star_diag f := by ext n; exact P.star_diag _
  mul_left f g := by
    classical
    ext n
    change P.diag (coeff n (coefficientMap P.diag f * g)) =
      coeff n (coefficientMap P.diag f * coefficientMap P.diag g)
    simp only [coeff_mul, map_sum]
    apply Finset.sum_congr rfl
    intro p hp
    exact P.mul_left _ _
  mul_right f g := by
    classical
    ext n
    change P.diag (coeff n (f * coefficientMap P.diag g)) =
      coeff n (coefficientMap P.diag f * coefficientMap P.diag g)
    simp only [coeff_mul, map_sum]
    apply Finset.sum_congr rfl
    intro p hp
    exact P.mul_right _ _

omit [Algebra ℚ A] [StarRing A] in
@[simp] theorem series_add_apply (f g : MvPowerSeries σ A) (n : σ →₀ ℕ) :
    (f + g) n = f n + g n := rfl
omit [Algebra ℚ A] [StarRing A] in
@[simp] theorem series_sub_apply (f g : MvPowerSeries σ A) (n : σ →₀ ℕ) :
    (f - g) n = f n - g n := rfl
omit [Algebra ℚ A] [StarRing A] in
@[simp] theorem series_neg_apply (f : MvPowerSeries σ A) (n : σ →₀ ℕ) :
    (-f) n = -f n := rfl
omit [StarRing A] in
@[simp] theorem series_smul_apply (c : ℚ) (f : MvPowerSeries σ A) (n : σ →₀ ℕ) :
    (c • f) n = c • f n := rfl
omit [Algebra ℚ A] in
@[simp] theorem series_star_apply (f : MvPowerSeries σ A) (n : σ →₀ ℕ) :
    (star f) n = star (f n) := rfl
omit [Algebra ℚ A] [StarRing A] in
@[simp] theorem series_zero_apply (n : σ →₀ ℕ) : (0 : MvPowerSeries σ A) n = 0 := rfl
omit [Algebra ℚ A] [StarRing A] in
@[simp] theorem series_mul_zero (f g : MvPowerSeries σ A) : (f * g) 0 = f 0 * g 0 := by
  classical
  change coeff 0 (f * g) = f 0 * g 0
  rw [coeff_mul]
  simp [coeff_apply]
@[simp] theorem series_herm_apply (f : MvPowerSeries σ A) (n : σ →₀ ℕ) :
    herm f n = herm (f n) := rfl
@[simp] theorem series_skew_apply (f : MvPowerSeries σ A) (n : σ →₀ ℕ) :
    skew f n = skew (f n) := rfl
@[simp] theorem series_diag_apply (P : BlockStructure A) (f : MvPowerSeries σ A) (n : σ →₀ ℕ) :
    P.series.diag f n = P.diag (f n) := rfl
@[simp] theorem series_off_apply (P : BlockStructure A) (f : MvPowerSeries σ A) (n : σ →₀ ℕ) :
    P.series.off f n = P.off (f n) := rfl

namespace JetEq

omit [Ring A] [Algebra ℚ A] [StarRing A] in
 theorem refl (d : ℕ) (f : MvPowerSeries σ A) : JetEq d f f := fun _ _ => rfl
omit [Algebra ℚ A] [StarRing A] in
 theorem add {d : ℕ} {f f' g g' : MvPowerSeries σ A}
    (hf : JetEq d f f') (hg : JetEq d g g') : JetEq d (f + g) (f' + g') := by
  intro n hn
  change f n + g n = f' n + g' n
  rw [hf n hn, hg n hn]
omit [Algebra ℚ A] [StarRing A] in
 theorem sub {d : ℕ} {f f' g g' : MvPowerSeries σ A}
    (hf : JetEq d f f') (hg : JetEq d g g') : JetEq d (f - g) (f' - g') := by
  intro n hn
  change f n - g n = f' n - g' n
  rw [hf n hn, hg n hn]
omit [Algebra ℚ A] [StarRing A] in
 theorem neg {d : ℕ} {f g : MvPowerSeries σ A} (h : JetEq d f g) : JetEq d (-f) (-g) := by
  intro n hn
  exact congrArg Neg.neg (h n hn)
omit [StarRing A] in
 theorem smul {d : ℕ} {f g : MvPowerSeries σ A} (h : JetEq d f g) (c : ℚ) :
    JetEq d (c • f) (c • g) := by
  intro n hn
  exact congrArg (fun a : A => c • a) (h n hn)
omit [Algebra ℚ A] in
 theorem star {d : ℕ} {f g : MvPowerSeries σ A} (h : JetEq d f g) : JetEq d (star f) (star g) := by
  intro n hn
  exact congrArg Star.star (h n hn)
omit [StarRing A] in
 theorem map {d : ℕ} {f g : MvPowerSeries σ A} (h : JetEq d f g) (L : A →ₗ[ℚ] A) :
    JetEq d (coefficientMap L f) (coefficientMap L g) := by
  intro n hn
  exact congrArg L (h n hn)
 theorem herm {d : ℕ} {f g : MvPowerSeries σ A} (h : JetEq d f g) : JetEq d (herm f) (herm g) :=
  (h.add h.star).smul _
 theorem skew {d : ℕ} {f g : MvPowerSeries σ A} (h : JetEq d f g) : JetEq d (skew f) (skew g) :=
  (h.sub h.star).smul _

end JetEq
end Pymablock
