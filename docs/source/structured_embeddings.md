---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 120
---

# Structured embeddings

An effective model can have different operators from its microscopic Hamiltonian.
For example, a spin flip can represent moving a fermion between two orbitals, or
exciting an oscillator from its ground state to its first excited state.
An embedding maps the effective model into the microscopic Hamiltonian, so we
call the effective model the **source** and the microscopic Hamiltonian the
**target**. The embedding defines the source operators by giving **their target
expressions and one reference state**. For a finite matrix result, an ordered
list of reference states defines the retained basis directly.

For a spin encoded in two fermions, the definition is

$$
s \longmapsto c_\downarrow^\dagger c_\uparrow,
\qquad
|0\rangle_s \longmapsto |n_\uparrow=0,n_\downarrow=1\rangle.
$$

Applying the adjoint generates the other spin state. The reference fixes which
representation is intended, while the generator determines its relative phase
and amplitude. Repeating this definition at each site requires one local
operator expression per spin, without listing a many-body basis or supplying a
projector.

```{code-cell} ipython3
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus
from pymablock.second_quantization import Embedding

up, down = FermionOp("up"), FermionOp("down")
s = SigmaMinus("s")
embedding = Embedding(
    {s: Dagger(down) * up},
    reference={up: 0, down: 1},
)
```

## What the definition means

The dictionary maps source **lowering generators** to target expressions.
Their adjoints generate source excitations from the reference, with the
normalization and allowed occupations prescribed by the source algebra.
A fermion or spin one half admits one excitation, a boson admits arbitrarily
many, and an integer ladder admits shifts in both directions. Every target mode
appears in `reference`, including modes that remain frozen.

This defines an isometry $W$ from the source representation into the target.
If $g$ is a source generator and $G$ is its supplied target expression, then

$$
WgW^\dagger=PGP,\qquad P=WW^\dagger.
$$

The target expression may act outside the generated space. For example, using
an oscillator lowering operator as the representative of a spin does not
remove the higher oscillator levels from the target Hamiltonian. Those levels
remain available for virtual transitions.

The constructor checks the physical occupation ranges, fermionic parity,
normalization, and consistency between the supplied generators. It rejects
incorrect amplitudes rather than silently normalizing them. The reference has
phase one; the generators fix all other phases, including signs from occupied
fermionic spectators and permutations of the source modes.

`embedding.restrict(A)` evaluates

$$
R(A)=W^\dagger A W
$$

and expresses the result using the source operators. It evaluates the complete
target expression before compression. In particular, $R(AB)$ need not equal
$R(A)R(B)$: the intermediate state in $AB$ can leave the retained space.

## Oscillator to spin, including a virtual correction

Consider an anharmonic oscillator with a weak drive,

$$
H_0=\omega N_a+\frac{\alpha}{2}N_a(N_a-1),\qquad
V=g(a+a^\dagger).
$$

The generator $s\mapsto a$ and the target vacuum identify the first two
oscillator levels with the source spin. The source algebra sets the upper
boundary, so no target cutoff is needed.

```{code-cell} ipython3
import sympy
from sympy.physics.quantum.boson import BosonOp
from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N

a = BosonOp("a")
omega, alpha, g = sympy.symbols("omega alpha g", positive=True)
H0 = omega * N(a) + alpha * N(a) * (N(a) - 1) / 2
V = g * (a + Dagger(a))

embedding = Embedding({s: a}, reference={a: 0})
H_eff, U, U_adjoint = block_diagonalize(
    [H0, V], subspace_eigenvectors=embedding
)
second_order = H_eff[0, 0, 2]
from pymablock.number_ordered_form import NumberOrderedForm
expected = NumberOrderedForm.from_expr(-2 * g**2 * N(s) / (omega + alpha))
assert (second_order - expected).applyfunc(sympy.cancel).is_zero
```

The effective Hamiltonian through second order is

$$
H_{\mathrm{eff}}=\omega N_s+g(s+s^\dagger)
-\frac{2g^2}{\omega+\alpha}N_s+O(g^3).
$$

The last term comes from $|1\rangle_a\to|2\rangle_a\to|1\rangle_a$.
Its matrix element is $\sqrt{2}g$ and its energy denominator is
$-(\omega+\alpha)$. Correspondingly,

$$
R(a)=s,\qquad R(aa^\dagger)=1+N_s,
\qquad R(a)R(a^\dagger)=1-N_s.
$$

The fixed embedding defines the source operators. The perturbative unitary
$\mathcal U$ subsequently accounts for virtual dressing:
$H_{\mathrm{eff}}=W^\dagger\mathcal U^\dagger H\mathcal U W$.

## Diagonalizing the retained block

The usual `fully_diagonalize` options also apply to embedded blocks. For example,
diagonalizing retained block zero includes transitions between the two source
spin states as well as the virtual transitions to higher oscillator levels:

```{code-cell} ipython3
H_diag, *_ = block_diagonalize(
    [H0, V], subspace_eigenvectors=embedding, fully_diagonalize=(0,)
)
expected = NumberOrderedForm.from_expr(
    -g**2 / omega + 2 * g**2 * N(s) * (1 / omega - 1 / (omega + alpha))
)
assert (H_diag[0, 0, 2] - expected).applyfunc(sympy.cancel).is_zero
```

Use `fully_diagonalize={0: s + Dagger(s)}` to select source operator powers
to eliminate. For a finite reference list, the mask instead selects entries
of the retained matrix. Block one uses the target operators on the complement.

## Dressed observables

`restrict(A)` gives the observable in the fixed reference representation.
To include virtual dressing, convert it with the same embedding and apply the
perturbative unitary returned by `block_diagonalize`:

```{code-cell} ipython3
from operator import mul
from pymablock import operator_to_BlockSeries
from pymablock.series import cauchy_dot_product

# The one-element order tuple gives the observable the same series dimension.
A = operator_to_BlockSeries({(0,): N(a)}, subspace_eigenvectors=embedding)
A_eff = cauchy_dot_product(U_adjoint, A, U, operator=mul)
expected = NumberOrderedForm.from_expr(2 * g**2 * N(s) / (omega + alpha)**2)
assert (A_eff[0, 0, 2] - expected).simplify().is_zero
```

The correction comes from the virtual population of oscillator level two.
Conversion accepts non-diagonal observables and preserves their zeroth-order
cross blocks. Use scalar multiplication (`operator=mul`) for generator
embeddings; it also multiplies the SymPy matrices from reference lists.

## Combining degrees of freedom

Independent generator definitions fit in one mapping. A tunable coupler uses
three target oscillators and two source spins:

```{code-cell} ipython3
a1, a2, coupler = (BosonOp(name) for name in ("a1", "a2", "coupler"))
s1, s2 = SigmaMinus("s1"), SigmaMinus("s2")
embedding = Embedding(
    {s1: a1, s2: a2},
    reference={a1: 0, a2: 0, coupler: 0},
)
```

The coupler stays in its vacuum in the reference representation, but can be
excited during a virtual process. Both computational oscillators can likewise
visit higher levels during perturbation theory.

A localized spin, conduction fermion, and cavity can coexist in one source:

```{code-cell} ipython3
f, conduction = FermionOp("f"), FermionOp("conduction")
b, cavity = BosonOp("b"), BosonOp("cavity")
embedding = Embedding(
    {s: Dagger(down) * up, f: conduction, b: cavity},
    reference={up: 0, down: 1, conduction: 0, cavity: 0},
)
```

The source mode `b` is a `BosonOp`, so its entire ladder is retained. Neither the
constructor nor the perturbative calculation enumerates that infinite ladder.
Number-dependent interactions and energy denominators retain their joint
dependence on all modes.

A hole is also a direct generator definition:

```{code-cell} ipython3
hole, electron = FermionOp("hole"), FermionOp("electron")
embedding = Embedding({hole: Dagger(electron)}, reference={electron: 1})
```

It gives $R(N_{\mathrm{electron}})=1-N_{\mathrm{hole}}$. A pair pseudospin uses
`{s: down * up}` with both target occupations initially zero.

## Integer ladders

A `LadderOp` represents a bilateral integer shift, such as a Floquet index.
It has no vacuum. Its reference represents source index zero, and its number
operator must be supplied independently because $L^\dagger L=1$:

```{code-cell} ipython3
from pymablock.number_ordered_form import LadderOp

ell, target_ell = LadderOp("ell"), LadderOp("target_ell")
embedding = Embedding(
    {ell: target_ell, N(ell): N(target_ell) - 3},
    reference={target_ell: 3},
)
```

## Finite matrices from a reference list

Omit the generator mapping and list the retained target states in the desired
matrix order. Selecting the first three oscillator levels gives

```{code-cell} ipython3
embedding = Embedding(reference=[{a: 0}, {a: 1}, {a: 2}])
assert embedding.restrict(N(a)) == sympy.diag(0, 1, 2)
H_eff, *_ = block_diagonalize([H0, V], subspace_eigenvectors=embedding)
second_order_matrix = H_eff[0, 0, 2]
```

The coefficients are ordinary SymPy matrices. This is sufficient for an
artificial higher spin: its ladder matrices and any engineered drive amplitudes
are expressed directly in this basis. There is no additional higher-spin
operator class. The target oscillator remains infinite, including the virtual
transition from level two to level three.

Each dictionary must declare the same complete set of target modes. References
are distinct product occupation states and need not form a contiguous range.
Their list order fixes both matrix rows and columns. The entire list defines
one retained subspace, including its internal transitions and degeneracies.

### Matrix Hamiltonians with operator entries

The target Hamiltonian may itself be a square SymPy matrix whose entries contain
second-quantized operators. Each reference then pairs a matrix-basis index with
an occupation dictionary:

```{code-cell} ipython3
b = BosonOp("b")
H0_matrix = sympy.diag(5 * N(b), 2 + 5 * N(b))
V_matrix = sympy.Matrix([[b + Dagger(b), 2 * b + 3 * Dagger(b)],
                        [3 * b + 2 * Dagger(b), 0]])
embedding = Embedding(reference=[(0, {b: 0}), (1, {b: 0})])
H_eff, *_ = block_diagonalize(
    [H0_matrix, V_matrix], subspace_eigenvectors=embedding
)
assert H_eff[0, 0, 2] == sympy.Matrix([[-sympy.Rational(27, 35), -sympy.Rational(4, 5)],
                                     [-sympy.Rational(4, 5), -3]])
```

Both matrix components are retained at boson occupation zero. Virtual processes
can change the component and excite the boson; their denominators use both.
The matrix index is zero-based. A dictionary without an index means component
zero, and an empty dictionary selects a component of an ordinary finite matrix.
All perturbative coefficients must have the same square target shape.

The provisional `matrix_index` symbol can instead put the target component
inside each reference dictionary. The constructor normalizes this spelling to
the same tuple representation:

```{code-cell} ipython3
from pymablock.second_quantization import matrix_index

assert Embedding(reference=[{matrix_index: 0, b: 0}, {matrix_index: 1, b: 0}]) == (
    Embedding(reference=[(0, {b: 0}), (1, {b: 0})])
)
```


The [cavity application](applications/cavity_spin.md) uses this construction with
a two-by-two dressed-ancilla matrix and cavity/Floquet operator entries. Its
three- or four-state reference list returns the artificial-spin matrix directly.

## Generator lattices with a matrix index

A reference list may also accompany generators. The source then has a matrix
index labeling disjoint copies of the source algebra. For example, retain the
two lowest oscillator levels in both ancilla branches:

```{code-cell} ipython3
embedding = Embedding({s: b}, reference=[(0, {b: 0}), (1, {b: 0})])
H_eff, *_ = block_diagonalize(
    [H0_matrix, V_matrix], subspace_eigenvectors=embedding
)
assert H_eff[0, 0, 0] == sympy.diag(
    NumberOrderedForm.from_expr(5 * N(s)),
    NumberOrderedForm.from_expr(2 + 5 * N(s)),
)
fourth_order = H_eff[0, 0, 4]
```

Each coefficient is a two-by-two matrix of source `NumberOrderedForm` entries.
A mapping reference still returns one NOF; a list of one reference returns a
one-by-one matrix. Omitting generators gives the finite matrices above.

References in one target row must generate disjoint occupation lattices.
A spectator offset gives a simple example:

```{code-cell} ipython3
spectator = BosonOp("spectator")
embedding = Embedding(
    {s: b}, reference=[{b: 0, spectator: 0}, {b: 0, spectator: 1}]
)
assert embedding.restrict(N(spectator)) == sympy.diag(
    0, NumberOrderedForm.from_expr(1, operators=(s,))
)
```

Offsets along modes moved by the generators are currently rejected with
`NotImplementedError`, even if the lattices are disjoint. Such translations can
require occupation-dependent bosonic normalization; their integration with
source lifting needs further work. Spectator offsets and different matrix rows
are supported. Every reference must still satisfy the same generator norms,
phase convention, occupation bounds, and independent ladder-number images.
Invalid translations and overlapping lattices raise clear errors during
construction. Fermion ordering signs and spectator `LadderOp` shifts are
included; no vacuum is assumed for a bilateral ladder. The overlap test is exact
for all supported independent integer generator shifts and product source domains.

## Supported representations and perturbation theory

Generator mappings return `NumberOrderedForm`, including when the source
contains infinite bosonic or Floquet modes. The effective Hamiltonian is an
ordinary source operator with no embedding attachment; `as_expr()` returns its
SymPy expression. Use `filter_terms(..., keep=True)` to select occupation shifts,
then inspect or display their `as_expr()` expressions. Reference lists return
matrices whose entries use that same source algebra; without generators their
entries are ordinary scalar expressions.

An embedding maps Fock states of the source to Fock states of the target. To
embed a linear combination of target modes, such as a bonding orbital, first
rewrite the Hamiltonian in terms of operators for that combination and the
combinations orthogonal to it:

```{code-cell} ipython3
c1, c2 = FermionOp("c1"), FermionOp("c2")
bonding, antibonding, f = (FermionOp(name) for name in ("bonding", "antibonding", "f"))
H0 = 3 * (N(c1) + N(c2)) + Dagger(c1) * c2 + Dagger(c2) * c1
modes = {
    c1: (bonding + antibonding) / sympy.sqrt(2),
    c2: (bonding - antibonding) / sympy.sqrt(2),
}
modes.update({Dagger(op): Dagger(image) for op, image in list(modes.items())})
# doit() writes number operators as products that the substitution can replace.
H0 = sympy.expand(H0.doit().xreplace(modes))
embedding = Embedding({f: bonding}, reference={bonding: 0, antibonding: 0})
embedding.restrict(H0).as_expr()  # 4 * N_f
```

Each target expression must make **one independent occupation shift per source
generator**, starting from a product occupation reference. Coefficients may
depend on target occupations. Finite source modes permit occupation-dependent
phases; infinite source modes require constant phases relative to their ladder
amplitudes. General nonlinear superpositions and entangled reference states are
not supported. Validation distinguishes a violated identity
from one it cannot establish symbolically.

The default perturbative solver requires Hermitian input and a target Hamiltonian
$H_0=E(N_1,\ldots,N_M)$ diagonal in the target occupations. For a matrix
target, H0 must also be diagonal in its matrix indices, with occupation-diagonal
entries. The selected target states
are then invariant under $H_0$. Each virtual transition uses its actual energy
difference; coupled degeneracies require changing the retained block or model.
For retained infinite modes, symbolic energy denominators require the usual
nonresonance assumption on the occupations where the effective model is used.

Symbolic division selects zero when the virtual amplitude vanishes, including
at a zero gap. An exposed zero gap with a nonzero amplitude raises an error.
Unresolved occupation-dependent resonances remain symbolic poles: evaluate
the effective model only in nonresonant sectors. Binary simplification preserves
such coefficients rather than evaluating their singular points.

[The developer documentation](developer.md) describes validation,
projectors, and the rectangular blocks returned outside the retained subspace.
