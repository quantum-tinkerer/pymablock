# Structured embeddings

An effective model can have different operators from its microscopic Hamiltonian.
For example, a spin flip can represent moving a fermion between two orbitals, or
exciting an oscillator from its ground state to its first excited state.
An embedding defines these effective operators by giving **their source
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

```python
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

The dictionary maps target **lowering generators** to source expressions.
Their adjoints generate target excitations from the reference, with the
normalization and allowed occupations prescribed by the target algebra.
A fermion or spin one half admits one excitation, a boson admits arbitrarily
many, and an integer ladder admits shifts in both directions. Every source mode
appears in `reference`, including modes that remain frozen.

This defines an isometry $W$ from the target representation into the source.
If $g$ is a target generator and $G$ is its supplied source expression, then

$$
WgW^\dagger=PGP,\qquad P=WW^\dagger.
$$

The source expression may act outside the generated space. For example, using
an oscillator lowering operator as the representative of a spin does not
remove the higher oscillator levels from the source Hamiltonian. Those levels
remain available for virtual transitions.

The constructor checks the physical occupation ranges, fermionic parity,
normalization, and consistency between the supplied generators. It rejects
incorrect amplitudes rather than silently normalizing them. The reference has
phase one; the generators fix all other phases, including signs from occupied
fermionic spectators and permutations of the target modes.

`embedding.restrict(A)` evaluates

$$
R(A)=W^\dagger A W
$$

and expresses the result using the target operators. It evaluates the complete
source expression before compression. In particular, $R(AB)$ need not equal
$R(A)R(B)$: the intermediate state in $AB$ can leave the retained space.

## Oscillator to spin, including a virtual correction

Consider an anharmonic oscillator with a weak drive,

$$
H_0=\omega N_a+\frac{\alpha}{2}N_a(N_a-1),\qquad
V=g(a+a^\dagger).
$$

The generator $s\mapsto a$ and the source vacuum identify the first two
oscillator levels with the target spin. The target algebra sets the upper
boundary, so no source cutoff is needed.

```python
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

The fixed embedding defines the target operators. The perturbative unitary
$\mathcal U$ subsequently accounts for virtual dressing:
$H_{\mathrm{eff}}=W^\dagger\mathcal U^\dagger H\mathcal U W$.

## Combining degrees of freedom

Independent generator definitions fit in one mapping. A tunable coupler uses
three source oscillators and two target spins:

```python
embedding = Embedding(
    {s1: a1, s2: a2},
    reference={a1: 0, a2: 0, coupler: 0},
)
```

The coupler stays in its vacuum in the reference representation, but can be
excited during a virtual process. Both computational oscillators can likewise
visit higher levels during perturbation theory.

A localized spin, conduction fermion, and cavity can coexist in one target:

```python
embedding = Embedding(
    {s: Dagger(down) * up, f: conduction, b: cavity},
    reference={up: 0, down: 1, conduction: 0, cavity: 0},
)
```

The cavity target is a `BosonOp`, so its entire ladder is retained. Neither the
constructor nor the perturbative calculation enumerates that infinite ladder.
Number-dependent interactions and energy denominators retain their joint
dependence on all modes.

A hole is also a direct generator definition:

```python
embedding = Embedding({hole: Dagger(electron)}, reference={electron: 1})
```

It gives $R(N_{\mathrm{electron}})=1-N_{\mathrm{hole}}$. A pair pseudospin uses
`{s: down * up}` with both source occupations initially zero.

## Integer ladders

A `LadderOp` represents a bilateral integer shift, such as a Floquet index.
It has no vacuum. Its reference represents target index zero, and its number
operator must be supplied independently because $L^\dagger L=1$:

```python
from pymablock.number_ordered_form import LadderOp

ell, source_ell = LadderOp("ell"), LadderOp("source_ell")
embedding = Embedding(
    {ell: source_ell, N(ell): N(source_ell) - 3},
    reference={source_ell: 3},
)
```

## Finite matrices from a reference list

Omit the generator mapping and list the retained source states in the desired
matrix order. Selecting the first three oscillator levels gives

```python
embedding = Embedding(reference=[{a: 0}, {a: 1}, {a: 2}])
assert embedding.restrict(N(a)) == sympy.diag(0, 1, 2)
H_eff, *_ = block_diagonalize([H0, V], subspace_eigenvectors=embedding)
second_order_matrix = H_eff[0, 0, 2]
```

The coefficients are ordinary SymPy matrices. This is sufficient for an
artificial higher spin: its ladder matrices and any engineered drive amplitudes
are expressed directly in this basis. There is no additional higher-spin
operator class. The source oscillator remains infinite, including the virtual
transition from level two to level three.

Each dictionary must declare the same complete set of source modes. References
are distinct product occupation states and need not form a contiguous range.
Their list order fixes both matrix rows and columns. The entire list defines
one retained subspace, including its internal transitions and degeneracies.

### Matrix Hamiltonians with operator entries

A source may itself be a square SymPy matrix whose entries contain
second-quantized operators. Each reference then pairs a matrix-basis index with
an occupation dictionary:

```python
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
All perturbative coefficients must have the same square source shape.

The [cavity application](applications/cavity_spin.md) uses this construction with
a two-by-two dressed-ancilla matrix and cavity/Floquet operator entries. Its
three- or four-state reference list returns the artificial-spin matrix directly.

## Supported representations and perturbation theory

Generator mappings return `NumberOrderedForm`, including when the target
contains infinite bosonic or Floquet modes. Its `to_matrix()` method uses the
full binary bases by default. For infinite modes, supply one occupation sequence
per operator, for example `[range(5), range(2)]` for an oscillator and a fermion.
Products are evaluated before this compression. Reference-list embeddings
return finite matrices and specify all occupations explicitly; they do not
provide matrices with symbolic target-operator entries.

Normalized linear mode mixing is supported directly:

```python
c1, c2, f = (FermionOp(name) for name in ("c1", "c2", "f"))
embedding = Embedding(
    {f: (c1 + sympy.I * c2) / sympy.sqrt(2)},
    reference={c1: 0, c2: 0},
)
embedding.restrict(c1).as_expr()  # f / sqrt(2)
```

The same construction works for bosons. The compiler completes orthonormal
linear images to a source basis rotation, retaining the orthogonal combinations
as virtual modes. Each mixed set must start in its empty Fock state. Symbolic
two-mode rotations are supported; larger rotations require the compiler to
establish nonzero norms when completing the basis. Images are checked for
normalization and mutual orthogonality, rather than silently renormalized.

After any linear rotation, the compiler supports source expressions with **one
independent occupation shift per target generator** and a product occupation
reference. Coefficients may depend on source occupations. Finite targets permit
occupation-dependent phases; infinite targets require constant phases relative
to their ladder amplitudes. General nonlinear superpositions and entangled
reference states are not supported. Validation distinguishes a violated identity
from one it cannot establish symbolically.

The default perturbative solver requires Hermitian input and a source Hamiltonian
$H_0=E(N_1,\ldots,N_M)$ diagonal in the compiled source occupations, including
after any mode rotation. For a matrix source, H0 must also be diagonal in its
matrix indices, with occupation-diagonal entries. The selected source states
are then invariant under $H_0$. Each virtual transition uses its actual energy
difference; coupled degeneracies require changing the retained block or model.
For retained infinite modes, symbolic energy denominators require the usual
nonresonance assumption on the occupations where the effective model is used.

The solver constructs the fixed occupation projector $P=WW^\dagger$ and its
complement $Q=1-P$, using equality-based `Piecewise` expressions. All virtual
products are evaluated as ordinary source operators, without enumerating or
truncating the discarded states. The retained block is compressed into the
target algebra during the standard Pymablock recurrence.

For each of `H_eff`, `U`, and `U_adjoint`, block `[0, 0, ...]` uses the target
operators or reference-list matrix basis. The other blocks are source-space
operators supported on $P$ or $Q$ on the corresponding side. With linear mode
mixing, these source operators use the compiler's rotated modes. The usual
`zero` and `one` series sentinels represent zero and the identity on the block's
space. Ordinary multiplication of the returned blocks requires lifting the
retained block back to the source space first.

`NumberOrderedForm.as_expr()` exports diagonal projectors as ordinary SymPy
`Piecewise` expressions, which can be read back with `from_expr()`. A SymPy
assumptions patch preserves noncommutativity when the conditions contain
operators; the scalar occupation coefficients inside a NOF remain commutative.

For infinite bosonic targets, the solver currently requires that nonnegative
target occupations follow from the physical source occupations. Other selections
raise `NotImplementedError` because they require occupation inequalities.
Finite reference lists and binary targets use only equality conditions.
Avoiding basis enumeration does not remove expression growth at high orders.

Executable physical checks for supercurrent, the Crépel–Fu interaction model,
the tunable coupler, and the artificial cavity spin are in
`pymablock/tests/test_embedding_models.py`. See also the
[API reference](documentation/pymablock.md#structured-embeddings).
