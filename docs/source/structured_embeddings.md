# Structured embeddings

An effective model can have different operators from its microscopic Hamiltonian.
For example, a spin flip can represent moving a fermion between two orbitals, or
exciting an oscillator from its ground state to its first excited state.
An embedding defines these effective operators by giving **their source
expressions and one reference state**.

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

## Integer ladders and higher spins

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

For a higher spin, specify its spin quantum number and the correct ladder
amplitude. The following spin-one generator acts on oscillator occupations
zero, one, and two, with both lowering amplitudes equal to $\sqrt{2}$:

```python
from pymablock.number_ordered_form import SpinOp

S = SpinOp("S", 1)
embedding = Embedding(
    {S: sympy.sqrt(2 - N(a)) * a},
    reference={a: 0},
)
```

The coefficient is to the left of `a`, so it is evaluated after lowering.
For dimension $d$, use `sqrt(d - 1 - N(a)) * a`; its action is
$\sqrt{n(d-n)}|n-1\rangle$.

In the artificial-spin cavity example, the ancilla eigenbasis depends on cavity
occupation. The model first rotates its Hamiltonian into that dressed basis.
The embedding then maps the normalized spin generator to the cavity expression,
with the dressed ancilla and Floquet index fixed in the reference. This preserves
the physical drive's occupation-dependent matrix elements; defining the target
spin does not replace the cavity drive by an ideal spin drive.

## Supported representations and perturbation theory

All target algebras, including higher spins, return `NumberOrderedForm`.
Higher spins compose with retained bosons, Floquet ladders, and fermions without
introducing a cutoff. Matrix conversion is explicit:

```python
embedding.restrict(N(a)).to_matrix()  # diag(0, 1, 2) for the spin-one example
```

For a target containing infinite modes, `to_matrix` requires an occupation
sequence for each operator. For example, `[range(5), range(3)]` selects five
oscillator levels and the complete spin-one representation. Products are
computed before this compression, so virtual excursions remain included.

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
after any mode rotation. The generated occupation states
are then invariant under $H_0$. Each virtual transition uses its actual energy
difference; coupled degeneracies require changing the retained block or model.
For retained infinite modes, symbolic energy denominators require the usual
nonresonance assumption on the occupations where the effective model is used.

Discarded states remain implicit through the identity

$$
W^\dagger A(1-P)BW
=W^\dagger ABW-(W^\dagger AW)(W^\dagger BW).
$$

The solver applies the standard Pymablock recurrence to these projected operator
products. Avoiding a basis enumeration does not remove the growth in operator
expressions at high perturbative orders.

Executable physical checks for supercurrent, the Crépel–Fu interaction model,
the tunable coupler, and the artificial cavity spin are in
`pymablock/tests/test_embedding_models.py`. See also the
[API reference](documentation/pymablock.md#structured-embeddings).
