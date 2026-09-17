# Structured embeddings

A many-body Hamiltonian is written in terms of physical degrees of freedom, but
its effective model may have a different natural description. Two states of an
oscillator can behave as a spin one half. A pair of fermionic modes with exactly
one particle can also behave as a spin. In both cases, the effective spin labels
selected physical states; it is not an additional physical particle.

An embedding makes that identification precise. It specifies **which source
occupation states are retained and which target operators act on their labels**.
For example, calling two oscillator states $|0\rangle_s$ and $|1\rangle_s$ lets us
define a spin lowering operator $s$ between them, even though the source
annihilation operator $a$ still acts on the whole oscillator ladder.

There are two distinct steps in constructing the effective model:

1. **Identify the retained states.** The embedding $W$ maps each target basis
   state to its selected source occupation state. This fixes the meaning of the
   new operators before perturbative corrections are calculated.
2. **Account for virtual excursions.** The perturbative unitary $\mathcal U$
   dresses those states with components outside the selected subspace. The
   effective Hamiltonian describes their dynamics using the target labels:

   $$
   H_{\mathrm{eff}}=W^\dagger\mathcal U^\dagger H\mathcal U W.
   $$

Thus $W$ is a fixed choice of states and operator labels, while $\mathcal U$
depends on the Hamiltonian and perturbation order. Keeping these roles separate
lets the embedding remain a simple state selection. The superpositions produced
by virtual transitions belong to perturbation theory.

In the interface, `target` declares the effective operators and `occupations`
expresses each source occupation in terms of the target occupations. The source
Hamiltonian keeps its original operators. Replacing those operators directly
would discard their action outside the retained states and lose the virtual
corrections.

## A complete example: two oscillator states become a spin

Consider an anharmonic oscillator with a weak drive,

$$
H_0=\omega N_a+\frac{\alpha}{2}N_a(N_a-1),
\qquad V=g(a+a^\dagger).
$$

We retain oscillator occupations zero and one. A target spin lowering operator
$s$ describes the transition between them. The rule $N_a=N_s$ specifies

$$
|0\rangle_s\longmapsto|0\rangle_a,
\qquad |1\rangle_s\longmapsto|1\rangle_a.
$$

Here is the calculation:

```python
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding

a, s = BosonOp("a"), SigmaMinus("s")
omega, alpha, g = sympy.symbols("omega alpha g", positive=True)
H0 = omega * N(a) + alpha * N(a) * (N(a) - 1) / 2
V = g * (a + Dagger(a))

embedding = Embedding(target=(s,), occupations={a: N(s)})
H_eff, U, U_adjoint = block_diagonalize(
    [H0, V], subspace_eigenvectors=embedding
)
second_order = H_eff[0, 0, 2]
```

Block zero is the retained spin. Through second order in the drive, its effective
Hamiltonian is

$$
H_{\mathrm{eff}}=
\omega N_s+g(s+s^\dagger)
-\frac{2g^2}{\omega+\alpha}N_s+O(g^3).
$$

The second-order term comes from the excursion $|1\rangle_a\to|2\rangle_a\to
|1\rangle_a$. The matrix element is $\sqrt{2}g$ and the energy difference is
$E_1-E_2=-(\omega+\alpha)$. Substituting $s$ for $a$ in the source Hamiltonian would
lose that excursion entirely. The embedding instead identifies the retained
states while preserving the source ladder amplitudes during the calculation.

`U` and `U_adjoint` are the corresponding perturbative unitary series in the
retained/complement block representation. The complement is represented by
operator actions, without constructing its full basis.

## State selections in physical models

The same interface describes several different effective degrees of freedom.
In the following declarations, source and target operators have already been
created with their appropriate SymPy types.

### Tunable coupler: two spins from three oscillators

```python
embedding = Embedding(
    target=(s1, s2),
    occupations={a1: N(s1), a2: N(s2), coupler: 0},
)
```

This retains occupations zero and one of each computational oscillator, with the
coupler in its vacuum. The effective Hamiltonian acts on two spins. Excited
coupler states and higher computational-oscillator occupations still mediate
virtual interactions. Fixing an occupation in the embedding therefore does not
remove that mode from the source Hamiltonian.

### Ring exchange: one spin per singly occupied site

```python
embedding = Embedding(
    target=(s,),
    occupations={up: N(s), down: 1 - N(s)},
)
```

The source modes `up` and `down` are fermions. Target occupation zero selects
`down` occupied, while target occupation one selects `up` occupied. Both target
states contain exactly one fermion. Applying this rule at every site gives a
spin model; empty and doubly occupied sites remain available as intermediate
states. On a Hubbard square, fourth-order hopping produces ring exchange.

### Localized spins and retained fermions together

```python
embedding = Embedding(
    target=(s, f),
    occupations={up: N(s), down: 1 - N(s), conduction: N(f)},
)
```

Here `s` is a spin and `f` is a fermion. The localized pair supplies the spin,
while the conduction mode remains fermionic. Pymablock accounts for source mode
ordering and occupied spectator modes when defining the target fermion's phase.
This preserves its correspondence with the retained source fermion, including
when the source modes are interleaved with the spin's constituent fermions.

## Higher spins and a preliminary basis rotation

For spin one, declare a three-dimensional representation:

```python
from sympy.physics.quantum.spin import JminusOp, JzOp

embedding = Embedding(
    target={JminusOp("S"): 3},
    occupations={cavity: JzOp("S") + 1, floquet: 0, dressed_ancilla: 0},
)
```

The magnetic quantum numbers $m=-1,0,1$ select cavity occupations $n=m+1=0,1,2$.
The Floquet index and dressed ancilla occupation are both fixed to zero.

In the dressed spin-1 example, the selected ancilla state in the original basis
is a superposition that depends on cavity occupation:

$$
|+_n\rangle=
\frac{e^{i\phi_n/2}|g\rangle+e^{-i\phi_n/2}|e\rangle}{\sqrt{2}}.
$$

The model first transforms the Hamiltonian into the dressed basis using

$$
R_n=\frac{1}{\sqrt{2}}
\begin{pmatrix}
e^{i\phi_n/2}&-e^{i\phi_n/2}\\
e^{-i\phi_n/2}&e^{-i\phi_n/2}
\end{pmatrix}.
$$

The first column is $|+_n\rangle$. A transition from cavity occupation $n$ to $n'$
with ancilla operator $A$ becomes $R_{n'}^\dagger A R_n$. Both the unperturbed
Hamiltonian and the perturbation must be transformed; observables must use that
same basis. In the transformed coordinates, selecting the dressed ancilla is
simply the occupation rule `dressed_ancilla: 0`.

The rotation belongs to the model construction. `Embedding` itself selects
occupation states and defines target operators; it does not diagonalize a source
Hamiltonian or construct arbitrary superpositions.

## What the calculation assumes and returns

The unperturbed Hamiltonian must be a function of source number operators,
$H_0=E(n_1,\ldots,n_M)$. Number-dependent interactions and anharmonicities are
allowed. Each virtual transition is divided by its initial-to-final energy
difference. A zero denominator in an uncoupled sector contributes zero; a
nonzero channel resonant with the retained states raises an error.

Every source mode must appear in `occupations`, including fixed modes. The rules
must be integer affine expressions, such as `N(s)`, `1 - N(s)`, or `1 + N(s)`.
They must give allowed source occupations and have full column rank, ensuring
that distinct target states select distinct source states. Direct target fermions
must each correspond to exactly one source fermion; other source occupations may
depend on target spins or be fixed.

Binary spin and fermion targets, including mixtures, return `NumberOrderedForm`
operators. If any target is declared using `JminusOp` and an explicit dimension,
the entire retained block is currently returned as a finite SymPy matrix.
Its basis follows the canonical target-generator order, with occupations
increasing for each generator; for higher spins this means increasing $m$.

## How virtual states remain implicit

Write the state selection as $W$, so $W^\dagger W=I$ on the target. The retained
projector is $P=WW^\dagger$ and the discarded projector is $Q=1-P$. The identity

$$
W^\dagger A Q B W
= W^\dagger A B W-(W^\dagger A W)(W^\dagger B W)
$$

expresses an excursion through discarded states using source operator products
and their restrictions to the target. The implementation stores complement
columns as sums of $QXWA$, where $X$ is a source operator and $A$ is a target
operator, and applies the standard Pymablock recurrence to these blocks.

This calculation uses the public `NumberOrderedForm` operations and term
interface. It does not require a particular NOF storage format or enumerate the
complementary Hilbert space. Its cost still depends on perturbative order and
operator-expression growth; avoiding basis enumeration does not guarantee
polynomial runtime.

See the [API reference](documentation/pymablock.md#structured-embeddings) for the
constructor contract. Executable physical examples and their analytic or
independent finite-matrix checks are in
`pymablock/tests/test_embedding_models.py`.
