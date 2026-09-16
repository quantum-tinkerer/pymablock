# Structured embeddings

A many-body effective model often keeps only a few states of each physical mode.
For example, the first two occupations of an anharmonic oscillator can represent
a spin one half. The discarded occupations still contribute through virtual
transitions, so simply truncating the Hamiltonian would miss their effect.

A structured embedding describes which source occupation states represent the
target degrees of freedom. Pymablock uses this description to compute the
effective Hamiltonian and unitary series while keeping transitions through the
discarded space symbolic. It does not enumerate a basis for that space.

## Defining the retained states

For an oscillator represented by a binary spin, the occupation rule is

$$
W|0\rangle_s=|0\rangle_a,\qquad W|1\rangle_s=|1\rangle_a.
$$

The isometry $W$ maps target states to source states. In code, declare the target
operator and express the source occupation in terms of the target occupation:

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

Here block zero is the retained spin. The occupation rule identifies states;
it is not a substitution of `s` for `a`. Source ladder amplitudes and excursions
to oscillator occupation two still enter the perturbative calculation.

Every source mode must appear in `occupations`, including modes fixed to a
single occupation. The rules must be injective integer affine functions of the
target occupations and must give physically allowed source occupations.

Binary spin and fermion targets return `NumberOrderedForm` operators. Higher
spins are declared with their dimension, such as `target={JminusOp("S"): 3}`,
and currently return finite matrices in increasing magnetic-quantum-number order.
Fermion targets support direct one-to-one source assignments with other source
occupations fixed. Mixed spin and fermion targets are not supported.

## Virtual transitions and energy denominators

The retained projector is $P=WW^\dagger$ and its complement is $Q=1-P$.
Compression gives a target operator $W^\dagger A W$, but compression alone is not
multiplicative: a product can leave the retained space and return. In particular,

$$
W^\dagger A Q B W
= W^\dagger A B W-(W^\dagger A W)(W^\dagger B W).
$$

Pymablock represents complement columns as sums of $QXWA$, where $X$ acts on the
source and $A$ acts on the target. These maps let the usual perturbative recurrence
account for virtual excursions without constructing the discarded basis.

The unperturbed Hamiltonian must be diagonal in the supplied occupation basis,
$H_0=E(n_1,\ldots,n_M)$. This includes number-dependent interactions and
anharmonicities. Each transition has an energy difference obtained by evaluating
$E$ on its initial and final occupations. The Sylvester solver divides by these
differences; a nonzero virtual channel resonant with the retained space cannot be
eliminated by this expansion.

This representation avoids explicit enumeration of the complementary Hilbert
space. Its computational cost still depends on perturbative order and the number
and complexity of generated operator terms.

## Physical examples

### Tunable coupler

For two computational oscillators and an auxiliary coupler, retain the first two
occupations of each computational oscillator and the coupler vacuum:

```python
embedding = Embedding(
    target=(s1, s2),
    occupations={a1: N(s1), a2: N(s2), coupler: 0},
)
```

The target is a pair of spins. Virtual excitation of the coupler and higher
oscillator occupations contributes to their effective interaction even though
those states are absent from the target. The physical tests check the mediated
exchange analytically and Hermiticity through fourth order.

### Ring exchange

At a singly occupied fermionic site, a spin can label which of two fermion modes
is occupied:

```python
embedding = Embedding(
    target=(s,),
    occupations={up: N(s), down: 1 - N(s)},
)
```

Applying this rule at each site retains one fermion per site. Hopping can create
virtual empty and doubly occupied sites. The fourth-order expansion on a square
then produces ring exchange; the physical test checks the coefficient
$40t^4/U^3$. Fermionic signs are evaluated in the source algebra before the result
is expressed in target spins.

### Dressed spin one

A cavity and a two-state ancilla can realize a spin-1 target using cavity
occupations zero, one, and two. In the example, the retained ancilla state depends
on cavity occupation:

$$
|+_n\rangle = \frac{e^{i\phi_n/2}|g\rangle
+e^{-i\phi_n/2}|e\rangle}{\sqrt{2}}.
$$

These are superpositions in the bare ancilla basis. The model first transforms
the Hamiltonian using the occupation-dependent matrices

$$
R_n=\frac{1}{\sqrt{2}}
\begin{pmatrix}
e^{i\phi_n/2}&-e^{i\phi_n/2}\\
e^{-i\phi_n/2}&e^{-i\phi_n/2}
\end{pmatrix}.
$$

Their first columns are the retained states. A transition from occupation $n$ to
$n'$ with ancilla operator $A$ transforms as $R_{n'}^\dagger A R_n$.
Transforming the perturbation as well as $H_0$ is essential because these rotations
differ between occupations.

In the rotated basis the selected ancilla has occupation zero, so the embedding
is an occupation-state map:

```python
from sympy.physics.quantum.spin import JminusOp, JzOp

embedding = Embedding(
    target={JminusOp("S"): 3},
    occupations={cavity: JzOp("S") + 1, floquet: 0, dressed_ancilla: 0},
)
```

The target magnetic quantum numbers $-1,0,1$ correspond to cavity occupations
$0,1,2$. The model supplies the preliminary rotation; `Embedding` does not
construct it. Observables must also be transformed into the same source basis.
The physical tests compare the first-order drive with an analytic expression and
the second-order correction with an independent finite occupation-matrix result.

## Implementation boundary

The public descriptor is `Embedding`, documented in the
[API reference](documentation/pymablock.md#structured-embeddings). Occupation
transitions, projected maps, and the solver are private implementation details
under `pymablock._embedding`. They use the public `NumberOrderedForm` operations
and term interface, so the embedding construction does not depend on packed
storage or a particular internal representation of number coefficients.

The executable physical models and their reference checks are grouped in
`pymablock/tests/test_operator_embedding/test_models.py`, alongside the descriptor,
transition, and projected-map tests.
