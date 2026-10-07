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
  execution_timeout: 300
---

# The implicit second-quantized algorithm

This page explains how Pymablock block-diagonalizes a second-quantized
Hamiltonian when the subspace is given by an {class}`~pymablock.operator_embedding.Embedding`.
It follows one small example through every step, using the code itself to
produce each intermediate object.

The [structured embeddings](structured_embeddings.md) page explains how to
*define* and *use* an embedding. Here we describe what happens inside.
Familiarity with the [block diagonalization algorithm](algorithms.md) and with
[number-ordered forms](second_quantization.md) helps, but the main ideas are
repeated below.

## The problem

The Hamiltonian $H = H_0 + V$ acts on a Fock space, the **target**.
The states we want an effective model for are described by a smaller
**source** algebra, for example a spin. The embedding gives a target expression
$G_j$ for each source lowering operator $g_j$, together with a reference target
state $|r\rangle$. It defines the isometry $W$ from source states to target
states by

$$
W \prod_j (g_j^\dagger)^{q_j} |0\rangle = \prod_j (G_j^\dagger)^{q_j} |r\rangle
$$

for every allowed source occupation $q$; a ladder source uses $G_j$ itself for
negative $q_j$. The constructor checks that the states
on the right are orthonormal and have the norms of the states on the left, so
that $W^\dagger W = 1$. We write

$$
P = WW^\dagger, \qquad Q = 1 - P,
$$

for the projectors onto the retained states and onto everything else.
Perturbation theory then finds a unitary $\mathcal{U}$ such that
$\tilde{H} = \mathcal{U}^\dagger H \mathcal{U}$ has no $P$–$Q$ blocks, and
reports the retained block as a source operator,

$$
\tilde{H}_{\text{eff}} = W^\dagger \tilde{H} W .
$$

Both subspaces are typically infinite: an oscillator has infinitely many
levels, and the retained states can be a whole sublattice of occupations.
The algorithm is **implicit** because it never lists states. Every operator
stays a number-ordered form, a sum of terms

$$
c(N_1, N_2, \dots)\,
(a_1^\dagger)^{k_1}\cdots\, a_1^{p_1}\cdots,
$$

whose coefficient $c$ is a function of the number operators. All the work
reduces to three operations on such forms:

- **compression** $R(A) = W^\dagger A W$, from target to source;
- **projection** $P = WW^\dagger$, an occupation-dependent indicator;
- **lifting** $L(a) = W a W^\dagger$, from source to target.

## Workflow

```{figure} implicit_algorithm_workflow.svg
:alt: Call flow of block_diagonalize with an Embedding

The call flow. Construction happens once per embedding. `_prepare` performs
every check that can fail before the lazy series starts. After that, the usual
block diagonalization runs unchanged: the embedding supplies the Hamiltonian
blocks and the Sylvester solver, and operators attached to $W$ handle the
products.
```

1. **Construction** compiles the embedding. Each source lowering operator $g$
   is given by a target expression $G$ that changes every target occupation by
   a fixed amount. Collecting these shifts as the columns of an integer matrix
   $M$, the retained states are

   $$
   n(q) = r + M q,
   $$

   where $r$ is the reference state and $q$ are the source occupations. A left
   inverse $L$ reads $q = L(n - r)$ back from target occupations. The
   constructor checks the occupation ranges, fermionic parity, normalization,
   and that the images obey the source algebra.

   ```{figure} implicit_algorithm_lattice.svg
   :alt: Retained states of a correlated boson embedding on the occupation grid

   The retained states are points on the grid of target occupations. Here a
   source boson $q$ is represented by two target bosons with equal occupation,
   $W|q\rangle = |q_a q_b\rangle$.
   Its image lowers both modes by one (orange), so $M$ has the single column
   $(1, 1)$ and the retained states lie on the diagonal (large dots). The
   projector selects exactly these points: $n - r$ must be orthogonal to the
   nullspace of $M^T$, and $q$ must lie in the spectrum of the source mode.
   ```
2. **Preparation** builds the frames $F_0 = W$ and $F_1 = Q$ and wraps the
   Hamiltonian in the $2\times 2$ block series $H_{ij} = F_i^\dagger H F_j$.
   It also checks that $H_0$ is diagonal in the occupations and builds the
   Sylvester solver.
3. **Perturbation theory** proceeds order by order as in the
   [general algorithm](algorithms.md). It requests blocks of $H$, solves
   Sylvester equations, and multiplies blocks. The off-diagonal blocks are
   rectangular: their entries are forms attached to $W$ on one side, such as
   $XW$ or $W^\dagger X$. Products of attached forms reduce to the three
   operations above:

   | product | result | operation |
   |---|---|---|
   | $(W^\dagger X)(YW)$ | $W^\dagger XYW$ | compression $R(XY)$ |
   | $(XW)(W^\dagger Y)$ | $XPY$ | projection |
   | $(XW)\,z$, $z$ a source operator | $X\,L(z)\,W$ | lifting |

4. **Sylvester equations.** The diagonal blocks use the ordinary
   second-quantized solver. The retained–complement block divides each
   transition by its *actual* energy difference: the term is applied to the
   symbolic retained occupations $n(q)$, the energies of the incoming and
   outgoing states are evaluated, and the coefficient is divided by their
   difference. The division is exact. When the coefficient and the energy
   difference vanish together, the transition is inactive and its contribution
   is zero, rather than the value $1$ that cancelling $n/n$ would give.

## A worked example

A qubit is stored in the two lowest levels of an anharmonic oscillator $a$,
which couples to an empty resonator $b$:

$$
H_0 = \omega N_a + \frac{U}{2} N_a (N_a - 1) + \Omega N_b, \qquad
V = g\,(a^\dagger b + b^\dagger a).
$$

The source is a spin $q$, represented by $q \mapsto a$ with both modes empty in
the reference state. The definition of $W$ above then gives
$W|0\rangle = |0_a 0_b\rangle$ and $W|1\rangle = a^\dagger |0_a 0_b\rangle = |1_a 0_b\rangle$.

```{code-cell} ipython3
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize, operator_to_BlockSeries
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.second_quantization import Embedding

a, b, q = BosonOp("a"), BosonOp("b"), SigmaMinus("q")
omega, Omega, U, g = sympy.symbols("omega Omega U g", positive=True)

H0 = omega * N(a) + U / 2 * N(a) * (N(a) - 1) + Omega * N(b)
V = g * (Dagger(a) * b + Dagger(b) * a)
embedding = Embedding({q: a}, reference={a: 0, b: 0})
```

```{figure} implicit_algorithm_levels.svg
:alt: Energy levels of the worked example

The lowest target states. The embedding keeps $|0_a 0_b\rangle$ and
$|1_a 0_b\rangle$ as the qubit states $q = 0, 1$. The coupling connects
$|1_a 0_b\rangle$ to the complement state $|0_a 1_b\rangle$. The $\sqrt{2}g$
coupling between $|2_a 0_b\rangle$ and $|1_a 1_b\rangle$ exists in $H$, but no
path from the retained states reaches it.
```

The cells below inspect internal attributes, whose names start with an
underscore. They show the intermediate objects and are not part of the public
interface.

### Step 1: construction

The image $a$ of $q$ lowers $n_a$ by one and leaves $n_b$ unchanged, so $M$ has
the single column $(1, 0)$ and the retained states are $n(q) = (q, 0)$:

```{code-cell} ipython3
print("modes:", embedding._target_operators)
print("reference r:", embedding._reference_state)
print("shift matrix M:", embedding._occupation_matrix.tolist())
print("left inverse L:", embedding._occupation_left_inverse.tolist())
print("retained occupations n(q):", embedding._target_occupations)
```

The symbol `_source_0` is the source occupation $q$.
The projector $P = WW^\dagger$ is the indicator of these states. It combines
the constraint $n_b = 0$, which comes from the nullspace of $M^T$, with the spin
spectrum $q \in \{0, 1\}$:

```{code-cell} ipython3
embedding._projector
```

### Step 2: compression

`restrict` computes $R(A) = W^\dagger A W$ in the source operators:

```{code-cell} ipython3
for name, operator in [
    ("H0", H0),
    ("V", V),
    ("a", a),
    ("a a†", a * Dagger(a)),
    ("restrict(a) restrict(a†)", None),
]:
    if operator is None:
        result = embedding.restrict(a) * embedding.restrict(Dagger(a))
    else:
        result = embedding.restrict(operator)
    print(f"{name:26} -> {result.simplify()}")
```

- $R(H_0) = \omega N_q$: the anharmonicity term $N_q(N_q - 1)$ vanishes because
  $N_q \in \{0, 1\}$.
- $R(V) = 0$: every term of $V$ changes $n_b$, which leaves the retained states.
- $R(a)$ is the source operator $q$, which SymPy prints as `SigmaMinus()`.
- $R(aa^\dagger) = 1 + N_q$, while $R(a)R(a^\dagger) = 1 - N_q$. The product is
  evaluated in the full target space first: on $q = 1$, $a^\dagger$ passes
  through $|2_a\rangle$, which is outside the retained states.

To compress a term, the code applies it to the symbolic retained occupations
with `NumberOrderedForm.act`:

```{code-cell} ipython3
NumberOrderedForm.from_expr(Dagger(b) * a).act(embedding._target_occupations)
```

The term $b^\dagger a$ takes $|q, 0\rangle$ to $|q - 1, 1\rangle$ with matrix
element $\sqrt{q}$. Its occupation shift $(-1, +1)$ is not a multiple of the
column of $M$, so the outgoing state is never retained and the term compresses
to zero. A term whose shift *is* in the range of $M$ becomes a source operator,
with its amplitude divided by the source ladder amplitude.

### Step 3: the block Hamiltonian

`operator_to_BlockSeries` performs the same conversion that `block_diagonalize`
does in `_prepare`:

```{code-cell} ipython3
from IPython.display import display

H = operator_to_BlockSeries([H0, V], subspace_eigenvectors=embedding)
for index, label in [
    ((0, 0, 0), "W† H0 W"),
    ((1, 1, 0), "Q H0 Q"),
    ((0, 0, 1), "W† V W"),
    ((1, 0, 1), "Q V W"),
]:
    print(f"H{index} = {label}:")
    display(H[index])
```

The retained block of $H_0$ is its compression. The complement block is $H_0$
with the retained energies removed. The off-diagonal block $QVW$ is a form
attached to $W$ on the right. The indicators in it come from
$Q = 1 - WW^\dagger$: the part of $a^\dagger b$ that would land back in a
retained state is removed.

### Step 4: order by order

```{code-cell} ipython3
H_tilde, U_series, U_adjoint = block_diagonalize(
    [H0, V], subspace_eigenvectors=embedding, symbols=[g]
)
for order in range(5):
    print(f"order {order}:")
    display(H_tilde[0, 0, order])
```

**Order 0** is the compression of $H_0$.

**Order 1** has no retained correction, because $R(V) = 0$. To continue,
the algorithm needs the first-order rotation, which solves a Sylvester equation
for the off-diagonal block:

```{code-cell} ipython3
U_series[1, 0, 1]
```

The solver applies $QVW$ to $n(q) = (q, 0)$. The term $g\,b^\dagger a$ goes to
$|q - 1, 1\rangle$, and the energy difference is

$$
E(q - 1, 1) - E(q, 0) = \Omega - \omega - U (q - 1).
$$

The amplitude $\sqrt{q}$ makes the transition active only at $q = 1$, where the
difference is $\Omega - \omega$. The indicators
$\mathbb{1}[N_a = 0]\,\mathbb{1}[N_b = 0]$ sit between $b^\dagger$ and $a$. They
pin the occupations after $a$ acts to $n_a = 0$, that is, to the incoming state
$q = 1$.

**Order 2** comes from products such as $H_{01}\,\mathcal{U}_{10}$. These are
products of $W^\dagger X$ with $YW$, so they reduce to the compression of the
target product $XY$:

$$
\tilde{H}^{(2)}_{\text{eff}} = -\frac{g^2}{\Omega - \omega}\, N_q .
$$

This is the dispersive shift of the qubit.

**Odd orders vanish**: every path that starts in a retained state must return
to $n_b = 0$, which takes an even number of photon exchanges.

**Order 4** combines both kinds of products. $(XW)(W^\dagger Y)$ places the
projector between target factors, and $(W^\dagger X)(YW)$ compresses at the end:

$$
\tilde{H}^{(4)}_{\text{eff}} = \frac{g^4}{(\Omega - \omega)^3}\, N_q .
$$

### Independent check

With the resonator empty, $|1_a 0_b\rangle$ couples only to $|0_a 1_b\rangle$.
The exact shift of the qubit frequency is therefore the lower eigenvalue of a
$2\times2$ matrix minus $\omega$:

$$
\delta E = \frac{\Delta - \sqrt{\Delta^2 + 4g^2}}{2}, \qquad \Delta = \Omega - \omega .
$$

The perturbative orders reproduce its expansion:

```{code-cell} ipython3
Delta = sympy.Symbol("Delta", positive=True)
exact = sympy.series((Delta - sympy.sqrt(Delta**2 + 4 * g**2)) / 2, g, 0, 7).removeO()


def qubit_shift(term):
    """Value of a correction on the excited qubit state; zero orders are not forms."""
    return term.as_expr().subs(N(q), 1) if isinstance(term, NumberOrderedForm) else 0


perturbative = sum(qubit_shift(H_tilde[0, 0, order]) for order in range(1, 7))
perturbative = perturbative.subs(Omega, omega + Delta)
assert sympy.simplify(exact - perturbative) == 0
exact
```

The anharmonicity $U$ drops out of all corrections for the same reason that
the $\sqrt{2}g$ coupling in the figure is never reached: no path from the
retained states reaches $n_a = 2$.

## Exact division by energy differences

The division in step 4 is the one place where an implicit treatment differs
from a matrix calculation. The occupations are symbols, so cancelling a common
factor could silently assign a value to a transition that does not occur.
For a coefficient $c$, an energy difference $\Delta E$, and the full amplitude
of the transition, the division

1. returns zero if the amplitude vanishes identically;
2. evaluates the quotient separately on any occupation value fixed by an
   indicator in the coefficient;
3. raises an error if $\Delta E = 0$ while the amplitude is nonzero, since the
   equation has no solution;
4. returns $c / \Delta E$ when $\Delta E$ cannot vanish on the allowed
   occupations, for example when it is a positive constant plus nonnegative
   occupations, or when it contains a generic parameter such as $\Omega$,
   which is assumed not to be tuned to a resonance;
5. otherwise returns $c/\Delta E$ conditional on the transition being active,
   and zero where an occupation-dependent factor of the amplitude vanishes.

Resonances that the rules above cannot detect remain poles of the result. The
effective Hamiltonian is then valid away from those parameter values.

## Requirements

The implicit algorithm currently requires a Hermitian Hamiltonian whose
unperturbed part $H_0$ is diagonal in the occupations. It uses exact symbolic
arithmetic, so the numerical options `atol`, `direct_solver`, and
`solver_options` are not supported. See
[structured embeddings](structured_embeddings.md) for the supported source
algebras and for embeddings given by an explicit list of states.
