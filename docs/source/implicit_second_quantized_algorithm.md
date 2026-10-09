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

This page explains what happens inside {func}`~pymablock.block_diagonalize` when
the subspace is an {class}`~pymablock.operator_embedding.Embedding`, and follows
one example through every step. See [structured embeddings](structured_embeddings.md)
for how to define and use an embedding.

## How it works

The original Hamiltonian $H = H_0 + V$ acts on the larger **target** Hilbert
space, while the effective model lives in the **source** Hilbert space. The embedding maps each
source lowering operator $g_j$ to a target expression $G_j$ and fixes a
reference target state $|r\rangle$. This defines the isometry

$$
W \prod_j (g_j^\dagger)^{q_j} |0\rangle = \prod_j (G_j^\dagger)^{q_j} |r\rangle ,
$$

where a ladder source uses $G_j$ itself for negative $q_j$. The retained states
have projector $P = WW^\dagger$, and $Q = 1 - P$. Block diagonalization removes
the $P$–$Q$ blocks of $H$ and returns $W^\dagger \tilde{H} W$ in source operators.

Both subspaces are usually infinite, so the algorithm never lists states: every
operator stays a [number-ordered form](second_quantization.md), and all the work
reduces to compression $R(A) = W^\dagger A W$, projection $P$, and lifting
$L(a) = W a W^\dagger$.

`block_diagonalize` proceeds in four steps:

1. **Construction.** Each $G_j$ shifts the target occupations by a fixed integer
   vector, the $j$-th column of a matrix $M$. The retained states are
   $n(q) = r + Mq$, and a left inverse recovers $q = L(n - r)$. The constructor
   also checks occupation ranges, fermionic parity, normalization, and the
   source algebra.

   ```{figure} implicit_algorithm_lattice.svg
   :alt: Retained states of a correlated boson embedding on the occupation grid

   Retained states (large dots) of a source boson stored in two target bosons,
   $W|q\rangle = |q_a q_b\rangle$. The image of $q$ (orange) lowers both
   occupations, so $M$ has the single column $(1, 1)$ and the projector selects
   $n_a = n_b$.
   ```
2. **Preparation.** The frames $F_0 = W$ and $F_1 = Q$ give the $2\times 2$ block
   Hamiltonian $H_{ij} = F_i^\dagger H F_j$. $H_0$ must be diagonal in the
   occupations.
3. **Products.** The off-diagonal blocks contain forms attached to $W$, such as
   $XW$ or $W^\dagger X$. Their products reduce to the three operations:

   | product | result | operation |
   |---|---|---|
   | $(W^\dagger X)(YW)$ | $W^\dagger XYW$ | compression $R(XY)$ |
   | $(XW)(W^\dagger Y)$ | $XPY$ | projection |
   | $(XW)\,z$, $z$ a source operator | $X\,L(z)\,W$ | lifting |

4. **Sylvester equations.** The off-diagonal block divides each term by its
   actual energy difference, evaluated on $n(q)$. A transition whose amplitude
   vanishes contributes zero, rather than the $1$ that cancelling $n/n$ would
   give; see [exact division](#exact-division-by-energy-differences).

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

The cells below inspect private implementation objects and their attributes.
They show intermediate calculations and are not part of the public interface.

### Step 1: construction

The image $a$ of $q$ lowers $n_a$ by one and leaves $n_b$ unchanged, so $M$ has
the single column $(1, 0)$ and the retained states are $n(q) = (q, 0)$:

```{code-cell} ipython3
lattice = embedding._first_lattice
print("target modes:", lattice.target_operators)
print("reference r:", lattice._reference_state)
print("shift matrix M:", lattice._occupation_matrix.tolist())
print("left inverse L:", lattice._occupation_left_inverse.tolist())
print("retained occupations n(q):", lattice._target_of_source)
```

The symbol `_source_0` is the source occupation $q$.
The projector $P = WW^\dagger$ is the indicator of these states. It combines
the constraint $n_b = 0$, which comes from the nullspace of $M^T$, with the spin
spectrum $q \in \{0, 1\}$:

```{code-cell} ipython3
lattice.projector
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
NumberOrderedForm.from_expr(Dagger(b) * a).act(lattice._target_of_source)
```

The term $b^\dagger a$ takes $|q, 0\rangle$ to $|q - 1, 1\rangle$ with matrix
element $\sqrt{q}$. Its occupation shift $(-1, +1)$ is not a multiple of the
column of $M$, so the outgoing state is never retained and the term compresses
to zero. A term whose shift *is* in the range of $M$ becomes a source operator,
with its amplitude divided by the source ladder amplitude.

### Step 3: the block Hamiltonian

`operator_to_BlockSeries` performs the same frame-based block conversion as
`block_diagonalize`. The latter also prepares the solver and validates $H_0$:

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
unperturbed part $H_0$ is diagonal in the occupations. Like other symbolic
second-quantized inputs, it ignores the numerical options `atol`,
`direct_solver`, and `solver_options`. See
[structured embeddings](structured_embeddings.md) for the supported source
algebras and for embeddings given by an explicit list of states.
