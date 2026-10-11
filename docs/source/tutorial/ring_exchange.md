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
---

# Ring exchange on a Hubbard plaquette

At half filling and strong repulsion, each site of the Hubbard model holds one
electron, and only its spin remains free. Virtual hopping of the electrons
couples the spins, and at second order in the hopping this gives the Heisenberg
model. Which interaction supplements Heisenberg exchange at fourth order?
An electron can then hop around a square plaquette, and this generates a
four-spin interaction called ring exchange.

In this tutorial we derive the spin model of a single square plaquette directly
from its Hubbard Hamiltonian. To fourth order in the hopping $t$, the result is

$$
H_{\textrm{eff}} = E_0
+ J_1 \sum_{\textrm{edges}} \mathbf{S}_i \cdot \mathbf{S}_j
+ J_2 \sum_{\textrm{diagonals}} \mathbf{S}_i \cdot \mathbf{S}_j
+ J_c \left[
(\mathbf{S}_1 \cdot \mathbf{S}_2)(\mathbf{S}_3 \cdot \mathbf{S}_4)
+ (\mathbf{S}_1 \cdot \mathbf{S}_4)(\mathbf{S}_2 \cdot \mathbf{S}_3)
- (\mathbf{S}_1 \cdot \mathbf{S}_3)(\mathbf{S}_2 \cdot \mathbf{S}_4)
\right],
$$

with

$$
J_1 = \frac{4t^2}{U} - \frac{20t^4}{U^3}, \qquad
J_2 = \frac{4t^4}{U^3}, \qquad
J_c = \frac{80t^4}{U^3}, \qquad
E_0 = -\frac{4t^2}{U} + \frac{13t^4}{U^3}.
$$

Here $\mathbf{S}_i$ is the spin-1/2 operator of site $i$, the sites $1, 2, 3, 4$
go around the square, and $U$ is the on-site repulsion.
The ring-exchange coefficient $J_c = 80t^4/U^3$ and the diagonal coupling $J_2$
agree with the square-lattice result of MacDonald, Girvin, and Yoshioka [^mgy].
The correction to $J_1$ differs from the lattice value, and we explain why
below.

[^mgy]: A. H. MacDonald, S. M. Girvin, and D. Yoshioka,
  [Phys. Rev. B **37**, 9753 (1988)](https://doi.org/10.1103/PhysRevB.37.9753).
  The explicit spin form used here is Eq. (11) of J.-Y. P. Delannoy *et al.*,
  [Phys. Rev. B **72**, 115114 (2005)](https://doi.org/10.1103/PhysRevB.72.115114).

## Model

The full Hamiltonian is $H = H_0 + V$ with

$$
H_0 = U \sum_{i=1}^{4} n_{i\uparrow} n_{i\downarrow}, \qquad
V = -t \sum_{\langle ij \rangle} \sum_{\sigma=\uparrow,\downarrow}
\left( c^\dagger_{i\sigma} c_{j\sigma} + c^\dagger_{j\sigma} c_{i\sigma} \right).
$$

Here $c_{i\sigma}$ annihilates an electron with spin $\sigma$ on site $i$,
$n_{i\sigma} = c^\dagger_{i\sigma} c_{i\sigma}$, and $\langle ij \rangle$ runs over
the four edges of the square. The unperturbed Hamiltonian $H_0$ counts doubly
occupied sites, and the hopping $V$ is the perturbation.

```{code-cell} ipython3
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus
from IPython.display import display

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.second_quantization import Embedding
```

We label the sites $0$ to $3$ in the code, following Python indexing.

```{code-cell} ipython3
t, U = sympy.symbols("t U", positive=True)
sites = range(4)
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

c_up = [FermionOp(rf"c_{{{i}\uparrow}}") for i in sites]
c_down = [FermionOp(rf"c_{{{i}\downarrow}}") for i in sites]
```

The Hamiltonian then reads

```{code-cell} ipython3
H_0 = U * sum(N(c_up[i]) * N(c_down[i]) for i in sites)
V = -t * sum(
    Dagger(c[i]) * c[j] + Dagger(c[j]) * c[i]
    for i, j in edges
    for c in (c_up, c_down)
)
```

## What is kept

The hopping conserves the number of electrons, so we stay at four electrons.
States with one electron per site have $H_0 = 0$, while any other state has an
empty and a doubly occupied site and costs at least $U$. For $t \ll U$ the
low-energy states are therefore the 16 singly occupied states, and the effective
model has one spin-1/2 per site.

We define this effective model with an
{class}`~pymablock.operator_embedding.Embedding`, which states how each spin
operator acts on the electrons. The lowering operator $s_i$ of spin $i$ flips the
electron on site $i$ from spin up to spin down:

$$
s_i \longmapsto c^\dagger_{i\downarrow} c_{i\uparrow}.
$$

The embedding also needs one reference state. It must be annihilated by every
lowering operator, so it is the state with all spins down: one spin-down
electron on every site.

```{code-cell} ipython3
spins = [SigmaMinus(str(i)) for i in sites]
embedding = Embedding(
    {spins[i]: Dagger(c_down[i]) * c_up[i] for i in sites},
    reference={**{c: 0 for c in c_up}, **{c: 1 for c in c_down}},
)
```

Raising spins one at a time from the reference moves an electron between the two
spin orbitals of its own site, so every state reached has exactly one electron
per site. Every state with an empty or doubly occupied site is a virtual state.
Pymablock never lists either set of states: it works only with the reference and
the four operator images above.

In the effective model, $s_i^\dagger$ raises spin $i$, and the number operator
$N_i = s_i^\dagger s_i$ equals one for spin up and zero for spin down. The spin
operators are therefore

$$
S^z_i = N_i - \tfrac{1}{2}, \qquad S^+_i = s_i^\dagger, \qquad S^-_i = s_i,
$$

and we use these relations to write the expected spin models in code.
In displayed output, Pymablock writes $s_i$ as $\sigma_-^{(i)}$ and
$s_i^\dagger$ as $\sigma_+^{(i)}$.

```{code-cell} ipython3
def spin_vector(i):
    s = spins[i]
    return (
        (Dagger(s) + s) / 2,
        (Dagger(s) - s) / (2 * sympy.I),
        N(s) - sympy.Rational(1, 2),
    )


def dot(i, j):
    return sum(a * b for a, b in zip(spin_vector(i), spin_vector(j)))
```

## Perturbation theory

We pass the Hamiltonian and the embedding to
{autolink}`~pymablock.block_diagonalize`. It returns the effective Hamiltonian
as a series, where `H_tilde[0, 0, n]` is the correction of order $n$ in $t$,
written in the spin operators $s_i$.

```{code-cell} ipython3
H_tilde, *_ = block_diagonalize([H_0, V], subspace_eigenvectors=embedding)
second_order = H_tilde[0, 0, 2].simplify()
```

At second order we expect the Heisenberg model,

$$
H^{(2)}_{\textrm{eff}} = \frac{4t^2}{U} \sum_{\textrm{edges}}
\left( \mathbf{S}_i \cdot \mathbf{S}_j - \tfrac{1}{4} \right).
$$

The constant $-1/4$ makes the energy vanish for parallel spins, which cannot hop
because of the Pauli principle. The second-order correction agrees with this
model exactly:

```{code-cell} ipython3
heisenberg = 4 * t**2 / U * sum(dot(i, j) - sympy.Rational(1, 4) for i, j in edges)
assert (second_order - NumberOrderedForm.from_expr(heisenberg)).simplify().is_zero
```

The full expression has a term for each pair of neighboring spins. One of them
exchanges spins $0$ and $1$ with amplitude $J_1/2 = 2t^2/U$, the coefficient of
$S^+_0 S^-_1$ in $\mathbf{S}_0 \cdot \mathbf{S}_1$.
In `filter_terms`, a power of $-1$ selects $s_i^\dagger$ and a power of $1$
selects $s_i$.

```{code-cell} ipython3
display(second_order.filter_terms(((-1, 1, 0, 0),), keep=True).as_expr())
```

The third-order correction vanishes, because an electron cannot return to its
starting configuration after three hops on a square.

```{code-cell} ipython3
assert H_tilde[0, 0, 3].simplify().is_zero
fourth_order = H_tilde[0, 0, 4].simplify()
```

## Result and check

Printed in spin operators, the fourth-order correction has many terms.
We compare it with the spin model of the introduction instead.

```{code-cell} ipython3
diagonals = [(0, 2), (1, 3)]
ring = dot(0, 1) * dot(2, 3) + dot(0, 3) * dot(1, 2) - dot(0, 2) * dot(1, 3)
spin_model = (t**4 / U**3) * (
    13
    - 20 * sum(dot(i, j) for i, j in edges)
    + 4 * sum(dot(i, j) for i, j in diagonals)
    + 80 * ring
)
assert (fourth_order - NumberOrderedForm.from_expr(spin_model)).simplify().is_zero
```

The two expressions agree. One term shows the ring exchange directly: the
process that flips all four spins, taking $\uparrow\downarrow\uparrow\downarrow$
around the square to $\downarrow\uparrow\downarrow\uparrow$. A two-spin interaction
flips at most two spins, so only the ring term contributes. Its coefficient is
$J_c/2$, because the first two products in the ring term each contain
$S^+_0 S^-_1 S^+_2 S^-_3 / 4$.

```{code-cell} ipython3
four_spin_flip = fourth_order.filter_terms(((-1, 1, -1, 1),), keep=True).as_expr()
display(four_spin_flip)
```

MacDonald, Girvin, and Yoshioka derived the same expansion for the infinite
square lattice. They found

$$
J_1 = \frac{4t^2}{U} - \frac{24t^4}{U^3}, \qquad
J_2 = J_3 = \frac{4t^4}{U^3}, \qquad
J_c = \frac{80t^4}{U^3},
$$

where $J_3$ couples sites two lattice spacings apart along a row.
At fourth order an electron hops four times, so each process stays within a
connected cluster of at most four sites. Each lattice coupling is therefore a
sum of contributions from three kinds of cluster:

- A single bond contributes $-16t^4/U^3$ to the exchange across it.
- An open chain of three sites adds nothing to its two bonds and couples its
  end sites with $+4t^4/U^3$.
- A plaquette, after subtracting the bonds and chains it contains, adds
  $-4t^4/U^3$ to all six pairs of its sites, and the ring term with $80t^4/U^3$.

Diagonal neighbors are the end sites of two three-site chains, one through each
of the other two corners, and they share one plaquette. This gives
$J_2 = (2 \times 4 - 4)t^4/U^3 = 4t^4/U^3$, the same on the lattice and on an
isolated plaquette, which contains the same chains. Sites two spacings apart
along a row are the ends of a single straight chain and share no plaquette, so
$J_3 = 4t^4/U^3$. An isolated plaquette has no such pairs.

The fourth-order correction to $J_1$ differs, because each lattice bond belongs
to two plaquettes. The isolated plaquette gives $(-16 - 4)t^4/U^3 = -20t^4/U^3$,
while the lattice gives $(-16 - 2 \times 4)t^4/U^3 = -24t^4/U^3$.

## Discussion

Pymablock never truncated the electronic Hilbert space and never enumerated the
singly occupied states. Doubly occupied and empty sites entered only as virtual
states, with their energies taken from $H_0$. The result contains every
process up to fourth order in $t$, stays symbolic in both $t$ and $U$, and is
written directly in spin operators.

The expansion is valid for $t \ll U$. Unequal hoppings on the edges, or a
diagonal hopping $t'$ that frustrates the antiferromagnetic order, require only
changing $V$. The same embedding applies to larger clusters, at a computational
cost that grows with the number of sites and the order.
