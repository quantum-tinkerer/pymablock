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
  execution_timeout: 480
---

# Interaction-assisted hopping in the Crépel–Fu model

We calculate assisted hopping and the induced pair interaction in the spinful
honeycomb model of Crépel and Fu.[^crepel-fu]

## Model

The cluster contains one doubly occupied A site and its three B neighbors.
Equal-spin dopants occupy the B sites. The Hamiltonian includes a sublattice
offset, on-site repulsions $U_A,U_B$, nearest-neighbor repulsion $V_0$, and
hopping $t_0$.

Omitted A neighbors remain doubly occupied. Each contributes $2V_0n_j$ to its
B neighbor, preserving the lattice energy denominators. Our gap convention is
$\Delta=\Delta_{\rm CF}+3V_0$, where the preprint defines
$\Delta_{\rm CF}=\delta_0-U_A$.

```{code-cell} ipython3
import sympy as sp
from IPython.display import display
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding

Delta, V0, UA, UB, t0 = sp.symbols("Delta V0 U_A U_B t_0", positive=True)

def cluster(edges, num_a, num_b):
    A = tuple((FermionOp(f"a{i}_up"), FermionOp(f"a{i}_down")) for i in range(num_a))
    B = tuple((FermionOp(f"b{i}_up"), FermionOp(f"b{i}_down")) for i in range(num_b))
    f = tuple(FermionOp(f"f{i}") for i in range(num_b))
    nA = [sum(N(op) for op in site) for site in A]
    nB = [sum(N(op) for op in site) for site in B]
    degrees = [sum(j == site for _, j in edges) for site in range(num_b)]
    delta0 = Delta + UA - 3 * V0
    H0 = (delta0 * (sum(nB) - sum(nA)) / 2
          + UA * sum(N(up) * N(down) for up, down in A)
          + UB * sum(N(up) * N(down) for up, down in B)
          + V0 * sum(nA[i] * nB[j] for i, j in edges)
          + 2 * V0 * sum((3 - z) * n for z, n in zip(degrees, nB)))
    V = -t0 * sum(
        Dagger(B[j][s]) * A[i][s] + Dagger(A[i][s]) * B[j][s]
        for i, j in edges for s in range(2)
    )
    reference = {**{op: 1 for site in A for op in site},
                 **{op: 0 for site in B for op in site}}
    embedding = Embedding({f[i]: B[i][0] for i in range(num_b)}, reference=reference)
    return H0, V, embedding, A, B, f

star = cluster(((0, 0), (0, 1), (0, 2)), 1, 3)
H0, V, embedding, A, B, f = star
display(sp.Eq(sp.Symbol("H_0", commutative=False), H0))
display(sp.Eq(sp.Symbol("V", commutative=False), V))
```

## Effective dopant Hamiltonian

The embedding retains the B up-spin fermions. The other spin orbitals enter
only through virtual states.

```{code-cell} ipython3
H, *_ = block_diagonalize([H0, V], subspace_eigenvectors=embedding)
h2 = H[0, 0, 2]
```

## Hopping and pair interaction

The occupation of $B_2$ distinguishes bare hopping from $B_0$ to $B_1$ from
its assisted contribution. The diagonal term gives the pair interaction after
subtracting the individual particle shifts.

```{code-cell} ipython3
# Powers (1, -1, 0) select hopping from B0 to B1, with B2 a spectator.
hopping = h2.filter_terms(((1, -1, 0),), keep=True).as_expr()
display(hopping)
bare = sp.factor(hopping.subs(N(f[2]), 0).coeff(Dagger(f[1]) * f[0]))
assisted = sp.factor(hopping.subs(N(f[2]), 1).coeff(Dagger(f[1]) * f[0]))
lam = sp.factor(assisted - bare)
display(sp.Eq(sp.Symbol("t"), bare), sp.Eq(sp.Symbol("lambda"), lam))

# Connected pair-density term with the third B site empty.
diagonal = h2.filter_terms(((0, 0, 0),), keep=True).as_expr()
pair_energy = diagonal.subs(N(f[2]), 0)
interaction = sp.factor(
    pair_energy.subs({N(f[0]): 1, N(f[1]): 1})
    - pair_energy.subs({N(f[0]): 1, N(f[1]): 0})
    - pair_energy.subs({N(f[0]): 0, N(f[1]): 1})
    + pair_energy.subs({N(f[0]): 0, N(f[1]): 0})
)
display(sp.Eq(sp.Symbol("W^{(2)}"), interaction))
```

These amplitudes reproduce the equal-spin sector of Crépel and Fu[^crepel-fu]
without a large-$U_B$ approximation. A spectator changes the virtual
charge-transfer energy and assists hopping; the induced pair interaction can
have a different sign from the microscopic repulsions.

[^crepel-fu]: V. Crépel and L. Fu,
    [Spin-triplet superconductivity from excitonic effect in doped insulators](https://doi.org/10.1073/pnas.2117735119),
    Proceedings of the National Academy of Sciences **119**, e2117735119 (2022).
    [Open preprint](https://arxiv.org/abs/2103.12060).
