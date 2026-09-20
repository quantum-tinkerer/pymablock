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

Virtual charge transfer can turn repulsive microscopic interactions into
assisted hopping and effective attraction between dopants. The spinful honeycomb
model of Crépel and Fu[^crepel-fu] provides a concrete setting: an A–B energy
offset and local repulsions favor doubly occupied A sites, while added electrons
occupy the B sublattice.

The retained particles are equal-spin dopants on B sites above a reference with
doubly occupied A sites. A filled neighboring B orbital changes the energy of an
intermediate charge-transfer excitation and therefore changes the hopping
amplitude. The second-order effective Hamiltonian gives both this assisted
hopping and the induced pair-density interaction.

## Cluster Hamiltonian and boundary terms

The cluster consists of one A site and its three B neighbors. Its symbolic
Hamiltonian includes the sublattice offset, on-site repulsion, nearest-neighbor
repulsion, and hopping along the explicit bonds.

$z_j$ counts the explicit A
neighbors of B site $j$. Each omitted A neighbor remains doubly occupied, giving
the boundary potential $2V_0n_j$. This term preserves the honeycomb coordination
in the local energy denominators. Omitting it changes the process being
calculated. The parameter $\Delta$ is the charge-transfer gap after the stated
interaction shift. In the linked preprint, $\Delta_{\rm CF}=\delta_0-U_A$,
so our convention is $\Delta=\Delta_{\rm CF}+3V_0$. Its denominators
$\Delta_{\rm CF}+4V_0$ and $\Delta_{\rm CF}+3V_0$ therefore become
$\Delta+V_0$ and $\Delta$, respectively.

The retained generators are $f_j\mapsto b_{j\uparrow}$, with every A spin orbital
occupied and every B spin orbital empty in the reference. B down-spin modes are
excluded from the target but remain available in virtual states.

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
H, *_ = block_diagonalize([H0, V], subspace_eigenvectors=embedding)
h2 = H[0, 0, 2]
```

## Bare and assisted hopping

The occupation of the spectator $B_2$ changes the hopping amplitude from
$B_0$ to $B_1$. Evaluating that occupation at zero and one separates the bare
hopping from its interaction-assisted contribution.

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

The hopping amplitudes reproduce the equal-spin sector of Crépel and Fu[^crepel-fu]
without taking a large-$U_B$ limit. Evaluating the diagonal expression at the four occupations and taking this
difference removes the reference energy and individual particle shifts,
leaving a pair-density interaction. Its sign need not follow
the sign of the bare repulsions. Both this term and the hopping arise from the
same charge-transfer processes.

The one-star calculation gives the hopping and pair interaction within the
equal-spin sector. Reconstructing the full spinful lattice Hamiltonian also
requires the other spin sectors and the remaining density terms. These local
coefficients describe the virtual processes behind the pairing mechanism;
they do not by themselves determine the superconducting phase diagram.

[^crepel-fu]: V. Crépel and L. Fu,
    [Spin-triplet superconductivity from excitonic effect in doped insulators](https://doi.org/10.1073/pnas.2117735119),
    Proceedings of the National Academy of Sciences **119**, e2117735119 (2022).
    [Open preprint](https://arxiv.org/abs/2103.12060).
