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

We reproduce local virtual processes in the spinful honeycomb interaction model
of [Crépel and Fu, *Spin-triplet superconductivity from excitonic effect in doped
insulators*, PNAS 119, e2117735119 (2022)](https://doi.org/10.1073/pnas.2117735119).
This is the lattice interaction example in the structured-embedding validation
battery. Its microscopic input consists of an A–B energy offset, on-site
repulsion, nearest-neighbor repulsion, and spin-conserving hopping.

The retained particles are equal-spin dopants on B sites above a reference with
doubly occupied A sites. A filled neighboring B orbital changes the energy of an
intermediate charge-transfer excitation and therefore changes the hopping
amplitude. We calculate that dependence at second order, then extend the cluster
to a hopping process that first appears at fourth order.

## Cluster Hamiltonian and boundary terms

For an explicit set of A–B bonds $\mathcal E$, we use

$$
H_0=\frac{\delta_0}{2}\left(\sum_{j\in B}n_j-\sum_{i\in A}n_i\right)
 +U_A\sum_{i\in A}n_{i\uparrow}n_{i\downarrow}
 +U_B\sum_{j\in B}n_{j\uparrow}n_{j\downarrow}
 +V_0\sum_{(i,j)\in\mathcal E}n_i n_j
 +2V_0\sum_{j\in B}(3-z_j)n_j,
$$

$$
V=-t_0\sum_{(i,j)\in\mathcal E,\sigma}
 (b_{j\sigma}^\dagger a_{i\sigma}+a_{i\sigma}^\dagger b_{j\sigma}),
\qquad \delta_0=\Delta+U_A-3V_0.
$$

Here $n_i=n_{i\uparrow}+n_{i\downarrow}$ and $z_j$ counts the explicit A
neighbors of B site $j$. Each omitted A neighbor remains doubly occupied, giving
the boundary potential $2V_0n_j$. This term preserves the honeycomb coordination
in the local energy denominators. Omitting it changes the process being
calculated. The parameter $\Delta$ is the charge-transfer gap after the stated
interaction shift.

The retained generators are $f_j\mapsto b_{j\uparrow}$, with every A spin orbital
occupied and every B spin orbital empty in the reference. B down-spin modes are
excluded from the target but remain available in virtual states.

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display
from itertools import product
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding
from validation import occupation_matrices, operator_matrix, occupation_indices, second_order

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
H, *_ = block_diagonalize([H0, V], subspace_eigenvectors=embedding)
h2 = H[0, 0, 2].to_matrix()
```

## Bare and assisted hopping

In the target basis $(n_0,n_1,n_2)$, the matrix elements from $B_0$ to $B_1$ are

$$
t=\langle010|H^{(2)}|100\rangle=\frac{t_0^2}{\Delta+V_0},
\qquad
t+\lambda=\langle011|H^{(2)}|101\rangle=\frac{t_0^2}{\Delta}.
$$

Thus the spectator on $B_2$ assists hopping by
$\lambda=t_0^2[1/\Delta-1/(\Delta+V_0)]$. The code checks both expressions
symbolically, without a large-$U_B$ approximation.

```{code-cell} ipython3
bare = sp.factor(h2[2, 4])
assisted = sp.factor(h2[3, 5])
assert sp.factor(bare - t0**2 / (Delta + V0)) == 0
assert sp.factor(assisted - t0**2 / Delta) == 0
lam = sp.factor(assisted - bare)
display(sp.Eq(sp.Symbol("t"), bare), sp.Eq(sp.Symbol("lambda"), lam))

# Connected pair-density term with the third B site empty.
interaction = sp.factor(h2[6, 6] - h2[4, 4] - h2[2, 2] + h2[0, 0])
reference_interaction = 2 * t0**2 * (
    4 / (Delta + V0) - 3 / (Delta + 2 * V0) - 1 / Delta
    + 1 / (Delta + UB + V0) - 1 / (Delta + UB)
)
assert sp.factor(interaction - reference_interaction) == 0
display(sp.Eq(sp.Symbol("W^{(2)}"), reference_interaction))
```

The diagonal combination removes the reference energy and the two individual
particle shifts, leaving a pair-density interaction. Its sign need not follow
the sign of the bare repulsions. Both this term and the hopping arise from the
same charge-transfer processes.

We compare the complete eight-state second-order block with direct resolvent
sums in the $2^8=256$ dimensional spinful source space.

```{code-cell} ipython3
parameters = {Delta: 10, V0: 2, UA: 7, UB: 11, t0: 1}
source = tuple(sorted((op for site in A + B for op in site), key=lambda op: str(op.name)))
occupations = [range(2)] * len(source)
matrices = occupation_matrices(source, occupations)
energy = operator_matrix(H0.subs(parameters), matrices).diagonal().real
perturbation = operator_matrix(V.subs(parameters), matrices).toarray()
selected = []
for state in product(range(2), repeat=3):
    values = {**{op: 1 for site in A for op in site},
              **{op: 0 for site in B for op in site}}
    values.update({site[0]: n for site, n in zip(B, state)})
    selected.append(tuple(values[op] for op in source))
kept = occupation_indices(occupations, selected)
reference = second_order(energy, perturbation, kept)
actual = np.asarray(h2.subs(parameters), dtype=complex)
error = np.max(np.abs(actual - reference))
assert error < 1e-12
print(f"Maximum second-order matrix error: {error:.2e}")
print("Pair-density coefficient at the reference point:", interaction.subs(parameters))
```

## A connected fourth-order path

Two overlapping A stars contain the four-hop path
$B_0-A_0-B_1-A_1-B_3$. The additional sites $B_2$ and $B_4$ retain the spectators
of both triangles. There are fourteen source fermion modes and five retained
fermions, giving a 32-state target with all dopant occupations retained.

```{code-cell} ipython3
edges = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 3), (1, 4))
positions_a = [(-1, 0), (1, 0)]
positions_b = [(-1.5, np.sqrt(3)/2), (0, 0), (-1.5, -np.sqrt(3)/2),
               (1.5, np.sqrt(3)/2), (1.5, -np.sqrt(3)/2)]
fig, ax = plt.subplots(figsize=(5.4, 3), constrained_layout=True)
for i, j in edges:
    x, y = zip(positions_a[i], positions_b[j])
    ax.plot(x, y, color="0.6", lw=2, zorder=0)
for label, positions, color in (("A", positions_a, "tab:red"), ("B", positions_b, "tab:blue")):
    for i, (x, y) in enumerate(positions):
        ax.scatter(x, y, s=140, color=color)
        ax.annotate(f"{label}{i}", (x, y), xytext=(6, 7), textcoords="offset points")
ax.set_aspect("equal")
ax.set_axis_off()
plt.show()
```

For the one-particle matrix element, all spectator occupations vanish. The
reference expression for the connected contribution is

$$
\langle B_3|H^{(4)}|B_0\rangle
=-\frac{t_0^4(2\Delta^2+4\Delta V_0+V_0^2)}
 {2(\Delta+V_0)^3(\Delta+2V_0)^2}.
$$

This is an extension of the local second-order calculation. The following cell
checks this expression at the exact rational parameter point above; it does
not establish the formula for arbitrary parameters. No one-A subcluster joins
these endpoints, so this matrix element needs no proper-subcluster subtraction.

```{code-cell} ipython3
H0_two, V_two, embedding_two, *_ = cluster(edges, 2, 5)
H_two, *_ = block_diagonalize(
    [H0_two.subs(parameters), V_two.subs(parameters)],
    subspace_eigenvectors=embedding_two,
)
# Lexicographic occupations: B0 alone is index 16, B3 alone is index 2.
assert H_two[0, 0, 2].to_matrix()[2, 16] == 0
connected = sp.factor(H_two[0, 0, 4].to_matrix()[2, 16])
reference_connected = -t0**4 * (2 * Delta**2 + 4 * Delta * V0 + V0**2) / (
    2 * (Delta + V0)**3 * (Delta + 2 * V0)**2
)
assert connected == reference_connected.subs(parameters)
assert connected == -sp.Rational(71, 169344)
display(sp.Eq(sp.Symbol("h^{(4)}_{B_3,B_0}"), connected))
```

This cluster result does not give every fourth-order lattice operator. That
requires the other connected four-bond clusters and the subtraction of their
proper subclusters. The calculation also fixes a single dopant spin sector;
it does not determine a superconducting phase diagram or a material-specific
moiré band structure.
