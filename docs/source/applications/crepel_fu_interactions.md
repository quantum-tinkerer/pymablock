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

Crépel and Fu proposed that electrons doped into a charge-transfer insulator
attract each other through virtual excitations of the insulator.[^crepel-fu]
How do the hopping and interaction of the doped electrons arise? The effective
model has spinless fermions $f_j$ on the B sites of a honeycomb lattice:

$$
H_{\rm eff} = \sum_{\langle ij\rangle}\left(t+\lambda n_k\right) f_i^\dagger f_j
+ \mathrm{h.c.} + \sum_{\langle ij\rangle} W n_i n_j + \dots,
$$

where $k$ is the third B site next to the A site that connects $i$ and $j$.
To second order in the A–B hopping $t_0$,

$$
t=\frac{t_0^2}{\Delta+V_0},\qquad
\lambda=\frac{V_0\,t_0^2}{\Delta\,(\Delta+V_0)}.
$$

A dopant on site $k$ makes the virtual excitation cheaper and so assists the
hopping between $i$ and $j$.

## Model

We use the smallest cluster that contains assisted hopping: one A site and its
three B neighbors $j=0,1,2$. Each site has spin-up and spin-down orbitals,
$a_\sigma$ and $b_{j\sigma}$, with occupations $n_A=n_{A\uparrow}+n_{A\downarrow}$
and $n_{B_j}=n_{B_j\uparrow}+n_{B_j\downarrow}$. The unperturbed Hamiltonian is

$$
H_0 = \frac{\delta_0}{2}\Big(\sum_j n_{B_j} - n_A\Big)
+ U_A n_{A\uparrow}n_{A\downarrow} + U_B\sum_j n_{B_j\uparrow}n_{B_j\downarrow}
+ V_0\, n_A\sum_j n_{B_j} + 4V_0\sum_j n_{B_j}.
$$

Here $\delta_0$ is the sublattice energy offset, $U_A$ and $U_B$ are on-site
repulsions, and $V_0$ is the nearest-neighbor repulsion. In the lattice, each B
site has two more A neighbors. They stay doubly occupied and each adds
$2V_0 n_{B_j}$, which gives the last term. We write
$\delta_0=\Delta+U_A-3V_0$; thus $\Delta=\Delta_{\rm CF}+3V_0$ in terms of the
gap $\Delta_{\rm CF}=\delta_0-U_A$ of Crépel and Fu.

```{code-cell} ipython3
import sympy as sp
from IPython.display import display
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding
```

```{code-cell} ipython3
Delta, V_0, U_A, U_B, t_0 = sp.symbols("Delta V_0 U_A U_B t_0", positive=True)
delta_0 = Delta + U_A - 3 * V_0

a_up, a_down = FermionOp("a_up"), FermionOp("a_down")
b_up = [FermionOp(f"b{j}_up") for j in range(3)]
b_down = [FermionOp(f"b{j}_down") for j in range(3)]

n_A = N(a_up) + N(a_down)
n_B = sum(N(b_up[j]) + N(b_down[j]) for j in range(3))
H_0 = (
    delta_0 / 2 * (n_B - n_A)
    + U_A * N(a_up) * N(a_down)
    + U_B * sum(N(b_up[j]) * N(b_down[j]) for j in range(3))
    + V_0 * n_A * n_B
    + 4 * V_0 * n_B
)
```

The perturbation is hopping between the A site and its neighbors,

$$
V = -t_0\sum_{j\sigma}\left(b_{j\sigma}^\dagger a_\sigma
+ a_\sigma^\dagger b_{j\sigma}\right).
$$

```{code-cell} ipython3
V = -t_0 * sum(
    Dagger(b_up[j]) * a_up + Dagger(b_down[j]) * a_down
    + Dagger(a_up) * b_up[j] + Dagger(a_down) * b_down[j]
    for j in range(3)
)
```

## What is kept

The insulator has a doubly occupied A site and empty B sites. We keep dopants
of one spin only, the equal-spin sector of Crépel and Fu: the effective fermion
$f_j$ is a spin-up electron on site $B_j$. All other configurations, including those with
an A electron moved to a B site, are reached only in virtual processes.

```{code-cell} ipython3
f = [FermionOp(f"f{j}") for j in range(3)]
embedding = Embedding(
    {f[0]: b_up[0], f[1]: b_up[1], f[2]: b_up[2]},
    reference={a_up: 1, a_down: 1,
               b_up[0]: 0, b_up[1]: 0, b_up[2]: 0,
               b_down[0]: 0, b_down[1]: 0, b_down[2]: 0},
)
```

## Perturbation theory

The hopping $V$ has no matrix elements inside the kept subspace, so the
dopant Hamiltonian first appears at second order in $t_0$.

```{code-cell} ipython3
H_eff, *_ = block_diagonalize([H_0, V], subspace_eigenvectors=embedding)
H_2 = H_eff[0, 0, 2]
```

## Result and check

We select the term that moves a dopant from $B_0$ to $B_1$. The `filter_terms`
powers $(1,-1,0)$ refer to $(f_0,f_1,f_2)$: one $f_0$ annihilated, one $f_1$
created, and $f_2$ untouched.

```{code-cell} ipython3
hopping = H_2.filter_terms(((1, -1, 0),), keep=True).as_expr()
display(hopping)
```

The amplitude without a spectator on $B_2$ is $t$; the spectator adds
$\lambda$. Both agree with the equal-spin amplitudes of Crépel and
Fu.[^crepel-fu]

```{code-cell} ipython3
t = sp.factor(hopping.subs(N(f[2]), 0).coeff(Dagger(f[1]) * f[0]))
lam = sp.factor(hopping.subs(N(f[2]), 1).coeff(Dagger(f[1]) * f[0]) - t)
display(sp.Eq(sp.Symbol("t"), t), sp.Eq(sp.Symbol("lambda"), lam))

assert sp.simplify(t - t_0**2 / (Delta + V_0)) == 0
assert sp.simplify(lam - V_0 * t_0**2 / (Delta * (Delta + V_0))) == 0
```

The pair interaction is the energy of two dopants minus the energies of each
dopant alone, $W=E_{11}-E_{10}-E_{01}+E_{00}$. Here $E_{n_0n_1}$ is the
second-order energy with $B_2$ empty.

```{code-cell} ipython3
diagonal = H_2.filter_terms(((0, 0, 0),), keep=True).as_expr().subs(N(f[2]), 0)

def E(n_0, n_1):
    return diagonal.subs({N(f[0]): n_0, N(f[1]): n_1})

W = sp.apart(sp.factor(E(1, 1) - E(1, 0) - E(0, 1) + E(0, 0)), Delta)
display(sp.Eq(sp.Symbol("W"), W))
```

As a check, we count virtual processes directly. Each energy is a sum of
$-t_0^2/\epsilon$ over the processes that move one A electron to a B site, with
$\epsilon$ the cost of that move. Without dopants there are six processes,
each costing $\Delta+2V_0$. With one dopant, a spin-down electron hopping onto
it costs $\Delta+V_0+U_B$, four other hops cost $\Delta+V_0$, and the hop onto the
occupied spin-up orbital is blocked. With two dopants, two hops cost
$\Delta+U_B$ and two cost $\Delta$.

```{code-cell} ipython3
E_00 = -6 * t_0**2 / (Delta + 2 * V_0)
E_10 = -t_0**2 / (Delta + V_0 + U_B) - 4 * t_0**2 / (Delta + V_0)
E_11 = -2 * t_0**2 / (Delta + U_B) - 2 * t_0**2 / Delta
assert sp.simplify(W - (E_11 - 2 * E_10 + E_00)) == 0
```

## Discussion

These amplitudes reproduce the equal-spin sector of Crépel and Fu without
assuming large $U_B$. A spectator dopant lowers the cost of the virtual charge
transfer, which makes $\lambda$ positive. The pair interaction $W$ combines
several virtual channels and can be attractive even though every microscopic
interaction is repulsive.

[^crepel-fu]: V. Crépel and L. Fu,
    [Spin-triplet superconductivity from excitonic effect in doped insulators](https://doi.org/10.1073/pnas.2117735119),
    Proceedings of the National Academy of Sciences **119**, e2117735119 (2022).
    [Open preprint](https://arxiv.org/abs/2103.12060).
