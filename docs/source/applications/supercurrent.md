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

# Supercurrent through an interacting quantum dot

How does the Josephson current through a Coulomb-blockaded quantum dot depend
on the dot's charge? The effective model keeps only the two spin orbitals of the
dot and eliminates the superconducting leads to fourth order in tunneling. Its
energies $E_n(\Phi)$ in each charge sector $n$ give the current
$I_n=(2e/\hbar)\,\partial_\Phi E_n$, where $\Phi$ is the superconducting phase
difference. For an empty dot,

$$
I_0=\frac{2e}{\hbar}\,
\frac{4t_L^2t_R^2\,u_Lv_Lu_Rv_R\,(2E_L+2E_R+b)}{b\,(E_L+E_R)(E_L+a)(E_R+a)}
\sin\Phi .
$$

A singly occupied dot carries a current of the opposite sign: it is a
$\pi$ junction.[^glazman]

## Model

The dot has spin orbitals $d_\uparrow,d_\downarrow$. Relative to the empty dot,
adding one electron costs $a$ and adding two costs $b$. With charging energy $U$
and offset charge $n_g$, these are $a=U(1/2-n_g)$ and $b=2U(1-n_g)$. As in the
[supercurrent tutorial](../tutorial/andreev_supercurrent.md), each lead
$\alpha=L,R$ is a single paired orbital. Its Bogoliubov quasiparticles
$\gamma_{\alpha\sigma}$ have energy $E_\alpha$. The full Hamiltonian is
$H=H_0+V_L+V_R$, with

$$
H_0 = a(n_\uparrow+n_\downarrow) + (b-2a)\,n_\uparrow n_\downarrow
+ \sum_{\alpha\sigma} E_\alpha N_{\gamma_{\alpha\sigma}}.
$$

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
a, b, Phi = sp.symbols("a b Phi", real=True)
E_L, E_R, t_L, t_R = sp.symbols("E_L E_R t_L t_R", positive=True)
u_L, v_L, u_R, v_R = sp.symbols("u_L v_L u_R v_R", positive=True)

d_up, d_down = FermionOp("d_up"), FermionOp("d_down")
gamma_L_up, gamma_L_down = FermionOp("gamma_L_up"), FermionOp("gamma_L_down")
gamma_R_up, gamma_R_down = FermionOp("gamma_R_up"), FermionOp("gamma_R_down")

H_0 = (
    a * (N(d_up) + N(d_down))
    + (b - 2 * a) * N(d_up) * N(d_down)
    + E_L * (N(gamma_L_up) + N(gamma_L_down))
    + E_R * (N(gamma_R_up) + N(gamma_R_down))
)
```

The lead electrons are $c_{\alpha\uparrow}=u_\alpha\gamma_{\alpha\uparrow}
-v_\alpha\gamma_{\alpha\downarrow}^\dagger$ and
$c_{\alpha\downarrow}=u_\alpha\gamma_{\alpha\downarrow}
+v_\alpha\gamma_{\alpha\uparrow}^\dagger$, with real coherence factors
$u_\alpha^2+v_\alpha^2=1$. Tunneling with amplitude $t_\alpha$ is

$$
V_\alpha = t_\alpha e^{i\phi_\alpha}
\sum_\sigma d_\sigma^\dagger c_{\alpha\sigma} + \mathrm{h.c.},
\qquad \phi_L=\Phi/2,\quad \phi_R=0.
$$

The left phase is half the condensate phase difference, because a Cooper pair
carries two electrons.

```{code-cell} ipython3
c_L_up = u_L * gamma_L_up - v_L * Dagger(gamma_L_down)
c_L_down = u_L * gamma_L_down + v_L * Dagger(gamma_L_up)
c_R_up = u_R * gamma_R_up - v_R * Dagger(gamma_R_down)
c_R_down = u_R * gamma_R_down + v_R * Dagger(gamma_R_up)

T_L = t_L * sp.exp(sp.I * Phi / 2) * (Dagger(d_up) * c_L_up + Dagger(d_down) * c_L_down)
T_R = t_R * (Dagger(d_up) * c_R_up + Dagger(d_down) * c_R_down)
V_L = T_L + Dagger(T_L)
V_R = T_R + Dagger(T_R)
```

## What is kept

We keep all four states of the dot and put both leads in their quasiparticle
vacuum. Each dot fermion therefore represents itself, and every lead
quasiparticle is reached only in virtual processes.

```{code-cell} ipython3
embedding = Embedding(
    {d_up: d_up, d_down: d_down},
    reference={d_up: 0, d_down: 0,
               gamma_L_up: 0, gamma_L_down: 0, gamma_R_up: 0, gamma_R_down: 0},
)
```

## Perturbation theory

We count left and right tunneling as separate perturbations, so the effective
Hamiltonian is a series in $t_L$ and $t_R$. The phase-dependent energy first
appears at order $t_L^2t_R^2$. At second order, tunneling also induces pairing
on the dot, which mixes the empty and doubly occupied states.
`fully_diagonalize=(0,)` removes this mixing perturbatively, so that the
effective Hamiltonian depends only on $n_\uparrow$ and $n_\downarrow$.

```{code-cell} ipython3
H_eff, *_ = block_diagonalize(
    {(0, 0): H_0, (1, 0): V_L, (0, 1): V_R},
    subspace_eigenvectors=embedding,
    fully_diagonalize=(0,),
)
H_22 = H_eff[0, 0, 2, 2].as_expr()
```

## Result and check

In each charge sector, we evaluate the energy and differentiate it with
respect to $\Phi$. We write currents in units of $2e/\hbar$.

```{code-cell} ipython3
def current(n_up, n_down):
    energy = H_22.subs({N(d_up): n_up, N(d_down): n_down})
    return sp.factor(sp.simplify(sp.diff(energy, Phi).rewrite(sp.cos)))

I_0, I_1, I_2 = current(0, 0), current(1, 0), current(1, 1)
display(sp.Eq(sp.Symbol("I_0"), I_0))
display(sp.Eq(sp.Symbol("I_1"), I_1))
display(sp.Eq(sp.Symbol("I_2"), I_2))
```

The current is odd in $\Phi$, as time-reversal symmetry requires, and the two
odd states carry the same current. The empty-dot current matches the formula at
the top of the page.

```{code-cell} ipython3
assert all(sp.simplify(I.subs(Phi, -Phi) + I) == 0 for I in (I_0, I_1, I_2))
assert sp.simplify(current(0, 1) - I_1) == 0

expected_I_0 = (4 * t_L**2 * t_R**2 * u_L * v_L * u_R * v_R * (2 * E_L + 2 * E_R + b)
                * sp.sin(Phi) / (b * (E_L + E_R) * (E_L + a) * (E_R + a)))
assert sp.simplify(I_0 - expected_I_0) == 0
```

Two further checks test the physics. A particle–hole transformation of the dot
exchanges the empty and doubly occupied states. It maps $a\to a-b$ and
$b\to-b$, so it must turn $I_0$ into $I_2$ and leave $I_1$ unchanged. Second,
the current must change sign between an empty dot ($n_g=0$) and a singly
occupied dot ($n_g=1$), the $\pi$-junction behavior found by Glazman and
Matveev.[^glazman]

```{code-cell} ipython3
particle_hole = {a: a - b, b: -b}
assert sp.simplify(I_0.subs(particle_hole, simultaneous=True) - I_2) == 0
assert sp.simplify(I_1.subs(particle_hole, simultaneous=True) - I_1) == 0

U, n_g = sp.symbols("U n_g", positive=True)
charge_energies = {a: U * (sp.Rational(1, 2) - n_g), b: 2 * U * (1 - n_g)}
point = {U: 1, E_L: 2, E_R: 3, Phi: sp.pi / 2}
assert I_0.subs(charge_energies).subs(n_g, 0).subs(point) > 0
assert I_1.subs(charge_energies).subs(n_g, 1).subs(point) < 0
```

## Discussion

Even charge states carry a positive current, and the odd state carries a
negative one. In the odd state, transferring a Cooper pair reverses the order of
the two spins on the dot, which contributes a fermionic minus sign. The
expressions hold away from charge degeneracies. Near $b=0$ the empty and doubly
occupied states are degenerate, so their mixing must be treated exactly rather
than perturbatively.

[^glazman]: L. I. Glazman and K. A. Matveev,
    [Resonant Josephson current through Kondo impurities in a tunnel barrier](http://jetpletters.ru/ps/1121/article_16988.pdf),
    JETP Letters **49**, 659–662 (1989).
