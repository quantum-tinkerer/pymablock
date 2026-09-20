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

We calculate the fourth-order Josephson current through a Coulomb-blockaded
dot, including its parity-dependent sign.[^glazman] Each superconducting lead
is represented by one paired orbital, as in the
[supercurrent tutorial](../tutorial/andreev_supercurrent.md).

## Model

The dot has charging energy $U$ and offset charge $n_g$. The lead
quasiparticles have energies $E_\alpha$ and real coherence factors satisfying
$u_\alpha^2+v_\alpha^2=1$, with pairing amplitudes
$\Gamma_\alpha=2E_\alpha u_\alpha v_\alpha$.
The tunneling phase is half the condensate phase difference $\Phi$,
so $I=(2e/\hbar)\partial_\Phi E$.

The energies $a$ and $b$ are measured from the empty dot. All parameters remain
symbolic.

```{code-cell} ipython3
import sympy as sp
from IPython.display import display
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding

Phi = sp.Symbol("Phi", real=True)
d_up, d_down = FermionOp("d_up"), FermionOp("d_down")
gamma = {
    lead: (FermionOp(f"gamma_{lead}_up"), FermionOp(f"gamma_{lead}_down"))
    for lead in ("L", "R")
}
U = sp.Symbol("U", positive=True)
ng = sp.Symbol("n_g", real=True)
a, b = sp.symbols("a b", real=True)
EL, ER, tL, tR = sp.symbols("E_L E_R t_L t_R", positive=True)
GammaL, GammaR = sp.symbols("Gamma_L Gamma_R", positive=True)
uL, uR, vL, vR = sp.symbols("u_L u_R v_L v_R", real=True)
charge = sp.Symbol("n", integer=True)
charging = U * (charge - ng)**2 / 2
charging_energies = {energy: sp.expand(charging.subs(charge, n) - charging.subs(charge, 0))
                     for energy, n in ((a, 1), (b, 2))}
for energy, expression in charging_energies.items():
    display(sp.Eq(energy, expression))
lead_parameters = {
    "L": (EL, uL, vL, tL, Phi / 2),
    "R": (ER, uR, vR, tR, sp.S.Zero),
}
H0 = a * (N(d_up) + N(d_down)) + (b - 2 * a) * N(d_up) * N(d_down)
couplings = {}
for lead, (energy, u, v, hopping, phase) in lead_parameters.items():
    up, down = gamma[lead]
    H0 += energy * (N(up) + N(down))
    forward = hopping * sp.exp(sp.I * phase) * (
        Dagger(d_up) * (u * up - v * Dagger(down))
        + Dagger(d_down) * (u * down + v * Dagger(up))
    )
    couplings[lead] = forward + Dagger(forward)
V = sum(couplings.values())
display(sp.Eq(sp.Symbol("H_0", commutative=False), H0))
for lead, coupling in couplings.items():
    display(sp.Eq(sp.Symbol(f"V_{lead}", commutative=False), coupling))
```

## Effective dot Hamiltonian

We retain both dot fermions and eliminate the lead quasiparticles. Counting
left and right tunneling separately selects the leading current contribution
at order $(2,2)$.

```{code-cell} ipython3
dot = tuple(sorted((d_up, d_down), key=lambda op: str(op.name)))
quasiparticles = tuple(op for pair in gamma.values() for op in pair)
source = tuple(sorted(dot + quasiparticles, key=lambda op: str(op.name)))
embedding = Embedding(
    {op: op for op in dot},
    reference=dict.fromkeys(source, 0),
)
H, *_ = block_diagonalize(
    {(0, 0): H0, (1, 0): couplings["L"], (0, 1): couplings["R"]},
    subspace_eigenvectors=embedding,
)
h0 = H[0, 0, 0, 0].as_expr()
hL, hR = H[0, 0, 2, 0], H[0, 0, 0, 2]
hLR = H[0, 0, 2, 2]
assert all(part.filter_terms(((1, -1), (-1, 1)), keep=True).is_zero
           for part in (hL, hR))
# Pair annihilation leaves the empty dot; its coefficient is Delta_ind.
pair_L, pair_R = [
    part.filter_terms(((1, 1),), keep=True).as_expr().coeff(d_up * d_down)
    for part in (hL, hR)
]
pairing = sp.factor(pair_L) + sp.factor(pair_R)
display(sp.Eq(sp.Symbol("Delta_ind"), pairing))
```

## Charge-resolved current

The induced pairing mixes the empty and doubly occupied dot states. Their
energies therefore include the pairing contribution as well as the diagonal
fourth-order term; the odd doublet does not mix. Differentiating the resulting
phase-dependent energies gives the three charge-sector currents below.

```{code-cell} ipython3
coherence = uL * vL * uR * vR
currents = []
diagonal_LR = hLR.filter_terms(((0, 0),), keep=True).as_expr()
pair_mixing = pair_L * sp.conjugate(pair_R) + pair_R * sp.conjugate(pair_L)
for occupations in ((0, 0), (0, 1), (1, 0), (1, 1)):
    charge = sum(occupations)
    values = dict(zip(map(N, dot), occupations))
    correction = diagonal_LR.subs(values)
    if charge != 1:
        # Pairing mixes the empty and doubly occupied dot states.
        other = dict(zip(map(N, dot), (1 - n for n in occupations)))
        gap = h0.subs(values) - h0.subs(other)
        correction += pair_mixing / gap
    # Expose the phase factor in each denominator before extracting harmonics.
    expanded = sp.Add(*(sp.factor_terms(term) for term in sp.Add.make_args(sp.expand(correction))))
    plus = expanded.coeff(sp.exp(sp.I * Phi))
    minus = expanded.coeff(sp.exp(sp.I * Phi), -1)
    constant = sp.expand(expanded - plus * sp.exp(sp.I * Phi) - minus * sp.exp(-sp.I * Phi))
    assert Phi not in constant.free_symbols
    assert sp.factor(minus - plus) == 0
    amplitude = sp.factor(plus / coherence)
    current = sp.diff(2 * amplitude * sp.cos(Phi), Phi)
    currents.append(current * GammaL * GammaR / (4 * EL * ER))
assert currents[1] == currents[2]

currents = [currents[i] for i in (0, 1, 3)]
# Restore charging energy and gate voltage symbolically, for arbitrary U and n_g.
currents_by_gate = [current.subs(charging_energies) for current in currents]
for charge, current in enumerate(currents_by_gate):
    display(sp.Eq(sp.Symbol(f"I_{charge}") / sp.Symbol("2e/hbar"), current))
```

The odd-charge ground state carries a $\pi$-junction current, opposite to the
even sectors. These branch expressions apply away from charge degeneracies;
near $b=0$, the retained even block must be diagonalized together.

[^glazman]: L. I. Glazman and K. A. Matveev,
    [Resonant Josephson current through Kondo impurities in a tunnel barrier](http://jetpletters.ru/ps/1121/article_16988.pdf),
    JETP Letters **49**, 659–662 (1989).
