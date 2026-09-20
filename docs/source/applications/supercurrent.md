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

A Coulomb-blockaded quantum dot carries a Josephson current whose sign depends
on its charge parity. This interaction effect was studied by Glazman and
Matveev.[^glazman] Here each superconducting lead is represented by one paired
orbital, as in the [supercurrent tutorial](../tutorial/andreev_supercurrent.md).
The finite model captures the parity-dependent current without integrating over
a continuum of lead states.

Eliminating the lead quasiparticles leaves an effective Hamiltonian in the two
dot fermions. Its fourth-order charge-sector energies give fully symbolic
current–phase relations for all four dot states, with gate voltage, charging
energy, lead energies, and tunneling amplitudes left as parameters.

## Hamiltonian and phase convention

We write the superconductors in their Bogoliubov basis:

$$
H_0=\frac{U}{2}(N_{d\uparrow}+N_{d\downarrow}-n_g)^2
  +\sum_{\alpha=L,R;\,\sigma} E_\alpha N_{\gamma_{\alpha\sigma}},
$$

$$
V=\sum_\alpha t_\alpha e^{i\theta_\alpha}
 \left[d_\uparrow^\dagger(u_\alpha\gamma_{\alpha\uparrow}
                  -v_\alpha\gamma_{\alpha\downarrow}^\dagger)
      +d_\downarrow^\dagger(u_\alpha\gamma_{\alpha\downarrow}
                  +v_\alpha\gamma_{\alpha\uparrow}^\dagger)\right]
 +\mathrm{h.c.}
$$

Here $U$ is the charging energy, $n_g$ the offset charge, and $E_\alpha>0$ the
quasiparticle excitation energy. Real coherence factors satisfy
$u_\alpha^2+v_\alpha^2=1$. They correspond to normal energies
$\xi_\alpha=E_\alpha(u_\alpha^2-v_\alpha^2)$ and real pairing amplitudes
$\Gamma_\alpha=2E_\alpha u_\alpha v_\alpha$. We omit the phase-independent
constant $\sum_\alpha(\xi_\alpha-E_\alpha)$.

We use the **condensate phase difference** $\Phi$ and put
$\theta_L=\Phi/2$, $\theta_R=0$. Consequently,
$I=(2e/\hbar)\partial_\Phi E$. The phase in a single-electron tunneling amplitude
is half the condensate phase; keeping this distinction fixes the current's
prefactor.

Subtracting the constant empty-dot energy $E_0$ leaves two independent dot
energies and simplifies the symbolic denominators:

$$
a=E_1-E_0=\frac{U}{2}(1-2n_g),\qquad
b=E_2-E_0=2U(1-n_g),
$$

and the dot Hamiltonian becomes
$a(N_{d\uparrow}+N_{d\downarrow})+(b-2a)N_{d\uparrow}N_{d\downarrow}$.
This choice of energy origin removes a redundant symbol without fixing the gate
voltage or charging energy. We leave the Bogoliubov coefficients symbolic until
the final current, where $u_\alpha v_\alpha=\Gamma_\alpha/(2E_\alpha)$.

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding
from validation import (
    occupation_matrices, operator_matrix, occupation_indices,
    fourth_order_degenerate,
)

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
charging_energies = {a: U * (1 - 2 * ng) / 2, b: 2 * U * (1 - ng)}
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
```

## Retaining the dot algebra

The generator map is the identity on the dot fermions. Its reference is the dot
vacuum with no lead quasiparticles. Applying the dot creation operators generates
empty, singly occupied, and doubly occupied dot states; lead excitations remain
available as virtual states.

We count left and right tunneling separately, using formal parameters
$\eta_L$ and $\eta_R$. A phase-dependent closed process must transfer a pair
between the leads, so its leading order is $\eta_L^2\eta_R^2$. The coefficients
retain the symbolic hopping amplitudes $t_L$ and $t_R$.

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

The displayed $\Delta_{\rm ind}=\langle00|h_L+h_R|11\rangle$ is the
second-order pairing matrix element. It mixes the even-charge states, so the
diagonal of $h_{LR}$ alone does not give the fourth-order energy. The mixed
coefficient is

$$
E_{n,LR}^{(4)}=(h_{LR})_{nn}
 +\sum_{m:E_m^{(0)}\ne E_n^{(0)}}
 \frac{(h_L)_{nm}(h_R)_{mn}+(h_R)_{nm}(h_L)_{mn}}
 {E_n^{(0)}-E_m^{(0)}}.
$$

For this spin-conserving model the odd-charge doublet does not mix internally.
The extra sum contributes in the even sectors and includes pair transfer
through the other retained charge state.

## Charge-resolved current

We expand the energy in its phase harmonics. At this order,
$E_{n,LR}^{(4)}=C_n+A_ne^{i\Phi}+A_ne^{-i\Phi}$, with real $A_n$ and
phase-independent $C_n$. Thus

$$
\frac{I_n^{(4)}}{2e/\hbar}=-2A_n\sin\Phi
 =\frac{t_L^2t_R^2\Gamma_L\Gamma_R}{E_LE_R}\,F_n\sin\Phi.
$$

Writing $S=E_L+E_R$, the empty and doubly occupied branches have

$$
F_0=\frac{1/S+2/b}{(E_L+a)(E_R+a)},\qquad
F_2=\frac{1/S-2/b}{(E_L+a-b)(E_R+a-b)}.
$$

The singly occupied branch has

$$
\begin{aligned}
F_1=-\frac{1}{2S}\bigg[&
 \frac{1}{(E_L-a)(E_R-a)}
 +\frac{1}{(E_L+b-a)(E_R+b-a)}\\
 &+\frac{2}{(E_L-a)(E_L+b-a)}
 +\frac{2}{(E_R-a)(E_R+b-a)}\bigg].
\end{aligned}
$$

These denominators distinguish virtual empty and doubly occupied dot states.
In the respective ground-state charge regions, $F_0$ and $F_2$ are positive
and $F_1$ is negative: the odd sector is a $\pi$ junction.

Extracting the two phase harmonics from the effective Hamiltonian gives these
three amplitudes directly. The remaining fourth-order energy is phase independent
and therefore contributes no current.

```{code-cell} ipython3
coherence = uL * vL * uR * vR
S = EL + ER
factors = [
    (1 / S + 2 / b) / ((EL + a) * (ER + a)),
    -(1 / ((EL - a) * (ER - a))
      + 1 / ((EL + b - a) * (ER + b - a))
      + 2 / ((EL - a) * (EL + b - a))
      + 2 / ((ER - a) * (ER + b - a))) / (2 * S),
    (1 / S - 2 / b) / ((EL + a - b) * (ER + a - b)),
]
phase_amplitudes = []
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
    amplitude = sp.factor(plus / (tL**2 * tR**2 * coherence))
    assert sp.factor(minus - plus) == 0
    assert sp.factor(-amplitude / 2 - factors[charge]) == 0
    phase_amplitudes.append(amplitude)
assert phase_amplitudes[1] == phase_amplitudes[2]

prefactor = tL**2 * tR**2 * GammaL * GammaR / (EL * ER)
currents = [prefactor * factor * sp.sin(Phi) for factor in factors]
# Restore charging energy and gate voltage symbolically, for arbitrary U and n_g.
currents_by_gate = [current.subs(charging_energies) for current in currents]
print("All three charge-sector currents agree symbolically.")
print("Parameters in the gate-dependent currents:",
      sorted(str(symbol) for symbol in set().union(*(x.free_symbols for x in currents_by_gate))))
```

## Comparison with a sector resolvent

We build the full $2^6=64$ dimensional Fock matrices directly. For each charge
sector, the resolvent excludes that sector alone, so its fourth-order expression
already includes the even-sector pairing correction. For a degenerate sector
with $PVP=0$, the reference uses

$$
E^{(4)}=PVRVRVRVP
 -\frac12\{PVR^2VP,\;PVRVP\},
\qquad R=Q(E_0-H_0)^{-1}Q.
$$

A direct Fock-space evaluation provides an independent comparison at three
gate offsets, one in each ground-state charge region. Subtracting the energy at
zero phase isolates the phase-dependent contribution; its finite-difference
derivative gives the current. Both quantities can then be compared with the
symbolic expressions above.

```{code-cell} ipython3
occupations = [range(2)] * len(source)
matrices = occupation_matrices(source, occupations)
selected_states = [
    tuple(dict(zip(dot, state)).get(op, 0) for op in source)
    for state in ((0, 0), (0, 1), (1, 0), (1, 1))
]
kept = occupation_indices(occupations, selected_states)
lead_values = {
    EL: sp.Rational(11, 10), ER: sp.Rational(7, 5),
    uL: sp.Rational(3, 5), vL: sp.Rational(4, 5),
    uR: sp.Rational(5, 13), vR: sp.Rational(12, 13),
    tL: sp.Rational(1, 5), tR: sp.Rational(3, 20),
}
for u, v in ((uL, vL), (uR, vR)):
    assert (u**2 + v**2).subs(lead_values) == 1
lead_values.update({
    GammaL: (2 * EL * uL * vL).subs(lead_values),
    GammaR: (2 * ER * uR * vR).subs(lead_values),
})
phi_sample, step = 0.74, 1e-5
for gate, charge in ((sp.Rational(1, 5), 0), (sp.Rational(9, 10), 1),
                     (sp.Rational(9, 5), 2)):
    parameters = {**lead_values, U: 10, ng: gate}
    parameters.update({symbol: value.subs(parameters) for symbol, value in charging_energies.items()})
    energy = operator_matrix(H0.subs(parameters), matrices).diagonal().real
    retained = [kept[i] for i in ((0,), (1, 2), (3,))[charge]]

    def reference_energy(phi):
        perturbation = operator_matrix(V.subs(parameters).subs(Phi, phi).evalf(), matrices).toarray()
        return fourth_order_degenerate(energy, perturbation, retained)[0, 0].real

    current = currents[charge].subs(parameters)
    actual_shift = float((current / sp.sin(Phi)).subs(Phi, phi_sample)) * (1 - np.cos(phi_sample))
    energy_error = abs(actual_shift - (reference_energy(phi_sample) - reference_energy(0)))
    reference_current = (reference_energy(phi_sample + step) - reference_energy(phi_sample - step)) / (2 * step)
    current_error = abs(float(current.subs(Phi, phi_sample)) - reference_current)
    assert energy_error < 1e-12
    assert current_error < 1e-10
    print(f"charge {charge}, n_g={gate}: energy error {energy_error:.2e}, current error {current_error:.2e}")
```

## Gate dependence and current–phase relation

We evaluate the symbolic expressions only when plotting. The left panel shows
the signed critical-current coefficient in the unperturbed ground-state charge
sector: $n=0$ for $n_g<1/2$, $n=1$ for $1/2<n_g<3/2$, and $n=2$ for $n_g>3/2$.
The right panel shows its sinusoidal phase dependence at one gate offset in each
region. Both panels use $I^{(4)}/(2e/\hbar)$; the physical current includes that
prefactor. The even-sector phase curves coincide at the particle-hole-related
offsets $n_g=0.2$ and $1.8$. The gate intervals exclude the charge-degeneracy points, where these
separate nondegenerate branches do not select a unique ground state.

```{code-cell} ipython3
plot_parameters = {**lead_values, U: 10}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
phases = np.linspace(-np.pi, np.pi, 301)
intervals = ((-0.5, 0.49), (0.51, 1.49), (1.51, 2.5))
for charge, (interval, gate) in enumerate(zip(intervals, (0.2, 0.9, 1.8))):
    coefficient = currents_by_gate[charge].subs(plot_parameters).subs(Phi, sp.pi / 2)
    gates = np.linspace(*interval, 200)
    axes[0].plot(gates, sp.lambdify(ng, coefficient, "numpy")(gates), label=f"charge {charge}")
    current = currents_by_gate[charge].subs(plot_parameters).subs(ng, gate)
    axes[1].plot(phases / np.pi, sp.lambdify(Phi, current, "numpy")(phases), label=f"charge {charge}, n_g={gate}")
for ax in axes:
    ax.axhline(0, color="0.6", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel(r"$I^{(4)}/(2e/\hbar)$")
axes[0].set(xlabel=r"Offset charge $n_g$", title=r"Signed critical current ($\Phi=\pi/2$)")
axes[1].set(xlabel=r"Condensate phase $\Phi/\pi$", title="Current–phase relation")
plt.show()
```

The expressions remain symbolic in $U,n_g,E_L,E_R,\Gamma_L,\Gamma_R,t_L,t_R$
and $\Phi$. Substituting $E_\alpha=\sqrt{\xi_\alpha^2+\Gamma_\alpha^2}$ restores
the original superconducting parameters when desired. Our phase convention is
$\Phi=2\phi$ relative to the single-electron tunneling phase in the tutorial;
$(2e/\hbar)\partial_\Phi=(e/\hbar)\partial_\phi$ gives the same physical current.

The generator embedding retains all dot charges. The subsequent charge-energy
expansion requires nonzero gaps to the states it eliminates; in particular,
near the even-charge degeneracy $b=0$, the retained even block should be
diagonalized together. The model represents each superconductor by one paired
orbital; a continuum junction requires the corresponding quasiparticle spectrum
and energy sums.

[^glazman]: L. I. Glazman and K. A. Matveev,
    [Resonant Josephson current through Kondo impurities in a tunnel barrier](http://jetpletters.ru/ps/1121/article_16988.pdf),
    JETP Letters **49**, 659–662 (1989).
