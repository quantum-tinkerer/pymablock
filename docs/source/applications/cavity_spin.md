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

# Artificial higher spins in a driven cavity

We reproduce the engineered spin-one and spin-three-halves matrix elements of
Roy et al.[^roy] and calculate the spin-one second-order correction.

## Model

A cavity with dispersive shift $\chi$ couples to a driven ancilla. A comb drive
of amplitude $\Omega$ and phases $\phi_k$ dresses the ancilla; a two-tone cavity
drive has amplitude $\epsilon$ and phase $\varphi$. We use the carrier
rotating-wave approximation and retain the other comb harmonics explicitly.

The source is a two-by-two ancilla matrix with cavity and Floquet operators in
its entries. The ladder $\ell$ shifts the Fourier index and contributes
$\chi N_\ell$ to the energy. The rotation below puts the resonant ancilla drive
in its dressed basis, with the positive-energy branch first.

```{code-cell} ipython3
import sympy as sp
from IPython.display import display
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import LadderOp
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding

chi, Omega, epsilon = sp.symbols("chi Omega epsilon", nonzero=True, real=True)
varphi = sp.Symbol("varphi", real=True)

def ancilla_rotation(phase):
    return sp.Matrix([[sp.exp(sp.I * phase / 2), -sp.exp(sp.I * phase / 2)],
                      [sp.exp(-sp.I * phase / 2), sp.exp(-sp.I * phase / 2)]]) / sp.sqrt(2)

phase = sp.Symbol("phi_n", real=True)
display(ancilla_rotation(phase))

def floquet_shift(ell, shift):
    return Dagger(ell)**shift if shift >= 0 else ell**(-shift)

def cavity_model(spin, virtual_shells=1):
    maximum = int(2 * spin)
    cutoff = maximum + virtual_shells
    c, ell = BosonOp("c"), LadderOp("ell")
    phases = sp.symbols(f"phi_0:{maximum + 1}", real=True)
    # These polynomials represent number projectors on 0,...,cutoff.
    projectors = [sp.prod((N(c) - j) / (n - j) for j in range(cutoff + 1) if j != n)
                  for n in range(cutoff + 1)]
    H0 = chi * N(ell) * sp.eye(2) + Omega * sum(projectors[:maximum + 1]) * sp.diag(1, -1) / 2
    ground, excited = sp.diag(1, 0), sp.diag(0, 1)
    lowering = sp.Matrix([[0, 1], [0, 0]])

    def rotation(n):
        if n > maximum:
            return sp.eye(2)
        return ancilla_rotation(phases[n])

    terms = []
    for n, projector in enumerate(projectors):
        R = rotation(n)
        qubit = R.adjoint() * lowering * R
        for tooth, phase in enumerate(phases):
            if tooth == n:
                continue  # This resonant component is already in H0.
            term = Omega * projector * sp.exp(sp.I * phase) * qubit * floquet_shift(ell, tooth - n) / 2
            terms.extend((term, Dagger(term)))
    for n in range(1, cutoff + 1):
        initial, final = rotation(n), rotation(n - 1)
        ground_path = final.adjoint() * ground * initial
        excited_path = final.adjoint() * excited * initial
        paths = ground_path * (1 + Dagger(ell)) + excited_path * (1 + ell)
        term = epsilon * sp.exp(sp.I * varphi) * c * projectors[n] * paths / 2
        terms.extend((term, Dagger(term)))
    embedding = Embedding(
        reference=[(0, {c: n, ell: 0}) for n in range(maximum + 1)]
    )
    return dict(H0=H0, V=sum(terms, sp.zeros(2)), embedding=embedding,
                operators=(c, ell), phases=phases, spin=spin)
```

The unperturbed spin-one input has two dressed ancilla branches:

```{code-cell} ipython3
models = [cavity_model(sp.Rational(1)), cavity_model(sp.Rational(3, 2))]
display(models[0]["H0"].applyfunc(sp.factor))
```

## Effective spin matrices

The references select the positive dressed branch, cavity occupations
$n=0,\ldots,2s$, and Fourier index zero. The output matrices use this order,
corresponding to spin projections $m=n-s$. One extra cavity level closes the
virtual paths through second order; the interpolated number projectors are
used only on that finite range.

The adjacent-level amplitudes reproduce Eq. (18) of Roy et al.[^roy]
Their cosine dependence comes from neighboring ancilla-state overlaps.

```{code-cell} ipython3
series = []
for model in models:
    H, *_ = block_diagonalize([model["H0"], model["V"]], subspace_eigenvectors=model["embedding"])
    series.append(H)
    phases = model["phases"]
    for n in range(1, len(phases)):
        # Separate the common drive phase before simplifying the ancilla overlap.
        amplitude = H[0, 0, 1][n-1, n] * sp.exp(-sp.I * varphi)
        amplitude = sp.trigsimp(sp.expand_complex(amplitude)) * sp.exp(sp.I * varphi)
        display(sp.Eq(sp.Symbol(f"h_{{{len(phases)},{n-1}{n}}}"), amplitude))
```

The phases prescribed in Eq. (27) of Roy et al.[^roy] give the spin-one and
spin-three-halves Hamiltonians:

```{code-cell} ipython3
spin_phases = []
for model, H in zip(models, series):
    maximum = int(2 * model["spin"])
    values = [sp.S.Zero]
    for n in range(1, maximum + 1):
        values.append(values[-1] + 2 * sp.acos(sp.sqrt(sp.Rational(maximum + 1 - n, maximum))))
    phase_choice = dict(zip(model["phases"], values))
    spin_phases.append(phase_choice)
    spin_matrix = H[0, 0, 1].subs(phase_choice)
    display(spin_matrix.applyfunc(sp.simplify))
```

The drive phase sets the rotation axis and its amplitude sets the rate.

## Second-order correction

For the spin-one phase choice, the six independent matrix entries are:

```{code-cell} ipython3
spin_one = models[0]
phase_choice = spin_phases[0]
h2 = series[0][0, 0, 2].subs(phase_choice)
for i in range(3):
    for j in range(i, 3):
        entry = h2[i, j].subs(sp.exp(-sp.I * sp.pi / 4), (1 - sp.I) / sp.sqrt(2))
        entry = sp.factor(sp.expand(entry))
        display(sp.Eq(sp.Symbol(f"H^{{(2)}}_{{{i}{j}}}"), entry))
```

The diagonal entries shift the dressed levels; the off-diagonal entries change
their couplings. The poles mark resonances with discarded ancilla or Floquet
states. This coherent model omits dissipation and carrier-frequency
counterrotating corrections.

[^roy]: S. Roy et al.,
    [Synthetic high angular momentum spin dynamics in a microwave oscillator](https://doi.org/10.1103/PhysRevX.15.021009),
    Physical Review X **15**, 021009 (2025).
    [Open preprint, Appendix B](https://arxiv.org/html/2405.15695v4#A2).
