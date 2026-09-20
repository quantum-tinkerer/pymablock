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

A driven dispersive ancilla can modify the cavity's ladder matrix elements,
turning selected oscillator levels into a spin representation. We reproduce the
phase-dependent amplitudes of Roy et al.[^roy] for spin one and spin three
halves. A second-order expansion extends that effective description to include
virtual excursions through discarded ancilla, cavity, and Floquet states.

The effective spins are ordinary three- and four-dimensional matrices. The
source is a two-by-two ancilla matrix with second-quantized cavity and Floquet
operators in its entries. An ordered reference list specifies the
retained basis.

## Driven Hamiltonian and dressed basis

The carrier rotating-wave approximation leaves a comb drive on the ancilla
and a two-tone cavity drive. The cavity lowering operator is $c$, the dispersive
shift is $\chi$, and the drive amplitudes are $\Omega$ and $\epsilon$.
The phases $\phi_k$ belong to the ancilla comb; $\varphi$ is the common
cavity-drive phase.

We transform to the interaction picture of the dispersive term and keep the
comb harmonics explicitly in a Sambe space: the bilateral ladder $\ell$ has
integer number $N_\ell$, and a Fourier factor $e^{ir\chi t}$ becomes
$(\ell^\dagger)^r$ for $r>0$ or $\ell^{-r}$ for $r<0$. The number term is
$\chi N_\ell$. This representation retains the off-resonant harmonics that
would be discarded in subsequent rotating-wave reductions.

At each retained cavity occupation, a rotation diagonalizes the resonant
ancilla drive. Its first column selects the positive-energy dressed branch.
The resonant splitting belongs to $H_0$; the remaining comb harmonics and
cavity drive enter $V$.

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

For spin one, the unperturbed input is the following matrix of number operators.
The two entries distinguish the dressed ancilla branches.

```{code-cell} ipython3
models = [cavity_model(sp.Rational(1)), cavity_model(sp.Rational(3, 2))]
display(models[0]["H0"].applyfunc(sp.factor))
```

The number-projector polynomials are interpolation identities on the displayed
cavity range. One extra cavity level includes every intermediate state reached
by one application of $V$, which suffices through second order. They are not
global projector identities on the infinite oscillator.

## Effective spin matrices

The reference list defines the columns of the isometry $W$ in order:

$$
W|n\rangle=|0\rangle_{\rm dressed}\otimes|n\rangle_c\otimes|0\rangle_\ell,
\qquad n=0,\ldots,2s.
$$

Each `(0, {c: n, ell: 0})` pairs an ancilla matrix index with occupations of
all source modes. The list selects the whole retained subspace, so transitions
between these states are retained together. Transitions to the other dressed
branch and to other cavity or Floquet occupations remain available in virtual
processes. No source ladder is truncated by the embedding.

The returned coefficients are ordinary SymPy matrices in this list order.
The spin interpretation, $m=n-s$, comes from their engineered matrix elements;
it requires no additional operator type. The ancilla is already in its
occupation-dependent dressed basis, so these references correspond to
superpositions of bare ancilla states.

The computed adjacent-level amplitudes reproduce Eq. (18) of Roy et al.[^roy]
The cosine dependence comes from the overlap of neighboring dressed ancilla
states. We display the upper-diagonal entries; Hermitian conjugation supplies
the reverse transitions.

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

The phase prescription in Eq. (27) of Roy et al.[^roy] turns these amplitudes
into spin ladder weights. Applying it to the computed matrices gives the
spin-one and spin-three-halves Hamiltonians:

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

The drive phase selects the rotation axis, while its amplitude sets the
rotation rate. The finite matrices encode the higher spin without introducing
another operator type.

## Second-order virtual corrections

The first-order spin Hamiltonian receives corrections from the other dressed
ancilla branch and from off-resonant comb harmonics. For spin one, the phase
choice above is $(\phi_0,\phi_1,\phi_2)=(0,0,\pi/2)$. Keeping $\chi$, $\Omega$,
$\epsilon$, and $\varphi$ symbolic gives the six independent entries of the
second-order Hermitian matrix:

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

The diagonal entries shift the dressed levels, while off-diagonal entries
modify their couplings. The denominators resolve transitions across the dressed
splitting $\Omega$ and its Floquet-shifted counterparts. The expansion requires
these virtual transitions to remain off resonance. One additional cavity level
contains every intermediate state reached in two perturbation steps; more
levels are needed at higher orders.

The effective Hamiltonian describes coherent dynamics after the carrier
rotating-wave approximation. Dissipation and carrier-frequency counterrotating
corrections are outside this model. Higher perturbative orders require more
cavity shells and Floquet indices to include their longer virtual paths.

[^roy]: S. Roy et al.,
    [Synthetic high angular momentum spin dynamics in a microwave oscillator](https://doi.org/10.1103/PhysRevX.15.021009),
    Physical Review X **15**, 021009 (2025).
    [Open preprint, Appendix B](https://arxiv.org/html/2405.15695v4#A2).
