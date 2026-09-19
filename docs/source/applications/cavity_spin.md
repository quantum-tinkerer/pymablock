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
phase-dependent amplitudes of [Roy et al., *Synthetic high angular momentum
spin dynamics in a microwave oscillator*, Physical Review X 15, 021009
(2025)](https://arxiv.org/html/2405.15695v4), and calculate second-order corrections
from the discarded ancilla, cavity, and Floquet states.

We treat spin one and spin three halves. The effective operators remain
symbolic `SpinOp` objects; finite matrices are requested only to display or
check the result.

## Driven Hamiltonian and dressed basis

After the rotating-wave approximation at the carrier frequencies, the source
Hamiltonian is

$$
\begin{aligned}
H_{\rm rot}(t)={}&\chi N_c |e\rangle\langle e|
 +\frac{\Omega}{2}\sum_{k=0}^{2s}e^{i(k\chi t+\phi_k)}|g\rangle\langle e|\\
 &+\frac{\epsilon}{2}e^{i\varphi}(1+e^{i\chi t})c+\mathrm{h.c.},
\end{aligned}
$$

where the Hermitian conjugate applies to the drive terms. The cavity lowering
operator is $c$, the dispersive shift is $\chi$, and the ancilla and cavity
drive amplitudes are $\Omega$ and $\epsilon$. The independent phases
$\phi_k$ belong to the ancilla comb; $\varphi$ is the common cavity-drive phase.
Our phase convention gives $e^{+i\varphi}$ on the cavity lowering term.

We transform to the interaction picture of the dispersive term and keep the
comb harmonics explicitly in a Sambe space: the bilateral ladder $\ell$ has
integer number $N_\ell$, and a Fourier factor $e^{ir\chi t}$ becomes
$(\ell^\dagger)^r$ for $r>0$ or $\ell^{-r}$ for $r<0$. The number term is
$\chi N_\ell$. This representation retains the off-resonant harmonics that
would be discarded in subsequent rotating-wave reductions.

At cavity occupation $n\leq2s$, we diagonalize the resonant ancilla drive with

$$
R_n=\frac{1}{\sqrt2}
\begin{pmatrix}
 e^{i\phi_n/2}&-e^{i\phi_n/2}\\
 e^{-i\phi_n/2}&e^{-i\phi_n/2}
\end{pmatrix}.
$$

Its first column is the positive-energy dressed branch. We write $d$ for the
ancilla lowering operator in this basis, so $N_d=0$ selects that branch. The
unperturbed Sambe Hamiltonian becomes

$$
H_0=\chi N_\ell+\frac{\Omega}{2}
 \sum_{n=0}^{2s}|n\rangle\langle n|(1-2N_d).
$$

All remaining comb components and the cavity drive enter $V$. We organize
$H_0+\eta V$ in powers of a formal $\eta$, preserving the nonzero dressed
splitting in $H_0$.

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize
from pymablock.number_ordered_form import LadderOp, SpinOp, NumberOrderedForm
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding
from validation import occupation_matrices, operator_matrix, occupation_indices, second_order

chi, Omega, epsilon = sp.symbols("chi Omega epsilon", nonzero=True, real=True)
varphi = sp.Symbol("varphi", real=True)

def binary_operator(matrix, d):
    return (matrix[0, 0] * (1 - N(d)) + matrix[1, 1] * N(d)
            + matrix[0, 1] * d + matrix[1, 0] * Dagger(d))

def floquet_shift(ell, shift):
    return Dagger(ell)**shift if shift >= 0 else ell**(-shift)

def cavity_model(spin, virtual_shells=1):
    maximum = int(2 * spin)
    cutoff = maximum + virtual_shells
    c, ell, d = BosonOp("c"), LadderOp("ell"), SigmaMinus("d")
    S = SpinOp("S", spin)
    phases = sp.symbols(f"phi_0:{maximum + 1}", real=True)
    # These polynomials represent number projectors on 0,...,cutoff.
    projectors = [sp.prod((N(c) - j) / (n - j) for j in range(cutoff + 1) if j != n)
                  for n in range(cutoff + 1)]
    H0 = chi * N(ell) + Omega * sum(projectors[:maximum + 1]) * (1 - 2 * N(d)) / 2
    ground, excited = sp.diag(1, 0), sp.diag(0, 1)
    lowering = sp.Matrix([[0, 1], [0, 0]])

    def rotation(n):
        if n > maximum:
            return sp.eye(2)
        phase = phases[n] / 2
        return sp.Matrix([[sp.exp(sp.I * phase), -sp.exp(sp.I * phase)],
                          [sp.exp(-sp.I * phase), sp.exp(-sp.I * phase)]]) / sp.sqrt(2)

    terms = []
    for n, projector in enumerate(projectors):
        R = rotation(n)
        qubit = binary_operator(R.adjoint() * lowering * R, d)
        for tooth, phase in enumerate(phases):
            if tooth == n:
                continue  # This resonant component is already in H0.
            term = Omega * projector * sp.exp(sp.I * phase) * qubit * floquet_shift(ell, tooth - n) / 2
            terms.extend((term, Dagger(term)))
    for n in range(1, cutoff + 1):
        initial, final = rotation(n), rotation(n - 1)
        ground_path = binary_operator(final.adjoint() * ground * initial, d)
        excited_path = binary_operator(final.adjoint() * excited * initial, d)
        paths = ground_path * (1 + Dagger(ell)) + excited_path * (1 + ell)
        term = epsilon * sp.exp(sp.I * varphi) * c * projectors[n] * paths / 2
        terms.extend((term, Dagger(term)))
    embedding = Embedding(
        {S: sp.sqrt(maximum - N(c)) * c}, reference={c: 0, ell: 0, d: 0}
    )
    return dict(H0=H0, V=sp.Add(*terms), embedding=embedding,
                operators=(c, ell, d), S=S, phases=phases, spin=spin)
```

The number-projector polynomials are interpolation identities on the displayed
cavity range. One extra cavity level includes every intermediate state reached
by one application of $V$, which suffices through second order. They are not
global projector identities on the infinite oscillator. We check the same
second-order result after adding another shell below.

## Spin generators and the source reference

The embedding is

$$
S_-\longmapsto\sqrt{2s-N_c}\,c,
\qquad (N_c,N_\ell,N_d)=(0,0,0)\quad\text{in the reference}.
$$

Because the coefficient stands to the left of $c$, the lowering amplitude on
occupation $n$ is $\sqrt{n(2s+1-n)}$. Thus $n=0,\ldots,2s$ represents
$m=n-s$. The ancilla is already in its occupation-dependent dressed basis;
the product reference corresponds to a superposition of bare ancilla states.

The physical cavity drive remains the one in $V$. Defining its target spin
algebra does not force that drive to have spin-like matrix elements. Those
amplitudes are determined by the comb phases.

## First-order matrix-element modification

The expected adjacent-level matrix element is

$$
\langle n-1|H^{(1)}|n\rangle
=\frac{\epsilon e^{i\varphi}}{2}\sqrt n
 \cos\!\left(\frac{\phi_n-\phi_{n-1}}{2}\right).
$$

The overlap between neighboring dressed ancilla states supplies the cosine.
This is the amplitude in Eq. (18) of the paper. We compare the full first-order
matrix with this expression for both spins.

```{code-cell} ipython3
models = [cavity_model(sp.Rational(1)), cavity_model(sp.Rational(3, 2))]
series = []
for model in models:
    H, *_ = block_diagonalize([model["H0"], model["V"]], subspace_eigenvectors=model["embedding"])
    series.append(H)
    phases = model["phases"]
    expected = sp.zeros(len(phases))
    for n in range(1, len(phases)):
        amplitude = epsilon * sp.exp(sp.I * varphi) * sp.sqrt(n) * sp.cos((phases[n] - phases[n-1]) / 2) / 2
        expected[n-1, n], expected[n, n-1] = amplitude, sp.conjugate(amplitude)
    difference = H[0, 0, 1].to_matrix() - expected
    assert difference.applyfunc(lambda x: sp.trigsimp(sp.expand_complex(x))) == sp.zeros(len(phases))
    print(f"spin {model['spin']}: first-order matrix agrees symbolically")
```

The phase choice

$$
\phi_0=0,\qquad
\phi_n-\phi_{n-1}=2\arccos\sqrt{\frac{2s+1-n}{2s}}
$$

gives

$$
H^{(1)}=\frac{\epsilon}{2\sqrt{2s}}
 (e^{i\varphi}S_-+e^{-i\varphi}S_+).
$$

We verify this operator identity and plot the resulting adjacent-level weights,
scaled by $\epsilon/2$, against the unmodified oscillator weights.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), constrained_layout=True)
for ax, model, H in zip(axes, models, series):
    maximum = int(2 * model["spin"])
    phases = model["phases"]
    values = {phases[0]: sp.S.Zero}
    angle = sp.S.Zero
    for n in range(1, maximum + 1):
        angle += 2 * sp.acos(sp.sqrt(sp.Rational(maximum + 1 - n, maximum)))
        values[phases[n]] = angle
    actual = H[0, 0, 1].applyfunc(lambda x: sp.trigsimp(sp.expand_complex(x.subs(values))))
    S = model["S"]
    expected = NumberOrderedForm.from_expr(
        epsilon * (sp.exp(sp.I * varphi) * S + sp.exp(-sp.I * varphi) * Dagger(S))
        / (2 * sp.sqrt(maximum))
    )
    assert (actual - expected).applyfunc(sp.simplify).is_zero
    n = np.arange(1, maximum + 1)
    ax.plot(n, np.sqrt(n), "o--", label="oscillator")
    ax.plot(n, np.sqrt(n * (maximum + 1 - n) / maximum), "s-", label="engineered spin")
    ax.set(xticks=n, xlabel="Initial occupation n", ylabel=r"Ladder weight / $(\epsilon/2)$",
           title=f"Spin {model['spin']}")
axes[0].legend(frameon=False)
plt.show()
```

## Second-order virtual corrections

At a generic phase choice, we compare against a direct Sambe-matrix resolvent.
The source ranges include $n=0,\ldots,2s+1$, both ancilla levels, and Floquet
indices $-(2s+1),\ldots,2s+1$. They contain every state reached by one action of
$V$ from the retained manifold. All retained states have energy $\Omega/2$.

The numerical point $(\chi,\Omega,\epsilon)=(11,3,1)$ is used to check the
coefficient of $\eta^2$. The asymptotic accuracy at $\eta=1$ is a separate
question governed by drive-to-gap ratios.

```{code-cell} ipython3
second_matrices = []
for model in models:
    size = len(model["phases"])
    parameters = {chi: 11, Omega: 3, epsilon: 1, varphi: sp.pi / 7}
    parameters.update({phase: sp.pi * i / (i + 2) for i, phase in enumerate(model["phases"])})
    source = [model[key].subs(parameters) for key in ("H0", "V")]
    H, *_ = block_diagonalize(source, subspace_eigenvectors=model["embedding"])
    actual = np.asarray(H[0, 0, 2].to_matrix().evalf(), dtype=complex)
    occupations = [range(size + 1), range(-size, size + 1), range(2)]
    matrices = occupation_matrices(model["operators"], occupations)
    energy = operator_matrix(source[0].evalf(), matrices).diagonal().real
    perturbation = operator_matrix(source[1].evalf(), matrices).toarray()
    kept = occupation_indices(occupations, [(n, 0, 0) for n in range(size)])
    reference = second_order(energy, perturbation, kept)
    error = np.max(np.abs(actual - reference))
    assert error < 1e-12
    second_matrices.append(actual)
    print(f"spin {model['spin']}: maximum second-order matrix error {error:.2e}")
    display(H[0, 0, 2].to_matrix().evalf(5))
```

Finally, increasing the polynomial interpolation range for spin one leaves its
second-order coefficient unchanged. This checks that the result does not depend
on the arbitrary continuation of those polynomials outside the reached states.

```{code-cell} ipython3
expanded = cavity_model(sp.Rational(1), virtual_shells=2)
parameters = {chi: 11, Omega: 3, epsilon: 1, varphi: sp.pi / 7}
parameters.update({phase: sp.pi * i / (i + 2) for i, phase in enumerate(expanded["phases"])})
H_expanded, *_ = block_diagonalize(
    [expanded[key].subs(parameters) for key in ("H0", "V")],
    subspace_eigenvectors=expanded["embedding"],
)
expanded_h2 = np.asarray(H_expanded[0, 0, 2].to_matrix().evalf(), dtype=complex)
shell_error = np.max(np.abs(expanded_h2 - second_matrices[0]))
assert shell_error < 1e-12
print(f"Change after adding a second cavity shell: {shell_error:.2e}")
```

These calculations validate the coherent effective Hamiltonian in the specified
carrier-rotated model. They omit dissipation and carrier-frequency
counterrotating corrections. Higher perturbative orders require enough cavity
shells and Floquet indices to include their longer virtual paths; the
second-order closure check does not establish those higher-order results.
