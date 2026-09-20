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

Its first column is the positive-energy dressed branch, which is matrix basis
index zero. In this dressed basis the unperturbed Sambe Hamiltonian becomes

$$
H_0=\chi N_\ell I_2+\frac{\Omega}{2}
 \sum_{n=0}^{2s}|n\rangle\langle n|\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
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

from pymablock import block_diagonalize
from pymablock.number_ordered_form import LadderOp
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding
from validation import occupation_matrices, operator_matrix, occupation_indices, second_order

chi, Omega, epsilon = sp.symbols("chi Omega epsilon", nonzero=True, real=True)
varphi = sp.Symbol("varphi", real=True)

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
        phase = phases[n] / 2
        return sp.Matrix([[sp.exp(sp.I * phase), -sp.exp(sp.I * phase)],
                          [sp.exp(-sp.I * phase), sp.exp(-sp.I * phase)]]) / sp.sqrt(2)

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

The number-projector polynomials are interpolation identities on the displayed
cavity range. One extra cavity level includes every intermediate state reached
by one application of $V$, which suffices through second order. They are not
global projector identities on the infinite oscillator. We check the same
second-order result after adding another shell below.

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

At first order, the adjacent-level matrix element is

$$
\langle n-1|H^{(1)}|n\rangle
=\frac{\epsilon e^{i\varphi}}{2}\sqrt n
 \cos\!\left(\frac{\phi_n-\phi_{n-1}}{2}\right).
$$

The overlap between neighboring dressed ancilla states supplies the cosine.
The result reproduces Eq. (18) of Roy et al.[^roy], restricted to the retained
manifold, and holds symbolically for both spins.

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
    difference = H[0, 0, 1] - expected
    assert difference.applyfunc(lambda x: sp.trigsimp(sp.expand_complex(x))) == sp.zeros(len(phases))
    print(f"spin {model['spin']}: first-order matrix agrees symbolically")
```

The phase choice in Eq. (27) of Roy et al.[^roy],

$$
\phi_0=0,\qquad
\phi_n-\phi_{n-1}=2\arccos\sqrt{\frac{2s+1-n}{2s}}
$$

gives

$$
H^{(1)}=\frac{\epsilon}{2\sqrt{2s}}
 (e^{i\varphi}S_-+e^{-i\varphi}S_+).
$$

The adjacent-level weights, scaled by $\epsilon/2$, now follow the spin ladder
rather than the oscillator's $\sqrt n$ dependence.

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
    S = sp.zeros(maximum + 1)
    for n in range(1, maximum + 1):
        S[n-1, n] = sp.sqrt(n * (maximum + 1 - n))
    expected = epsilon * (sp.exp(sp.I * varphi) * S + sp.exp(-sp.I * varphi) * S.adjoint()) / (2 * sp.sqrt(maximum))
    assert (actual - expected).applyfunc(sp.simplify) == sp.zeros(maximum + 1)
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
    actual = np.asarray(H[0, 0, 2].evalf(), dtype=complex)
    occupations = [range(size + 1), range(-size, size + 1)]
    matrices = occupation_matrices(model["operators"], occupations)
    energy = operator_matrix(source[0].evalf(), matrices).diagonal().real
    perturbation = operator_matrix(source[1].evalf(), matrices).toarray()
    # operator_matrix stacks the two ancilla components as outer blocks.
    # The positive dressed branch is the first block (component zero).
    kept = occupation_indices(occupations, [(n, 0) for n in range(size)])
    reference = second_order(energy, perturbation, kept)
    error = np.max(np.abs(actual - reference))
    assert error < 1e-12
    second_matrices.append(actual)
    print(f"spin {model['spin']}: maximum second-order matrix error {error:.2e}")
    # Separate real and imaginary parts to keep the four-state matrix readable.
    matrix = H[0, 0, 2].evalf(5, chop=True)
    print("Real part:")
    display(sp.re(matrix))
    print("Imaginary part:")
    display(sp.im(matrix))
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
expanded_h2 = np.asarray(H_expanded[0, 0, 2].evalf(), dtype=complex)
shell_error = np.max(np.abs(expanded_h2 - second_matrices[0]))
assert shell_error < 1e-12
print(f"Change after adding a second cavity shell: {shell_error:.2e}")
```

The effective Hamiltonian describes coherent dynamics after the carrier
rotating-wave approximation. Dissipation and carrier-frequency counterrotating
corrections are outside this model. Higher perturbative orders require more
cavity shells and Floquet indices to include their longer virtual paths.

[^roy]: S. Roy et al.,
    [Synthetic high angular momentum spin dynamics in a microwave oscillator](https://doi.org/10.1103/PhysRevX.15.021009),
    Physical Review X **15**, 021009 (2025).
    [Open preprint, Appendix B](https://arxiv.org/html/2405.15695v4#A2).
