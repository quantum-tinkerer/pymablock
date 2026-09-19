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

# Two qubits connected through a tunable coupler

A detuned oscillator mediates exchange between two qubits. Changing its frequency
changes the virtual energy denominators and therefore the effective coupling.
We use three weakly anharmonic oscillators, retain two levels of each qubit, and
eliminate the coupler. The resulting two-spin Hamiltonian includes excursions
through higher qubit levels as well as through the coupler.

This circuit architecture is described by [Yan et al., *Tunable Coupling Scheme
for Implementing High-Fidelity Two-Qubit Gates*, Physical Review Applied 10,
054062 (2018)](https://doi.org/10.1103/PhysRevApplied.10.054062). The parameters
below define an illustrative Hamiltonian; they are not a device calibration.
We reproduce its second-order exchange analytically and check the complete
fourth-order matrix using finite oscillator calculations.

## Three-oscillator Hamiltonian

We count qubit–coupler couplings at first order and direct qubit–qubit coupling
at second order: $H=H_0+\eta V_1+\eta^2 V_2$.

$$
H_0=\sum_{j=1,2,c}\left[\omega_j N_j+
 \frac{\alpha_j}{2}N_j(N_j-1)\right],
\qquad
V_1=\sum_{j=1,2}g_{jc}(a_j^\dagger-a_j)(a_c^\dagger-a_c),
\qquad
V_2=g_{12}(a_1^\dagger-a_1)(a_2^\dagger-a_2).
$$

The $\alpha_j$ are anharmonicities. We retain the pair-creation and
pair-annihilation terms along with excitation exchange. In this sign convention,
the direct qubit exchange amplitude is $-g_{12}$.

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display
from itertools import product
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding
from validation import occupation_matrices, operator_matrix, occupation_indices

a1, a2, ac = (BosonOp(name) for name in ("a1", "a2", "ac"))
q1, q2 = SigmaMinus("q1"), SigmaMinus("q2")
w1, w2, wc = sp.symbols("omega_1 omega_2 omega_c", positive=True)
alpha1, alpha2, alphac = sp.symbols("alpha_1 alpha_2 alpha_c", real=True, nonzero=True)
g12, g1c, g2c = sp.symbols("g_12 g_1c g_2c", real=True)
modes = (a1, a2, ac)
frequencies = (w1, w2, wc)
anharmonicities = (alpha1, alpha2, alphac)
H0 = sum(w * N(a) + alpha * N(a) * (N(a) - 1) / 2
         for a, w, alpha in zip(modes, frequencies, anharmonicities))
V1 = sum(g * (Dagger(a) - a) * (Dagger(ac) - ac)
         for a, g in ((a1, g1c), (a2, g2c)))
V2 = g12 * (Dagger(a1) - a1) * (Dagger(a2) - a2)
source = {(0,): H0, (1,): V1, (2,): V2}
embedding = Embedding({q1: a1, q2: a2}, reference={a1: 0, a2: 0, ac: 0})
H, *_ = block_diagonalize(source, subspace_eigenvectors=embedding)
```

The maps $q_1\mapsto a_1$ and $q_2\mapsto a_2$, together with the oscillator
vacuum, identify the qubit lowering operators. The source oscillators remain
unbounded. For example, annihilation after creation can visit level two even
when both endpoints lie in the qubit subspace; the embedding retains such
virtual processes.

## Exchange including counterrotating processes

The coefficient of $q_2^\dagger q_1+q_1^\dagger q_2$ is

$$
J=\eta^2[-g_{12}+J^{(2)}]+O(\eta^4),
\qquad
J^{(2)}=\frac{g_{1c}g_{2c}}{2}
 \left[\frac{1}{\omega_1-\omega_c}+\frac{1}{\omega_2-\omega_c}
       -\frac{1}{\omega_1+\omega_c}-\frac{1}{\omega_2+\omega_c}\right].
$$

The sum-frequency denominators come from counterrotating processes. The
symmetrized difference-frequency denominators account for unequal qubit
energies. We extract the exchange between $|10\rangle$ and $|01\rangle$ directly
from the effective matrix.

```{code-cell} ipython3
h2 = H[0, 0, 2].to_matrix()
exchange = sp.factor(h2[1, 2])
reference_exchange = g1c * g2c / 2 * (
    1 / (w1 - wc) + 1 / (w2 - wc) - 1 / (w1 + wc) - 1 / (w2 + wc)
)
assert sp.factor(exchange + g12 - reference_exchange) == 0
display(sp.Eq(sp.Symbol("J^{(2)}"), reference_exchange))
```

An illustrative scan shows cancellation between the direct and mediated terms.
Here $\eta=1$, $(\omega_1,\omega_2)=(3,5)$,
$(g_{12},g_{1c},g_{2c})=(10^{-4},1/50,1/60)$, and the coupler stays away from
both qubit resonances. These scan parameters are separate from the fourth-order
reference point below.

```{code-cell} ipython3
scan = {w1: 3, w2: 5, g12: sp.Rational(1, 10000),
        g1c: sp.Rational(1, 50), g2c: sp.Rational(1, 60)}
coupler_frequencies = np.linspace(0.8, 2.0, 250)
mediated = sp.lambdify(wc, reference_exchange.subs(scan), "numpy")(coupler_frequencies)
fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
ax.plot(coupler_frequencies, mediated, label="mediated exchange")
ax.plot(coupler_frequencies, mediated - float(scan[g12]), label="direct + mediated")
ax.axhline(0, color="0.6", lw=0.7)
ax.set(xlabel=r"Coupler frequency $\omega_c$", ylabel=r"Exchange $J$")
ax.legend(frameon=False)
plt.show()
```

## Full fourth-order Hamiltonian

We compute the complete fourth-order coefficient symbolically and convert the
two-spin output to a $4\times4$ matrix. With this ordering, fourth order includes
terms quadratic in the direct coupling, mixed direct and mediated processes,
and terms quartic in the qubit–coupler couplings.

```{code-cell} ipython3
h4_symbolic = H[0, 0, 4].to_matrix()
assert (h4_symbolic - h4_symbolic.adjoint()).applyfunc(sp.cancel) == sp.zeros(4)
conditional_symbolic = (h4_symbolic[3, 3] - h4_symbolic[2, 2]
                        - h4_symbolic[1, 1] + h4_symbolic[0, 0])
```

Only after obtaining the symbolic result do we substitute exact rational
parameters for display and the independent finite-matrix comparison.

```{code-cell} ipython3
parameters = {
    w1: 3, w2: 5, wc: 9,
    alpha1: sp.Rational(-1, 5), alpha2: sp.Rational(-1, 4), alphac: sp.Rational(-1, 7),
    g12: sp.Rational(1, 11), g1c: sp.Rational(1, 5), g2c: sp.Rational(1, 6),
}
numeric_source = {order: value.subs(parameters) for order, value in source.items()}
h4 = h4_symbolic.subs(parameters)
assert (h4 - h4.adjoint()).applyfunc(sp.simplify) == sp.zeros(4)
display(h4.evalf(6))
conditional_coefficient = sp.factor(conditional_symbolic.subs(parameters))
print("Fourth-order coefficient of N(q1) N(q2):", float(conditional_coefficient))
```

The displayed diagonal combination is the coefficient multiplying
$N_{q_1}N_{q_2}$ in this effective basis. To obtain a spectroscopic conditional
frequency shift, one must also include the mixing from the off-diagonal terms
when diagonalizing the retained Hamiltonian.

We now construct ordinary oscillator matrices independently of the symbolic
operator arithmetic and run Pymablock's matrix interface. Increasing the number
of levels per oscillator from four to five checks the fourth-order truncation.
This comparison tests the source algebra, compression, and energy denominators;
both calculations use the same perturbative recurrence.

```{code-cell} ipython3
actual = np.asarray(h4, dtype=complex)
errors = []
for levels in (4, 5):
    occupations = [range(levels)] * 3
    matrices = occupation_matrices(modes, occupations)
    matrix_source = {order: operator_matrix(value, matrices).toarray()
                     for order, value in numeric_source.items()}
    kept = occupation_indices(occupations, [(n1, n2, 0) for n1, n2 in product(range(2), repeat=2)])
    labels = np.ones(levels**3, dtype=int)
    labels[kept] = 0
    matrix_series, *_ = block_diagonalize(matrix_source, subspace_indices=labels)
    reference = matrix_series[0, 0, 4]
    error = np.max(np.abs(actual - reference))
    assert error < 1e-12
    errors.append((levels, error))
for levels, error in errors:
    print(f"{levels} oscillator levels: maximum fourth-order matrix error {error:.2e}")
```

The Duffing Hamiltonian is a local approximation to the circuit spectrum. In
particular, negative anharmonicity should not be extrapolated to arbitrarily
high occupations. This finite-order result uses the low-lying levels reached
by the virtual paths. The dispersive expansion also requires nonzero gaps to
coupled discarded states; near a coupler or higher-level resonance, those states
must be retained explicitly.
