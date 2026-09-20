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

The model follows the tunable-coupler architecture of Yan et al.[^yan]
The second-order exchange exhibits cancellation between direct and mediated
coupling, including counterrotating processes. Extending the symbolic expansion
to fourth order also gives conditional interactions and corrections from higher
oscillator levels. The numerical parameters illustrate these effects rather than
representing a calibrated device.

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
energies. This reproduces the dispersive exchange in Eq. (33) of the Yan et al.
preprint[^yan], with $g_{12}^{\rm Yan}=-g_{12}$ in our coupling convention.
In number-ordered form it is the coefficient of the term that transfers one
excitation from qubit 1 to qubit 2.

```{code-cell} ipython3
h2 = H[0, 0, 2]
exchange_term = h2.filter_terms(((1, -1),), keep=True).as_expr()
display(exchange_term)
exchange = sp.factor(exchange_term.coeff(Dagger(q2) * q1))
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

Fourth order includes terms quadratic in the direct coupling, mixed direct and
mediated processes, and terms quartic in the qubit–coupler couplings. From the
full symbolic operator, the diagonal combination
$h_{11}-h_{10}-h_{01}+h_{00}$ isolates the conditional interaction by removing
the reference energy and individual excitation shifts. Here
$h_{n_1n_2}=\langle n_1n_2|H^{(4)}|n_1n_2\rangle$.

```{code-cell} ipython3
h4_symbolic = H[0, 0, 4]
assert (h4_symbolic - h4_symbolic.adjoint()).applyfunc(sp.cancel).is_zero
diagonal = h4_symbolic.filter_terms(((0, 0),), keep=True).as_expr()
conditional_symbolic = (
    diagonal.subs({N(q1): 1, N(q2): 1})
    - diagonal.subs({N(q1): 1, N(q2): 0})
    - diagonal.subs({N(q1): 0, N(q2): 1})
    + diagonal.subs({N(q1): 0, N(q2): 0})
)
```

The symbolic result can be evaluated at a rational parameter point to display
the effective operator and compare it with a finite oscillator calculation.

```{code-cell} ipython3
parameters = {
    w1: 3, w2: 5, wc: 9,
    alpha1: sp.Rational(-1, 5), alpha2: sp.Rational(-1, 4), alphac: sp.Rational(-1, 7),
    g12: sp.Rational(1, 11), g1c: sp.Rational(1, 5), g2c: sp.Rational(1, 6),
}
numeric_source = {order: value.subs(parameters) for order, value in source.items()}
h4 = h4_symbolic.subs(parameters)
display(h4.as_expr().evalf(6))
conditional_coefficient = sp.factor(conditional_symbolic.subs(parameters))
print("Fourth-order coefficient of N(q1) N(q2):", float(conditional_coefficient))
```

The selected density term is the coefficient multiplying
$N_{q_1}N_{q_2}$ in this effective basis. To obtain a spectroscopic conditional
frequency shift, one must also include the mixing from the off-diagonal terms
when diagonalizing the retained Hamiltonian.

Pymablock's matrix interface gives the same fourth-order block for ordinary
oscillator matrices with four or five levels per mode. This comparison checks
the finite-order truncation and symbolic operator arithmetic; it uses the same
perturbative recurrence in both representations.

```{code-cell} ipython3
target_matrices = occupation_matrices((q1, q2), [range(2)] * 2)
actual = operator_matrix(h4, target_matrices).toarray()
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

[^yan]: F. Yan et al.,
    [Tunable coupling scheme for implementing high-fidelity two-qubit gates](https://doi.org/10.1103/PhysRevApplied.10.054062),
    Physical Review Applied **10**, 054062 (2018).
    [Open preprint](https://arxiv.org/abs/1803.09813).
