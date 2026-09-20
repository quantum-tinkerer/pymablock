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

We calculate exchange and fourth-order ZZ coupling for two qubits connected
through a detuned oscillator, following Yan et al.[^yan]

## Model

The three modes are Duffing oscillators with frequencies $\omega_j$ and
anharmonicities $\alpha_j$. Both exchange and counterrotating couplings are
included. Qubit–coupler coupling enters at first order; direct qubit–qubit
coupling enters at second order, with exchange amplitude $-g_{12}$.

```{code-cell} ipython3
import sympy as sp
from IPython.display import Math, display
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding

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
for name, expression in (("H_0", H0), ("V_1", V1), ("V_2", V2)):
    display(sp.Eq(sp.Symbol(name, commutative=False), expression))
```

## Effective qubit Hamiltonian

We retain two levels of each qubit and the coupler vacuum. Higher oscillator
levels remain available in virtual processes.

```{code-cell} ipython3
source = {(0,): H0, (1,): V1, (2,): V2}
embedding = Embedding({q1: a1, q2: a2}, reference={a1: 0, a2: 0, ac: 0})
H, *_ = block_diagonalize(source, subspace_eigenvectors=embedding)
```

## Exchange

The coefficient below transfers an excitation between the qubits.

```{code-cell} ipython3
h2 = H[0, 0, 2]
exchange_term = h2.filter_terms(((1, -1),), keep=True).as_expr()
exchange = exchange_term.coeff(Dagger(q2) * q1)
display(sp.Eq(sp.Symbol("J^{(2)}"), exchange))
```

The sum-frequency denominators describe counterrotating processes. The result
reproduces Eq. (33) of Yan et al.[^yan], with $g_{12}^{\rm Yan}=-g_{12}$.
Tuning the coupler frequency changes the mediated term and can cancel the
direct exchange.

## ZZ coupling

The combination $h_{11}-h_{10}-h_{01}+h_{00}$ extracts the conditional shift
from the diagonal fourth-order Hamiltonian.

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

Mixing within each parity block cancels in this energy combination. The ZZ
coefficient is one quarter of the conditional shift for $Z_j=1-2N_{q_j}$.
The following direct, mixed, and mediated contributions sum to that shift;
repeated energy gaps are abbreviated for readability.

```{code-cell} ipython3
# Abbreviate repeated one-step energy denominators for display.
gaps = {}
for j, (omega, alpha) in enumerate(((w1, alpha1), (w2, alpha2)), 1):
    for prefix, gap in (("d", wc - omega), ("s", wc + omega), ("t", wc + omega + alpha)):
        symbol = sp.Symbol(f"{prefix}_{j}")
        gaps[gap] = symbol
        display(sp.Eq(symbol, gap))
short_coefficient = conditional_symbolic.xreplace(gaps)
monomials = (g12**2, g12 * g1c * g2c, g1c**2 * g2c**2)
contributions = sp.collect(short_coefficient, monomials, evaluate=False, exact=True)
for label, monomial in zip(("direct", "mixed", "mediated"), monomials):
    coefficient = contributions[monomial]
    poles = sorted((p for p in coefficient.atoms(sp.Pow) if p.exp == -1 and p.base.is_Add),
                   key=sp.default_sort_key)
    terms = []
    for pole, numerator in sp.collect(coefficient, poles, evaluate=False).items():
        terms.extend(sp.Add.make_args(numerator) if pole == 1 else [sp.factor(numerator) * pole])
    lines = [sp.latex(term) for term in terms]
    lhs = sp.latex(sp.Symbol(f"zeta_{{{label}}}") / monomial)
    display(Math(r"\begin{aligned}" + lhs + r"={}&"
                 + r"\\ &{}+".join(lines) + r"\end{aligned}"))
```

The shift vanishes when all anharmonicities are zero. Near a resonance with a
coupler or higher qubit level, that level must be retained explicitly.

[^yan]: F. Yan et al.,
    [Tunable coupling scheme for implementing high-fidelity two-qubit gates](https://doi.org/10.1103/PhysRevApplied.10.054062),
    Physical Review Applied **10**, 054062 (2018).
    [Open preprint](https://arxiv.org/abs/1803.09813).
