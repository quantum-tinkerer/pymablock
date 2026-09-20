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
oscillator levels.

## Three-oscillator Hamiltonian

We count qubit–coupler couplings at first order and direct qubit–qubit coupling
at second order: $H=H_0+\eta V_1+\eta^2 V_2$.


The $\alpha_j$ are anharmonicities. We retain the pair-creation and
pair-annihilation terms along with excitation exchange. In this sign convention,
the direct qubit exchange amplitude is $-g_{12}$.

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

The second-order exchange transfers one excitation from qubit 1 to qubit 2.
We extract its coefficient directly from the effective operator.

```{code-cell} ipython3
h2 = H[0, 0, 2]
exchange_term = h2.filter_terms(((1, -1),), keep=True).as_expr()
exchange = exchange_term.coeff(Dagger(q2) * q1)
display(sp.Eq(sp.Symbol("J^{(2)}"), exchange))
```

The sum-frequency denominators come from counterrotating processes; the
symmetrized difference-frequency denominators account for unequal qubit
energies. This reproduces Eq. (33) of the Yan et al. preprint[^yan], with
$g_{12}^{\rm Yan}=-g_{12}$ in our coupling convention.

## Fourth-order ZZ interaction

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

The coefficient of $N_{q_1}N_{q_2}$ is the conditional shift. Since the retained
Hamiltonian preserves excitation parity, its even and odd blocks each have
dimension two. Their eigenvalue sums equal their traces, so mixing cancels in
$E_{11}-E_{10}-E_{01}+E_{00}$. With $Z_j=1-2N_{q_j}$, the ZZ coefficient is
one quarter of this shift.

The three contributions distinguish direct coupling, interference with a path
through the coupler, and purely mediated coupling. Their sum is the conditional
shift. All are obtained from the same computed coefficient, including
counterrotating processes. SymPy abbreviates repeated gaps and groups terms by
the remaining virtual-state denominator for display.

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

The anharmonicities remain in the virtual energy denominators. This fourth-order
result extends the second-order exchange of Yan et al.[^yan] In the harmonic
limit the conditional shift vanishes, as required for a quadratic Hamiltonian.

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
