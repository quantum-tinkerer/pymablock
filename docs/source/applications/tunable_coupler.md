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

$$
\begin{aligned}
H_0&=\sum_{j=1,2,c}\left[\omega_j N_j+
 \frac{\alpha_j}{2}N_j(N_j-1)\right],\\
V_1&=\sum_{j=1,2}g_{jc}(a_j^\dagger-a_j)(a_c^\dagger-a_c),\\
V_2&=g_{12}(a_1^\dagger-a_1)(a_2^\dagger-a_2).
\end{aligned}
$$

The $\alpha_j$ are anharmonicities. We retain the pair-creation and
pair-annihilation terms along with excitation exchange. In this sign convention,
the direct qubit exchange amplitude is $-g_{12}$.

```{code-cell} ipython3
import sympy as sp
from IPython.display import display
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

The conditional shift is

$$
\zeta^{(4)}=h_{11}-h_{10}-h_{01}+h_{00}.
$$

It is also the coefficient of $N_{q_1}N_{q_2}$. Since the retained Hamiltonian
preserves excitation parity, its even and odd blocks each have dimension two.
Their eigenvalue sums equal their traces, so mixing within those blocks cancels
in $E_{11}-E_{10}-E_{01}+E_{00}$. Thus this coefficient already gives the
spectroscopic conditional shift. With $Z_j=1-2N_{q_j}$, the corresponding term
is $\eta^4\zeta^{(4)}Z_1Z_2/4$.

A compact expression keeps the virtual energy denominators separate. Define

$$
r_j=\frac1{\omega_j-\omega_c},\qquad
s_j=\frac1{\omega_j+\omega_c},\qquad
u_j=\frac1{\omega_j+\omega_c+\alpha_j},
$$

and $G=g_{1c}g_{2c}$, $W=\omega_1+\omega_2$, $C=2\omega_c+\alpha_c$.
The first denominator describes excitation exchange with the coupler; the
other two describe pair creation from an empty or occupied qubit. For repeated
contributions through a virtual qubit state and its two-coupler-excitation
counterpart, write

$$
F(x,y)=\frac{(g_{12}+Gy)^2}{x}+\frac{2G^2y^2}{x+C}.
$$

Then the full fourth-order result, including counterrotating processes, is

$$
\begin{aligned}
\zeta^{(4)}={}&-4F(W+\alpha_1+\alpha_2,u_1+u_2)\\
&+2\sum_{j\ne k}\left[
 F(W+\alpha_j,u_j+s_k)
 -F(\omega_j-\omega_k+\alpha_j,u_j-r_k)\right]\\
&-4g_{12}G\left[(s_1+r_1-u_1)(s_2+r_2-u_2)+u_1u_2\right]
+G^2 R,
\end{aligned}
$$

where the sum contains $(j,k)=(1,2),(2,1)$ and

$$
\begin{aligned}
R={}&-\frac{2(s_1+s_2)^2}{C+W}
     -\frac{2(r_1+r_2)^2}{C-W}
     -\frac{4(2u_1-s_1-r_1)(2u_2-s_2-r_2)}{C}\\
&+2\sum_{j\ne k}\left[
 \frac{(s_j-r_k)^2}{C+\omega_j-\omega_k}
 +u_j\left(2u_k^2-u_j(s_k+r_k)+r_k^2-s_k^2\right)
 \right].
\end{aligned}
$$

This separates direct coupling ($g_{12}^2$), interference between direct and
mediated paths ($g_{12}G$), and purely mediated coupling ($G^2$). The
anharmonicities remain in the denominators, including the two-excitation coupler
energy $C$; no rotating-wave approximation is used. This fourth-order expression
extends the second-order exchange result cited above. For harmonic modes
($\alpha_1=\alpha_2=\alpha_c=0$), all terms cancel and $\zeta^{(4)}=0$,
as required for a quadratic Hamiltonian.

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
