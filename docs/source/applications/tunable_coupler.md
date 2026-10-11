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

How does a tunable coupler switch the interaction between two transmon qubits
on and off? The effective model keeps two levels of each qubit:

$$
H_{\rm eff} = \tilde\omega_1 N_{q_1} + \tilde\omega_2 N_{q_2}
+ J\left(q_1^\dagger q_2 + q_2^\dagger q_1\right) + \zeta N_{q_1}N_{q_2}.
$$

The exchange coupling $J$ combines a direct term and a term mediated by the
coupler,

$$
J = -g_{12} + \frac{g_{1c}g_{2c}}{2}\sum_{j=1,2}
\left(\frac{1}{\omega_j-\omega_c}-\frac{1}{\omega_j+\omega_c}\right),
$$

so tuning the coupler frequency $\omega_c$ can make $J$ vanish.[^yan] We also
compute the residual $ZZ$ coupling $\zeta$ to fourth order, including all
counter-rotating processes.

## Model

The two qubits and the coupler are weakly anharmonic oscillators, $j=1,2,c$,
with frequencies $\omega_j$ and anharmonicities $\alpha_j$:

$$
H_0 = \sum_{j=1,2,c}\left[\omega_j N_j + \frac{\alpha_j}{2}N_j(N_j-1)\right].
$$

They couple through their charges, which are proportional to
$x_j=a_j^\dagger-a_j$:

$$
V_1 = g_{1c}\,x_1x_c + g_{2c}\,x_2x_c,\qquad V_2 = g_{12}\,x_1x_2.
$$

The direct coupling $g_{12}$ is much weaker than $g_{1c}$ and $g_{2c}$. We
therefore count $V_1$ as first order and $V_2$ as second order, so that the
direct and the mediated exchange appear at the same order.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from IPython.display import display
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding
```

```{code-cell} ipython3
omega_1, omega_2, omega_c = sp.symbols("omega_1 omega_2 omega_c", positive=True)
alpha_1, alpha_2, alpha_c = sp.symbols("alpha_1 alpha_2 alpha_c", real=True, nonzero=True)
g_12, g_1c, g_2c = sp.symbols("g_12 g_1c g_2c", real=True)
a_1, a_2, a_c = BosonOp("a_1"), BosonOp("a_2"), BosonOp("a_c")

H_0 = (
    omega_1 * N(a_1) + alpha_1 / 2 * N(a_1) * (N(a_1) - 1)
    + omega_2 * N(a_2) + alpha_2 / 2 * N(a_2) * (N(a_2) - 1)
    + omega_c * N(a_c) + alpha_c / 2 * N(a_c) * (N(a_c) - 1)
)
x_1, x_2, x_c = Dagger(a_1) - a_1, Dagger(a_2) - a_2, Dagger(a_c) - a_c
V_1 = g_1c * x_1 * x_c + g_2c * x_2 * x_c
V_2 = g_12 * x_1 * x_2
```

## What is kept

We keep the ground and first excited states of each qubit, with the coupler
in its ground state. The two-level lowering operator $q_j$ then represents
$a_j$ on these states. Higher qubit levels and coupler excitations appear only
in virtual processes.

```{code-cell} ipython3
q_1, q_2 = SigmaMinus("q_1"), SigmaMinus("q_2")
embedding = Embedding({q_1: a_1, q_2: a_2}, reference={a_1: 0, a_2: 0, a_c: 0})
```

## Perturbation theory

The effective Hamiltonian is a series in the coupling strength. Exchange
appears at second order and $ZZ$ coupling at fourth order.

```{code-cell} ipython3
H_eff, *_ = block_diagonalize(
    {(0,): H_0, (1,): V_1, (2,): V_2}, subspace_eigenvectors=embedding
)
H_2, H_4 = H_eff[0, 0, 2], H_eff[0, 0, 4]
```

## Result and check

### Exchange

The exchange coupling $J$ is the coefficient of $q_2^\dagger q_1$ in the
second-order Hamiltonian. It reproduces Eq. (33) of Yan et al.,[^yan] whose
direct coupling is $g_{12}^{\rm Yan}=-g_{12}$.

```{code-cell} ipython3
J = H_2.filter_terms(((1, -1),), keep=True).as_expr().coeff(Dagger(q_2) * q_1)
display(sp.Eq(sp.Symbol("J"), J))

J_yan = -g_12 + g_1c * g_2c / 2 * (
    1 / (omega_1 - omega_c) - 1 / (omega_1 + omega_c)
    + 1 / (omega_2 - omega_c) - 1 / (omega_2 + omega_c)
)
assert sp.simplify(J - J_yan) == 0
```

The terms with $\omega_j+\omega_c$ come from counter-rotating processes, which
create or destroy two excitations at once.

### ZZ coupling

The $ZZ$ coupling is the energy of the doubly excited state minus the energies
of the singly excited states, $\zeta=E_{11}-E_{10}-E_{01}+E_{00}$. In terms of
Pauli operators $Z_j=1-2N_{q_j}$, it adds $(\zeta/4)Z_1Z_2$ to the Hamiltonian.

```{code-cell} ipython3
diagonal = H_4.filter_terms(((0, 0),), keep=True).as_expr()

def E(n_1, n_2):
    return diagonal.subs({N(q_1): n_1, N(q_2): n_2})

zeta = E(1, 1) - E(1, 0) - E(0, 1) + E(0, 0)
```

Two limits check this result. Without the coupler, only the direct coupling
remains. Summing its virtual transitions from $|11\rangle$, $|10\rangle$,
$|01\rangle$ and $|00\rangle$ gives

$$
\zeta_{\rm direct} = 2g_{12}^2\left[
\frac{1}{\omega_2-\omega_1-\alpha_1}+\frac{1}{\omega_1-\omega_2-\alpha_2}
+\frac{1}{\omega_1+\omega_2+\alpha_1}+\frac{1}{\omega_1+\omega_2+\alpha_2}
-\frac{2}{\omega_1+\omega_2+\alpha_1+\alpha_2}\right].
$$

The first two terms are the familiar rotating-wave result
$2g_{12}^2(\alpha_1+\alpha_2)/[(\Delta+\alpha_1)(\Delta-\alpha_2)]$ with
$\Delta=\omega_1-\omega_2$. Second, coupled harmonic oscillators have no $ZZ$
coupling, so $\zeta$ must vanish when all anharmonicities are zero. We check
this at one choice of nonresonant frequencies, keeping the couplings symbolic.

```{code-cell} ipython3
zeta_direct = 2 * g_12**2 * (
    1 / (omega_2 - omega_1 - alpha_1) + 1 / (omega_1 - omega_2 - alpha_2)
    + 1 / (omega_1 + omega_2 + alpha_1) + 1 / (omega_1 + omega_2 + alpha_2)
    - 2 / (omega_1 + omega_2 + alpha_1 + alpha_2)
)
assert sp.simplify(zeta.subs({g_1c: 0, g_2c: 0}) - zeta_direct) == 0

harmonic = {alpha_1: 0, alpha_2: 0, alpha_c: 0, omega_1: 5, omega_2: 7, omega_c: 11}
assert sp.simplify(zeta.subs(harmonic)) == 0
```

The full $\zeta$ has dozens of terms, so we plot it instead of printing it.
The parameters are illustrative values of $\omega/2\pi$ in GHz:
$\omega_1=4.0$, $\omega_2=4.1$, all $\alpha_j=-0.2$, $g_{1c}=g_{2c}=0.1$ and
$g_{12}=-0.007$.

```{code-cell} ipython3
parameters = {omega_1: 4.0, omega_2: 4.1, alpha_1: -0.2, alpha_2: -0.2,
              alpha_c: -0.2, g_1c: 0.1, g_2c: 0.1, g_12: -0.007}
J_of_wc = sp.lambdify(omega_c, J.subs(parameters))
zeta_of_wc = sp.lambdify(omega_c, zeta.subs(parameters))

wc = np.linspace(4.7, 7.5, 200)
fig, ax = plt.subplots()
ax.plot(wc, 1e3 * J_of_wc(wc), label=r"$J$")
ax.plot(wc, 1e3 * zeta_of_wc(wc), label=r"$\zeta$")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel(r"$\omega_c$ (GHz)")
ax.set_ylabel("coupling (MHz)")
ax.legend();
```

## Discussion

Above the qubit frequencies, the mediated exchange is negative and decreases
in magnitude as the coupler moves away. It cancels the direct exchange at one
coupler frequency, which switches the qubit interaction off. The $ZZ$ coupling
requires anharmonicity and is much smaller than $J$ over most of this range.
For these parameters it is also smallest close to the point where $J$
vanishes. Near a resonance between a qubit state and a coupler or higher qubit
level, that level must be kept in the effective model instead of being
eliminated.

[^yan]: F. Yan et al.,
    [Tunable coupling scheme for implementing high-fidelity two-qubit gates](https://doi.org/10.1103/PhysRevApplied.10.054062),
    Physical Review Applied **10**, 054062 (2018).
    [Open preprint](https://arxiv.org/abs/1803.09813).
