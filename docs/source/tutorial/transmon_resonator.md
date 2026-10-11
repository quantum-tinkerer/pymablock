---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# A transmon qubit in a resonator

At low energies, a transmon is an oscillator with a weak negative anharmonicity.
We use its two lowest levels as a qubit, and couple the transmon to a resonator.
What is the effective Hamiltonian of the qubit and the resonator?
The higher transmon levels are not part of the qubit, but they still contribute through virtual transitions.

In this tutorial we keep all levels of the transmon model and of the resonator, without truncating either.
A structured embedding tells Pymablock which states form the qubit.
Pymablock then returns the effective Hamiltonian as an operator in the qubit excitation number and the resonator photon number, with every parameter symbolic.
At second order it reproduces the dispersive shift of Koch et al. [^koch].
At fourth order it gives a resonator Kerr nonlinearity that depends on the qubit state.
When the qubit is excited, this Kerr nonlinearity involves the third excited level of the transmon, which a three-level model of the transmon would miss.

[^koch]: J. Koch et al., [Phys. Rev. A 76, 042319 (2007)](https://doi.org/10.1103/PhysRevA.76.042319).

## The Hamiltonian

We use the same low-energy transmon model as the [dispersive shift tutorial](dispersive_shift.md):

$$
    \mathcal{H} =
    \omega_t a^{\dagger}_{t} a_{t}
    - \frac{\alpha}{2} a^{\dagger}_{t} a^{\dagger}_{t} a_{t} a_{t} +
    \omega_r a^{\dagger}_{r} a_{r} -
    g (a^{\dagger}_{t} - a_{t}) (a^{\dagger}_{r} - a_{r}).
$$

Here $a_t$ and $a_r$ are the bosonic annihilation operators of the transmon and the resonator.
The parameters $\omega_t$ and $\omega_r$ are their frequencies, $-\alpha$ with $\alpha > 0$ is the anharmonicity of the transmon, and $g$ is the coupling, which we treat as the perturbation.
The energy of transmon level $n$ is $E_n = \omega_t n - \alpha n (n - 1) / 2$.
Therefore the qubit transition frequency is $\omega_{01} = E_1 - E_0 = \omega_t$, and the transition from level one to level two has frequency $\omega_{12} = E_2 - E_1 = \omega_t - \alpha$.
We keep the counter-rotating terms $a_t a_r$ and $a_t^\dagger a_r^\dagger$.

```{code-cell} ipython3
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator
from pymablock.second_quantization import Embedding

symbols = sympy.symbols(r"\omega_{t} \omega_{r} \alpha g", real=True, positive=True)
omega_t, omega_r, alpha, g = symbols

a_t, a_r = BosonOp("a_t"), BosonOp("a_r")

H_0 = (
    omega_t * Dagger(a_t) * a_t + omega_r * Dagger(a_r) * a_r
    - alpha * Dagger(a_t)**2 * a_t**2 / 2
)

H_p = (
    -g * (Dagger(a_t) - a_t) * (Dagger(a_r) - a_r)
)
```

```{code-cell} ipython3
:tags: [hide-input]

def display_eq(title, expr):
    """Print a sympy expression as an equality."""
    display(sympy.Eq(sympy.Symbol(title), expr, evaluate=False))
```

## Embedding a qubit in the transmon

The effective model has its own operators: a qubit lowering operator $\sigma_-$ and a resonator annihilation operator $r$.
An {autolink}`~pymablock.operator_embedding.Embedding` defines them through the operators of the full Hamiltonian:

$$
\sigma_- \longmapsto a_t, \qquad r \longmapsto a_r, \qquad
|0\rangle \longmapsto |n_t = 0, n_r = 0\rangle .
$$

The last relation is the reference state: the effective vacuum corresponds to the transmon and the resonator in their ground states.
Applying $\sigma_+$ and $r^\dagger$ to the reference generates the other states of the effective model.
Because $\sigma_+^2 = 0$, these are the transmon levels $n_t = 0, 1$ with any number of photons $n_r$.
The resonator keeps its entire ladder, so it is not truncated.

```{code-cell} ipython3
sigma, r = SigmaMinus("q"), BosonOp("r")
embedding = Embedding({sigma: a_t, r: a_r}, reference={a_t: 0, a_r: 0})
```

The embedding only selects states; it does not change the full Hamiltonian.
Acting with $a_t^\dagger$ on transmon level one still gives level two, which lies outside the qubit.
Pymablock treats all such states as virtual states of perturbation theory, so every transmon level can contribute.

Restricted to the qubit states, the coupling becomes a qubit–resonator interaction, including its counter-rotating terms:

```{code-cell} ipython3
display_eq("H_{p}", embedding.restrict(H_p).as_expr())
```

## The dispersive Hamiltonian

We pass the embedding to {autolink}`~pymablock.block_diagonalize` as `subspace_eigenvectors`.
Block 0 of the result is the effective model, and block 1 contains all other states.
By default Pymablock only removes the couplings between the two blocks, so the effective model would keep this interaction.
To obtain the dispersive Hamiltonian, we also eliminate all off-diagonal terms inside the effective model with `fully_diagonalize=(0,)`.

```{code-cell} ipython3
H_tilde, U, U_adjoint = block_diagonalize(
    H_0 + H_p, subspace_eigenvectors=embedding, symbols=[g],
    fully_diagonalize=(0,),
)
```

The result is diagonal: a function of the qubit excitation number $N_q = \sigma_+ \sigma_-$, which is either 0 or 1, and the photon number $N_r = r^\dagger r$.
The zeroth order is the restricted $H_0$, and the second-order correction is

```{code-cell} ipython3
N_q, N_r = NumberOperator(sigma), NumberOperator(r)

display_eq("H_{0}", H_tilde[0, 0, 0].as_expr())
display_eq(r"H^{(2)}_{\textrm{eff}}", H_tilde[0, 0, 2].as_expr())
```

The denominators $\omega_t - \alpha \pm \omega_r$ are the energies $\omega_{12} \pm \omega_r$ of virtual transitions from transmon level one to level two.
Even though level two is not part of the qubit, it enters the effective model.

To read off the dispersive shift, we evaluate the diagonal effective Hamiltonian on the state with qubit excitation $n_q$ and $n_r$ photons:

```{code-cell} ipython3
def energy(term, n_q, n_r):
    """Value of a diagonal effective Hamiltonian term on the state |n_q, n_r>."""
    return term.subs({N_q: n_q, N_r: n_r}).as_expr()
```

To second order the effective Hamiltonian is linear in $N_r$.
Its term $2\chi N_q N_r$ equals $\chi \sigma_z N_r$, up to a shift of the resonator frequency, where $\sigma_z = 2 N_q - 1$.
This is the convention of Koch et al. for the dispersive shift $\chi$.
Hence $2\chi = E_{11} - E_{10} - E_{01} + E_{00}$, where $E_{n_q n_r}$ is the energy of the state with $n_q$ qubit excitations and $n_r$ photons.
The dispersive shift tutorial computes the same combination.

Koch et al. derive $\chi = \chi_{01} - \chi_{12}/2$, with $\chi_{ij} = g_{ij}^2 / (\omega_{ij} - \omega_r)$, within the rotating-wave approximation.
Here $g_{01} = g$ and $g_{12} = \sqrt{2} g$ are the coupling matrix elements of the two transitions.
The counter-rotating terms add the same expression with $\omega_r \to -\omega_r$, so

$$
\chi = \sum_{\pm} g^2 \left( \frac{1}{\omega_{01} \mp \omega_r} - \frac{1}{\omega_{12} \mp \omega_r} \right).
$$

```{code-cell} ipython3
H_2 = H_tilde[0, 0, 2]
two_chi = energy(H_2, 1, 1) - energy(H_2, 1, 0) - energy(H_2, 0, 1) + energy(H_2, 0, 0)

omega_01, omega_12 = omega_t, omega_t - alpha
chi_koch = sum(
    g**2 * (1 / (omega_01 - sign * omega_r) - 1 / (omega_12 - sign * omega_r))
    for sign in (1, -1)
)
assert sympy.simplify(two_chi - 2 * chi_koch) == 0

display_eq(r"\chi", sympy.factor(two_chi / 2))
```

The dispersive shift is proportional to $\alpha$: in a harmonic transmon, the contributions of the two transitions cancel.

## Photon-number dependence at fourth order

At fourth order the effective Hamiltonian contains $N_r^2$, so the resonator acquires a Kerr nonlinearity.
We write the photon-number dependence for qubit state $n_q$ as $\frac{K_{n_q}}{2} N_r (N_r - 1)$ plus terms linear in $N_r$.
The Kerr coefficient $K_{n_q}$ is then the second difference of the energy in $n_r$.
A Kerr coefficient that depends on the qubit state also makes the dispersive shift depend on the photon number: the qubit frequency $E_{1 n_r} - E_{0 n_r}$ has curvature $K_1 - K_0$ in $n_r$.

```{code-cell} ipython3
H_4 = H_tilde[0, 0, 4]
K = [
    sympy.factor(energy(H_4, n_q, 2) - 2 * energy(H_4, n_q, 1) + energy(H_4, n_q, 0))
    for n_q in (0, 1)
]
display_eq("K_{0}", K[0])
```

Like the dispersive shift, the Kerr nonlinearity vanishes when $\alpha = 0$, since two coupled harmonic oscillators remain harmonic.

The expression for $K_1$ is longer, so we show only its denominator.
Each factor is the energy of a virtual transition:

```{code-cell} ipython3
display_eq(r"\mathrm{denominator}(K_{1})", sympy.denom(K[1]))
```

Up to sign, the factors $3\alpha - 2\omega_t$ and $3\alpha - 2\omega_t \pm 2\omega_r$ are the transmon energy $E_3 - E_1 = 2\omega_t - 3\alpha$, plus a change of the resonator energy by $0$ or $\pm 2\omega_r$.
A fourth-order process that starts from the excited qubit can pass through the third excited level of the transmon.
A process that starts from the qubit ground state reaches only level two, so $K_0$ has no such factor:

```{code-cell} ipython3
level_3_resonance = {alpha: 2 * omega_t / 3}  # Where E_3 = E_1.
assert sympy.denom(K[1]).subs(level_3_resonance) == 0
assert sympy.denom(K[0]).subs(level_3_resonance) != 0
```

A matrix calculation that obtains $K_1$ therefore needs at least four transmon levels and, for each photon number used to extract the curvature, two more resonator levels.
It also needs several photon numbers to fit the polynomial in $n_r$.
The embedding returns the dependence on $N_r$ as a single operator, and no truncation needs to be chosen.

## Keeping the exchange coupling

Without `fully_diagonalize`, Pymablock only eliminates the transmon levels outside the qubit.
The effective model then keeps the qubit–resonator exchange interaction at first order, which is useful when the qubit is close to resonance with the resonator.
The second-order correction now contains only the virtual transitions through transmon level two:

```{code-cell} ipython3
H_tilde_exchange, *_ = block_diagonalize(
    H_0 + H_p, subspace_eigenvectors=embedding, symbols=[g]
)
display_eq(r"H^{(1)}_{\textrm{eff}}", H_tilde_exchange[0, 0, 1].as_expr())
display_eq(r"H^{(2)}_{\textrm{eff}}", H_tilde_exchange[0, 0, 2].simplify().as_expr())
```

These corrections act only on the excited qubit state.
Besides shifting its energy, they contain two-photon terms $N_q r^2$ and $N_q r^{\dagger 2}$: the excited qubit can emit or absorb two photons through transmon level two.

## Summary

A single embedding $\sigma_- \mapsto a_t$, $r \mapsto a_r$ turns the transmon into a qubit while keeping all its higher levels available as virtual states.
The resulting effective Hamiltonian is an operator in $N_q$ and $N_r$, valid for every photon number and symbolic in all parameters.
The [dispersive shift tutorial](dispersive_shift.md) computes the same energies either as a function of the transmon occupation, or with a truncated matrix.
The embedding instead returns the result directly in qubit operators, and it allows choosing which couplings to eliminate: all of them for the dispersive Hamiltonian, or only those to the higher transmon levels.
See [structured embeddings](../structured_embeddings.md) for other embeddings and their requirements.
