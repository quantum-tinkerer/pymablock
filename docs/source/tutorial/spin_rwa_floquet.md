---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Rotating wave approximation in Floquet Formalism

In this tutorial, we demonstrate how to apply the rotating wave approximation (RWA) to a spin system under periodic driving using the Floquet formalism in Pymablock.

## Time-Dependent Hamiltonian

We start with a time-dependent Hamiltonian for a spin-1/2 system under periodic driving:

$$H(t) = \frac{\omega_0}{2}\sigma_z + g \sigma_x \cos(\Omega t)$$

Decomposing the cosine term into exponentials:

$$\cos(\Omega t) = \frac{e^{i\Omega t} + e^{-i\Omega t}}{2}$$

gives us:

$$H(t) = \frac{\omega_0}{2}\sigma_z + \frac{g}{2} \sigma_x (e^{i\Omega t} + e^{-i\Omega t})$$

## Floquet Hamiltonian with Ladder Operators

In the Floquet formalism, we represent the exponential terms using ladder operators. These follow the commutation relations described in the [Second Quantization Tools](../second_quantization.md#ladder-operators) section.

Fourier transforming the Hamiltonian leads to a Floquet Hamiltonian of the form:
The Floquet Hamiltonian as an infinite matrix is:

$$H_{Floquet} = \begin{pmatrix}
\ddots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
\cdots & H_0 - 2\Omega & V_+ & 0 & 0 & 0 & \cdots \\
\cdots & V_- & H_0 - \Omega & V_+ & 0 & 0 & \cdots \\
\cdots & 0 & V_- & H_0 & V_+ & 0 & \cdots \\
\cdots & 0 & 0 & V_- & H_0 + \Omega & V_+ & \cdots \\
\cdots & 0 & 0 & 0 & V_- & H_0 + 2\Omega & \cdots \\
& \vdots & \vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix},$$

with $H_0$ and $V_\pm$ $2 \times 2$ matrices.
By introducing the ladder operator $a$ and its number operator $N_a$ that satisfy:

$$
a = \begin{pmatrix}
\ddots & \vdots & \vdots & \vdots & \vdots & \vdots \\
\cdots & 0 & 1 & 0 & 0 & 0 & \cdots \\
\cdots & 0 & 0 & 1 & 0 & 0 & \cdots \\
\cdots & 0 & 0 & 0 & 1 & 0 & \cdots \\
\cdots & 0 & 0 & 0 & 0 & 1 & \cdots \\
\cdots & 0 & 0 & 0 & 0 & 0 & \cdots \\
& \vdots & \vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}, \quad
N_a = \begin{pmatrix}
\ddots & \vdots & \vdots & \vdots & \vdots & \vdots \\
\cdots & -2 & 0 & 0 & 0 & 0 & \cdots \\
\cdots & 0 & -1 & 0 & 0 & 0 & \cdots \\
\cdots & 0 & 0 & 0 & 0 & 0 & \cdots \\
\cdots & 0 & 0 & 0 & 1 & 0 & \cdots \\
\cdots & 0 & 0 & 0 & 0 & 2 & \cdots \\
& \vdots & \vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix},
$$

we arrive at the second-quantized form of the Floquet Hamiltonian:

$$
H_{Floquet} = \frac{\omega_0}{2} \sigma_z + \Omega N_a + \frac{g}{2} \sigma_x (a + a^\dagger).$$

Let us now implement this in Pymablock using {autolink}`~pymablock.number_ordered_form.LadderOp` and {autolink}`~pymablock.number_ordered_form.NumberOperator`.

```{code-cell} ipython3
from IPython.display import display
import numpy as np
from sympy import Matrix, Symbol, symbols, Eq, simplify, cancel, I
from sympy.physics.quantum import Dagger
from sympy.physics.quantum import pauli
from pymablock.block_diagonalization import block_diagonalize
from pymablock.number_ordered_form import LadderOp, NumberOperator

# Helper function to display equations nicely
def display_eq(title, value):
    display(Eq(Symbol(title), value, evaluate=False))

# System parameters
omega_0, Omega, g = symbols('omega_0 Omega g', real=True)

# Define ladder operators for the Floquet modes
a = LadderOp("a")
N_a = NumberOperator(a)  # Note: Dagger(a) * a = 1 for ladder ops

# Define the Pauli matrices
sigma_x = pauli.SigmaX("s")
sigma_z = pauli.SigmaZ("s")

# Define the Floquet Hamiltonian
H_0 = omega_0/2 * sigma_z + Omega * N_a
H_p = g/2 * sigma_x * (a + Dagger(a))

# Display the full Hamiltonian
display_eq('H_{Floquet}', H_0 + H_p)
```

## Full Perturbation Theory

First, let's perform block diagonalization without any term elimination to see the full perturbative expansion:

```{code-cell} ipython3
# Apply full block diagonalization without RWA
H_full, U_full, U_adjoint_full = block_diagonalize(
    [H_0, H_p],
    symbols=[g],
)

# Examine different orders of the effective Hamiltonian
display_eq('H_{eff}^{(0)}', H_full[0, 0, 0])
display_eq('H_{eff}^{(1)}', H_full[0, 0, 1])
display_eq('H_{eff}^{(2)}', H_full[0, 0, 2])
```

The first order correction is zero: the perturbation is off-diagonal and therefore we eliminated it completely.
The second order correction is non-zero and contains contributions from both the rotating and counter-rotating terms with energy denominators $\omega \pm \Omega$.

We see that the result contains $N_s \equiv (\sigma_z^{(s)} + 1) / 2$, the number operator for the spin.
This is a consequence of Pymablock using the number ordered form to perform the calculations, but we can also substitute the original Pauli operators back in:

```{code-cell} ipython3
display_eq('H_{eff}^{(2)}', simplify(H_full[0, 0, 2]).doit().expand())
```

## Applying the Rotating Wave Approximation

Pymablock offers two ways to apply the rotating wave approximation (RWA): either by separating the perturbation into different terms or by specifying which terms to eliminate.
Which method to use in general depends on the specific problem at hand.

### Specifying what to eliminate

The rotating wave approximation involves eliminating rapidly oscillating terms. We can achieve this by only eliminating some of the terms in the perturbation expansion:

```{code-cell} ipython3
# Create elimination rules matrix for RWA
k = Symbol('k', integer=True, positive=True)
sigma_plus = pauli.SigmaPlus("s")
to_eliminate = sigma_plus * a + Dagger(sigma_plus) * Dagger(a) + sigma_z * (a**k + Dagger(a)**k)
```

This eliminates:

- The terms $\sigma_+ a$ and $\sigma_- a^\dagger$ (counter-rotating terms)
- The terms with any power of the ladder operator on the diagonal.

The latter is needed to avoid generating terms like $\sigma_z (a^2 + a^{\dagger 2})$.

```{code-cell} ipython3
# Apply block diagonalization with RWA filtering
H_rwa, U_rwa, U_adjoint_rwa = block_diagonalize(
    [H_0, H_p],
    symbols=[g],
    fully_diagonalize=to_eliminate,
)

# Examine different orders of the effective Hamiltonian with RWA
display_eq('H_{RWA}^{(1)}', H_rwa[0, 0, 1])
display_eq('H_{RWA}^{(2)}', simplify(H_rwa[0, 0, 2]).doit().expand())
```

The first order effective Hamiltonian is now non-zero because we only eliminate the counter-rotating terms.
On the other hand, the second order effective Hamiltonian now only has contributions from the near-resonant terms with energy denominators $\omega - \Omega$.

### Separating perturbation terms

Instead of treating the entire interaction $H_p = \frac{g}{2} \sigma_x (a + a^\dagger)$ as a single perturbation, we can separate it into co-rotating and counter-rotating terms.
The co-rotating terms are those that create excitations plus their Hermitian conjugate ($\sigma_+ a^\dagger + \sigma_- a$), while the counter-rotating terms are the rest ($\sigma_+ a + \sigma_- a^\dagger$).
This allows us to treat each part independently and specify different orders for each perturbation.

```{code-cell} ipython3
# Separate the perturbation into co-rotating and counter-rotating parts
sigma_plus = pauli.SigmaPlus("s")
sigma_minus = pauli.SigmaMinus("s")

# Co-rotating terms: σ₊a† + σ₋a (terms that create excitations + h.c.)
H_co = g/4 * (sigma_plus * Dagger(a) + sigma_minus * a)

# Counter-rotating terms: σ₊a + σ₋a† (the rest)
H_counter = g/4 * (sigma_plus * a + sigma_minus * Dagger(a))

# Verify that H_co + H_counter = H_p
display_eq('H_{co}', H_co)
display_eq('H_{counter}', H_counter)
```

Now we can perform block diagonalization with two separate perturbative parameters.
The order specification $(i, j)$ corresponds to $g_{co}^i \cdot g_{counter}^j$:

```{code-cell} ipython3
H_sep, U_sep, U_adjoint_sep = block_diagonalize(
    [H_0, H_co, H_counter],
)

# Examine terms with different orders in co- and counter-rotating terms
# (1,0): First order in co-rotating, zero order in counter-rotating
display_eq('H_{eff}^{(1,0)}', H_sep[0, 0, 1, 0])

# (0,1): Zero order in co-rotating, first order in counter-rotating
display_eq('H_{eff}^{(0,1)}', H_sep[0, 0, 0, 1])

# (2,0): Second order in co-rotating only
display_eq('H_{eff}^{(2,0)}', H_sep[0, 0, 2, 0])

# (0,2): Second order in counter-rotating only
display_eq('H_{eff}^{(0,2)}', H_sep[0, 0, 0, 2])

# (1,1): First order in both (cross term)
display_eq('H_{eff}^{(1,1)}', H_sep[0, 0, 1, 1])
```

This separation allows us to identify which terms arise from purely co-rotating interactions, purely counter-rotating interactions, or cross-terms between them.
As expected, all first-order terms are zero since both perturbations are off-diagonal.

## Conclusion

This tutorial demonstrates two key concepts:

1. **Floquet formalism**: Using ladder operators to represent the time-dependent Hamiltonian in a second-quantized form
2. **Selective diagonalization**: Using elimination rules to eliminate only some of the perturbation terms, specifically the counter-rotating terms in our case.
