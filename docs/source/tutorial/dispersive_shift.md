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

# Dispersive shift

The need for analytical effective Hamiltonians often arises in circuit quantum electrodynamics (cQED) problems.
In this tutorial, we illustrate how to use Pymablock to compute the the frequency shift of a resonator due to its coupling to the qubit, a phenomenon used to measure the qubit's state [^1^].

[^1^]: [Blais et al.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.69.062320)

The Hamiltonian of the system is given by

$$
    \mathcal{H} =
    -\omega_t (a^{\dagger}_{t} a_{t} - \frac{1}{2})
    + \frac{\alpha}{2} a^{\dagger}_{t} a^{\dagger}_{t} a_{t} a_{t} +
    \omega_r (a^{\dagger}_{r} a_{r} + \frac{1}{2}) -
    g (a^{\dagger}_{t} - a_{t}) (a^{\dagger}_{r} - a_{r}),
$$

where $a_t$ and $a_r$ are bosonic annihilation operators of the transmon and resonator, respectively, and $\omega_t$ and $\omega_r$ are their frequencies.
The transmon has an anharmonicity $\alpha$, so that its energy levels are not equally spaced.
In presence of both the coupling $g$ between the transmon and the resonator and the anharmonicity, this Hamiltonian admits no analytical solution.
We therefore treat $g$ as a perturbative parameter.

We start by importing all necessary packages and defining the Hamiltonian.

```{code-cell} ipython3

from itertools import product

import numpy as np
import sympy
from sympy.physics.quantum.operatorordering import normal_ordered_form
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
import matplotlib
import matplotlib.pyplot as plt

from pymablock import block_diagonalize

symbols = sympy.symbols(
    r"\omega_{t} \omega_{r} \alpha g",
    real=True,
    commutative=True,
    positive=True,
)

omega_t, omega_r, alpha, g = symbols

a_t, a_r = BosonOp("a_t"), BosonOp("a_r")

H_0 = -omega_t * (Dagger(a_t) * a_t - sympy.Rational(1) / 2) + omega_r * (
    Dagger(a_r) * a_r + sympy.Rational(1) / 2
)
H_0 += alpha * Dagger(a_t) * Dagger(a_t) * a_t * a_t / sympy.Rational(2)

H_p = (
    -g * a_t * a_r
    - g * Dagger(a_t) * Dagger(a_r)
    + g * a_t * Dagger(a_r)
    + g * Dagger(a_t) * a_r
)
```


```{code-cell} ipython3
:tags: [hide-input]

def display_eq(title, expr):
    """Print a sympy expression as an equality."""
    display(sympy.Eq(sympy.Symbol(title), expr))

display_eq("H_{0}", H_0)
display_eq("H_{p}", H_p)
```

To deal with the infinite dimensional Hilbert space, we observe that the perturbation only changes the occupation numbers of the transmon and the resonator by $\pm 1$.
Therefore computing $n$-th order corrections to the $n_0$-th state allows to disregard states with any occupation numbers larger than $n_0 + n/2$.
We want to compute the second order correction to the levels with occupation numbers of either the transmon or the resonator being $0$ and $1$.
We accordingly truncate the Hilbert space to the lowest 3 levels of the transmon and the resonator.
The resulting Hamiltonian is a $9 \times 9$ matrix, which we construct by computing the matrix elements of $H_0$ and $H_p$ in the truncated basis.

```{code-cell} ipython3
:tags: [hide-input]

def collect_constant(expr):
    expr = normal_ordered_form(expr.expand(), independent=True)
    constant_terms = []
    for term in expr.as_ordered_terms():
        if not term.has(sympy.physics.quantum.Operator):
            constant_terms.append(term)
    return sum(constant_terms)


def to_matrix(ham, basis):
    """Generate the matrix elements"""
    N = len(basis)
    ham = normal_ordered_form(ham.expand(), independent=True)
    all_brakets = product(basis, basis)
    flat_matrix = [
        collect_constant(braket[0] * ham * Dagger(braket[1])) for braket in all_brakets
    ]
    return sympy.Matrix(np.array(flat_matrix).reshape(N, N))
```

```{code-cell} ipython3
# Construct the matrix Hamiltonian
basis = [
    sympy.Rational(1),
    a_t,
    a_r,
    a_t * a_r,
    a_t * a_t / sympy.sqrt(2),
    a_r * a_r / sympy.sqrt(2),
    a_t * a_t * a_r / sympy.sqrt(2),
    a_t * a_r * a_r / sympy.sqrt(2),
    a_t * a_t * a_r * a_r / sympy.Rational(2),
]

H_0_matrix = to_matrix(H_0, basis)
H_p_matrix = to_matrix(H_p, basis)

H = H_0_matrix + H_p_matrix
```
Because the dispersive shift is

$$
\chi = (E^{(2)}_{11} - E^{(2)}_{10}) - (E^{(2)}_{01} - E^{(2)}_{00})
$$

we need to compute the energy corrections of the lowest $4$ levels.
Therefore, we call `block_diagonalize` to separate the Hamiltonian into multiple $1 \times 1$ subspaces, replicating a regular perturbation theory calculation for single wavefunctions.
To do this, we observe that $H_0$ is diagonal, and use `subspace_indices` to assign the elements of its eigenbasis to the desired states.

```{code-cell} ipython3
subspaces = {
    sympy.Rational(1): 0,
    a_t: 1,
    a_r: 2,
    a_t * a_r: 3,
}
subspace_indices = [subspaces.get(element, 4) for element in basis]
H_tilde, U, U_adjoint = block_diagonalize(
    H, subspace_indices=subspace_indices, symbols=[g]
)
```

Finally, we compute the dispersive shift from the second order correction to the energies

```{code-cell} ipython3
xi = (H_tilde[0, 0, 2] - H_tilde[1, 1, 2] - H_tilde[2, 2, 2] + H_tilde[3, 3, 2])[0, 0]

display_eq(r"\chi", xi)
```

In this example, we have not used the rotating wave approximation, including the frequently omitted counter-rotating terms $\sim a_{r} a_{t}$ to illustrate the extensibility of Pymablock.
Computing higher order corrections to the qubit frequency only requires increasing the size of the truncated Hilbert space and requesting `H_tilde[0, 0, n]` to the desired order $n$.
