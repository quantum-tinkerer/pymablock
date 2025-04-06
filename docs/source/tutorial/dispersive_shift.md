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

# Dispersive shift of a resonator coupled to a transmon

The need for analytical effective Hamiltonians often arises in circuit quantum electrodynamics (cQED) problems.
In this tutorial, we illustrate how to use Pymablock to compute the the frequency shift of a resonator due to its coupling to the qubit, a phenomenon used to measure the qubit's state [^1^].

[^1^]: [Blais et al.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.69.062320)

The Hamiltonian of the system is given by

$$
    \mathcal{H} =
    -\omega_t a^{\dagger}_{t} a_{t}
    + \frac{\alpha}{2} a^{\dagger}_{t} a^{\dagger}_{t} a_{t} a_{t} +
    \omega_r a^{\dagger}_{r} a_{r} -
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
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

from pymablock import block_diagonalize

symbols = sympy.symbols(r"\omega_{t} \omega_{r} \alpha g", real=True, positive=True)
omega_t, omega_r, alpha, g = symbols

a_t, a_r = BosonOp("a_t"), BosonOp("a_r")

H_0 = (
    -omega_t * Dagger(a_t) * a_t + omega_r * Dagger(a_r) * a_r
    + alpha * Dagger(a_t)**2 * a_t**2 / 2
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

display_eq("H_{0}", H_0)
display_eq("H_{p}", H_p)
```

The frequency shift of the resonator is given by:

$$
\chi = \frac{E^{(2)}_{11} - E^{(2)}_{10}}{2} - \frac{E^{(2)}_{01} - E^{(2)}_{00}}{2}
$$

where $E^{(2)}_{ij}$ is the second order correction to the energy of the state with $i$ excitations in the transmon and $j$ in the resonator.
To find the corrections, we may use second quantization directly, or compute the matrix representation of the Hamiltonian in a truncated Hilbert space.
We demonstrate both approaches.

## Approach I: second quantized form

To compute the effective Hamiltonian in second quantized form, we provide the Hamiltonian following Pymablock's API: wrapped in a `sympy.Matrix` and with `BosonOp` elements.

```{code-cell} ipython3
H_tilde, U, U_adjoint = block_diagonalize(
    sympy.Matrix([[H_0 + H_p]]), symbols=[g]
)
```

The matrix has a single element, because we are interested in the corrections to the energy of a single state.

:::{admonition} Only diagonal unperturbed Hamiltonians are supported
:class: warning

Pymablock only supports bosonic Hamiltonians whose unperturbed part is diagonal: diagonal in the matrix representation and diagonal in the bosonic basis.
When calling `block_diagonalize`, the unperturbed Hamiltonian must be provided as a `sympy.Matrix` with `BosonOp` elements in its entries.
:::

The effective Hamiltonian is a $1 \times 1$ matrix, whose entry is a function of the number of excitations in the transmon $N_{a_t} = a_t^\dagger a_t$ and the resonator $N_{a_r} = a_r^\dagger a_r$.

```{code-cell} ipython3
E_eff = H_tilde[0, 0, 2][0, 0]
display_eq("E_{eff}", E_eff)
```

The expression is long, but it becomes simpler if we evaluate it for specific occupation numbers.
Pymablock uses number operators to simplify the expressions that contain bosonic operators throughout the algorithm execution.

To compute the dispersive shift, we need to evaluate $E_{eff}$ for the states with $N_{a_t} = 0, 1$ and $N_{a_r} = 0, 1$.
We do this by first defining the number operators for the transmon and resonator:

```{code-cell} ipython3
from pymablock.second_quantization import NumberOperator

N_a_t = NumberOperator(a_t)
N_a_r = NumberOperator(a_r)
```

and then substituting their values in $E_{eff}$:

```{code-cell} ipython3
E_eff_00 = E_eff.subs({N_a_t: 0, N_a_r: 0})
E_eff_01 = E_eff.subs({N_a_t: 0, N_a_r: 1})
E_eff_10 = E_eff.subs({N_a_t: 1, N_a_r: 0})
E_eff_11 = E_eff.subs({N_a_t: 1, N_a_r: 1})

xi = E_eff_11 - E_eff_10 - E_eff_01 + E_eff_00

display_eq(r"\chi", xi)
```

## Approach II: matrix representation

Alternatively, we can compute the effective Hamiltonian in a matrix representation.
To deal with the infinite dimensional Hilbert space, we observe that the perturbation only changes the occupation numbers of the transmon and the resonator by $\pm 1$.
Therefore computing $n$-th order corrections to the $n_0$-th state allows to disregard states with any occupation numbers larger than $n_0 + n/2$.
We want to compute the second order correction to the levels with occupation numbers of either the transmon or the resonator being $0$ and $1$.
We accordingly truncate the Hilbert space to the lowest 3 levels of the transmon and the resonator.
The resulting Hamiltonian is a $9 \times 9$ matrix, which we construct by computing the matrix elements of $H_0$ and $H_p$ in the truncated basis.

```{code-cell} ipython3
N = 4  # Number of levels for each boson
a = sympy.zeros(N, N)
for i in range(N-1):
    a[i, i+1] = sympy.sqrt(i+1)
n = sympy.diag(*[i for i in range(N)])

a_t = sympy.KroneckerProduct(a, sympy.eye(N))
a_r = sympy.KroneckerProduct(sympy.eye(N), a)

H_0 = (
    -omega_t * Dagger(a_t) * a_t + omega_r * Dagger(a_r) * a_r
    + alpha * Dagger(a_t)**2 * a_t**2 / 2
)
H_p = (
    -g * (Dagger(a_t) - a_t) * (Dagger(a_r) - a_r)
)
H = H_0 + H_p
```

To compute the dispersive shift, we need to compute the energy corrections of the lowest $4$ levels.
Therefore, we call `block_diagonalize` to separate the Hamiltonian into multiple $1 \times 1$ subspaces, replicating a regular perturbation theory calculation for single wavefunctions.
To do this, we observe that $H_0$ is diagonal, and use `subspace_indices` to assign the elements of its eigenbasis to the desired states.

```{code-cell} ipython3
subspaces = {state: n for n, state in enumerate([0, 1, N, N+1])}
subspace_indices = [subspaces.get(state, 4) for state in range(N**2)]
H_tilde, U, U_adjoint = block_diagonalize(
    H, subspace_indices=subspace_indices, symbols=[g]
)
```

Here `subspaces_indices` is a list with integers from $0$ to $4$ that indicate to which subspace each basis element belongs.
The first four elements of `subspaces_indices` correspond to the states $|0 0\rangle$, $|1 0\rangle$, $|0 1\rangle$, and $|1 1\rangle$, and each of them corresponds to a different subspace.
The number $4$ is used to indicate the subspace of the remaining states.
This yields the effective Hamiltonian `H_tilde` with $5$ blocks, all decoupled from each other.

```{code-cell} ipython3
H_tilde.shape
```

Finally, we compute the dispersive shift from the second order correction to the energies

```{code-cell} ipython3
xi = (H_tilde[0, 0, 2] - H_tilde[1, 1, 2] - H_tilde[2, 2, 2] + H_tilde[3, 3, 2])[0, 0]

display_eq(r"\chi", xi)
```

In this example, we have not used the rotating wave approximation, including the frequently omitted counter-rotating terms $\sim a_{r} a_{t}$ to illustrate the extensibility of Pymablock.
Computing higher order corrections to the qubit frequency only requires increasing the size of the truncated Hilbert space and requesting `H_tilde[0, 0, n]` to the desired order $n$.
