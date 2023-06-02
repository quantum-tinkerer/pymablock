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

# Jaynes-Cummings model

In this tutorial we demonstrate how to get a CQED effective Hamiltonian using
{{Pymablock}} with bosonic operators.
As an example, we use the Jaynes-Cummings model, which describes a two-level
bosonic system coupled by ladder operators.

Let's start by importing the `sympy` functions we need to define the Hamiltonian.
We will make use of `sympy`'s
[quantum mechanics module](https://docs.sympy.org/latest/modules/physics/quantum/index.html)
and its
[matrices](https://docs.sympy.org/latest/tutorials/intro-tutorial/matrices.html).

```{code-cell} ipython3
import sympy
from sympy import Matrix, Symbol, sqrt, Eq
from sympy.physics.quantum.boson import BosonOp, BosonFockKet
from sympy.physics.quantum import qapply, Dagger
```

## Define a second quantization Hamiltonian

We define the onsite energy $\omega_r$, the energy gap $\omega_q$,
the perturbative parameter $g$, and $a$, the bosonic annihilation
operator.

```{code-cell} ipython3
# resonator frequency, qubit frequency, Rabi coupling
wr = Symbol(r'\omega_r', real=True)
wq = Symbol(r'\omega_q', real=True)
g = Symbol(r'g', real=True)

# resonator photon annihilation operator
a = BosonOp("a")
```

The Hamiltonian reads

```{code-cell} ipython3
H_0 = [[wr * Dagger(a) * a + wq / 2, 0], [0, wr * Dagger(a) * a - wq / 2]]
H_p = [[0,  g * Dagger(a)], [g * a, 0]]

Eq(Symbol('H'), Matrix(H_0) + Matrix(H_p), evaluate=False)
```

where the basis is the one of the occupied and unoccupied subspaces.

## Custom Sylvester's equation solver

To use {{Pymablock}}, we need a custom solver for Sylvester's equation that can
compute the energies of the subspaces using bosonic operators.
We need to define a `solve_sylvester` function that takes $Y_{n+1}$ and returns
$V_{n+1}$,

```{math}
H_0^{AA} V_{n+1}^{AB} - V_{n+1}^{AB} H_0^{BB} = Y_{n+1} \\
(V_{n+1}^{AB})_{x,y} = (Y_{n+1})_{x,y} / (E_x - E_y),
```
where $Y_{n+1}$ is the right hand side of Sylvester's equation, and $V_{n+1}$
is the block off-diagonal block of the transformation that block diagonalizes
the Hamiltonian.

We implement

```{code-cell} ipython3
n = Symbol("n", integer=True, positive=True)
basis_ket = BosonFockKet(n)

def expectation_value(v, operator):
    return qapply(Dagger(v) * operator * v).simplify()

def solve_sylvester(Y):
    """
    Solves Sylvester's Equation
    Y : sympy expression

    Returns:
    V : sympy expression for off-diagonal block of unitary transformation
    """
    E_i = expectation_value(basis_ket, H_0[0][0])
    V = []
    for term in Y.expand().as_ordered_terms():
        term_on_basis = qapply(term * basis_ket).doit()
        normalized_ket = term_on_basis.as_ordered_factors()[-1]
        E_j = expectation_value(normalized_ket, H_0[1][1])
        V.append(term / (E_j - E_i))
    return sum(V)
```

```{important}
Note that this Sylvester's solver is specific to the Jaynes-Cummings Hamiltonian.
Adapting this tutorial to a different CQED Hamiltonian would require adapting
`solve_sylvester` accordingly.
```

## Get the Hamiltonian corrections

We can now set the block-diagonalization routine by defining,

```{code-cell} ipython3
%%time

from pymablock import block_diagonalize

H_tilde, U, U_adjoint = block_diagonalize(
    [H_0, H_p], solve_sylvester=solve_sylvester, symbols=[g]
)
```

where `H_tilde` is the transformed Hamiltonian, `U` is the unitary
transformation, and `U_adjoint` it the conjugate transpose of `U`.

For example, to request the 2nd order correction to the occupied subspace of
the Hamiltonian, you may execute:

```{code-cell} ipython3
%%time

Eq(Symbol(r'\tilde{H}_{2}^{AA}'), H_tilde[0, 0, 2].expand().simplify(), evaluate=False)
```

The result is computed in **less than a second!**
We may also request 4th and 6th order corrections without waiting much longer.

```{code-cell} ipython3
%%time

Eq(Symbol(r'\tilde{H}_{4}^{AA}'), H_tilde[0, 0, 4].expand().simplify(), evaluate=False)
```

```{code-cell} ipython3
%%time

Eq(Symbol(r'\tilde{H}_{6}^{AA}'), H_tilde[0, 0, 6].expand().simplify(), evaluate=False)
```
