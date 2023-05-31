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
_Pymablock_ with bosonic operators.
As an example, we use the Jaynes-Cummings model, which describes a two-level
bosonic system coupled by ladder operators.

Let's start by importing the `sympy` functions we need to define the Hamiltonian. We will make use of `sympy`'s [quantum mechanics module](https://docs.sympy.org/latest/modules/physics/quantum/index.html) and [matrices](https://docs.sympy.org/latest/tutorials/intro-tutorial/matrices.html)

```{code-cell} ipython3
import sympy
from sympy import Matrix, Symbol, sqrt
from sympy.physics.quantum.boson import BosonOp, BosonFockKet
from sympy.physics.quantum import qapply, Dagger
```

## Define the Hamiltonian

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

Matrix(H_0) + Matrix(H_p)
```

where the basis is the one of the occupied and unoccupied subspaces.

## Custom Sylvester's equation solver

To use _Pymablock_, we need a custom solver for Sylvester's equation that can
compute the energies of the subspaces using bosonic operators.
We need to define a `solve_sylvester` function that takes $Y$, the right hand side containing only lower orders of the expansion, and returns
$V$, the off-diagonal block of the block diagonalizing transformation of the next higher order, such that

```{math}
H_0^{AA} V_{n+1}^{AB} - V_{n+1}^{AB} H_0^{BB} = Y_{n+1} \\
(V_{n+1}^{AB})_{x,y} = (Y_{n+1})_{x,y} / (E_x - E_y).
```

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

## Perturbative results

We can now set the block-diagonalization routine by defining,

```{code-cell} ipython3
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
H_tilde[0, 0, 2].expand().simplify()
```

The 4th order correction is

```{code-cell} ipython3
H_tilde[0, 0, 4].expand().simplify()
```
