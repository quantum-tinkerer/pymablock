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

In this tutorial we demonstrate how to get a CQED effective Hamiltonian using Pymablock with bosonic operators.
As an example, we use the Jaynes-Cummings model, which describes a spin coupled to a boson.
This tutorial shows how to use Pymablock with arbitrary object types by defining a custom Sylvester's equation solver.

Let's start by importing the `sympy` functions we need to define the Hamiltonian.
We will make use of `sympy`'s [quantum mechanics module](https://docs.sympy.org/latest/modules/physics/quantum/index.html)
and its [matrices](https://docs.sympy.org/latest/tutorials/intro-tutorial/matrices.html).

```{code-cell} ipython3
import sympy
from sympy import Matrix, Symbol, sqrt, Eq
from sympy.physics.quantum.boson import BosonOp, BosonFockKet
from sympy.physics.quantum import qapply, Dagger
```

## Define a second quantization Hamiltonian

We define the onsite energy $\omega_r$, the energy gap $\omega_q$, the perturbative parameter $g$, and $a$, the bosonic annihilation operator.

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

where the basis corresponds to the two spin states.

## Custom Sylvester's equation solver

To compute perturbative expansions Pymablock needs three things:

- Having the input Hamiltonian and perturbations
- Being able to add and multiply operators
- Being able to solve the Sylvester's equation $[H_0, V] = Y$.

In the current version of Pymablock, solving Sylvester equation with second-quantized operators is not yet directly supported, and it requires either casting the operators to matrices (as done in the [dispersive shift tutorial](dispersive_shift.md)) or defining a custom solver, which we will do here.
Sylvester's equation provides a solution for $V$, the antihermitian part of the unitary transformation that block-diagonalizes the Hamiltonian, and it needs to be solved for each perturbative order.
If the unperturbed Hamiltonian is diagonal, the solution is straightforward:

$$
V_{n,ij} = \frac{Y_{n,ij}}{E_i - E_j}
$$

where $E_i$ and $E_j$ are the diagonal elements of the unperturbed Hamiltonian corresponding to different subspaces.

Therefore, to use Pymablock with second-quantized operators, we define a custom Sylvester's equation solver that takes $Y$ as input and returns $V$.
To compute the energy denominators we only need to count the amount of raising or lowering operators in $Y$ and use this to determine the energy denominators.

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
This Sylvester's solver is specific to the Jaynes-Cummings Hamiltonian.
Using a different CQED Hamiltonian would require adapting
`solve_sylvester` accordingly.
```

## Get the Hamiltonian corrections

We can now define the block-diagonalization routine by calling {autolink}`~pymablock.block_diagonalize`

```{code-cell} ipython3
%%time

from pymablock import block_diagonalize

H_tilde, U, U_adjoint = block_diagonalize(
    [H_0, H_p], solve_sylvester=solve_sylvester, symbols=[g]
)
```

For example, to compute the 2nd order correction of the Hamiltonian of the $\uparrow$ subspace (the `(0, 0)` block) we use

```{code-cell} ipython3
%%time

Eq(Symbol(r'\tilde{H}_{2}^{AA}'), H_tilde[0, 0, 2].expand().simplify(), evaluate=False)
```

Higher order corrections work exactly the same:

```{code-cell} ipython3
%%time

Eq(Symbol(r'\tilde{H}_{4}^{AA}'), H_tilde[0, 0, 4].expand().simplify(), evaluate=False)
```

```{code-cell} ipython3
%%time

Eq(Symbol(r'\tilde{H}_{6}^{AA}'), H_tilde[0, 0, 6].expand().simplify(), evaluate=False)
```

We see that also computing the 6th order correction takes effectively no time.
