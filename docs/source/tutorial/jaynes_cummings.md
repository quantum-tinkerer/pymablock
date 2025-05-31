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
This tutorial shows how to use Pymablock with second-quantized operators, without the need to transform the Hamiltonian to a matrix representation.

Let's start by importing the `sympy` functions we need to define the Hamiltonian.
We will make use of `sympy`'s [quantum mechanics module](https://docs.sympy.org/latest/modules/physics/quantum/index.html)
and its [matrices](https://docs.sympy.org/latest/tutorials/intro-tutorial/matrices.html).

```{code-cell} ipython3
from sympy import Matrix, Symbol, symbols, Eq, simplify
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum import Dagger
```

## Define a second quantization Hamiltonian

We define the onsite energy $\omega_r$, the energy gap $\omega_q$, the perturbative parameter $g$, and $a$, the bosonic annihilation operator.

```{code-cell} ipython3
# resonator frequency, qubit frequency, Rabi coupling
wr, wq, g = symbols(r'\omega_r \omega_q g', real=True)

# resonator photon annihilation operator
a = BosonOp("a")
```

The Hamiltonian reads

```{code-cell} ipython3
H_0 = Matrix([[wr * Dagger(a) * a + wq / 2, 0], [0, wr * Dagger(a) * a - wq / 2]])
H_p = Matrix([[0,  g * a], [g * Dagger(a), 0]])

Eq(Symbol('H'), H_0 + H_p, evaluate=False)
```

where the basis corresponds to the two spin states.

## Get the Hamiltonian corrections

We can now define the block-diagonalization routine by calling {autolink}`~pymablock.block_diagonalize`

```{code-cell} ipython3
%%time

import numpy as np
from pymablock import block_diagonalize

H_tilde, U, U_adjoint = block_diagonalize([H_0, H_p], symbols=[g])
```

The function {autolink}`block_diagonalize` takes the Hamiltonian and the perturbative parameter as input.
Differently from the rest of the tutorials, here we do not provide `susbpace_vectors` or `subspace_indices`.
Pymablock treats the Hamiltonian as a **single block**, where the goal is to remove all terms that are not diagonal.
The output therefore is a $2 \times 2$ diagonal Hamiltonian that only contains one block with number operators.

```{note}
Pymablock only supports diagonal unperturbed Hamiltonians when using bosonic operators.
This means that $H_0$ must be block-diagonal and its entries need to be convertible to functions of the number operator, without single boson terms.
This limitation may be lifted using advanced functionality, by providing a custom `solve_sylvester` input to the {autolink}`block_diagonalize` function.
```

For example, to compute the 2nd order correction of the Hamiltonian of the $↑, ↓$ subspaces we use

```{code-cell} ipython3
%%time

Eq(Symbol(r'\tilde{H}_2'), H_tilde[0, 0, 2], evaluate=False)
```

:::{admonition} Pymablock's number operator
:class: dropdown

Pymablock's output contains $N_a = a^\dagger a$, the number operator for the bosonic mode, which is a {autolink}`~pymablock.number_ordered_form.NumberOperator` object.
Furthermore, the output is stored in the {autolink}`~pymablock.number_ordered_form.NumberOrderedForm` class that Pymablock uses for efficient manipulation of second quantized expressions.
In the example above, the diagonal entries of the Hamiltonian are {autolink}`~pymablock.number_ordered_form.NumberOrderedForm` objects.

To see the result in terms of individual bosonic operators, we may use the `doit` method:

```python

simplify(H_tilde[0, 0, 2])[0, 0].doit()
```

Check out the [documentation](../second_quantization.md) for more information on how to use number operators and simplify expressions that contain them.

:::

Higher order corrections to the Hamiltonian work exactly the same:

```{code-cell} ipython3
%%time

Eq(Symbol(r'\tilde{H}_4'), H_tilde[0, 0, 4], evaluate=False)
```

```{code-cell} ipython3
%%time

Eq(Symbol(r'\tilde{H}_6'), H_tilde[0, 0, 6], evaluate=False)
```

We see that also computing the 6th order correction takes effectively no time.

## Spin operators

Instead of defining the Hamiltonian as a 2x2 matrix, we can use the spin operators.

```{code-cell} ipython3
from sympy.physics.quantum import pauli

H_0 = wr * Dagger(a) * a + wq * pauli.SigmaZ("s") / 2
H_p = g * (pauli.SigmaPlus("s") * a + pauli.SigmaMinus("s") * Dagger(a))

H_tilde, *_ = block_diagonalize([H_0, H_p], symbols=[g])

Eq(Symbol(r'\tilde{H}_4'), simplify(H_tilde[0, 0, 4]), evaluate=False)
```

This gives the same result (here $N_s$ is the number operator for the spin).
