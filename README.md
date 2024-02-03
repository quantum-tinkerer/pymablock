# Pymablock: quasi-degenerate perturbation theory in Python

`Pymablock` (Python matrix block-diagonalization) is a Python package that constructs
effective models using quasi-degenerate perturbation theory.
It handles both numerical and symbolic inputs, and it efficiently
block-diagonalizes Hamiltonians with multivariate perturbations to arbitrary
order.

Building an effective model using Pymablock is a three step process:

* Define a Hamiltonian
* Call `pymablock.block_diagonalize`
* Request the desired order of the effective Hamiltonian

```python
from pymablock import block_diagonalize

# Define perturbation theory
H_tilde, *_ = block_diagonalize([h_0, h_p], subspace_eigenvectors=[vecs_A, vecs_B])

# Request correction to the effective Hamiltonian
H_AA_4 = H_tilde[0, 0, 4]
```

Here is why you should use Pymablock:

* Do not reinvent the wheel

  Pymablock provides a tested reference implementation

* Apply to any problem

  Pymablock supports `numpy` arrays, `scipy` sparse arrays, `sympy` matrices and
  quantum operators

* Speed up your code

  Due to several optimizations, Pymablock can reliably handle both higher orders
  and large Hamiltonians

For more details see the Pymablock [documentation](https://pymablock.readthedocs.io/en/latest/).
