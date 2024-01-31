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

```{toctree}
:hidden:
:maxdepth: 4
:caption: Contents

tutorial/tutorial.md
algorithms.md
documentation/pymablock.rst
CHANGELOG.md
authors.md
```

# Pymablock

## What is Pymablock?

Pymablock (Python matrix block-diagonalization) is a Python package that constructs
effective models using quasi-degenerate perturbation theory.
It handles both numerical and symbolic inputs, and it efficiently
block-diagonalizes Hamiltonians with multivariate perturbations to arbitrary
order.

Building an effective model using Pymablock is a three step process:
* Define a Hamiltonian
* Call {autolink}`~pymablock.block_diagonalize`
* Request the desired order of the effective Hamiltonian

```python
from pymablock import block_diagonalize

# Define perturbation theory
H_tilde, *_ = block_diagonalize([h_0, h_p], subspace_eigenvectors=[vecs_A, vecs_B])

# Request correction to the effective Hamiltonian
H_AA_4 = H_tilde[0, 0, 4]
```

## Why Pymablock?
Here is why you should use Pymablock:

* Do not reinvent the wheel

  Pymablock provides a tested reference implementation

* Apply to any problem

  Pymablock supports `numpy` arrays, `scipy` sparse arrays, `sympy` matrices and
  quantum operators

* Speed up your code

  Due to several optimizations, Pymablock can reliably handle both higher orders
  and large Hamiltonians

## How does Pymablock work?

Pymablock considers a Hamiltonian as a series of $2\times 2$ block operators
with the zeroth order block-diagonal.
To carry out the block-diagonalization procedure, Pymablock finds a minimal
unitary transformation $U$ that cancels the off-diagonal block of the
Hamiltonian order by order.

\begin{gather}
H = \begin{pmatrix}H_0^{AA} & 0 \\ 0 & H_0^{BB}\end{pmatrix} + \sum_{i\geq 1} H_i,\quad
U = \sum_{i=0}^\infty U_n
\end{gather}

The result of this procedure is a perturbative series of the transformed
block-diagonal Hamiltonian.

\begin{gather}
\tilde{H} = U^\dagger H U=\sum_{i=0}
\begin{pmatrix}
\tilde{H}_i^{AA} & 0 \\
0 & \tilde{H}_i^{BB}
\end{pmatrix}.
\end{gather}

Similar to Lowdin perturbation theory or the Schrieffer–Wolff transformation,
Pymablock solves Sylvester's equation and imposes unitarity at every order.
However, differently from other approaches, Pymablock uses efficient algorithms
by choosing an appropriate parametrization of the series of the unitary
transformation.
As a consequence, the computational cost of every order scales linearly with
the order, while the algorithms are still mathematically equivalent.

To see Pymablock in action, check out the [tutorial](tutorial/tutorial.md).
See its [algorithms](algorithms.md) to learn about the underlying ideas, or read
the [reference documentation](documentation/pymablock.rst) for the package API.

## What does Pymablock not do yet?

* Pymablock is not able to treat time-dependent perturbations yet
* Pymablock does not block diagonalize more than two subspaces simultaneously

## Installation

The preferred way of installing `pymablock` is to use `mamba`/`conda`:

```
mamba install pymablock -c conda-forge
```

Or use `pip`

```
pip install pymablock
```

```{important}
Be aware that the using `pymablock` on large Hamiltonians requires
[MUMPS](https://mumps-solver.org/index.php) support via ``python-mumps`` package.
It is only pip-installable on Linux, use conda on other platforms.
```

## Citing

If you have used Pymablock for work that has lead to a scientific publication,
please cite it as

```
@misc{Pymablock,
author = {{Araya Day}, Isidora and Miles, Sebastian and Varjas, Daniel and Akhmerov, Anton R.},
doi = {10.5281/zenodo.7995684},
month = {6},
title = {Pymablock},
year = {2023}
}
```

## Contributing

Pymablock is an open source package, and we invite you to contribute!
You contribute by opening [issues](https://gitlab.kwant-project.org/qt/pymablock/-/issues),
fixing them, and spreading the word about `pymablock`.
