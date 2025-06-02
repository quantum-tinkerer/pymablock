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

# Pymablock

```{toctree}
:hidden:
:maxdepth: 1
:caption: Tutorials

tutorial/getting_started.md
tutorial/bilayer_graphene.md
Induced superconducting gap <tutorial/induced_gap.md>
tutorial/jaynes_cummings.md
Dispersive shift of a resonator <tutorial/dispersive_shift.md>
Rabi model <tutorial/spin_rwa_floquet.md>
tutorial/andreev_supercurrent.md
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Documentation

algorithms.md
Comparison to Schrieffer-Wolff <radius.md>
second_quantization.md
documentation/pymablock.md
CHANGELOG.md
authors.md
developer.md
```

::::{admonition} **✨ NEW! Second Quantization Support ✨**
:class: tip

Pymablock now works with second-quantized operators: fermions, bosons, spins, and ladder (Floquet)!
Check out our [dispersive shift tutorial](tutorial/dispersive_shift.md) and [Jaynes-Cummings model tutorial](tutorial/jaynes_cummings.md) to see second quantization in action, and see the theoretical background in [second quantization notes](second_quantization.md).
::::

## What is Pymablock?

Pymablock (Python matrix block-diagonalization) is a Python package that constructs effective models using quasi-degenerate perturbation theory.
It handles both numerical and symbolic inputs, and it efficiently block-diagonalizes Hamiltonians with multivariate perturbations to arbitrary order.

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

  Pymablock supports `numpy` arrays, `scipy` sparse arrays, `sympy` matrices and quantum operators

* Speed up your code

  Due to several optimizations, Pymablock can reliably handle both higher orders and large Hamiltonians

## How does Pymablock work?

Pymablock considers a Hamiltonian as a series of block operators with the zeroth order block-diagonal.
To carry out the block-diagonalization procedure, Pymablock finds a minimal unitary transformation $U$ that cancels the off-diagonal blocks of the Hamiltonian order by order.

\begin{gather}
\mathcal{H} = \begin{pmatrix}H_0^{AA} & 0 \\ 0 & H_0^{BB}\end{pmatrix} + \sum_{n\geq 1} H'_n,\quad
\mathcal{U} = \sum_{n=0}^\infty U_n
\end{gather}

The result of this procedure is a perturbative series of the transformed block-diagonal Hamiltonian.

\begin{gather}
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U}=\sum_{n=0}
\begin{pmatrix}
\tilde{H}_n^{AA} & 0 \\
0 & \tilde{H}_n^{BB}
\end{pmatrix}.
\end{gather}

Similar to Lowdin perturbation theory or the Schrieffer–Wolff transformation, Pymablock solves Sylvester's equation and imposes unitarity at every order.
However, Pymablock is unique because it uses efficient algorithms by choosing a different parametrization of the series of the unitary transformation.
As a consequence, the computational cost of every order scales linearly with the order, while the algorithms are still mathematically equivalent.
Additionally, this parametrization allows Pymablock to perform selective diagonalization and eliminate an arbitrary subset of offdiagonal matrix elements, and to go beyond the standard $2\times 2$ block-diagonalization.

To see Pymablock in action, check out the [tutorial](tutorial/getting_started.md).
See its [algorithms](algorithms.md) to learn about the underlying ideas, or read the [reference documentation](documentation/pymablock.md) for the package API.

## What does Pymablock not do yet?

* Pymablock is not able to treat time-dependent perturbations yet

## Installation

The preferred way of installing `pymablock` is to use `mamba`/`conda`:

```bash
mamba install pymablock -c conda-forge
```

Or use `pip`

```bash
pip install pymablock
```

```{important}
Be aware that the using `pymablock` on large Hamiltonians requires [MUMPS](https://mumps-solver.org/index.php) support via ``python-mumps`` package.
It is only pip-installable on Linux, use conda on other platforms.
```

## Citing

If you have used Pymablock for work that has lead to a scientific publication, please cite the accompanying [paper](https://doi.org/10.21468/SciPostPhysCodeb.50) as

```bibtex
@Article{10.21468/SciPostPhysCodeb.50,
  title={{Pymablock: An algorithm and a package for quasi-degenerate perturbation theory}},
  author={Isidora {Araya Day} and Sebastian Miles and Hugo K. Kerstens and Daniel Varjas and Anton R. Akhmerov},
  journal={SciPost Phys. Codebases},
  pages={50},
  year={2025},
  publisher={SciPost},
  doi={10.21468/SciPostPhysCodeb.50},
  url={https://scipost.org/10.21468/SciPostPhysCodeb.50},
}

@Article{10.21468/SciPostPhysCodeb.50-r2.1,
  title={{Codebase release 2.1 for Pymablock}},
  author={Isidora {Araya Day} and Sebastian Miles and Hugo K. Kerstens and Daniel Varjas and Anton R. Akhmerov},
  journal={SciPost Phys. Codebases},
  pages={50-r2.1},
  year={2025},
  publisher={SciPost},
  doi={10.21468/SciPostPhysCodeb.50-r2.1},
  url={https://scipost.org/10.21468/SciPostPhysCodeb.50-r2.1},
}
```

## Contributing

Pymablock is an open source package, and we invite you to contribute!
You contribute by opening [issues](https://gitlab.kwant-project.org/qt/pymablock/-/issues), fixing them, and spreading the word about `pymablock`.
If you want to contribute code, please read the [developer documentation](developer.md).
