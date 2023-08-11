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

# Introduction

## Effective models

**Effective models enable the study of complex physical systems by reducing the space of interest to a low energy one.**

**To find an effective Hamiltonian, we use perturbative approaches, like a SW transformation or Lowdin perturbation theory.**

**Even though these methods are standard, their algorithm is computationally expensive, scaling poorly for large systems and high orders.**

**We develop an efficient algorithm capable of symbolic and numeric computations and make it available in Pymablock.**

**Building an effective model with Pymablock is easy, its core is a versatile block diagonalization routine.**

## Installing Pymablock

Pymablock is a Python package that requires Python $3.9$ or higher.
We recommend to use `mamba`/`conda` and run the following command in the
terminal to install Pymablock

```{code}
mamba install pymablock -c conda-forge
```

Alternatively, Pymablock can be installed with `pip` by running

```{code}
pip install pymablock
```

```{important}
Using Pymablock on large Hamiltonians requires the Kwant package
[kwant](doi:10.1088/1367-2630/16/6/063065) with MUMPS sparse solver
support [mumps1](doi:10.1137/S0895479899358194),
[mumps2](doi:10.1016/j.parco.2005.07.004).
For this purpose, we recommend to install Kwant via conda in Linux or MAC OS.
Unfortunately, MUMPS support in Kwant is not available for Windows.
As an alternative, pymablock can be installed in Windows Subsystem for Linux
(WSL).
```
