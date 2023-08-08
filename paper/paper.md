---
title: Pymablock
authors:
  - name: person
    orcid: orcid
    affiliation: aaaa
    corresponding: true
    email: myemail@gmail.com
  - name: personita
    orcid: orcid
    affiliation: bbbb
site:
  template: scipost-template
  draft: true
  # title:
  # logo:
  nav: []
  actions:
    - title: Learn More
      url: https://mystmd.org/guide
  domains: []
exports:
  - format: tex+pdf
    drafts: true
    template: scipost-template
    output: exports/paper.pdf
---

# Pymablock

+++ {"part": "abstract"}
Numerical simulations play a key role in the study of complex physical systems.
There are many tools to simulate a system, study its symmetries, and extract its properties efficiently.
All of these are limited by the size of the system that can be simulated and the computational resources available.
However, many physical systems are sufficiently well described by a smaller, low energy, subspace.
Here we introduce Pymablock, a Python package that constructs effective models using quasi-degenerate perturbation theory.
It handles both numerical and symbolic inputs, and it efficiently block-diagonalizes Hamiltonians with multivariate perturbations to arbitrary order.
+++

# Introduction

Effective models enable the study of complex physical systems by reducing the space of interest to a low energy one.

To find an effective Hamiltonian, we use perturbative approaches, like a SW transformation or Lowdin perturbation theory.

Even though these methods are standard, their algorithm is computationally expensive, scaling poorly for large systems and high orders.

We develop an efficient algorithm capable of symbolic and numeric computations and make it available in Pymablock.

Building an effective model with Pymablock is easy, its core is a versatile block diagonalization routine.

(Include figure with 2 subpanels: 3 lines of code that show how to build a mode, and a scheme of the steps.)

# Finding an effective model

## Pymablock workflow

The workflow of Pymablock consists of three steps.

Depending on the input Hamiltonian, Pymablock uses specific routines to find the effective model, so that symbolic expressions are compact and numerics are efficient.

## k.p model of bilayer graphene

To illustrate the use of Pymablock in analytic models, we consider a k.p model of bilayer graphene.

## Induced gap in a double quantum dot

Large systems pose an additional challenge due to the scaling of linear algebra routines for large matrices.

Pymablock handles large systems by using sparse matrices and avoiding the construction of the full Hamiltonian.

We illustrate this with a model of a double quantum dot.

# Pymablock algorithms

Pymablock considers a Hamiltonian as a series of $2 \times 2$ block operators and finds a minimal unitary transformation that separates its subspaces.

The result of this procedure is a perturbative series of the transformed block-diagonal Hamiltonian.

The transformed Hamiltonian is equivalent to that of other perturbative methods, but the algorithm is efficient.

Pymablock finds the unitary transformation recursively, using unitarity and solving Sylvester's equation at every order.

Pymablock has two algorithms, general and expanded, tailored for different use cases.


# Benchmark

# Conclusion
