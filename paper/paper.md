---
title: Pymablock
authors:
  - name: person
    orcid: 0000-0000-0000-0000
    affiliations: [aaaa]
    corresponding: true
    email: myemail@gmail.com
  - name: personita
    orcid: 0000-0000-0000-0000
    affiliations: [bbbb]
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


```{include} introduction.md
```

```{include} finding_effective_model.md
```

```{include} algorithms.md
```

```{include} implementation.md
```

```{include} benchmark.md
```

```{include} conclusion.md
```
