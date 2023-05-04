```{toctree}
:hidden:
:caption: Contents

tutorial/tutorial.md
algorithm/algorithm.md
documentation/lowdin.rst
CITING.md
CHANGELOG.md
```

# Lowdin
TODO:
This is the initial page and it contains explanations on:
- Goal of the package
- Usage context
- Setting of the perturbation theory: Hamiltonian + gap + 2 subspaces
- Statement about how results are the same as SW and Lowdin.
- Overall idea behind general and expanded
- Implicit and its two solvers KPM and direct
- How to install it
- Where to find docs

Things to clarify:
- We do not support more than 2 blocks
- We do not support time-dependent perturbation theory


## What is Lowdin?

Lowdin is a Python package that does quasi-degenerate perturbation theory.
It provides series of block operators that may contain numerical or symbolic
values, and various algorithms that allow to block-diagonalize numerical and
symbolic Hamiltonians.

## Why Lowdin?

The goal of this package is to...




## How does Lowdin work?

Lowdin provides series of block operators to do perturbation theory.
By decomposing a Hamiltonian into a block-diagonal unperturbed component and
perturbative orders, Lowdin allows to access any order and block in the
transformed block-diagonalized Hamiltonian. The results are cached and
additional orders may be requested by reusing previously computed orders.

To carry out the block-diagonalization procedure, Lowdin finds a minimal unitary
transformation that iteratively block-diagonalizes the Hamiltonian at every
order.
Like other approaches, Sylvester's equation needs to be solved at every order
to find the off-diagonal terms of the transformation.
However, differently from other approaches, Lowdin uses efficient algorithms
that do not waste computational efforts by choosing an appropriate
parametrization of the series for the unitary transformation.
As a consequence, the computational cost of every order scales linearly with
the order, while the resulting block-diagonalized Hamiltonian is still the same.

The two main algorithms, `general` and `expanded`, rely on decomposing the
unitary transformation {math}`U` as a series of Hermitian block diagonal
{math}`U` and skew-Hermitian block off-diagonal {math}`V` terms,

```{math}
\tilde{H} = (U + V)^\dagger H (U + V),
\quad U = \sum_{i=0}^\infty U_n,
\quad V = \sum_{i=0}^\infty V_n,
```

where {math}`H` is the original Hamiltonian and {math}`\tilde{H}` is its
block-diagonalized form.
It follows that every order of the transformed
Hamiltonian is computed as a Cauchy product between the series

```{math}
\tilde{H}^{(n)} = \sum_{i=0}^n (U_{n-i} - V_{n-i}) H_0 (U_i + V_i) +
\sum_{i=0}^{n-1} (U_{n-i-1} - V_{n-i-1}) H_p (U_i + V_i).
```

While `general` implements the procedure outlined here directly, `expanded`
initializes a fully symbolic Hamiltonian and derives the general expressions
for {math}`\tilde{H}`.
Additionaly, it simplifies {math}`\tilde{H}^{(n)}` such
that it only depends on {math}`V` and the perturbative orders in {math}`H`.
This makes `expanded` efficient for lower order numerical computations and
symbolic ones, while `general` is suitable for higher orders.

More details on the procedure are explained in [algorithms](algorithm/algorithm.md).

**TODO:** update all notation.

## Installation


## Citing

Follow the instructions in [citing](CITING.md) if you want to cite us.

## Contributing
`lowdin` is on Gitlab.
