---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
```{toctree}
:hidden:
:caption: Contents

tutorial/tutorial.md
documentation/lowdin.rst
CITING.md
CHANGELOG.md
```

# _Lowdin_
TODO:
This is the initial page and it contains explanations on:
- Goal of the package
- Usage context
- Setting of the perturbation theory: Hamiltonian + gap + 2 subspaces
- Statement about how results are the same as SW and _Lowdin_.
- Overall idea behind general and expanded
- Implicit and its two solvers KPM and direct
- How to install it
- Where to find docs

Things to clarify:
- We do not support more than 2 blocks
- We do not support time-dependent perturbation theory


## What is _Lowdin_?

_Lowdin_ is a Python package that does quasi-degenerate perturbation theory.
It provides series of block operators that may contain numerical or symbolic
values, and two algorithms that allow to efficiently block-diagonalize
numerical and symbolic Hamiltonians with multivariate perturbations.


Doing perturbation theory is a three step process:
* First, a Hamiltonian that depends on perturbative parameters is required.
{math}`\lambda`

```{code-cell} ipython3
:tags: [hide-input]

from sympy import Matrix, Symbol, Eq, Add

h_0 = -Matrix([[1, 0], [0, -1]])  # sigma z

lamba = Symbol('lambda', real=True)
h_p = lamba * Matrix([[0, 1], [1, 0]]) # sigma_x
hamiltonian = [h_0, h_p]

h_0 + h_p
```

* Second, the perturbation theory is defined by calling `block_diagonalized`.

```{code-cell} ipython3
from lowdin.block_diagonalization import block_diagonalize

subspace_indices = [0, 1]
H_tilde, *_ = block_diagonalize(hamiltonian, subspace_indices=subspace_indices)
```

* Finally, the perturbative corrections to the Hamiltonian are called, computed
and cached for each block and order.

```{code-cell} ipython3
H_tilde[0, 0, 2]  # AA block, 2nd order
```

## Why _Lowdin_?

* _Lowdin_ is efficient

  It provides taylored algorithms for different Hamiltonians.
* _Lowidn_ handles symbolic and numeric computations

  It works with `numpy` arrays, `scipy` sparse arrays, and `sympy` matrices and
  quantum operators.
* _Lowdin_ is well tested

  Its tests make it reliable for an arbitrary number of perturbations.

## How does _Lowdin_ work?

_Lowdin_ provides series of block operators to do perturbation theory.
By decomposing a Hamiltonian into a block-diagonal unperturbed component and
perturbative orders, _Lowdin_ allows to access any block and order of the
transformed block-diagonalized Hamiltonian. The results are cached and
additional orders may be requested by reusing previously computed orders.

To carry out the block-diagonalization procedure, _Lowdin_ finds a minimal
unitary transformation that iteratively block-diagonalizes the Hamiltonian at
every order.
Like with Lowdin perturbation theory or the Schrieffer–Wolff transformation,
_Lowdin_ solves Sylvester's equation at every order to find the off-diagonal
terms of the transformation.
At the same time, it imposes unitarity at every order to find the diagonal
terms of the transformation.
However, differently from other approaches, _Lowdin_ uses efficient algorithms
that do not waste computational efforts by choosing an appropriate
parametrization of the series for the unitary transformation.
As a consequence, the computational cost of every order scales linearly with
the order, while the resulting block-diagonalized Hamiltonian is still the same.

The two main algorithms, `general` and `expanded`, rely on decomposing the
unitary transformation as a series of Hermitian block diagonal {math}`U` and
skew-Hermitian block off-diagonal {math}`V` terms,

```{math}
\tilde{H} = (U + V)^\dagger H (U + V),
\quad U = \sum_{i=0}^\infty U_n,
\quad V = \sum_{i=0}^\infty V_n,
```
where {math}`H` is the original Hamiltonian and {math}`\tilde{H}` is its
block-diagonalized form.
It follows that every order {math}`n` of the transformed Hamiltonian is
computed as a Cauchy product between the series
```{math}
\tilde{H}_{n} = \sum_{i=0}^n (U_{n-i} - V_{n-i}) H_0 (U_i + V_i) +
\sum_{i=0}^{n-1} (U_{n-i-1} - V_{n-i-1}) H_p (U_i + V_i),
```
where {math}`H_0` is the unperturbed Hamiltonian and {math}`H_p` is a first
order univariate perturbation.

Consequently, the orders of the unitary transformation are solutions to
```{math}
U_{n} = - \frac{1}{2} \sum_{i=1}^{n-1}(U_{n-i}U_i - V_{n-i}V_i), \quad \text{unitarity} \\
H_0^{AA} V_{n}^{AB} - V_{n}^{AB} H_0^{BB} = Y_{n}, \quad \text{Sylvester's equation}
```
where
```{math}
Y_{n} = \sum_{i=1}^{n-1}\left[U_{n-i}^{AA}H_0^{AA}V_i^{AB}-V_{n-i}^{AB} H_0^{BB}U_i^{BB}\right].
```

While `general` implements the procedure outlined here directly, `expanded`
initializes a fully symbolic Hamiltonian and derives general expressions
for {math}`\tilde{H}`.
Additionaly, it simplifies {math}`\tilde{H}_{n}` and the unitary transformation
such that they only depend on {math}`V` and the perturbation {math}`H_p`.
As an example, these are the corrections to the effective Hamiltonian up to fourth
order using `expanded`.

```{code-cell} ipython3
:tags: [hide-input]

from operator import mul

from lowdin.block_diagonalization import BlockSeries, symbolic

H = BlockSeries(
    data={
        (0, 0, 0): Symbol('{H_{0}^{AA}}'),
        (1, 1, 0): Symbol('{H_{0}^{BB}}'),
        (0, 0, 1): Symbol('{H_{p}^{AA}}'),
        (0, 1, 1): Symbol('{H_{p}^{AB}}'),
        (1, 1, 1): Symbol('{H_{p}^{BB}}'),
    },
    shape=(2, 2),
    n_infinite=1,
)

max_order = 5
hamiltonians = {
  Symbol(f'H_{{{index}}}'): value for index, value in H._data.items()
}
offdiagonals = {
  Symbol(f'V_{{({order},)}}'): Symbol(f'V_{order}') for order in range(max_order)
}

H_tilde, *_ = symbolic(H)

for order in range(max_order):
    result = Symbol(fr'\tilde{{H}}_{order}^{{AA}}')
    display(Eq(result, H_tilde[0, 0, order].subs({**hamiltonians, **offdiagonals})))
```
Finally, `expanded` replaces the specifics of `H` into the simplified expressions,
never requiring to compute products within the auxiliary `B` subspace.
This makes `expanded` efficient for lower order numerical computations and
symbolic ones, while `general` is suitable for higher orders.

##  How to use _Lowdin_ on large numerical Hamiltonians?

The most expensive parts of the algorithm for large matrices are solving
Sylvester's equation and the matrix products within the auxiliary `B` subspace.
By calling `block_diagonalize` and providing the eigenvectors of the effective
`A` subspace, _Lowdin_ runs in `implicit` mode without needing the eigenvectors
of the auxiliary subspace.
For this, _Lowdin_ wraps the `B` subspace components of the Hamiltonian into
``scipy.sparse.LinearOperator`` and chooses an efficient
[MUMPS](https://mumps-solver.org/index.php)-based solver.
This allows an efficient calculation of the perturbative corrections to the
effective subspace.

Additionaly, there is an experimental solver that uses the
[Hybrid Kernel Polynomial Method](https://arxiv.org/abs/1909.09649).

## Installation


## Citing

Follow the instructions in [citing](CITING.md) if you want to cite us.

## Contributing
`lowdin` is on Gitlab.
