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

## What is _Lowdin_?

_Lowdin_ is a Python package that constructs effective models using
quasi-degenerate perturbation theory.
It handles both numerical and symbolic inputs, and it efficiently
block-diagonalizes Hamiltonians with multivariate perturbations to arbitrary
order.

Building an effective model using _Lowdin_ is a three step process:
* Define a Hamiltonian
* Call `block_diagonalize`
* Request the desired order of the effective Hamiltonian

```python
from lowdin import block_diagonalize

# Define perturbation theory
H_tilde, *_ = block_diagonalize([h_0, h_p], subspace_eigenvectors=[vecs_A, vecs_B])

# Request correction to the effective Hamiltonian
H_AA_4 = H_tilde[0, 0, 4]
```

## Why _Lowdin_?
Here is why you should use _Lowdin_:

* Do not reinvent the wheel

  _Lowdin_ provides a tested reference implementation

* Apply to any problem

  _Lowdin_ supports `numpy` arrays, `scipy` sparse arrays, `sympy` matrices and
  quantum operators

* Speed up your code

  Due to several optimizations, _Lowdin_ can reliable handle both higher orders
  and large Hamiltonians

## How does _Lowdin_ work?

_Lowdin_ considers a Hamiltonian as a series of {math}`2\times 2` block operators
with the zeroth order block-diagonal.
To carry out the block-diagonalization procedure, _Lowdin_ finds a minimal
unitary transformation that cancels the off-diagonal block of the Hamiltonian
order by order.

```{math}
\begin{gather}
H = \begin{pmatrix}H_0^{AA} & 0 \\ 0 & H_0^{BB}\end{pmatrix} + \sum_{i\geq 1} H_i,\quad
U = \sum_{i=0}^\infty U_n
\end{gather}
```

The result of this procedure is a perturbative series of the transformed
block-diagonal Hamiltonian.

```{math}
\begin{gather}
\tilde{H} = U^\dagger H U=\sum_{i=0}\begin{pmatrix}\tilde{H}_i^{AA} & 0 \\ 0 & \tilde{H}_i^{BB}\end{pmatrix}.
\end{gather}
```

Similar to Lowdin perturbation theory or the Schrieffer–Wolff transformation,
_Lowdin_ solves Sylvester's equation and imposes unitarity at every order.
However, differently from other approaches, _Lowdin_ uses efficient algorithms
by choosing an appropriate parametrization of the series of the unitary
transformation.
As a consequence, the computational cost of every order scales linearly with
the order, while the algorithms are still mathematically equivalent.

## The algorithms

_Lowdin_ algorithms, `general` and `expanded`, rely on decomposing {math}`U` as
a series of Hermitian block diagonal {math}`W` and skew-Hermitian block
off-diagonal {math}`V` terms.
The transformed Hamiltonian is a Cauchy product between the series of
{math}`U^\dagger`, {math}`H`, and {math}`U`.
For example, for a single first order perturbation {math}`H_p`, the transformed
Hamiltonian at order {math}`n` is
```{math}
\tilde{H}_{n} = \sum_{i=0}^n (W_{n-i} - V_{n-i}) H_0 (W_i + V_i) +
\sum_{i=0}^{n-1} (W_{n-i-1} - V_{n-i-1}) H_p (W_i + V_i).
```

To block diagonalize {math}`H_0 + H_p`, _Lowdin_ finds the orders of {math}`W`
and {math}`V` as a solution to
```{math}
W_{n} = - \frac{1}{2} \sum_{i=1}^{n-1}(W_{n-i}W_i - V_{n-i}V_i), \quad \text{unitarity} \\
H_0^{AA} V_{n}^{AB} - V_{n}^{AB} H_0^{BB} = Y_{n}, \quad \text{Sylvester's equation}
```
where
```{math}
Y_{n} = \sum_{i=1}^{n-1}\left[W_{n-i}^{AA}H_0^{AA}V_i^{AB}-V_{n-i}^{AB} H_0^{BB}W_i^{BB}\right].
```

While the `general` algorithm implements the procedure outlined here directly,
`expanded` initializes a fully symbolic Hamiltonian and derives general
expressions for {math}`\tilde{H}`.
Additionaly, it simplifies {math}`\tilde{H}_{n}` and the unitary transformation
such that they only depend on {math}`V` and the perturbation {math}`H_p`, but
not on {math}`H_0`.
As an example, the corrections to the effective Hamiltonian up to fourth
order using `expanded` are

```{code-cell} ipython3
:tags: [hide-input]

from operator import mul

from sympy import Symbol, Eq

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
Finally, `expanded` replaces the problem-specific `H` into the simplified
{math}`\tilde{H}`, without computing products within the auxiliary `B` subspace.
This makes `expanded` efficient for lower order numerical computations and
symbolic ones, while `general` is suitable for higher orders.


##  How to use _Lowdin_ on large numerical Hamiltonians?

Solving Sylvester's equation and computing the matrix products within the
auxiliary subspace are the most expensive steps of the algorithms for large
matrices.
If the eigenvectors of the effective subspace are provided, _Lowdin_ will
choose an efficient
[MUMPS](https://mumps-solver.org/index.php)-based solver to speed up.

## What does _Lowdin_ not do?

* _Lowdin_ is not able to treat time-dependent perturbations yet
* _Lowdin_ does not block diagonalize on more than two subspaces simultaneously

## Installation


## Citing

Follow the instructions in [citing](CITING.md) if you want to cite us.

## Contributing
`lowdin` is on Gitlab.
