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

+++ {"tags": [], "user_expressions": []}

```{toctree}
:hidden:
:caption: Contents

tutorial/tutorial.md
background/background.md
documentation/pymablock.rst
CITING.md
CHANGELOG.md
```

# _Pymablock_

## What is _Pymablock_?

_Pymablock_ is a Python package that constructs effective models using
quasi-degenerate perturbation theory.
It handles both numerical and symbolic inputs, and it efficiently
block-diagonalizes Hamiltonians with multivariate perturbations to arbitrary
order.

Building an effective model using _Pymablock_ is a three step process:
* Define a Hamiltonian
* Call `block_diagonalize`
* Request the desired order of the effective Hamiltonian

```python
from pymablock import block_diagonalize

# Define perturbation theory
H_tilde, *_ = block_diagonalize([h_0, h_p], subspace_eigenvectors=[vecs_A, vecs_B])

# Request correction to the effective Hamiltonian
H_AA_4 = H_tilde[0, 0, 4]
```

## Why _Pymablock_?
Here is why you should use _Pymablock_:

* Do not reinvent the wheel

  _Pymablock_ provides a tested reference implementation

* Apply to any problem

  _Pymablock_ supports `numpy` arrays, `scipy` sparse arrays, `sympy` matrices and
  quantum operators

* Speed up your code

  Due to several optimizations, _Pymablock_ can reliable handle both higher orders
  and large Hamiltonians

## How does _Pymablock_ work?

_Pymablock_ considers a Hamiltonian as a series of $2\times 2$ block operators
with the zeroth order block-diagonal.
To carry out the block-diagonalization procedure, _Pymablock_ finds a minimal
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
_Pymablock_ solves Sylvester's equation and imposes unitarity at every order.
However, differently from other approaches, _Pymablock_ uses efficient algorithms
by choosing an appropriate parametrization of the series of the unitary
transformation.
As a consequence, the computational cost of every order scales linearly with
the order, while the algorithms are still mathematically equivalent.

## The algorithms

The algorithms of _Pymablock_ rely on decomposing $U$, the unitary transformation
that block diagonalizes the Hamiltonian, as a series of Hermitian
block diagonal $W$ and skew-Hermitian, block off-diagonal, $V$ terms.
The transformed Hamiltonian is a [Cauchy product](https://en.wikipedia.org/wiki/Cauchy_product)
between the series of $U^\dagger$, $H$, and $U$.
For example, for a single first order perturbation $H_p$, the transformed
Hamiltonian at order $n$ is
```{math}
\begin{align}
\tilde{H}_{n} = \sum_{i=0}^n (W_{n-i} - V_{n-i}) H_0 (W_i + V_i) +
\sum_{i=0}^{n-1} (W_{n-i-1} - V_{n-i-1}) H_p (W_i + V_i).
\end{align}
```

To block diagonalize $H_0 + H_p$, _Pymablock_ finds the orders of $W$
and $V$ as a solution to
```{math}
\begin{align}
W_{n} &= - \frac{1}{2} \sum_{i=1}^{n-1}(W_{n-i}W_i - V_{n-i}V_i), & \quad &\text{unitarity} \\
H_0^{AA} V_{n}^{AB} - V_{n}^{AB} H_0^{BB} &= Y_{n}, & \quad &\text{Sylvester's equation}
\end{align}
```
where
```{math}
Y_{n} = \sum_{i=1}^{n-1}\left[W_{n-i}^{AA}H_0^{AA}V_i^{AB}-V_{n-i}^{AB} H_0^{BB}W_i^{BB}\right].
```

_Pymablock_ has two principal algorithms, `general` and `expanded`.
While the `general` algorithm implements the procedure outlined here directly,
`expanded` initializes a fully symbolic Hamiltonian and derives general
expressions for $\tilde{H}$.
Additionaly, it simplifies $\tilde{H}_{n}$ and the unitary transformation by 
using Sylvester's equation for $H_0$ to eliminate it from the equations
and regroup the remaining terms. That way, $\tilde{H}_n$ only depends on $V$
and the perturbation $H_p$, but not on $H_0$, reducing the overall number of 
matrix-vector products that need to be performed.
As an example, the corrections to the effective Hamiltonian up to fourth
order using `expanded` are

```{code-cell} ipython3
:tags: [hide-input]

from operator import mul

from sympy import Symbol, Eq

from pymablock.block_diagonalization import BlockSeries, symbolic

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

+++ {"user_expressions": []}

Finally, `expanded` replaces the problem-specific $H$ into the simplified
$\tilde{H}$, without computing products within the auxiliary $B$ subspace.
This makes `expanded` efficient for lower order numerical computations and
symbolic ones, while `general` is suitable for higher orders.


##  How to use _Pymablock_ on large numerical Hamiltonians?

Solving Sylvester's equation and computing the matrix products are the most
expensive steps of the algorithms for large Hamiltonians.
_Pymablock_ can efficiently construct an effective Hamiltonian of a small subspace
even when the full Hamiltonian is a sparse matrix that is too costly to
diagonalize. This functionality is provided by the `implicit` function.
It exploits the low rank structure of $U$, and
by using the sparse solver [MUMPS](https://mumps-solver.org/index.php) to
compute the Green's function.

## What does _Pymablock_ not do?

* _Pymablock_ is not able to treat time-dependent perturbations yet
* _Pymablock_ does not block diagonalize on more than two subspaces simultaneously

## Installation


## Citing

Follow the instructions in [citing](CITING.md) if you want to cite us.

## Contributing
`pymablock` is on Gitlab.

```{code-cell} ipython3

```
