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
documentation/pymablock.rst
derivation.md
CHANGELOG.md
```

# {{Pymablock}}

## What is {{Pymablock}}?

{{Pymablock}} (Python matrix block-diagonalization) is a Python package that constructs
effective models using quasi-degenerate perturbation theory.
It handles both numerical and symbolic inputs, and it efficiently
block-diagonalizes Hamiltonians with multivariate perturbations to arbitrary
order.

Building an effective model using {{Pymablock}} is a three step process:
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

## Why {{Pymablock}}?
Here is why you should use {{Pymablock}}:

* Do not reinvent the wheel

  {{Pymablock}} provides a tested reference implementation

* Apply to any problem

  {{Pymablock}} supports `numpy` arrays, `scipy` sparse arrays, `sympy` matrices and
  quantum operators

* Speed up your code

  Due to several optimizations, {{Pymablock}} can reliable handle both higher orders
  and large Hamiltonians

## How does {{Pymablock}} work?

{{Pymablock}} considers a Hamiltonian as a series of $2\times 2$ block operators
with the zeroth order block-diagonal.
To carry out the block-diagonalization procedure, {{Pymablock}} finds a minimal
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
{{Pymablock}} solves Sylvester's equation and imposes unitarity at every order.
However, differently from other approaches, {{Pymablock}} uses efficient algorithms
by choosing an appropriate parametrization of the series of the unitary
transformation.
As a consequence, the computational cost of every order scales linearly with
the order, while the algorithms are still mathematically equivalent.

## The algorithms

The algorithms of {{Pymablock}} rely on decomposing $U$, the unitary transformation
that block diagonalizes the Hamiltonian, as a series of Hermitian
block diagonal $W$ and skew-Hermitian and block off-diagonal $V$ terms.
The transformed Hamiltonian is a
[Cauchy product](https://en.wikipedia.org/wiki/Cauchy_product)
between the series of $U^\dagger$, $H$, and $U$.

For example, for a single first order perturbation $H_p$, the transformed
Hamiltonian at order $n$ is

\begin{align}
\tilde{H}_{n} = \sum_{i=0}^n (W_{n-i} - V_{n-i}) H_0 (W_i + V_i) +
\sum_{i=0}^{n-1} (W_{n-i-1} - V_{n-i-1}) H_p (W_i + V_i).
\end{align}

To block diagonalize $H_0 + H_p$, {{Pymablock}} finds the orders of $W$
such that $U$ is unitary

\begin{equation}
W_{n} = - \frac{1}{2} \sum_{i=1}^{n-1}(W_{n-i}W_i - V_{n-i}V_i),
\end{equation}

and the orders of $V$ by ensuring that the $\tilde{H}^{AB}=0$ to any order

\begin{equation}
H_0^{AA} V_{n}^{AB} - V_{n}^{AB} H_0^{BB} = Y_{n}.
\end{equation}

This is known as [Sylvester's equation](https://en.wikipedia.org/wiki/Sylvester_equation)
and $Y_{n}$ is recursive on $n$.

{{Pymablock}} has two algorithms, {autolink}`~pymablock.general` and {autolink}`~pymablock.expanded`.
While the {autolink}`~pymablock.general` algorithm implements the procedure outlined here directly,
{autolink}`~pymablock.expanded` initializes a fully symbolic Hamiltonian and derives general
expressions for $\tilde{H}$.
Additionaly, it simplifies $\tilde{H}_{n}$ and the unitary transformation
such that they only depend on $V$ and the perturbation $H_p$, but not on $H_0$.
This requires using Sylvester's equation for every order.

As an example, the corrections to the effective Hamiltonian up to fourth
order using {autolink}`~pymablock.expanded` are

```{code-cell} ipython3
:tags: [remove-input]

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

Finally, {autolink}`~pymablock.expanded` replaces the problem-specific $H$ into the simplified
$\tilde{H}$, without computing products within the auxiliary $B$ subspace.
This makes {autolink}`~pymablock.expanded` efficient for lower order numerical computations and
symbolic ones, while {autolink}`~pymablock.general` is suitable for higher orders.


##  How to use {{Pymablock}} on large numerical Hamiltonians?

Solving Sylvester's equation and computing the matrix products are the most
expensive steps of the algorithms for large Hamiltonians.
{{Pymablock}} can efficiently construct an effective Hamiltonian of a small subspace
even when the full Hamiltonian is a sparse matrix that is too costly to
diagonalize. This functionality is provided by the
{autolink}`~pymablock.implicit` function.
It exploits the low rank structure of $U$, and
by using the sparse solver [MUMPS](https://mumps-solver.org/index.php) to
compute the Green's function.

## What does {{Pymablock}} not do?

* {{Pymablock}} is not able to treat time-dependent perturbations yet
* {{Pymablock}} does not block diagonalize on more than two subspaces simultaneously

## Installation

To install `pymablock`, prefer using `conda`

```
conda install pymablock
```

Alternatively, you use `pip`

```
pip install pymablock
```

```{important}
Be aware that the using `pymablock` on large Hamiltonians requires `Kwant`
installed [via conda](https://kwant-project.org/install#conda) in order to use
[MUMPS](https://mumps-solver.org/index.php). Make sure you have the correct
`Kwant` installed if you use `pip` to install `pymablock`.
```

## Citing

If you have used {{Pymablock}} for work that has lead to a scientific publication,
please cite it as

```
TODO

```

## Contributing

{{Pymablock}} is an open source package, and we invite you to contribute!
You contribute by opening [issues](https://gitlab.kwant-project.org/qt/pymablock/-/issues),
fixing them, and spreading the word about `pymablock`.
