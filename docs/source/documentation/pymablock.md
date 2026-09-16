# Package reference

## Block diagonalization

```{eval-rst}
.. automodule:: pymablock
   :members:
   :show-inheritance:
```

## Solvers of Sylvester equation

```{eval-rst}
.. automodule:: pymablock.block_diagonalization
   :members: solve_sylvester_diagonal, solve_sylvester_direct, solve_sylvester_KPM
   :show-inheritance:
```

## Series

```{eval-rst}
.. autoclass:: pymablock.series.BlockSeries

.. automodule:: pymablock.series
   :members: cauchy_dot_product
   :show-inheritance:

.. autodata:: pymablock.series.zero

.. autodata:: pymablock.series.one
```

## Linear algebra

```{eval-rst}
.. automodule:: pymablock.linalg
   :members: direct_greens_function
   :show-inheritance:
```

## Number ordered form

```{eval-rst}
.. autoclass:: pymablock.number_ordered_form.LadderOp
   :members:
   :show-inheritance:
   :class-doc-from: class
```

```{eval-rst}
.. autoclass:: pymablock.number_ordered_form.NumberOperator
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pymablock.number_ordered_form.NumberOrderedForm
   :members:
   :show-inheritance:
   :class-doc-from: class
```

```{eval-rst}
.. autofunction:: pymablock.number_ordered_form.find_operators
```

## Second quantization

```{eval-rst}
.. automodule:: pymablock.second_quantization
   :members: solve_sylvester_2nd_quant, apply_mask_to_operator
   :show-inheritance:
```

### Structured embeddings

`Embedding` declares the target operator algebra and source occupation constraints.
For example, a spin represented by the first two levels of a boson is

```python
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.pauli import SigmaMinus
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding

a, s = BosonOp("a"), SigmaMinus("s")
embedding = Embedding(target=(s,), occupations={a: N(s)})
# block_diagonalize([H0, V], subspace_eigenvectors=embedding)
```

The same constructor accepts `FermionOp` targets. A source spin encoded in two
fermionic modes uses `occupations={up: N(s), down: 1 - N(s)}`. The expressions
specify occupations, not replacements for source annihilation operators.

A finite spin-1 target is declared as `target={JminusOp("S"): 3}`. Its occupation
coordinate is `JzOp("S") + 1`, in units with hbar equal to one. The effective
operator is a matrix in increasing magnetic-quantum-number order. Binary spins
and fermions instead return `NumberOrderedForm` objects in their named generators.

The constructor supports full-rank integer affine occupation maps. It validates
source occupation ranges and direct fermion assignments without enumerating the
Hilbert space. Relative phases follow the canonical source Fock convention; direct
fermion targets include frozen-mode and permutation signs. It does not accept a
particle-number-sector option or perform basis rotations. For superpositions,
rotate the source Hamiltonian first or supply explicit orthonormal basis columns
through the existing matrix `subspace_eigenvectors` interface.

```{eval-rst}
.. autoclass:: pymablock.operator_embedding.Embedding(*, target, occupations)
   :members: encode
   :class-doc-from: class
```

## Kernel polynomial method (KPM)

```{eval-rst}
.. automodule:: pymablock.kpm
   :members: greens_function, rescale
   :show-inheritance:
```

## Algorithms

```{eval-rst}
.. automodule:: pymablock.algorithm_parsing
   :members: series_computation
   :show-inheritance:
```
