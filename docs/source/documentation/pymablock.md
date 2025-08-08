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
