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
```

```{eval-rst}
.. automodule:: pymablock.series
   :members: cauchy_dot_product
   :show-inheritance:
```

## Linear algebra

```{eval-rst}
.. automodule:: pymablock.linalg
   :members: direct_greens_function
   :show-inheritance:
```

## Second quantization

```{eval-rst}
.. automodule:: pymablock.second_quantization
   :members: solve_monomial, solve_sylvester_bosonic, apply_mask_to_operator
   :show-inheritance:
```

## Number ordered form

```{eval-rst}
.. autoclass:: pymablock.number_ordered_form.NumberOperator
   :members: name, doit
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: pymablock.number_ordered_form.NumberOrderedForm
   :members: operators, terms, from_expr, as_expr, apply_sympy_func, is_particle_conserving filter_terms
   :no-index: __new__
```

```{eval-rst}
.. automodule:: pymablock.number_ordered_form
   :members: find_operators
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
