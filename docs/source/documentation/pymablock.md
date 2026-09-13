# Package reference

## Block diagonalization

```{autodoc} pymablock
:summary-only: true
```

```{autodoc} pymablock.block_diagonalize
```
```{autodoc} pymablock.operator_to_BlockSeries
```

## Solvers of Sylvester equation

```{autodoc} pymablock.block_diagonalization
:summary-only: true
```

```{autodoc} pymablock.block_diagonalization.solve_sylvester_diagonal
```
```{autodoc} pymablock.block_diagonalization.solve_sylvester_direct
```
```{autodoc} pymablock.block_diagonalization.solve_sylvester_KPM
```

## Series

```{autodoc} pymablock.series
:summary-only: true
```

```{autodoc} pymablock.series.BlockSeries
```
```{autodoc} pymablock.series.cauchy_dot_product
```
```{autodoc} pymablock.series.zero
```
```{autodoc} pymablock.series.one
```

## Linear algebra

```{autodoc} pymablock.linalg
:summary-only: true
```

```{autodoc} pymablock.linalg.direct_greens_function
```

## Number ordered form

```{autodoc} pymablock.number_ordered_form.LadderOp
```
```{autodoc} pymablock.number_ordered_form.NumberOperator
```
```{autodoc} pymablock.number_ordered_form.NumberOrderedForm
```
```{autodoc} pymablock.number_ordered_form.find_operators
```

## Second quantization

```{autodoc} pymablock.second_quantization
:summary-only: true
```

```{autodoc} pymablock.second_quantization.solve_sylvester_2nd_quant
```
```{autodoc} pymablock.second_quantization.apply_mask_to_operator
```

## Kernel polynomial method (KPM)

```{autodoc} pymablock.kpm
:summary-only: true
```

```{autodoc} pymablock.kpm.greens_function
```
```{autodoc} pymablock.kpm.rescale
```

## Algorithms

```{autodoc} pymablock.algorithm_parsing
:summary-only: true
```

```{autodoc} pymablock.algorithm_parsing.series_computation
```
