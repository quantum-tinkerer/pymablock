```{include} ../../CONTRIBUTING.md
```

## Structured embeddings

`Embedding` selects a compiled generator representation or an ordered reference
basis at construction. Each owns source normalization and compression
$W^\dagger A W$. The generator representation works symbolically in target
occupations; the reference basis evaluates matrix elements in the supplied order.
The latter normalizes scalar sources to 1×1 matrices at the input boundary.

Both use the same P/Q block recurrence in `_operator_embedding.py`. Their
Sylvester solvers differ only in how they resolve energy denominators. The
reference solver and compression share `_actions`, which applies source
transitions to references without truncating the source space. Its source factor
already includes the ladder amplitude; division must not multiply that amplitude
into the target coefficient again.

Complement operators store an action and its adjoint action. Sums, products, and
scalar multiples compose these functions, and an explicit block multiplication
table connects them to the recurrence. Source products remain uncompressed until
the retained matrix element is evaluated, preserving virtual excursions.
