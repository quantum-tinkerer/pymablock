```{include} ../../CONTRIBUTING.md
```

## Structured embeddings

The public `Embedding` object provides construction and `restrict()`. Its private
compiled basis owns source normalization and compression
$W^\dagger A W$. The generator representation works symbolically in target
occupations; the reference basis evaluates matrix elements in the supplied order.
The latter normalizes scalar sources to 1×1 matrices at the input boundary.

One function in `_operator_embedding.py` constructs the P/Q block series and runs
the recurrence. Energy-division functions capture the data for each representation;
there is no solver class hierarchy. The
reference solver and compression share `_actions`, which applies source
transitions to references without truncating the source space. Its source factor
already includes the ladder amplitude; division must not multiply that amplitude
into the target coefficient again.

`_CouplingBlock` maps retained states to discarded states, and
`_AdjointCouplingBlock` maps back. `_ComplementBlock` acts within the discarded
space, storing an action and its adjoint action. Sums, products, and
scalar multiples compose these functions, and an explicit block multiplication
table connects them to the recurrence. Source products remain uncompressed until
the retained matrix element is evaluated, preserving virtual excursions.
