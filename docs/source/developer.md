```{include} ../../CONTRIBUTING.md
```

## Structured embeddings

`Embedding` owns source-projector construction, compression through `restrict()`,
and preparation of the perturbative blocks through `_prepare()`. The standard
`block_diagonalize` driver calls `_prepare()` directly and runs the recurrence.

Two private compiled representations share source normalization:
`_GeneratorBasis` converts transitions into symbolic target occupations, while
`_ReferenceBasis` evaluates matrix elements in an ordered list of source states.
The reference representation converts scalar sources to 1×1 matrices at its
input boundary.

`_convert_operator()` normalizes one source operator, `_convert_source()` handles
the source's scalar or matrix representation, and `_compress()` evaluates the
converted source in the target representation. Block preparation computes the
incoming energies and coordinate substitutions once; transition division then
uses the same path for generator and reference-list embeddings.

`_NOFTransition` stores a term's operators, powers, and coefficient directly.
Its occupation action supplies the destination and ladder-weighted amplitude
used by both compression and the Sylvester solver. Division uses that amplitude
to identify inactive transitions, but does not multiply it into the solved NOF
coefficient again.

Rectangular blocks are NOFs with an embedding attached on the left or right;
ordinary NOF arithmetic composes them. Contracting `W† X W` returns an ordinary
target operator or matrix. Source products remain uncompressed until contraction,
preserving intermediate excursions outside the retained space. Reference lists
use matrices of NOFs with a shared single-reference attachment.
