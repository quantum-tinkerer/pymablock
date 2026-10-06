```{include} ../../CONTRIBUTING.md
```

## Structured embeddings

`Embedding` is a structural SymPy expression: its arguments specify the source
images and references, while private compiled bases implement them. Reconstructing
or unpickling equal arguments produces the same source modes. The generator and
reference-list representations share source normalization and compression.

The implementation separates three operations:

- `operator_embedding._EmbeddingBlocks` constructs the retained/complement frames
  and converts general operators. It preserves zeroth-order cross blocks.
- `second_quantization._make_embedding_sylvester_solver` validates diagonal H0
  and divides transitions by their actual energy differences. The driver can then omit
  zeroth-order cross blocks of that validated Hamiltonian.
- `number_ordered_form` composes rectangular operators attached to an
  `Embedding` and supplies scalar projectors and support reduction. Both
  Sylvester solvers share coefficient division in `second_quantization`.

`_NOFTransition` describes a term's shift, destination, and ladder amplitude.
Compression and division share this action. Division uses the amplitude to
identify inactive transitions, without multiplying it into the NOF coefficient
again. The zero series sentinel is handled before attempting matrix operations.
For `fully_diagonalize`, transitions within either diagonal block use the
ordinary second-quantized Sylvester solver, with retained or source energies.
Transitions between the blocks use the embedding-aware solver.

Method caches belong to their compiled instance, so discarding an embedding
also discards its basis caches. The scalar projector cache has a fixed size.
SymPy compatibility patches live in `number_ordered_form`; the condition
workaround is feature-detected and installed once, including across reloads.

### Validation and rotations

These checks apply to the compressed generators $PGP$. Each generator must
have the target ladder norm, and each pair must commute (anticommute for two
fermions) on the retained occupation lattice. The reference and occupation
boundaries fix the vacuum and finite truncation; the norm and pair relations
then also determine the relations involving adjoints. Thus checking the source
algebra outside the retained space is unnecessary: a boson can represent a
two-state target even though their uncompressed commutators differ.

Validation leaves spectator occupations symbolic. Binary polynomial identities
are reduced modulo $n^2-n$, rather than checked separately at every combination
of occupations. There are quadratically many generator relations, but their
symbolic coefficients can still grow; this is not a polynomial-time guarantee.

The compiler groups overlapping source modes and collects the linear images
as matrix rows. It checks their Gram matrix and completes only the orthogonal
complement. Disconnected groups stay separate, including bosonic and fermionic
groups. A direct two-mode completion avoids singular denominators at special
rotation angles.

Rotated-mode names depend on the source group, target group, and rotation matrix.
SymPy's boson and fermion adjoints stringify names, so the compiler uses ordinary
symbols with deterministic names. An attached NOF parameter substitution first
undoes the compiled rotation, substitutes in the declared source expression, and reattaches through the new
embedding. This also handles reordered groups and disappearing rotations.
Plain source operands are converted into this basis before multiplying an
attachment; mixing original and rotated names would represent extra modes.

### Rectangular arithmetic

The solver constructs the fixed occupation projector $P=WW^\dagger$ and its
complement $Q=1-P$, using equality-based `Piecewise` expressions. It prepares
Hamiltonian blocks and an energy-gap solver for the standard `block_diagonalize`
driver. Virtual products do not enumerate or truncate the discarded states.

For each of `H_eff`, `U`, and `U_adjoint`, block `[0, 0, ...]` uses the target
operators or reference-list matrix basis. Block `[1, 1, ...]` uses source
operators on the complement. The off-diagonal blocks are rectangular maps:
$XW$ or $W^\dagger X$. Their NOFs carry the fixed `embedding` and a `side`
(`1` for a right attachment, `-1` for a left attachment). The `source` property
returns the ordinary operator $X$.

These blocks support ordinary addition, multiplication, and adjoints. In
particular, $W^\dagger XW$ contracts immediately to an ordinary target operator,
and $(XW)(W^\dagger Y)=XPY$. Source multiplication preserves the attachment;
it does not project or normalize after each operation. Support is reduced
before dividing by an energy gap, so cancelling virtual amplitudes do not
produce spurious resonance errors.

For a generator embedding, algebraic use looks like this:

```python
from pymablock.number_ordered_form import NumberOrderedForm

embedding = Embedding({s: a}, reference={a: 0})
W = NumberOrderedForm.from_expr(embedding)
X = NumberOrderedForm.from_expr(a + Dagger(a))
rectangular = X * W
compressed = W.adjoint() * rectangular  # embedding.restrict(X)
```

A reference-list embedding is prepared
as a matrix of NOFs sharing one vacuum attachment. Matrix indices select the
source components, and normalized creation monomials prepare the listed
occupations. There is no separate embedding wrapper around the matrix.

With linear mode mixing, source operators use the compiler's rotated modes.
The usual `zero` and `one` series sentinels represent zero and the identity on
the block's space. Embedding attachments occur only on rectangular blocks;
the effective Hamiltonian has already been contracted into the target algebra.

`NumberOrderedForm.as_expr()` exports diagonal projectors as ordinary SymPy
`Piecewise` expressions, which can be read back with `from_expr()`. A SymPy
assumptions patch preserves noncommutativity when the conditions contain
operators; the scalar occupation coefficients inside a NOF remain commutative.
NOF coefficient arithmetic applies point-projector constraints directly, so
embedding products use the same reduction as ordinary NOF products.

The source projector selects a joint spectrum of commuting number operators.
Writing the occupation map as $n=r+Mm$, let $LM=I$ and let the rows of $C$
form a basis of the left nullspace of $M$. The projector imposes
$C(n-r)=0$ and selects the allowed spectrum of each target number operator
in $L(n-r)$. Using independent nullspace constraints avoids redundant
occupation equations. A reference state is the special case that selects one
eigenvalue of every source number operator. Finite spectral selections are sums
of equality indicators; integer spectra use the condition $x=\lfloor x\rfloor$.

The Sylvester solver evaluates each source transition on the incoming occupations
$n=r+Mm$, then divides by the difference between its outgoing and incoming
energies. The embedding already fixes the incoming support, so the solver does
not multiply by the source projector. Reference-list columns use their vacuum
attachment and creation monomial in the same transition calculation.
Coefficient division is shared with regular second quantization. It evaluates
occupations already fixed by explicit indicators, then divides the coefficient
by the energy gap. Where the full transition amplitude is defined and zero,
it chooses zero even if the gap also vanishes. An explicit zero gap with a
nonzero amplitude raises an error; unresolved gaps remain symbolic poles.
The helper does not search binary occupation sectors or solve for integer roots.
Scalar parameters are treated as generic, and arithmetic shortcuts avoid
conditional expressions when the gap cannot vanish as occupations change.
A returned expression therefore does not certify nonresonance in every sector.

For infinite bosonic targets, the solver currently requires that nonnegative
target occupations follow from the physical source occupations. Other selections
raise `NotImplementedError` because they require occupation inequalities.
Finite reference lists and binary targets use only equality conditions.
Avoiding basis enumeration does not remove expression growth at high orders.

The [application documents](applications/index.md) reproduce the supercurrent,
Crépel–Fu interaction model, tunable coupler, and artificial cavity spin.
See also the
[API reference](documentation/pymablock.md#structured-embeddings).
