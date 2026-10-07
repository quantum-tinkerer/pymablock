```{include} ../../CONTRIBUTING.md
```

## Structured embeddings

As in the user documentation, the *source* is the effective model and the
*target* is the Hamiltonian passed to `block_diagonalize`. The embedding is an
isometry $W$ from source states to target states.

`Embedding` is a structural SymPy expression whose constructor selects one of two subclasses.
`_GeneratorEmbedding` compiles generator images and the affine occupation map.
`_ReferenceEmbedding` compiles an ordered list of target states.
Both store compiled data directly and recompile when reconstructed or unpickled.
Shared target normalization, attachments, and block conversion live on `Embedding`.
The subclasses implement compression, projectors, lifting, and frame columns.

The implementation separates three operations:

- `Embedding._convert` constructs the retained/complement frames
  and converts general operators. It preserves zeroth-order cross blocks.
- `Embedding._sylvester_solver` validates diagonal H0
  and divides transitions by their actual energy differences. `Embedding._prepare`
  builds it for `block_diagonalize` and then omits zeroth-order cross blocks of that
  validated Hamiltonian.
- `number_ordered_form` composes rectangular operators attached to an
  `Embedding` and supplies scalar projectors and support reduction. Both
  Sylvester solvers share coefficient division in `second_quantization`.

`NumberOrderedForm.act` applies each term to a Fock state, returning its
destination and matrix element. Compression and division share this action.
Division uses the matrix element only to identify inactive transitions; it
divides the bare NOF coefficient, so ladder factors are not applied twice. The zero series sentinel is handled before attempting matrix operations.
For `fully_diagonalize`, transitions within either diagonal block use the
ordinary second-quantized Sylvester solver, with retained or target energies.
Transitions between the blocks use the embedding-aware solver.

Method caches belong to their compiled instance, so discarding an embedding
also discards its conversion caches. The scalar projector cache has a fixed size.
SymPy compatibility patches live in `number_ordered_form`; the condition
workaround is feature-detected and installed once, including across reloads.

### Validation

These checks apply to the compressed generators $PGP$. Each generator must
have the source ladder norm, and each pair must commute (anticommute for two
fermions) on the retained occupation lattice. The reference and occupation
boundaries fix the vacuum and finite truncation; the norm and pair relations
then also determine the relations involving adjoints. Thus checking the target
algebra outside the retained space is unnecessary: a boson can represent a
two-state source even though their uncompressed commutators differ.

Validation leaves spectator occupations symbolic. Binary polynomial identities
are reduced modulo $n^2-n$, rather than checked separately at every combination
of occupations. There are quadratically many generator relations, but their
symbolic coefficients can still grow; this is not a polynomial-time guarantee.

Each generator image has one fixed occupation change, so retained states are
target Fock states and target operators keep their declared names. Substitution
and replacement in an attached NOF act on its arguments, including the embedding.
Renaming a mode also renames its number-operator placeholder, and the constructor
reorders the result to match the replaced embedding.

### Rectangular arithmetic

The solver constructs the fixed occupation projector $P=WW^\dagger$ and its
complement $Q=1-P$, using equality-based `Piecewise` expressions. It prepares
Hamiltonian blocks and an energy-gap solver for the standard `block_diagonalize`
driver. Virtual products do not enumerate or truncate the discarded states.

For each of `H_eff`, `U`, and `U_adjoint`, block `[0, 0, ...]` uses the source
operators or reference-list matrix basis. Block `[1, 1, ...]` uses target
operators on the complement. The off-diagonal blocks are rectangular maps:
$XW$ or $W^\dagger X$. Their NOFs carry the fixed `embedding` and a `side`
(`1` for a right attachment, `-1` for a left attachment). The `target` property
returns the ordinary target operator $X$.

These blocks support ordinary addition, multiplication, and adjoints. In
particular, $W^\dagger XW$ contracts immediately to an ordinary source operator,
and $(XW)(W^\dagger Y)=XPY$. Multiplying by a target operator preserves the
attachment; it does not project or normalize after each operation. Support is
reduced before dividing by an energy gap, so cancelling virtual amplitudes do
not produce spurious resonance errors.

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
target components, and normalized creation monomials prepare the listed
occupations. There is no separate embedding wrapper around the matrix.

The usual `zero` and `one` series sentinels represent zero and the identity on
the block's space. Embedding attachments occur only on rectangular blocks;
the effective Hamiltonian has already been contracted into the source algebra.

`NumberOrderedForm.as_expr()` exports diagonal projectors as ordinary SymPy
`Piecewise` expressions, which can be read back with `from_expr()`. A SymPy
assumptions patch preserves noncommutativity when the conditions contain
operators; the scalar occupation coefficients inside a NOF remain commutative.
NOF coefficient arithmetic applies point-projector constraints directly, so
embedding products use the same reduction as ordinary NOF products.

The target projector selects a joint spectrum of commuting number operators.
Writing the occupation map as $n=r+Mm$, let $LM=I$ and let the rows of $C$
form a basis of the left nullspace of $M$. The projector imposes
$C(n-r)=0$ and selects the allowed spectrum of each source number operator
in $L(n-r)$. Using independent nullspace constraints avoids redundant
occupation equations. A reference state is the special case that selects one
eigenvalue of every target number operator. Finite spectral selections are sums
of equality indicators; integer spectra use the condition $x=\lfloor x\rfloor$.

The solver reads `energy_states`, `target_occupations`, `coordinate_symbols`,
and `source_coordinates` from the embedding; reference lists have no coordinates.
Generator embeddings compute `source_coordinates`, $L(n-r)$, once and reuse them
in the projector, lifting, and the solver's coordinate substitution.
Reference-list frame entries share an internal vacuum embedding.
Matrix shape consistency belongs to each block conversion, so one embedding can
be reused for operators of different target matrix sizes.

The Sylvester solver evaluates each target transition on the incoming occupations
$n=r+Mm$, then divides by the difference between its outgoing and incoming
energies. The embedding already fixes the incoming support, so the solver does
not multiply by the target projector. Reference-list columns use their vacuum
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

For infinite bosonic source modes, the solver currently requires that nonnegative
source occupations follow from the physical target occupations. Other selections
raise `NotImplementedError` because they require occupation inequalities.
Finite reference lists and binary source modes use only equality conditions.
Avoiding basis enumeration does not remove expression growth at high orders.

The [application documents](applications/index.md) reproduce the supercurrent,
Crépel–Fu interaction model, tunable coupler, and artificial cavity spin.
See also the
[API reference](documentation/pymablock.md#structured-embeddings).
