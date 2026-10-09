```{include} ../../CONTRIBUTING.md
```

## Structured embeddings

The **source** is the effective model's Hilbert space; the **target** is the
larger Hilbert space of the original Hamiltonian passed to `block_diagonalize`.
The embedding is an isometry $W:\mathcal H_\mathrm{source}\to\mathcal H_\mathrm{target}$.
The retained subspace is the image of $W$ inside the target; its complement also
belongs to the target space.

`Embedding` is the structural SymPy expression defining the isometry. Every
instance contains an ordered `(row, lattice)` collection and transfer operators.
A mapping reference gives one lattice; a list adds a source matrix index.
Column $j$ maps $q$ to target row $i_j$ and occupations $r_j+Mq$.

Each private `_Lattice` owns the compiled generator images, occupation maps,
phases, validation, and single-lattice operator arithmetic. It is derived data,
excluded from the embedding's symbolic arguments and rebuilt when an embedding
is reconstructed or unpickled. Lattices in one embedding share symbolic source
occupations, with nonnegative assumptions for physical modes and signed indices
for `LadderOp`. NOF number-placeholder assumptions are unchanged.

`Embedding._first_embedding` supplies the single-reference symbolic identity
used by rectangular NOF attachments. For a list it shares the first compiled
lattice without recompiling it. The attachment remains an `Embedding`, so SymPy
substitution and pickling rebuild it from its public constructor arguments.
NOF arithmetic accesses the scalar lattice through `_attachment_lattice`, which
rejects reference-list frames. Unprefixed members of the private `_Lattice` class
form its package-internal contract. Lattice transition mapping owns coordinate
conversion; the solver supplies energy-gap division and resonance handling.
Solver results preserve the incoming NOF attachment through `_rebuild`.

Lifting substitutes source NOF terms into target NOFs. Scalar number
placeholders map directly to the compiled source coordinates; generator NOFs
and their adjoints supply the ladder factors and graded signs. Existing NOFs
are extended to the target mode order structurally. Embedding arithmetic never
converts NOFs to expressions and back. `_Lattice.parse_target` parses expressions
at the input boundary; `NumberOrderedForm._expand_operators` changes the operator
list of an existing NOF structurally and refuses to drop an operator that a term uses.
`Embedding` handles construction, format boundaries, and full-frame operations;
`_Lattice` handles the occupation maps and single-lattice mathematics.

Every frame entry uses the same scalar isometry $W_1$ at the first reference.
The constructor builds target operators $T_j$ such that $W_j=T_jW_1$.
Their occupation shifts are $r_j-r_1$. If the bare shift monomial has
matrix element $t_j(q)$, its coefficient is
$\phi_j(q)/(\phi_1(q)t_j(q))$, expressed at the intermediate NOF occupations.
`NumberOrderedForm.act` supplies the ladder factors and fermion signs;
the generator validation supplies each phase $\phi_j$.
`restrict` always multiplies the frames $W^\dagger XW$, then unwraps a mapping
reference's one-by-one result. Restriction calls `_retained_frame` directly;
Hamiltonian preparation and block conversion also call `_complement_frame`.
Both frames are cached on the owning embedding. Attached scalar contractions call
`_Lattice.restrict` directly, avoiding recursion through `Embedding.restrict`.
Compression always returns a
NOF, including when there are no source operators. `_format_output` unwraps
operator-free entries and the mapping reference's matrix axis at the output
boundary. Later perturbative arithmetic may retain zero-mode NOFs in scalar
matrix entries. Compression of source entry $(i,j)$ acts on
$T_i^\dagger X_{\rho_i,\rho_j}T_j$, with $\rho_j$ the declared target row.
Frame products also give
$T_iP_1T_j^\dagger$ without a separate list projector or compression algorithm.

The implementation separates three operations:

- `block_diagonalization._split_embedding_series` uses the embedding's
  retained/complement frames to convert general operators. It preserves
  zeroth-order cross blocks.
- `second_quantization.solve_sylvester_embedding(h0, embedding)` validates diagonal
  H0, reads retained energies directly from $W^\dagger H_0W$, and dispatches matrix
  blocks. Its helper `_divide_transitions` takes a lattice, a target NOF, and
  target/source energy expressions and returns a target NOF divided by its
  transition gaps. `block_diagonalize`
  constructs the solver, validates the complement projector, and splits the
  Hamiltonian into blocks, omitting the validated zeroth-order cross blocks.
- `number_ordered_form` composes rectangular operators attached to an
  `Embedding` and supplies scalar projectors and support reduction. Both
  Sylvester solvers share coefficient division in `second_quantization`.

`NumberOrderedForm.act` applies each term to a Fock state, returning its
destination and matrix element. Compression and division share this action.
Division uses the matrix element only to identify inactive transitions; it
divides the bare NOF coefficient, so ladder factors are not applied twice. The zero series sentinel is handled before attempting matrix operations.
For `fully_diagonalize`, transitions within either diagonal block use the
ordinary second-quantized Sylvester solver, with source or target energies.
Transitions between the blocks use the embedding-aware solver.

Frame caches belong to the embedding; lattice conversion caches belong to the
compiled lattice. Discarding the embedding and its attached operators therefore
allows both to be collected. The scalar projector cache has a fixed size.
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

A reference-list frame is a matrix of NOFs attached to $W_1$.
The zero-generator case uses the same transfer construction: a shift monomial
maps the first listed occupation state to each other state, including negative
bilateral ladder indices. A vacuum first reference gives normalized creation
monomials as a special case.

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

The solver reads `_column_target_states` from the embedding to evaluate the source
energy of column $j$ at $r_j+Mq$. The attached target operator already includes
$T_j$. The lattice's `map_transition_coefficients` evaluates its transitions
from $r_1+Mq$ and translates transformed coefficients back using $L(n-r_1)$.
The solver evaluates the target energy at each final target occupation and
divides by the target-minus-source energy gap; it does not inspect occupation maps.
Matrix shape consistency belongs to each block conversion, so an embedding can
be reused for operators of different target matrix sizes.

Every reference lattice independently passes domain, generator-algebra, phase,
and ladder-number validation. These bounds also ensure the transfer monomial
never annihilates a state of the first lattice. Translations along modes moved
by a generator are currently rejected. Spectator boson normalizations are
constant, while phase ratios include occupation-dependent fermion ordering signs.
For references in the same target row, independent columns of $M$ give a unique
candidate displacement $d=L(r_i-r_j)$. The lattices overlap exactly when
$Md=r_i-r_j$, $d$ is integral, and $|d_a|<D_a$ for every finite source mode of
size $D_a$. Infinite bosonic and bilateral domains have all integers as their
difference set. Different target rows are disjoint automatically.

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
