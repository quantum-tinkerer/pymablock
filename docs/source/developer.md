```{include} ../../CONTRIBUTING.md
```

## Structured embeddings

| Name | Meaning |
|---|---|
| Source | Hilbert space of the effective model, with occupations $q$ |
| Target | Hilbert space of the input Hamiltonian, with occupations $n$ |
| $W$ | Isometry from source to retained target states |
| $P=WW^\dagger$, $Q=1-P$ | Retained and complement target projectors |
| Lattice | Target occupations $n=r+Mq$ reached from reference $r$ |
| Transfer $T_j$ | Target operator carrying the first lattice to lattice $j$ |

`Embedding` is the structural SymPy expression, publicly imported from
`pymablock.operator_embedding`. Its arguments are generators and required
reference occupations. A mapping reference defines one lattice; a list adds an
ordered source matrix index. Column $j$ has target row $\rho_j$ and occupations
$r_j+Mq$. Reconstruction, substitution, and pickling rebuild derived data from
these arguments. Different target rows are disjoint; lattices in the same row
must not overlap.

`_Lattice` compiles occupation maps, phases, generators, and single-lattice
arithmetic. Its unprefixed methods are package-internal contracts. The columns
of $M$ are independent integer shifts. A left inverse $L$ gives $q=L(n-r)$;
the left nullspace supplies the remaining occupation constraints. Validation
checks domains, reference vacua, ladder norms, and pair commutation or fermion
anticommutation on retained states. Binary identities reduce modulo $q^2-q$;
spectator occupations stay symbolic. No target Fock basis is enumerated.

`_EmbeddedMap` stores a target NOF $X$, a scalar embedding, and a side: $XW$
or $W^\dagger X$. It supports addition, adjoints, scalar arithmetic, and
composition. Products contract eagerly: $(W^\dagger X)(YW)$ restricts $XY$,
while $(XW)(W^\dagger Y)=XPY$. A source factor is lifted to $WSW^\dagger$;
scalars multiply $X$ directly. Zero detection tests $XP$ or $PX$, depending on
the side. Equality and hashing use structural arguments. Maps are rectangular
and have no `act` method.

The full frame uses $W_j=T_jW_1$. Transfers include normalization and fermionic
phases, so restricting a matrix entry means restricting
$T_i^\dagger X_{\rho_i,\rho_j}T_j$ on the first lattice. `Embedding.restrict`
forms $W^\dagger XW$ and unwraps a mapping reference's one-by-one result.
`_Lattice.restrict` returns a source NOF even without source modes; only output
boundaries unwrap operator-free entries. `lift` substitutes source generators
and number coordinates structurally. Existing NOFs change mode order through
`_expand_operators`; they never make a round trip through printed expressions.

`_split_embedding_series` converts general operators with the retained and
complement frames, preserving zeroth-order cross blocks. Hamiltonian preparation
validates diagonal $H_0$ and its vanishing cross blocks before omitting them.
The embedding solver gets retained energies from $W^\dagger H_0W$.
`map_transition_coefficients` handles coordinate conversion; the solver supplies
energy-gap division. Diagonal blocks selected by `fully_diagonalize` use the
ordinary second-quantized solver. Both solvers share the division policy.

`NumberOrderedForm.act` supplies destination occupations and ladder amplitudes
for concrete integer ladder powers. Division uses the amplitude to identify
inactive transitions but divides the bare coefficient, avoiding double-counted
ladder factors. It evaluates target energies at the destination and source
energies at the incoming state. A binary-zero source amplitude is discarded
before division. Incoming lattice guards prevent off-lattice evaluation, and
the solver multiplies the divided target operator by the projector afterwards.

Occupation conditions have five `Piecewise` shapes in NOF coefficients:

- Indicators: `((1, Eq(x, v)), (0, True))`; products and sums build projectors.
- Boundary guards: `((0, Eq(n, i) | ...), (c, True))`, suppressing poles where
  downward ladder factors vanish.
- Inactive transitions: `((0, inactive), (c / gap, True))`.
- Point splits: `((value at v, Eq(n, v)), (rest, True))`.
- Lattice guards: `((c, Ne(projector, 0)), (0, True))`.

Only indicators are read back as fixed occupations. The other shapes must
remain branches: multiplying a pole by a zero indicator would evaluate
$0\cdot\infty$. `as_expr` exports operator-valued conditions. A feature-detected,
global SymPy `Piecewise` patch preserves their noncommutativity; occupation
placeholders inside NOF coefficients remain commutative.

Division first applies fixed occupations. A defined zero transition has zero
solution even at zero gap. An exposed zero gap with nonzero amplitude raises
`ValueError`; unresolved resonances stay symbolic poles. Scalar parameters are
generic. The solver does not enumerate binary sectors or integer roots, so a
returned expression does not certify nonresonance in every sector.

Supported embeddings map source states to individual target occupation states,
up to phases. Arbitrary superpositions and dependent generator shifts are
unsupported. Infinite bosonic source domains must follow from nonnegative target
occupations; otherwise projector construction raises `NotImplementedError`.
Transfers along generator directions are unsupported. Finite lists and binary
sources use equality conditions. Symbolic ladder powers in `act` and `restrict`
are rejected. Avoiding basis enumeration does not prevent expression growth.

The [applications](applications/index.md) and
[API reference](documentation/pymablock.md#structured-embeddings) demonstrate
the public contracts; finite-matrix oracle tests check results independently.

## Writing application pages

The [application pages](applications/index.md) are executable MyST notebooks
for physicists who know perturbation theory but not Pymablock internals.
New pages follow the same standard.

- **Question and answer first.** The opening paragraph states the physical
  question, the effective model, and the main result as a formula.
- **Fixed structure.** Use the sections Model, What is kept, Perturbation
  theory, Result and check, and Discussion.
- **Math, then code.** Write each Hamiltonian as an equation, then as code whose
  names match its symbols. Avoid generic builders and loops that hide terms.
  A short loop or comprehension that mirrors a sum in the equation is fine.
- **Short cells.** Keep cells to about 15 lines, each doing the one thing that
  the preceding prose announces. Comment only what the prose does not explain.
- **Public API only.** Extract results with `restrict`, `as_expr`,
  `filter_terms`, and `simplify`; never read private attributes.
- **Explain the embedding.** Before `Embedding(...)`, say in one or two
  sentences which states are kept and why.
- **Check physics.** Assert agreement with the literature, a limiting case, a
  symmetry, or another physical constraint such as Hermiticity or a
  conservation law. Cite references in the prose. Leave out checks that only
  verify an implementation step.
- **Plain prose.** Use short sentences and define every symbol at first use.
  Avoid implementation terms such as lattice, frame, or NOF. Say "effective
  model" and "full Hamiltonian".
- **Runtime.** Each page runs in under about a minute. If it does not, reduce
  the order or the model where the physics survives, and say what was dropped.
- **Readable output.** Display the key equations with `display`. Plot or
  abbreviate results that are too long to read; do not print them raw.
