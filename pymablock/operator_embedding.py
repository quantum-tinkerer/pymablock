"""Operator embeddings generated from a source reference state."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Self

import sympy
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOperator,
    NumberOrderedForm,
    OperatorType,
    _allowed_values_indicator,
    _NOFTransition,
    _number_symbols,
    _occupation_dimension,
    find_operators,
    generator_types,
)
from pymablock.series import BlockSeries, zero

if TYPE_CHECKING:
    from typing import Any

__all__ = ["Embedding"]


def _cache_on_instance[Result](method: Callable[..., Result]) -> Callable[..., Result]:
    """Memoize a method with hashable positional arguments on its owning instance.

    Unlike a global method cache, this does not keep discarded compiled bases alive.
    The wrapped methods accept positional arguments only.
    """

    @wraps(method)
    def cached(self: object, *args: Hashable) -> Result:
        memo = self.__dict__.setdefault("_memo_" + method.__name__, {})
        if args not in memo:
            memo[args] = method(self, *args)
        return memo[args]

    return cached


def _operator_sort_key(operator: OperatorType) -> tuple[int, str]:
    """Order modes by algebra type, then name, matching NOF's fermionic convention."""
    return generator_types.index(type(operator)), str(operator.name)


class Embedding(sympy.Expr):
    r"""Select the states of an effective model inside a second-quantized Hamiltonian.

    Below, *source* refers to the operators of the Hamiltonian passed to
    `~pymablock.block_diagonalize`, and *target* refers to the operators of the
    effective model. An embedding either describes how each target operator acts
    on the source states, or lists source states that form a finite matrix basis.

    Pass the embedding as ``subspace_eigenvectors`` to
    `~pymablock.block_diagonalize` or `~pymablock.operator_to_BlockSeries`.
    Block 0 contains the embedded states and block 1 contains all other source
    states, which remain available for virtual transitions.

    Parameters
    ----------
    generators : collections.abc.Mapping
        Maps each target lowering operator to a source expression. The keys may be
        ``SigmaMinus``, ``FermionOp``, ``BosonOp``, or ``LadderOp`` operators. A
        ``LadderOp`` target also needs an entry mapping its ``NumberOperator`` to a
        source expression, because its lowering operator does not determine it.
        Omit ``generators`` to select a finite list of states instead.
    reference : collections.abc.Mapping or collections.abc.Sequence
        If ``generators`` is given, maps every source operator to its occupation in
        the source state that represents the target state with all occupations and
        ``LadderOp`` indices equal to zero. Applying the adjoints of the source
        expressions to this state defines all other embedded states and their
        phases. ``LadderOp`` targets also use the source expressions themselves to
        reach negative indices.

        Otherwise, lists the embedded states in the order of the matrix rows and
        columns. Each state maps every source operator to its occupation, and all
        states must use the same operators. If the Hamiltonian is a matrix, give
        ``(matrix_index, occupations)`` pairs; a mapping without an index refers to
        index 0. For a matrix Hamiltonian without operators, use pairs with empty
        mappings, such as ``[(0, {}), (2, {})]``.

    Notes
    -----
    Boson occupations are nonnegative, fermion and spin occupations are 0 or 1,
    and ``LadderOp`` indices may be any integer. Each source expression must
    change every source occupation by a fixed amount, possibly zero, as a
    product of creation and annihilation operators does. Embedded states are
    therefore Fock states of the source operators. To embed a combination such
    as ``(c1 + c2) / sqrt(2)``, first rewrite the Hamiltonian in terms of new
    operators for that combination and those orthogonal to it.

    The matrix elements of each source expression between embedded states must
    equal those of the target operator, up to a phase factor. For example,
    ``{s: 2 * a}`` raises an error instead of being normalized. Phases may depend
    on occupations, such as fermion signs, only for spin and fermion targets.

    With ``generators``, the block 0 entries are
    `~pymablock.number_ordered_form.NumberOrderedForm` objects in the target
    operators. This includes bosonic or ``LadderOp`` targets with infinitely many
    states. With a list of states, block 0 entries are SymPy matrices in the
    order of the list. Block 1 uses the source operators. Blocks (0, 1) and
    (1, 0) map between target and source states.

    Perturbation theory with an embedding requires:

    - A Hermitian Hamiltonian.
    - An unperturbed Hamiltonian that is diagonal in the source occupations.
    - For matrix Hamiltonians, an unperturbed Hamiltonian that is also diagonal
      in the matrix index, with all perturbative orders of the same shape.
    - For each bosonic target, an occupation that is nonnegative for every source
      state. For example, a boson defined in terms of a ``LadderOp`` is not
      supported.

    The solver uses exact symbolic arithmetic. It treats a floating-point number
    as the exact fraction that the computer stores, so ``0.1`` differs slightly
    from ``sympy.Rational(1, 10)``. Use rational numbers when exact decimal
    values matter. The numerical options ``atol``, ``direct_solver``, and
    ``solver_options`` are not supported.

    See the :doc:`structured embeddings documentation <../structured_embeddings>`
    for examples with physical models.

    Examples
    --------
    >>> import sympy
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.pauli import SigmaMinus
    >>> from pymablock.number_ordered_form import NumberOperator
    >>> a, s = BosonOp("a"), SigmaMinus("s")

    Represent a spin by the two lowest levels of the oscillator ``a``:

    >>> embedding = Embedding({s: a}, reference={a: 0})
    >>> embedding.restrict(a).as_expr() == s
    True

    Select the three lowest levels as a finite matrix basis:

    >>> embedding = Embedding(reference=[{a: 0}, {a: 1}, {a: 2}])
    >>> embedding.restrict(NumberOperator(a)) == sympy.diag(0, 1, 2)
    True

    """

    is_commutative = False

    def __new__(
        cls,
        generators: Mapping | sympy.Expr | None = None,
        reference: Mapping | Sequence | None = None,
    ) -> Self:
        """Compile a structurally reconstructible generator or reference map."""
        if reference is None:
            raise TypeError("Specify reference occupations for the embedding")
        if generators is None or generators is sympy.S.NaN:
            reference = tuple(
                (0, state) if isinstance(state, (Mapping, sympy.Dict)) else state
                for state in reference
            )
            basis = _ReferenceBasis([(i, dict(state)) for i, state in reference])
            generators = sympy.S.NaN
            reference = sympy.Tuple(
                *(sympy.Tuple(i, sympy.Dict(state)) for i, state in reference)
            )
        else:
            basis = _GeneratorBasis(dict(generators), reference=dict(reference))
            generators, reference = sympy.Dict(generators), sympy.Dict(reference)
        result = sympy.Expr.__new__(cls, generators, reference)
        result._basis = basis
        return result

    @cached_property
    def _projector(self) -> NumberOrderedForm:
        """Return ``W W†``, the source-space projector onto retained states.

        Scalar indicators enforce the affine occupation constraints and each target
        mode's allowed numbers. No source occupation states are enumerated.
        """
        basis = self._basis
        numbers = basis._source_placeholders
        if isinstance(basis, _ReferenceBasis):
            spectra = list(zip(numbers, ((n,) for n in basis._references[0][1])))
        else:
            offsets = sympy.Matrix(numbers) - sympy.Matrix(basis.reference)
            spectra = [
                (sympy.expand(normal.dot(offsets)), (0,))
                for normal in basis._occupation_matrix.T.nullspace()
            ]
            target = basis._occupation_left_inverse * offsets
            physical = {
                n: sympy.Dummy(integer=True, nonnegative=True)
                for source, n in zip(basis.operators, numbers)
                if not isinstance(source, LadderOp)
            }
            for q, op, size in zip(
                target, basis._target_operators, basis._target_dimensions
            ):
                if (
                    isinstance(op, BosonOp)
                    and q.xreplace(physical).is_nonnegative is not True
                ):
                    raise NotImplementedError(
                        "This bosonic embedding requires an occupation inequality"
                    )
                spectra.append((q, range(size) if size is not None else sympy.S.Integers))
        indicator = sympy.prod(
            _allowed_values_indicator(q, spectrum) for q, spectrum in spectra
        )
        return NumberOrderedForm(basis.operators, {(0,) * len(numbers): indicator}) * 1

    @property
    def _source_operators(self) -> tuple[OperatorType, ...]:
        """Return source lowering modes in the compiled basis order."""
        return self._basis.operators

    def _convert_operator(self, value: sympy.Expr) -> NumberOrderedForm:
        """Normalize one source operator in the compiled mode basis."""
        return self._basis._convert_operator(value)

    def _contract(self, value: NumberOrderedForm) -> NumberOrderedForm | sympy.Expr:
        """Compute ``W† value W``, unwrapping a single-reference 1x1 result."""
        result = self.restrict(value)
        return result[0, 0] if isinstance(result, sympy.MatrixBase) else result

    @_cache_on_instance
    def _lift(self, value: NumberOrderedForm) -> NumberOrderedForm:
        """Represent a target operator in source space, as ``W value W†``.

        Replace target generators and numbers by their source images, then project
        both sides onto the retained space. A single-reference attachment already
        uses source operators and only needs normalization.
        """
        basis = self._basis
        if isinstance(basis, _ReferenceBasis):
            return basis._convert_operator(value)
        coordinates = basis._occupation_left_inverse * (
            sympy.Matrix([NumberOperator(op) for op in basis.operators])
            - sympy.Matrix(basis.reference)
        )
        images = dict(zip(map(NumberOperator, basis._target_operators), coordinates))
        for op, generator in zip(basis._target_operators, basis._generators):
            image = NumberOrderedForm(
                generator.operators,
                {generator.powers: generator.coefficient},
                validate=False,
            )
            images[op] = image.as_expr()
            images[op.adjoint()] = image.adjoint().as_expr()
        result = NumberOrderedForm.from_expr(
            value.as_expr().xreplace(images), basis.operators
        )
        return self._projector * result * self._projector

    def _attach(self, value: sympy.Expr, side: int) -> NumberOrderedForm:
        """Represent ``value W`` for side +1, or ``W† value`` for side -1.

        Normalize the source expression but retain the attachment until arithmetic
        contracts it. Reference lists use matrices of single-reference attachments.
        """
        if isinstance(self._basis, _ReferenceBasis) and (
            len(self._basis._references) != 1 or self._basis._references[0][0] != 0
        ):
            raise ValueError(
                "Reference lists use matrices of single-reference attachments"
            )
        value = self._basis._convert_operator(value)
        return NumberOrderedForm(
            value.operators, value.args[1], self, side, validate=False
        )

    def restrict(
        self, expression: sympy.Expr | sympy.MatrixBase
    ) -> NumberOrderedForm | sympy.MatrixBase:
        """Express a source operator in terms of the embedded states.

        The result contains the matrix elements of ``expression`` between embedded
        states, without perturbative corrections. Products are evaluated in the full
        source space before taking these matrix elements, so intermediate states
        outside the embedding contribute. For example, with ``{s: a}``,
        ``restrict(a * Dagger(a))`` is ``1 + N_s``, while
        ``restrict(a) * restrict(Dagger(a))`` is ``1 - N_s``.

        With ``generators``, the result is a
        `~pymablock.number_ordered_form.NumberOrderedForm` in the target operators.
        It may contain products such as ``N_s * (1 - N_s)`` that vanish because
        fermion and spin occupations are 0 or 1; call ``simplify()`` to remove
        them. With a list of states, the result is a SymPy matrix in the order of
        the list.
        """
        return self._basis._compress(self._basis._convert_source(expression))


class _EmbeddingBlocks:
    """Reusable conversion from source expressions to retained/complement blocks."""

    def __init__(self, embedding: Embedding) -> None:
        """Store a compiled embedding; infer source matrix size on first conversion."""
        self.embedding, self.basis = embedding, embedding._basis
        self.finite = isinstance(self.basis, _ReferenceBasis)
        self.source_shape: tuple[int, int] | None = None

    def source_matrix(
        self, expression: sympy.Expr | sympy.MatrixBase
    ) -> sympy.MatrixBase:
        """Normalize a source coefficient as a square matrix with a consistent size."""
        source = self.basis._convert_source(expression)
        if not self.finite:
            source = sympy.ImmutableMatrix([[source]])
        if self.source_shape is None:
            self.source_shape = source.shape
        elif source.shape != self.source_shape:
            raise ValueError(
                "All operator coefficients must have the same source matrix shape"
            )
        return source

    @cached_property
    def entry_embedding(self) -> Embedding:
        """Return the attachment shared by entries of the retained-space frame."""
        if not self.finite:
            return self.embedding
        return Embedding(reference=[dict.fromkeys(self.basis.operators, 0)])

    @cached_property
    def frames(self) -> tuple[sympy.MatrixBase, sympy.MatrixBase]:
        """Return retained frame W and complement projector Q = 1 - W W†.

        Source matrix size must already be set by ``source_matrix``. Reference
        columns are normalized creation monomials acting on a vacuum attachment.
        """
        basis = self.basis
        modes = basis.operators
        vacuum = (0,) * len(modes)
        if self.finite:
            w = sympy.zeros(self.source_shape[0], len(basis._references))
            for col, (row, state) in enumerate(basis._references):
                monomial = NumberOrderedForm(
                    modes, {tuple(-n for n in state): sympy.S.One}
                )
                (transition,) = _NOFTransition.from_form(monomial)
                w[row, col] = self.entry_embedding._attach(
                    monomial / transition.apply(vacuum).weight, 1
                )
            w = sympy.ImmutableMatrix(w)
        else:
            w = sympy.ImmutableMatrix([[self.embedding._attach(sympy.S.One, 1)]])
        return w, sympy.eye(self.source_shape[0]) - w * w.adjoint()

    def convert(
        self, operator: BlockSeries, *, diagonal_origin: bool = False
    ) -> BlockSeries:
        """Convert an unseparated series to retained/complement operator blocks.

        Preserve every block by default, including zeroth-order observable cross
        blocks. ``diagonal_origin=True`` omits those cross blocks only for a
        Hamiltonian whose H0 has already been validated by the solver.
        """
        if operator.shape:
            raise ValueError("Structured embeddings require an unseparated operator.")
        origin = (0,) * operator.n_infinite

        def evaluate(i: int, j: int, *order: int) -> Any:
            source = operator[tuple(order)]
            if source is zero or (diagonal_origin and i != j and tuple(order) == origin):
                return zero
            source = self.source_matrix(source)
            result = self.frames[i].adjoint() * source * self.frames[j]
            return (
                zero if result.is_zero_matrix else result if self.finite else result[0, 0]
            )

        return BlockSeries(
            eval=evaluate,
            shape=(2, 2),
            n_infinite=operator.n_infinite,
            dimension_names=operator.dimension_names,
            name=operator.name,
        )


class _SourceBasis:
    """Shared normalization of source expressions and occupation symbols."""

    operators: tuple[OperatorType, ...]

    @cached_property
    def _source_placeholders(self) -> tuple[sympy.Symbol, ...]:
        """Return scalar number symbols in source mode order."""
        return _number_symbols(self.operators)

    def _at_occupations(
        self, expression: sympy.Expr, occupations: Sequence[int | sympy.Expr]
    ) -> sympy.Expr:
        """Evaluate a source number coefficient at the supplied occupations."""
        return expression.xreplace(
            dict(zip(self._source_placeholders, occupations, strict=True))
        )

    def _convert_operator(self, expression: sympy.Expr) -> NumberOrderedForm:
        """Convert an expression to NOF in the compiled source mode order.

        Rationalize stored floating coefficients before cancellation and reject
        modes missing from the reference declaration.
        """
        if isinstance(expression, NumberOrderedForm):
            if expression.operators == self.operators:
                if expression.has(sympy.Float):
                    return expression.applyfunc(
                        lambda c: c.xreplace(
                            {v: sympy.Rational(v) for v in c.atoms(sympy.Float)}
                        )
                    )
                return expression
            expression = expression.as_expr()
        expression = sympy.sympify(expression)
        expression = expression.xreplace(
            {v: sympy.Rational(v) for v in expression.atoms(sympy.Float)}
        )
        if set(find_operators(expression)) - set(self.operators):
            raise ValueError("Every source mode must be declared in the reference")
        return NumberOrderedForm.from_expr(expression, operators=self.operators)


class _GeneratorBasis(_SourceBasis):
    """Symbolic target algebra generated from one reference state."""

    def __init__(
        self,
        generators: Mapping[sympy.Expr, sympy.Expr],
        *,
        reference: Mapping[OperatorType, int],
    ) -> None:
        """Compile the representation generated from the reference."""
        if not isinstance(generators, Mapping) or not isinstance(reference, Mapping):
            raise TypeError("Generators and reference must be mappings")
        if not reference:
            raise ValueError("Specify the source reference occupations")
        if any(
            target in reference and target != image
            for target, image in generators.items()
        ):
            raise ValueError("Target modes must not shadow distinct source modes")
        self.operators, self.reference = _ordered_reference_state(reference)
        numbers = {
            op: value
            for op, value in generators.items()
            if isinstance(op, NumberOperator)
        }
        operators = tuple(op for op in generators if op not in numbers)
        if not all(isinstance(op, generator_types) for op in operators):
            raise TypeError("Keys must be target lowering generators or ladder numbers")
        if any(not op.is_annihilation for op in operators):
            raise ValueError("Target generators must be lowering operators")
        operators = tuple(sorted(operators, key=_operator_sort_key))
        self._target_operators = operators
        self._target_dimensions = tuple(map(_occupation_dimension, operators))
        self.coordinate_symbols = tuple(
            sympy.Dummy(
                f"target_{i}",
                integer=True,
                **({} if isinstance(op, LadderOp) else {"nonnegative": True}),
            )
            for i, op in enumerate(operators)
        )
        transitions = []
        for op in operators:
            form = self._convert_source(generators[op])
            terms = tuple(_NOFTransition.from_form(form))
            if len(terms) != 1 or not any(terms[0].powers):
                raise ValueError(
                    f"Image of {op} must change source occupations by one nonzero shift"
                )
            transition = terms[0]
            parity = (
                sum(
                    power
                    for source, power in zip(
                        self.operators, transition.powers, strict=True
                    )
                    if isinstance(source, FermionOp)
                )
                % 2
            )
            if parity != isinstance(op, FermionOp):
                raise ValueError("Generator images must preserve fermionic parity")
            transitions.append(transition)
        self._generators = tuple(transitions)
        self._occupation_matrix = sympy.Matrix(
            len(self.operators), len(operators), lambda i, j: transitions[j].powers[i]
        )
        if self._occupation_matrix.rank() != len(operators):
            raise ValueError("Generator shifts must be independent")
        matrix = self._occupation_matrix
        self._occupation_left_inverse = (matrix.T * matrix).inv() * matrix.T
        self.source_occupations = tuple(
            origin + sum(matrix[i, j] * q for j, q in enumerate(self.coordinate_symbols))
            for i, origin in enumerate(self.reference)
        )
        self._validate_domains()
        self._validate_generator_algebra()
        self.phase = self._reference_phase()
        expected_numbers = {
            NumberOperator(op) for op in operators if isinstance(op, LadderOp)
        }
        if set(numbers) != expected_numbers:
            raise ValueError(
                "Supply the independent number image for each target LadderOp"
            )
        for op, number in numbers.items():
            index = next(
                i for i, target in enumerate(operators) if NumberOperator(target) == op
            )
            form = self._convert_source(number)
            if any(any(powers) for powers in form.terms):
                raise ValueError("A ladder number image must be occupation diagonal")
            expression = form.terms.get((0,) * len(self.operators), sympy.S.Zero)
            expression = self._at_occupations(expression, self.source_occupations)
            self._validate_identity(
                expression - self.coordinate_symbols[index],
                f"Ladder number image {op} must count from the target reference index zero",
            )

    def _convert_source(self, expression: sympy.Expr) -> NumberOrderedForm:
        """Normalize a scalar source; matrix sources require reference lists."""
        if isinstance(expression, sympy.MatrixBase):
            raise TypeError("Matrix sources require a list of reference states")
        return self._convert_operator(expression)

    def _validate_domains(self) -> None:
        """Check source occupation bounds over the full target occupation domain.

        Use each affine map's minimum and maximum, rather than enumerating states.
        Raise ValueError if a generator can overfill or underfill a source mode.
        """
        for i, op in enumerate(self.operators):
            lower = upper = self.reference[i]
            for coefficient, target, size in zip(
                self._occupation_matrix.row(i),
                self._target_operators,
                self._target_dimensions,
                strict=True,
            ):
                if not coefficient:
                    continue
                if size is None:
                    if isinstance(target, LadderOp):
                        lower, upper = -sympy.oo, sympy.oo
                    elif coefficient > 0:
                        upper = sympy.oo
                    else:
                        lower = -sympy.oo
                else:
                    lower += min(0, coefficient) * (size - 1)
                    upper += max(0, coefficient) * (size - 1)
            if not isinstance(op, LadderOp) and lower < 0:
                raise ValueError("Generators leave the physical source occupation domain")
            if _occupation_dimension(op) is not None and upper >= _occupation_dimension(
                op
            ):
                raise ValueError("Generators overfill a source spin or fermion")

    def _lowering_weight(self, index: int) -> sympy.Expr:
        """Return the target lowering amplitude for mode ``index`` at symbolic numbers."""
        powers = tuple(int(i == index) for i in range(len(self._target_operators)))
        transition = _NOFTransition(self._target_operators, powers, sympy.S.One)
        return transition.symbolic_action(self.coordinate_symbols).weight

    def _reference_phase(self) -> sympy.Expr:
        """Return the phase relating normalized source and target occupation states.

        Compare source and target lowering amplitudes in a fixed creation order.
        Infinite modes require a number-independent ratio; otherwise raise
        NotImplementedError. The reference state's phase is one.
        """
        phase = sympy.S.One
        for i, (transition, q, size) in enumerate(
            zip(
                self._generators,
                self.coordinate_symbols,
                self._target_dimensions,
                strict=True,
            )
        ):
            ratio = (
                self._lowering_weight(i)
                / transition.symbolic_action(self.source_occupations).weight
            )
            ratio = ratio.xreplace(
                dict.fromkeys(self.coordinate_symbols[:i], sympy.S.Zero)
            )
            if size is None:
                ratio = sympy.simplify(ratio)
                if ratio.free_symbols.intersection(self.coordinate_symbols):
                    raise NotImplementedError(
                        "Infinite target generators require a constant phase relative to their ladder weights"
                    )
                phase *= ratio**q
            else:
                phase *= 1 - q + q * sympy.simplify(ratio.xreplace({q: sympy.S.One}))
        return sympy.factor(phase)

    def _validate_identity(self, expression: sympy.Expr, context: str) -> None:
        """Require a residual to vanish on the retained occupation domain.

        Reduce binary polynomials modulo n² - n before testing zero. A disproved
        identity raises ValueError; an undecidable identity raises NotImplementedError.
        """
        expression = sympy.expand(expression)
        for q, size in zip(self.coordinate_symbols, self._target_dimensions):
            if size == 2 and q in expression.free_symbols and expression.is_polynomial(q):
                expression = sympy.rem(expression, q**2 - q, q)
        binary = {
            q
            for q, size in zip(self.coordinate_symbols, self._target_dimensions)
            if size == 2
        }
        if (
            expression != 0
            and expression.free_symbols <= binary
            and expression.is_polynomial(*binary)
        ):
            raise ValueError(f"{context}; residual: {expression}")
        _require_zero(expression, context)

    def _validate_generator_algebra(self) -> None:
        """Check local norms and graded commutation on the retained lattice.

        The occupation boundaries fix the vacuum and binary truncation. Norms
        then fix the same-mode algebra; pairwise lowering relations also fix
        the mixed adjoint relations by reversing an edge of each lattice square.
        """
        weights = [
            g.symbolic_action(self.source_occupations).weight for g in self._generators
        ]
        coordinates = self.coordinate_symbols
        active = {
            q: sympy.S.One if size == 2 else q + 1
            for q, size in zip(coordinates, self._target_dimensions)
        }
        for i, q in enumerate(coordinates):
            norm = (
                weights[i] * sympy.conjugate(weights[i]) - self._lowering_weight(i) ** 2
            )
            self._validate_identity(
                norm.xreplace({q: active[q]}),
                "Generator images must produce normalized target states",
            )
            for j, other in enumerate(coordinates[:i]):
                sign = (
                    -1
                    if all(
                        isinstance(self._target_operators[k], FermionOp) for k in (i, j)
                    )
                    else 1
                )
                relation = weights[j] * weights[i].xreplace(
                    {other: other - 1}
                ) - sign * weights[i] * weights[j].xreplace({q: q - 1})
                self._validate_identity(
                    relation.xreplace({q: active[q], other: active[other]}),
                    "Generator images must obey the target algebra",
                )

    @cached_property
    def _target_placeholders(self) -> tuple[sympy.Symbol, ...]:
        """Return scalar number symbols in target mode order."""
        return _number_symbols(self._target_operators)

    @cached_property
    def _target_zero(self) -> NumberOrderedForm:
        """Return zero carrying the target operator basis."""
        return NumberOrderedForm(self._target_operators, {}, validate=False)

    @_cache_on_instance
    def _target_shift(self, source_shift: tuple[int, ...]) -> tuple[int, ...] | None:
        """Return the induced target shift, or None if compression annihilates it.

        The source shift must lie in the occupation map's image and correspond to
        integer target steps within each finite mode's range.
        """
        source = sympy.Matrix(source_shift)
        result = self._occupation_left_inverse * source
        if self._occupation_matrix * result != source:
            return None
        if any(
            not value.is_Integer or (size is not None and abs(value) >= size)
            for value, size in zip(result, self._target_dimensions, strict=True)
        ):
            return None
        return tuple(map(int, result))

    @_cache_on_instance
    def _project_transition(self, transition: _NOFTransition) -> NumberOrderedForm:
        """Compress one source term, including ladder amplitudes and reference phases.

        Convert its shift and middle coefficient to target coordinates. Spectator
        numbers stay symbolic; a transition outside the retained lattice gives zero.
        """
        powers = self._target_shift(transition.powers)
        if powers is None:
            return self._target_zero
        target_transition = _NOFTransition(self._target_operators, powers, sympy.S.One)
        source_weight = transition.symbolic_action(self.source_occupations).weight
        target_weight = target_transition.symbolic_action(self.coordinate_symbols).weight
        shifted = {q: q - p for q, p in zip(self.coordinate_symbols, powers)}
        # The phase has unit modulus on the retained domain. Its ratio cancels
        # unchanged factors without expanding binary occupation identities.
        amplitude = (
            source_weight / target_weight * self.phase / self.phase.xreplace(shifted)
        )
        # NOF coefficients sit between creation and annihilation operators.
        # A binary transition fixes its input occupation; spectators stay symbolic.
        initial = {
            q: sympy.Integer(p > 0) if p and size == 2 else n + max(p, 0)
            for q, n, p, size in zip(
                self.coordinate_symbols,
                self._target_placeholders,
                powers,
                self._target_dimensions,
            )
        }
        coefficient = sympy.factor_terms(amplitude.xreplace(initial))
        return NumberOrderedForm(
            self._target_operators, {powers: coefficient}, validate=False
        )

    @_cache_on_instance
    def _compress(self, source: NumberOrderedForm) -> NumberOrderedForm:
        """Return ``W† source W`` by translating each term to the target algebra."""
        result = self._target_zero
        for transition in _NOFTransition.from_form(source):
            result += self._project_transition(transition)
        return result


class _ReferenceBasis(_SourceBasis):
    """Finite target matrix in an ordered source occupation basis."""

    def __init__(self, reference: Sequence[Mapping | tuple[int, Mapping]]) -> None:
        """Use an ordered orthonormal product basis for a finite matrix target."""
        if isinstance(reference, Mapping):
            raise TypeError("A matrix target requires a list of reference states")
        references = list(reference)
        if not references:
            raise ValueError("Specify at least one reference state")
        states = []
        for reference in references:
            component, occupations = (
                (0, reference) if isinstance(reference, Mapping) else reference
            )
            component = sympy.sympify(component)
            if not component.is_Integer or component < 0:
                raise ValueError("Matrix basis indices must be nonnegative integers")
            operators, state = _ordered_reference_state(occupations)
            if states and operators != self.operators:
                raise ValueError("Every reference must declare the same source modes")
            self.operators = operators
            states.append((int(component), state))
        if len(set(states)) != len(states):
            raise ValueError("Reference states must be distinct")
        self._references = tuple(states)
        self._reference_indices = {state: i for i, state in enumerate(states)}

    def _convert_source(
        self, expression: sympy.Expr | sympy.MatrixBase
    ) -> sympy.MatrixBase:
        """Normalize square matrix entries, promoting scalar sources to 1x1 matrices."""
        if not isinstance(expression, sympy.MatrixBase):
            if any(c for c, _ in self._references):
                raise ValueError(
                    "Nonzero reference matrix indices require a matrix source"
                )
            expression = sympy.ImmutableSparseMatrix([[expression]])
        if expression.rows != expression.cols:
            raise ValueError("Source matrices must be square")
        if any(component >= expression.rows for component, _ in self._references):
            raise ValueError("Reference matrix index lies outside the source matrix")
        return sympy.ImmutableSparseMatrix(expression.applyfunc(self._convert_operator))

    @_cache_on_instance
    def _compress(self, source: sympy.MatrixBase) -> sympy.MatrixBase:
        """Evaluate matrix elements between the listed reference states, in list order."""
        entries = {}
        for (row, col), entry in source.todok().items():
            for transition in _NOFTransition.from_form(entry):
                for j, (component, state) in enumerate(self._references):
                    if component != col or (action := transition.apply(state)) is None:
                        continue
                    if (
                        i := self._reference_indices.get((row, action.output_state))
                    ) is not None:
                        entries[i, j] = entries.get((i, j), 0) + action.weight
        return sympy.ImmutableSparseMatrix(
            len(self._references), len(self._references), entries
        )


def _ordered_reference_state(
    reference: Mapping[OperatorType, int | sympy.Integer],
) -> tuple[tuple[OperatorType, ...], tuple[sympy.Integer, ...]]:
    """Validate a product state and return modes and numbers in canonical order.

    Require lowering-mode keys and integer occupations in each mode's domain.
    Fermionic matrix elements use this ordering for their parity signs.
    """
    if not isinstance(reference, Mapping) or not all(
        isinstance(op, generator_types) and op.is_annihilation for op in reference
    ):
        raise TypeError("Reference keys must be source lowering generators")
    operators = tuple(sorted(reference, key=_operator_sort_key))
    state = tuple(sympy.sympify(reference[op]) for op in operators)
    for op, n in zip(operators, state, strict=True):
        size = _occupation_dimension(op)
        if (
            not n.is_Integer
            or (not isinstance(op, LadderOp) and n < 0)
            or (size is not None and n >= size)
        ):
            raise ValueError("Reference occupations lie outside the source algebra")
    return operators, state


def _require_zero(expression: sympy.Expr, context: str) -> None:
    """Require a simplified residual to be zero, reporting ``context`` on failure.

    Raise ValueError when the residual is known nonzero, or NotImplementedError
    when SymPy cannot decide. Unknown identities are never accepted silently.
    """
    residual = sympy.simplify(expression)
    if residual == 0:
        return
    message = f"{context}; residual: {residual}"
    if residual.is_zero is False:
        raise ValueError(message)
    raise NotImplementedError(f"Cannot establish {message}")
