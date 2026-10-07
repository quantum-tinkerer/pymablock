"""Operator embeddings generated from a target reference state."""

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
    _number_operator_to_placeholder,
    _occupation_dimension,
    _operator_sort_key,
    find_operators,
    generator_types,
)
from pymablock.series import BlockSeries, zero

if TYPE_CHECKING:
    from typing import Any

__all__ = ["Embedding"]


def _cache_on_instance[Result](method: Callable[..., Result]) -> Callable[..., Result]:
    """Memoize a method with hashable positional arguments on its owning instance.

    Unlike a global method cache, this does not keep discarded embeddings alive.
    The wrapped methods accept positional arguments only.
    """

    @wraps(method)
    def cached(self: object, *args: Hashable) -> Result:
        memo = self.__dict__.setdefault("_memo_" + method.__name__, {})
        if args not in memo:
            memo[args] = method(self, *args)
        return memo[args]

    return cached


class Embedding(sympy.Expr):
    r"""Select the states of an effective model inside a second-quantized Hamiltonian.

    Below, *source* refers to the effective model, and *target* refers to the
    Hamiltonian passed to `~pymablock.block_diagonalize`. The embedding maps
    source states to target states. It either maps each source operator to an
    expression in the target operators, or lists target states that form a finite
    matrix basis.

    Pass the embedding as ``subspace_eigenvectors`` to
    `~pymablock.block_diagonalize` or `~pymablock.operator_to_BlockSeries`.
    Block 0 contains the embedded states and block 1 contains all other target
    states, which remain available for virtual transitions. See the
    :doc:`structured embeddings documentation <../structured_embeddings>` for
    requirements and examples with physical models.

    Parameters
    ----------
    generators : collections.abc.Mapping
        Maps each source lowering operator to a target expression. The keys may be
        ``SigmaMinus``, ``FermionOp``, ``BosonOp``, or ``LadderOp`` operators. A
        ``LadderOp`` key also needs an entry mapping its ``NumberOperator`` to a
        target expression, because its lowering operator does not determine it.
        Omit ``generators`` to select a finite list of states instead.
    reference : collections.abc.Mapping or collections.abc.Sequence
        If ``generators`` is given, maps every target operator to its occupation in
        the target state that represents the source state with all occupations and
        ``LadderOp`` indices equal to zero. Applying the adjoints of the target
        expressions to this state defines all other embedded states and their
        phases. ``LadderOp`` source operators also use the target expressions
        themselves to reach negative indices.

        Otherwise, lists the embedded states in the order of the matrix rows and
        columns. Each state maps every target operator to its occupation, and all
        states must use the same operators. If the Hamiltonian is a matrix, give
        ``(matrix_index, occupations)`` pairs; a mapping without an index refers to
        index 0. For a matrix Hamiltonian without operators, use pairs with empty
        mappings, such as ``[(0, {}), (2, {})]``.

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
        """Dispatch to the representation specified by the structural arguments."""
        if reference is None:
            raise TypeError("Specify reference occupations for the embedding")
        if generators is None or generators is sympy.S.NaN:
            return _ReferenceEmbedding(generators, reference)
        return _GeneratorEmbedding(generators, reference)

    def _printed_arguments(self, printer) -> list[str]:
        """Print the constructor arguments."""
        return [printer._print(arg) for arg in self.args]

    def _sympystr(self, printer) -> str:
        """Keep the public expression name when printing either subclass."""
        return f"Embedding({', '.join(self._printed_arguments(printer))})"

    def _latex(self, printer) -> str:
        """Keep the public expression name in rendered equations."""
        arguments = ", ".join(self._printed_arguments(printer))
        return rf"\operatorname{{Embedding}}\left({arguments}\right)"

    @cached_property
    def _target_numbers(self) -> tuple[sympy.Symbol, ...]:
        """Return scalar number symbols in target mode order."""
        return tuple(
            _number_operator_to_placeholder(NumberOperator(op))
            for op in self._target_operators
        )

    def _evaluate_numbers(
        self, expression: sympy.Expr, occupations: Sequence[int | sympy.Expr]
    ) -> sympy.Expr:
        """Evaluate a target number coefficient at the supplied occupations."""
        return expression.xreplace(
            dict(zip(self._target_numbers, occupations, strict=True))
        )

    def _convert_operator(self, expression: sympy.Expr) -> NumberOrderedForm:
        """Convert an expression to NOF in the compiled target mode order.

        Reject modes missing from the reference declaration.
        """
        if isinstance(expression, NumberOrderedForm):
            if expression.operators == self._target_operators:
                return expression
            expression = expression.as_expr()
        expression = sympy.sympify(expression)
        if set(find_operators(expression)) - set(self._target_operators):
            raise ValueError("Every target mode must be declared in the reference")
        return NumberOrderedForm.from_expr(expression, operators=self._target_operators)

    def _contract(self, value: NumberOrderedForm) -> NumberOrderedForm | sympy.Expr:
        """Compute ``W† value W``, unwrapping a single-reference 1x1 result."""
        result = self.restrict(value)
        return result[0, 0] if isinstance(result, sympy.MatrixBase) else result

    def _attach(self, value: sympy.Expr, side: int) -> NumberOrderedForm:
        """Represent ``value W`` for side +1, or ``W† value`` for side -1.

        Normalize the target expression but retain the attachment until arithmetic
        contracts it. Reference lists use matrices of single-reference attachments.
        """
        value = self._convert_operator(value)
        return NumberOrderedForm(
            value.operators, value.args[1], self, side, validate=False
        )

    def restrict(
        self, expression: sympy.Expr | sympy.MatrixBase
    ) -> NumberOrderedForm | sympy.MatrixBase:
        """Express a target operator in terms of the embedded states.

        The result contains the matrix elements of ``expression`` between embedded
        states, without perturbative corrections. Products are evaluated in the full
        target space before taking these matrix elements, so intermediate states
        outside the embedding contribute. For example, with ``{s: a}``,
        ``restrict(a * Dagger(a))`` is ``1 + N_s``, while
        ``restrict(a) * restrict(Dagger(a))`` is ``1 - N_s``.

        With ``generators``, the result is a
        `~pymablock.number_ordered_form.NumberOrderedForm` in the source operators.
        It may contain products such as ``N_s * (1 - N_s)`` that vanish because
        fermion and spin occupations are 0 or 1; call ``simplify()`` to remove
        them. With a list of states, the result is a SymPy matrix in the order of
        the list.
        """
        return self._compress(self._convert_target(expression))

    def _target_matrix(
        self, expression: sympy.Expr | sympy.MatrixBase
    ) -> sympy.MatrixBase:
        """Normalize a target coefficient, promoting scalar NOFs to 1x1 matrices."""
        target = self._convert_target(expression)
        return (
            sympy.ImmutableMatrix([[target]])
            if isinstance(target, NumberOrderedForm)
            else target
        )

    def _occupation_projector(self, spectra: Sequence) -> NumberOrderedForm:
        """Return the diagonal target operator selecting ``(expression, spectrum)`` pairs.

        Linearizing turns indicators of fermion and spin numbers into polynomials.
        """
        indicator = sympy.prod(
            _allowed_values_indicator(q, spectrum) for q, spectrum in spectra
        )
        return NumberOrderedForm(
            self._target_operators, {(0,) * len(self._target_operators): indicator}
        )._linearize_binary_operators()

    @_cache_on_instance
    def _frames(self, rows: int) -> tuple[sympy.MatrixBase, sympy.MatrixBase]:
        """Return retained frame W and complement Q for a target matrix size."""
        w = self._frame_columns(rows)
        return w, sympy.eye(rows) - w * w.adjoint()

    def _convert(
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
        target_shape = None
        frames = None

        def evaluate(i: int, j: int, *order: int) -> Any:
            nonlocal target_shape, frames
            target = operator[tuple(order)]
            if target is zero or (diagonal_origin and i != j and tuple(order) == origin):
                return zero
            target = self._target_matrix(target)
            if target_shape is None:
                target_shape = target.shape
                frames = self._frames(target.rows)
            elif target.shape != target_shape:
                raise ValueError(
                    "All operator coefficients must have the same target matrix shape"
                )
            result = frames[i].adjoint() * target * frames[j]
            return zero if result.is_zero_matrix else self._block_result(result)

        return BlockSeries(
            eval=evaluate,
            shape=(2, 2),
            n_infinite=operator.n_infinite,
            dimension_names=operator.dimension_names,
            name=operator.name,
        )

    def _prepare(self, hamiltonian: BlockSeries) -> tuple[BlockSeries, Callable]:
        """Return the 2x2 retained/complement Hamiltonian and its Sylvester solver.

        Check H0 and build the frames W and 1 - W W† now, so an off-diagonal H0 or
        an unsupported projector raises here instead of during lazy evaluation.
        The validated H0 has no zeroth-order cross blocks, so they are omitted.
        """
        h0 = self._target_matrix(hamiltonian[(0,) * hamiltonian.n_infinite])
        solve_sylvester = self._sylvester_solver(h0)
        self._frames(h0.rows)
        return self._convert(hamiltonian, diagonal_origin=True), solve_sylvester

    def _sylvester_solver(
        self, h0: sympy.MatrixBase
    ) -> Callable[[Any, tuple[int, ...]], Any]:
        """Validate H0 and return a solver for this embedding's Sylvester equations.

        H0 must be a target matrix of NOF entries, diagonal in both matrix indices and
        occupation numbers. Intra-block solves use the ordinary second-quantized solver.
        Rectangular solves evaluate outgoing and incoming energies on the reference
        states or symbolic source occupations, then divide each transition.
        The callback preserves the series zero sentinel before accessing matrix entries.
        """
        # second_quantization imports this module to re-export Embedding.
        from pymablock.second_quantization import (
            _divide_by_energy_gap,
            solve_sylvester_2nd_quant,
        )

        modes, numbers = self._target_operators, self._target_numbers
        if any(
            i != j or any(any(p) for p in x.terms) for (i, j), x in h0.todok().items()
        ):
            raise ValueError("Structured embeddings currently require diagonal H0")
        vacuum = (0,) * len(modes)
        energies = [
            x.terms.get(vacuum, sympy.S.Zero) if x != 0 else sympy.S.Zero
            for x in h0.diagonal()
        ]
        occupations, coordinates = self._target_occupations, self._coordinate_symbols
        nonnegative = tuple(q for q in coordinates if q.is_nonnegative)
        incoming_energies = [
            self._evaluate_numbers(energies[row], state)
            for row, state in self._energy_states
        ]

        # Within each diagonal block the operators already use source or target
        # coordinates. Only rectangular blocks require embedding-aware division.
        retained_energies = self.restrict(self._block_result(h0))
        retained_energies = (
            retained_energies.diagonal()
            if isinstance(retained_energies, sympy.MatrixBase)
            else [retained_energies]
        )
        diagonal_solver = solve_sylvester_2nd_quant([retained_energies, h0.diagonal()])

        def divide_transition_entry(
            value: NumberOrderedForm | sympy.Expr, row: int, col: int
        ) -> NumberOrderedForm | sympy.Expr:
            """Solve one retained-to-target matrix entry using its actual transition gap."""
            if value == 0 or value.is_zero:
                return sympy.S.Zero
            value = self._convert_operator(value.target)
            terms, coefficients = {}, value.terms
            for powers, (output, matrix_element) in value.act(occupations).items():
                # Divide the coefficient, not the full matrix element; ladder factors
                # only determine whether the transition is active.
                middle_occupations = [n - max(p, 0) for n, p in zip(occupations, powers)]
                coefficient = self._evaluate_numbers(
                    coefficients[powers], middle_occupations
                )
                denominator = sympy.expand(
                    self._evaluate_numbers(energies[row], output) - incoming_energies[col]
                )
                # A literal zero gap still needs the amplitude interpreted in the
                # source algebra: binary numbers obey n² = n, including indicators.
                if denominator == 0 and coordinates:
                    amplitude = NumberOrderedForm(
                        self._source_operators,
                        {
                            (0,) * len(coordinates): matrix_element.xreplace(
                                dict(zip(coordinates, self._source_placeholders))
                            )
                        },
                        validate=False,
                    )._linearize_binary_operators()
                    if amplitude.is_zero:
                        continue
                try:
                    result = _divide_by_energy_gap(
                        coefficient,
                        denominator,
                        coordinates,
                        matrix_element,
                        nonnegative,
                    )
                except ValueError as error:
                    raise ZeroDivisionError(str(error)) from error
                incoming = {n: n + max(p, 0) for n, p in zip(numbers, powers)}
                terms[powers] = result.xreplace(
                    {
                        q: expression.xreplace(incoming)
                        for q, expression in zip(coordinates, self._source_coordinates)
                    }
                )
            return NumberOrderedForm(
                self._target_operators,
                terms,
                self._entry_embedding,
                1,
                validate=False,
            )

        def solve(value: Any, index: tuple[int, ...]) -> Any:
            """Dispatch by block; use anti-Hermitian symmetry for the reverse cross block."""
            if value is zero:
                return zero
            if index[0] == index[1]:
                return diagonal_solver(value, index)
            reverse = index[:2] == (0, 1)
            block = value.adjoint() if reverse else value
            result = sympy.ImmutableMatrix(
                block.rows,
                block.cols,
                lambda i, j: divide_transition_entry(block[i, j], i, j),
            )
            return -result.adjoint() if reverse else result

        return solve


class _GeneratorEmbedding(Embedding):
    """Symbolic source algebra generated from one reference state."""

    def __new__(cls, generators: Mapping, reference: Mapping) -> Self:
        """Recompile the generator map from its SymPy arguments."""
        if not all(isinstance(x, (Mapping, sympy.Dict)) for x in (generators, reference)):
            raise TypeError("Generators and reference must be mappings")
        self = sympy.Expr.__new__(cls, sympy.Dict(generators), sympy.Dict(reference))
        self._compile(dict(generators), dict(reference))
        return self

    def _compile(self, generators: Mapping, reference: Mapping) -> None:
        """Compile and validate the affine occupation map."""
        if any(
            source in reference and source != image
            for source, image in generators.items()
        ):
            raise ValueError("Source modes must not shadow distinct target modes")
        self._target_operators, self._reference_state = _ordered_reference_state(
            reference
        )
        numbers = {
            op: value
            for op, value in generators.items()
            if isinstance(op, NumberOperator)
        }
        operators = tuple(op for op in generators if op not in numbers)
        if not all(isinstance(op, generator_types) for op in operators):
            raise TypeError("Keys must be source lowering generators or ladder numbers")
        if any(not op.is_annihilation for op in operators):
            raise ValueError("Source generators must be lowering operators")
        operators = tuple(sorted(operators, key=_operator_sort_key))
        self._source_operators = operators
        self._source_dimensions = tuple(map(_occupation_dimension, operators))
        self._coordinate_symbols = tuple(
            sympy.Dummy(
                f"source_{i}",
                integer=True,
                **({} if isinstance(op, LadderOp) else {"nonnegative": True}),
            )
            for i, op in enumerate(operators)
        )
        images, shifts = [], []
        for op in operators:
            image = self._convert_target(generators[op])
            if len(image.terms) != 1 or not any(shift := next(iter(image.terms))):
                raise ValueError(
                    f"Image of {op} must change target occupations by one nonzero shift"
                )
            parity = sum(
                power
                for target, power in zip(self._target_operators, shift, strict=True)
                if isinstance(target, FermionOp)
            )
            if parity % 2 != isinstance(op, FermionOp):
                raise ValueError("Generator images must preserve fermionic parity")
            images.append(image)
            shifts.append(shift)
        self._generators = tuple(images)
        self._occupation_matrix = sympy.Matrix(
            len(self._target_operators), len(operators), lambda i, j: shifts[j][i]
        )
        if self._occupation_matrix.rank() != len(operators):
            raise ValueError("Generator shifts must be independent")
        matrix = self._occupation_matrix
        self._occupation_left_inverse = (matrix.T * matrix).inv() * matrix.T
        self._source_coordinates = tuple(
            self._occupation_left_inverse
            * (sympy.Matrix(self._target_numbers) - sympy.Matrix(self._reference_state))
        )
        self._target_occupations = tuple(
            origin + sum(matrix[i, j] * q for j, q in enumerate(self._coordinate_symbols))
            for i, origin in enumerate(self._reference_state)
        )
        self._validate_domains()
        self._validate_generator_algebra()
        self._phase = self._reference_phase()
        expected_numbers = {
            NumberOperator(op) for op in operators if isinstance(op, LadderOp)
        }
        if set(numbers) != expected_numbers:
            raise ValueError(
                "Supply the independent number image for each source LadderOp"
            )
        for op, number in numbers.items():
            index = next(
                i for i, source in enumerate(operators) if NumberOperator(source) == op
            )
            form = self._convert_target(number)
            if any(any(powers) for powers in form.terms):
                raise ValueError("A ladder number image must be occupation diagonal")
            expression = form.terms.get((0,) * len(self._target_operators), sympy.S.Zero)
            expression = self._evaluate_numbers(expression, self._target_occupations)
            self._validate_identity(
                expression - self._coordinate_symbols[index],
                f"Ladder number image {op} must count from the source reference index zero",
            )

    def _convert_target(self, expression: sympy.Expr) -> NumberOrderedForm:
        """Normalize a scalar target; matrix targets require reference lists."""
        if isinstance(expression, sympy.MatrixBase):
            raise TypeError("Matrix targets require a list of reference states")
        return self._convert_operator(expression)

    def _validate_domains(self) -> None:
        """Check target occupation bounds over the full source occupation domain.

        Use each affine map's minimum and maximum, rather than enumerating states.
        Raise ValueError if a generator can overfill or underfill a target mode.
        """
        for i, op in enumerate(self._target_operators):
            lower = upper = self._reference_state[i]
            for coefficient, source, size in zip(
                self._occupation_matrix.row(i),
                self._source_operators,
                self._source_dimensions,
                strict=True,
            ):
                if not coefficient:
                    continue
                if size is None:
                    if isinstance(source, LadderOp):
                        lower, upper = -sympy.oo, sympy.oo
                    elif coefficient > 0:
                        upper = sympy.oo
                    else:
                        lower = -sympy.oo
                else:
                    lower += min(0, coefficient) * (size - 1)
                    upper += max(0, coefficient) * (size - 1)
            if not isinstance(op, LadderOp) and lower < 0:
                raise ValueError("Generators leave the physical target occupation domain")
            if _occupation_dimension(op) is not None and upper >= _occupation_dimension(
                op
            ):
                raise ValueError("Generators overfill a target spin or fermion")

    @_cache_on_instance
    def _source_weight(self, powers: tuple[int, ...]) -> sympy.Expr:
        """Return the source ladder amplitude of a shift at symbolic occupations."""
        term = NumberOrderedForm(
            self._source_operators, {powers: sympy.S.One}, validate=False
        )
        return _matrix_element(term, self._coordinate_symbols)

    def _lowering_weight(self, index: int) -> sympy.Expr:
        """Return the source lowering amplitude for mode ``index`` at symbolic numbers."""
        return self._source_weight(
            tuple(int(i == index) for i in range(len(self._source_operators)))
        )

    def _reference_phase(self) -> sympy.Expr:
        """Return the phase relating normalized target and source occupation states.

        Compare target and source lowering amplitudes in a fixed creation order.
        Infinite modes require a number-independent ratio; otherwise raise
        NotImplementedError. The reference state's phase is one.
        """
        phase = sympy.S.One
        for i, (image, q, size) in enumerate(
            zip(
                self._generators,
                self._coordinate_symbols,
                self._source_dimensions,
                strict=True,
            )
        ):
            ratio = self._lowering_weight(i) / _matrix_element(
                image, self._target_occupations
            )
            ratio = ratio.xreplace(
                dict.fromkeys(self._coordinate_symbols[:i], sympy.S.Zero)
            )
            if size is None:
                ratio = sympy.simplify(ratio)
                if ratio.free_symbols.intersection(self._coordinate_symbols):
                    raise NotImplementedError(
                        "Infinite source generators require a constant phase relative to their ladder weights"
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
        binary = tuple(
            q
            for q, size in zip(self._coordinate_symbols, self._source_dimensions)
            if size == 2
        )
        expression = sympy.expand(expression)
        for q in binary:
            if q in expression.free_symbols and expression.is_polynomial(q):
                expression = sympy.rem(expression, q**2 - q, q)
        if (
            expression != 0
            and expression.free_symbols <= set(binary)
            and expression.is_polynomial(*binary)
        ):
            raise ValueError(f"{context}; residual: {expression}")
        residual = sympy.simplify(expression)
        if residual == 0:
            return
        message = f"{context}; residual: {residual}"
        if residual.is_zero is False:
            raise ValueError(message)
        raise NotImplementedError(f"Cannot establish {message}")

    def _validate_generator_algebra(self) -> None:
        """Check local norms and graded commutation on the retained lattice.

        The occupation boundaries fix the vacuum and binary truncation. Norms
        then fix the same-mode algebra; pairwise lowering relations also fix
        the mixed adjoint relations by reversing an edge of each lattice square.
        """
        weights = [
            _matrix_element(image, self._target_occupations) for image in self._generators
        ]
        coordinates = self._coordinate_symbols
        active = {
            q: sympy.S.One if size == 2 else q + 1
            for q, size in zip(coordinates, self._source_dimensions)
        }
        for i, q in enumerate(coordinates):
            norm = (
                weights[i] * sympy.conjugate(weights[i]) - self._lowering_weight(i) ** 2
            )
            self._validate_identity(
                norm.xreplace({q: active[q]}),
                "Generator images must produce normalized source states",
            )
            for j, other in enumerate(coordinates[:i]):
                sign = (
                    -1
                    if all(
                        isinstance(self._source_operators[k], FermionOp) for k in (i, j)
                    )
                    else 1
                )
                relation = weights[j] * weights[i].xreplace(
                    {other: other - 1}
                ) - sign * weights[i] * weights[j].xreplace({q: q - 1})
                self._validate_identity(
                    relation.xreplace({q: active[q], other: active[other]}),
                    "Generator images must obey the source algebra",
                )

    @cached_property
    def _source_placeholders(self) -> tuple[sympy.Symbol, ...]:
        """Return scalar number symbols in source mode order."""
        return tuple(
            _number_operator_to_placeholder(NumberOperator(op))
            for op in self._source_operators
        )

    @cached_property
    def _source_zero(self) -> NumberOrderedForm:
        """Return zero carrying the source operator basis."""
        return NumberOrderedForm(self._source_operators, {}, validate=False)

    @_cache_on_instance
    def _source_shift(self, target_shift: tuple[int, ...]) -> tuple[int, ...] | None:
        """Return the induced source shift, or None if compression annihilates it.

        The target shift must lie in the occupation map's image and correspond to
        integer source steps within each finite mode's range.
        """
        target = sympy.Matrix(target_shift)
        result = self._occupation_left_inverse * target
        if self._occupation_matrix * result != target:
            return None
        if any(
            not value.is_Integer or (size is not None and abs(value) >= size)
            for value, size in zip(result, self._source_dimensions, strict=True)
        ):
            return None
        return tuple(map(int, result))

    @_cache_on_instance
    def _project_term(
        self, target_shift: tuple[int, ...], target_weight: sympy.Expr
    ) -> NumberOrderedForm:
        """Compress one target term, given its matrix element on retained states.

        Convert its shift and matrix element to source coordinates. Spectator
        numbers stay symbolic; a term outside the retained lattice gives zero.
        """
        powers = self._source_shift(target_shift)
        if powers is None:
            return self._source_zero
        shifted = {q: q - p for q, p in zip(self._coordinate_symbols, powers)}
        # The phase has unit modulus on the retained domain. Its ratio cancels
        # unchanged factors without expanding binary occupation identities.
        amplitude = (
            target_weight
            / self._source_weight(powers)
            * self._phase
            / self._phase.xreplace(shifted)
        )
        # NOF coefficients sit between creation and annihilation operators.
        # A binary transition fixes its input occupation; spectators stay symbolic.
        initial = {
            q: sympy.Integer(p > 0) if p and size == 2 else n + max(p, 0)
            for q, n, p, size in zip(
                self._coordinate_symbols,
                self._source_placeholders,
                powers,
                self._source_dimensions,
            )
        }
        coefficient = sympy.factor_terms(amplitude.xreplace(initial))
        return NumberOrderedForm(
            self._source_operators, {powers: coefficient}, validate=False
        )

    @_cache_on_instance
    def _compress(self, target: NumberOrderedForm) -> NumberOrderedForm:
        """Return ``W† target W`` by translating each term to the source algebra."""
        result = self._source_zero
        for shift, (_, weight) in target.act(self._target_occupations).items():
            result += self._project_term(shift, weight)
        return result

    @cached_property
    def _projector(self) -> NumberOrderedForm:
        """Return ``W W†``, the target-space projector onto retained states.

        Scalar indicators enforce the affine occupation constraints and each source
        mode's allowed numbers. No target occupation states are enumerated.
        """
        numbers = self._target_numbers
        offsets = sympy.Matrix(numbers) - sympy.Matrix(self._reference_state)
        spectra = [
            (sympy.expand(normal.dot(offsets)), (0,))
            for normal in self._occupation_matrix.T.nullspace()
        ]
        source = self._source_coordinates
        physical = {
            n: sympy.Dummy(integer=True, nonnegative=True)
            for target, n in zip(self._target_operators, numbers)
            if not isinstance(target, LadderOp)
        }
        for q, op, size in zip(source, self._source_operators, self._source_dimensions):
            if (
                isinstance(op, BosonOp)
                and q.xreplace(physical).is_nonnegative is not True
            ):
                raise NotImplementedError(
                    "This bosonic embedding requires an occupation inequality"
                )
            spectra.append((q, range(size) if size is not None else sympy.S.Integers))
        return self._occupation_projector(spectra)

    @_cache_on_instance
    def _lift(self, value: NumberOrderedForm) -> NumberOrderedForm:
        """Represent a source operator in target space, as ``W value W†``.

        Replace source generators and numbers by their target images, then project
        both sides onto the retained space.
        """
        target_numbers = dict(
            zip(self._target_numbers, map(NumberOperator, self._target_operators))
        )
        coordinates = [q.xreplace(target_numbers) for q in self._source_coordinates]
        images = dict(zip(map(NumberOperator, self._source_operators), coordinates))
        for op, image in zip(self._source_operators, self._generators):
            images[op] = image.as_expr()
            images[op.adjoint()] = image.adjoint().as_expr()
        result = NumberOrderedForm.from_expr(
            value.as_expr().xreplace(images), self._target_operators
        )
        return self._projector * result * self._projector

    @property
    def _energy_states(self) -> tuple:
        """Target occupations at which each retained energy is evaluated."""
        return ((0, self._target_occupations),)

    @property
    def _entry_embedding(self) -> Embedding:
        """Attachment used by scalar entries of the retained frame."""
        return self

    def _frame_columns(self, _rows: int) -> sympy.MatrixBase:
        """Represent the generator isometry as one attached scalar."""
        return sympy.ImmutableMatrix([[self._attach(sympy.S.One, 1)]])

    def _block_result(self, result: sympy.MatrixBase) -> NumberOrderedForm:
        """Unwrap the scalar block used by a generator embedding."""
        return result[0, 0]


class _ReferenceEmbedding(Embedding):
    """Finite source matrix in an ordered target occupation basis."""

    # A finite matrix has no source coordinates; the shared solver reads these.
    _coordinate_symbols = _source_coordinates = ()

    def __new__(cls, generators=None, reference=None) -> Self:  # noqa: ARG004
        """Recompile the listed states from their SymPy arguments."""
        # SymPy rebuilds expressions as type(self)(*args), which also passes the
        # unused generators argument (NaN for reference lists).
        if isinstance(reference, Mapping):
            raise TypeError("A matrix source requires a list of reference states")
        reference = tuple(
            (0, state) if isinstance(state, (Mapping, sympy.Dict)) else state
            for state in reference
        )
        args = sympy.Tuple(*(sympy.Tuple(i, sympy.Dict(state)) for i, state in reference))
        self = sympy.Expr.__new__(cls, sympy.S.NaN, args)
        self._compile(args)
        return self

    def _printed_arguments(self, printer) -> list[str]:
        """Print the reference list as passed, without the generators sentinel."""
        states = [state if i == 0 else (i, state) for i, state in self.args[1]]
        keyword = r"\text{reference}" if printer.printmethod == "_latex" else "reference"
        return [f"{keyword}={printer._print(states)}"]

    def _compile(self, reference: sympy.Tuple) -> None:
        """Validate an ordered list of orthonormal target product states."""
        if not reference:
            raise ValueError("Specify at least one reference state")
        states = []
        for component, occupations in reference:
            if not component.is_Integer or component < 0:
                raise ValueError("Matrix basis indices must be nonnegative integers")
            operators, state = _ordered_reference_state(dict(occupations))
            if states and operators != self._target_operators:
                raise ValueError("Every reference must declare the same target modes")
            self._target_operators = operators
            states.append((int(component), state))
        if len(set(states)) != len(states):
            raise ValueError("Reference states must be distinct")
        self._references = tuple(states)
        self._reference_indices = {state: i for i, state in enumerate(states)}

    def _convert_target(
        self, expression: sympy.Expr | sympy.MatrixBase
    ) -> sympy.MatrixBase:
        """Normalize square matrix entries, promoting scalar targets to 1x1 matrices."""
        if not isinstance(expression, sympy.MatrixBase):
            if any(c for c, _ in self._references):
                raise ValueError(
                    "Nonzero reference matrix indices require a matrix target"
                )
            expression = sympy.ImmutableSparseMatrix([[expression]])
        if expression.rows != expression.cols:
            raise ValueError("Target matrices must be square")
        if any(component >= expression.rows for component, _ in self._references):
            raise ValueError("Reference matrix index lies outside the target matrix")
        return sympy.ImmutableSparseMatrix(expression.applyfunc(self._convert_operator))

    @_cache_on_instance
    def _compress(self, target: sympy.MatrixBase) -> sympy.MatrixBase:
        """Evaluate matrix elements between the listed reference states, in list order."""
        entries = {}
        for (row, col), entry in target.todok().items():
            for j, (component, state) in enumerate(self._references):
                if component != col:
                    continue
                for output, weight in entry.act(state).values():
                    if (i := self._reference_indices.get((row, output))) is not None:
                        entries[i, j] = entries.get((i, j), 0) + weight
        return sympy.ImmutableSparseMatrix(
            len(self._references), len(self._references), entries
        )

    @cached_property
    def _target_occupations(self) -> tuple:
        """Entry attachments act on the target vacuum."""
        return (0,) * len(self._target_operators)

    @property
    def _energy_states(self) -> tuple:
        """Matrix components and occupations of the listed retained states."""
        return self._references

    @cached_property
    def _entry_embedding(self) -> Embedding:
        """Vacuum attachment shared by entries of the retained frame."""
        return Embedding(reference=[dict.fromkeys(self._target_operators, 0)])

    def _frame_columns(self, rows: int) -> sympy.MatrixBase:
        """Prepare each listed state with a normalized creation monomial."""
        w = sympy.zeros(rows, len(self._references))
        for col, (row, state) in enumerate(self._references):
            monomial = NumberOrderedForm(
                self._target_operators, {tuple(-n for n in state): sympy.S.One}
            )
            w[row, col] = self._entry_embedding._attach(
                monomial / _matrix_element(monomial, self._target_occupations), 1
            )
        return sympy.ImmutableMatrix(w)

    def _block_result(self, result: sympy.MatrixBase) -> sympy.MatrixBase:
        """Reference-list blocks retain their matrix indices."""
        return result

    @cached_property
    def _projector(self) -> NumberOrderedForm:
        """Project an entry attachment onto its single reference state."""
        return self._occupation_projector(
            [(q, (n,)) for q, n in zip(self._target_numbers, self._references[0][1])]
        )

    @_cache_on_instance
    def _lift(self, value: NumberOrderedForm) -> NumberOrderedForm:
        """Normalize a reference entry, which already uses target operators."""
        return self._convert_operator(value)

    def _attach(self, value: sympy.Expr, side: int) -> NumberOrderedForm:
        """Attach one reference; lists use matrices of vacuum attachments."""
        if len(self._references) != 1 or self._references[0][0] != 0:
            raise ValueError(
                "Reference lists use matrices of single-reference attachments"
            )
        return super()._attach(value, side)


def _matrix_element(term: NumberOrderedForm, occupations: Sequence) -> sympy.Expr:
    """Return the matrix element of a single-term NOF, or zero if it annihilates."""
    actions = term.act(occupations)
    if not actions:
        return sympy.S.Zero
    ((_, matrix_element),) = actions.values()
    return matrix_element


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
        raise TypeError("Reference keys must be target lowering generators")
    operators = tuple(sorted(reference, key=_operator_sort_key))
    state = tuple(sympy.sympify(reference[op]) for op in operators)
    for op, n in zip(operators, state, strict=True):
        size = _occupation_dimension(op)
        if (
            not n.is_Integer
            or (not isinstance(op, LadderOp) and n < 0)
            or (size is not None and n >= size)
        ):
            raise ValueError("Reference occupations lie outside the target algebra")
    return operators, state
