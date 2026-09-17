"""Pymablock adapter for algebraic occupation-basis embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import TYPE_CHECKING

import numpy as np
import sympy

from pymablock._embedding.maps import (
    AdjointOperatorMap,
    ModuleEndomorphism,
    OperatorMap,
    TargetOperator,
    multiply_projected,
)
from pymablock._embedding.transitions import BasisMap, NOFTransition, number_symbols
from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main
from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.series import BlockSeries, zero

if TYPE_CHECKING:
    from pymablock.operator_embedding import Embedding


def _one(operators) -> NumberOrderedForm:
    operators = tuple(operators)
    return NumberOrderedForm(
        operators,
        {(0,) * len(operators): sympy.S.One},
        validate=False,
    )


def _one_term(
    operators,
    powers: tuple[int, ...],
    coefficient: sympy.Expr,
) -> NumberOrderedForm:
    return NumberOrderedForm(
        tuple(operators),
        {powers: coefficient},
        validate=False,
    )


def _target_is_zero(value: TargetOperator) -> bool:
    if isinstance(value, sympy.MatrixBase):
        return not value.todok()
    return not any(coefficient != 0 for coefficient in value.terms.values())


def _immutable(matrix: sympy.MatrixBase) -> sympy.ImmutableMatrix:
    if isinstance(matrix, sympy.ImmutableMatrix):
        return matrix
    return sympy.ImmutableMatrix(matrix)


@dataclass(frozen=True)
class _ComplementSpace:
    """The unenumerated complement fixed by one structured embedding."""

    embedding: Embedding


class _EmbeddingBackend:
    """Compile a public embedding to generic weighted transitions."""

    def __init__(self, embedding: Embedding):
        self.descriptor = embedding
        self.source_operators = tuple(embedding.operators)
        self.coordinates = embedding.coordinate_symbols
        self.target_operators = embedding.target.operators
        self.target_is_nof = embedding.target_is_nof

        self.source_placeholders = number_symbols(self.source_operators)
        target_placeholders = (
            number_symbols(self.target_operators) if self.target_is_nof else ()
        )
        self.basis_map = BasisMap(
            embedding.source_occupations,
            embedding.coordinate_symbols,
            target_placeholders,
            embedding.phase,
        )

        self.target_space = embedding.target
        self.complement_space = _ComplementSpace(embedding)
        if self.target_is_nof:
            self.target_identity = _one(self.target_operators)
            self.target_zero = NumberOrderedForm(
                self.target_operators, {}, validate=False
            )
        else:
            self.target_identity = _immutable(sympy.eye(embedding.target.dimension))
            self.target_zero = _immutable(sympy.zeros(embedding.target.dimension))

    def source_form(self, expression) -> NumberOrderedForm:
        """Convert an expression into the source algebra of the embedding."""
        if isinstance(expression, NumberOrderedForm):
            if expression.operators == self.source_operators:
                return expression
            expression = expression.as_expr()
        return NumberOrderedForm.from_expr(
            sympy.sympify(expression),
            operators=self.source_operators,
        )

    @cache
    def project_transition(
        self,
        transition: NOFTransition,
    ) -> NumberOrderedForm:
        """Pull back one transition and encode it as a target NOF."""
        target_powers = self.basis_map.target_shift(transition.powers)
        if target_powers is None:
            return self.target_zero
        amplitude = self.basis_map.pullback_weight(transition, target_powers)
        if amplitude == 0:
            return self.target_zero
        # The source action includes its Fock sign. A target fermion monomial
        # supplies a Fock sign of its own; remove it from the coefficient so
        # that evaluating the target operator does not count it twice.
        target_form = _one_term(self.target_operators, target_powers, sympy.S.One)
        (target_transition,) = NOFTransition.from_form(target_form)
        target_weight = target_transition.symbolic_action(self.coordinates).weight
        target_weight = target_weight.xreplace(
            self.basis_map.transition_support(target_powers)
        ).xreplace(self.basis_map.initial_to_middle(target_powers))
        amplitude *= target_weight * target_transition.scalar  # Inverse of a sign.
        return (
            _one_term(self.target_operators, target_powers, amplitude) * transition.scalar
        )

    @cached_property
    def finite_source_index(self):
        """Source occupations belonging to a finite target, without coefficients."""
        return {
            self.descriptor.encode(state): index
            for index, state in enumerate(self.descriptor.target.states)
        }

    @cache
    def finite_transition_action(
        self,
        transition: NOFTransition,
        retained_row: int,
    ):
        """Apply a transition to the source image of one retained state."""
        target_state = self.descriptor.target.states[retained_row]
        return transition.apply(self.descriptor.encode(target_state))

    @cache
    def pullback(self, source: NumberOrderedForm) -> TargetOperator:
        """Return ``W† source W`` without enumerating the source Hilbert space."""
        source = self.source_form(source)
        transitions = tuple(NOFTransition.from_form(source))
        if self.target_is_nof:
            result = self.target_zero
            for transition in transitions:
                result += self.project_transition(transition)
            return NumberOrderedForm(
                result.operators,
                {
                    powers: coefficient
                    for powers, coefficient in result.terms.items()
                    if coefficient != 0
                },
                validate=False,
            )

        target = self.descriptor.target
        source_to_target = self.finite_source_index
        matrix = sympy.MutableSparseMatrix(target.dimension, target.dimension, {})
        for column, state in enumerate(target.states):
            source_state = self.descriptor.encode(state)
            for transition in transitions:
                action = transition.apply(source_state)
                if action is None:
                    continue
                if (row := source_to_target.get(action.output_state)) is not None:
                    initial_phase = self.descriptor.phase.xreplace(
                        dict(zip(self.coordinates, state, strict=True))
                    )
                    final_phase = self.descriptor.phase.xreplace(
                        dict(zip(self.coordinates, target.states[row], strict=True))
                    )
                    matrix[row, column] += (
                        sympy.conjugate(final_phase)
                        * initial_phase
                        * action.weight
                        * transition.scalar
                    )
        return _immutable(matrix)


class _EmbeddingProblem:
    """Lower one scalar Hamiltonian series to the formal P/Q block algebra."""

    def __init__(self, hamiltonian: BlockSeries, embedding: Embedding):
        if hamiltonian.shape:
            raise ValueError("Structured embeddings require an unseparated Hamiltonian.")
        self.embedding = _EmbeddingBackend(embedding)
        self.hamiltonian = hamiltonian
        zero_order = (0,) * hamiltonian.n_infinite
        h0 = self.embedding.source_form(hamiltonian[zero_order])
        zero_powers = (0,) * len(self.embedding.source_operators)
        if any(tuple(map(int, powers)) != zero_powers for powers in h0.terms):
            raise ValueError("Structured embeddings currently require diagonal H0")
        self.source_energy = sympy.simplify(h0.terms.get(zero_powers, sympy.S.Zero))
        self.source_energy_placeholders = number_symbols(tuple(h0.operators))

        target_h0 = self.embedding.pullback(h0)
        if isinstance(target_h0, NumberOrderedForm):
            if any(any(map(int, powers)) for powers in target_h0.terms):
                raise ValueError("The embedding image must be invariant under H0")
            self.target_energy = sympy.simplify(
                self.source_energy.xreplace(
                    dict(
                        zip(
                            self.embedding.source_placeholders,
                            self.embedding.basis_map.source_occupations,
                            strict=True,
                        )
                    )
                )
            )
            self.finite_target_energy = None
        else:
            if any(row != column for row, column in target_h0.todok()):
                raise ValueError("The retained finite block must diagonalize H0")
            self.target_energy = None
            self.finite_target_energy = tuple(
                target_h0[index, index]
                for index in range(self.embedding.target_space.dimension)
            )

    @cache
    def _source_energy_at(self, state: tuple[int, ...]) -> sympy.Expr:
        return self.source_energy.xreplace(
            dict(
                zip(
                    self.source_energy_placeholders,
                    map(sympy.Integer, state),
                    strict=True,
                )
            )
        )

    @cache
    def _target_reciprocal(self, denominator: sympy.Expr) -> NumberOrderedForm:
        return _one_term(
            self.embedding.target_operators,
            (0,) * len(self.embedding.target_operators),
            sympy.S.One / denominator,
        )

    def _channel_denominator(
        self,
        source: NOFTransition,
        target: NOFTransition,
    ) -> sympy.Expr:
        return self.embedding.basis_map.energy_denominator(
            source,
            target.powers,
            source_energy=self.source_energy,
            source_placeholders=self.source_energy_placeholders,
            target_energy=self.target_energy,
        )

    def _zero_denominator_is_projected_out(
        self,
        source: NOFTransition,
        target: NumberOrderedForm | sympy.MatrixBase,
    ) -> bool:
        projected = self.embedding.pullback(source.form)
        leakage = (
            self.embedding.pullback(source.form.adjoint() * source.form)
            - projected.adjoint() * projected
        )
        norm = target.adjoint() * leakage * target
        return _target_is_zero(norm)

    def solve_sylvester(self, value, index):
        """Solve the P-Q Sylvester equation transition by transition."""
        if value is zero:
            return zero
        if index[:2] != (0, 1) or not isinstance(value, AdjointOperatorMap):
            raise TypeError("The embedding solver expects the P-Q block")
        if not self.embedding.target_is_nof:
            return self._solve_finite(value)

        solved_terms = []
        for source_form, target_form in value.column.terms:
            for source in NOFTransition.from_form(source_form):
                for target in NOFTransition.from_form(target_form):
                    denominator = self._channel_denominator(source, target)
                    target_term = target.form * target.scalar
                    quotient = self._divide_channel(source, target_term, denominator)
                    if not _target_is_zero(quotient):
                        solved_terms.append((source.form, quotient * source.scalar))
        column = OperatorMap(self.embedding, solved_terms).or_zero()
        return zero if column is zero else column.adjoint()

    def _divide_channel(
        self, source: NOFTransition, target: NumberOrderedForm, denominator: sympy.Expr
    ) -> NumberOrderedForm:
        """Divide on binary sectors, discarding unsupported resonances first."""
        variables = tuple(
            symbol
            for symbol in self.embedding.basis_map.target_placeholders
            if symbol in denominator.free_symbols
        )

        sectors = []
        has_zero = False

        def visit(expression, mask, remaining):
            nonlocal has_zero
            expression = sympy.expand(expression)
            remaining = tuple(x for x in remaining if x in expression.free_symbols)
            if remaining:
                symbol, *rest = remaining
                for value in (0, 1):
                    visit(
                        expression.xreplace({symbol: sympy.Integer(value)}),
                        mask * (symbol if value else 1 - symbol),
                        rest,
                    )
                return
            expression = sympy.simplify(expression)
            if expression == 0:
                has_zero = True
                sector = _one_term(
                    self.embedding.target_operators,
                    (0,) * len(self.embedding.target_operators),
                    mask,
                )
                if not self._zero_denominator_is_projected_out(source, target * sector):
                    raise ZeroDivisionError(
                        "A virtual channel is degenerate with the retained space"
                    )
            else:
                sectors.append(mask / expression)

        visit(denominator, sympy.S.One, variables)
        if not has_zero:
            return target * self._target_reciprocal(denominator)
        inverse = _one_term(
            self.embedding.target_operators,
            (0,) * len(self.embedding.target_operators),
            sympy.Add(*sectors),
        )
        return target * inverse

    def _solve_finite(self, value: AdjointOperatorMap):
        solved_terms = []
        for source_form, target in value.column.terms:
            for source in NOFTransition.from_form(source_form):
                for (row, column), target_coefficient in target.todok().items():
                    action = self.embedding.finite_transition_action(source, row)
                    # A weighted transition has one output occupation. If it
                    # returns to P, the outer Q removes it exactly, regardless
                    # of how complicated its symbolic coefficient is.
                    if (
                        action is None
                        or action.output_state in self.embedding.finite_source_index
                    ):
                        continue
                    denominator = sympy.factor(
                        self.finite_target_energy[column]
                        - self._source_energy_at(action.output_state)
                    )
                    target_term = sympy.ImmutableSparseMatrix(
                        self.embedding.target_space.dimension,
                        self.embedding.target_space.dimension,
                        {(row, column): target_coefficient},
                    )
                    if denominator == 0:
                        if self._zero_denominator_is_projected_out(source, target_term):
                            continue
                        raise ZeroDivisionError(
                            "A virtual channel is degenerate with the retained space"
                        )
                    solved_terms.append(
                        (
                            source.form,
                            target_term / denominator * source.scalar,
                        )
                    )
        column = OperatorMap(self.embedding, solved_terms).or_zero()
        return zero if column is zero else column.adjoint()

    def block_series(self) -> BlockSeries:
        """Represent the Hamiltonian in the generic compression module."""
        embedding = self.embedding
        zero_order = (0,) * self.hamiltonian.n_infinite

        def evaluate(*index):
            row, column, *order = index
            source = self.hamiltonian[tuple(order)]
            if source is zero:
                return zero
            source = embedding.source_form(source)
            if _target_is_zero(source):
                return zero
            if row == column == 0:
                result = embedding.pullback(source)
                return zero if _target_is_zero(result) else result
            if tuple(order) == zero_order and row != column:
                return zero
            if (row, column) == (1, 0):
                return OperatorMap.from_source(embedding, source)
            if (row, column) == (0, 1):
                column_map = OperatorMap.from_source(embedding, source)
                return zero if column_map is zero else column_map.adjoint()
            return ModuleEndomorphism.source(embedding, source)

        return BlockSeries(
            eval=evaluate,
            shape=(2, 2),
            n_infinite=self.hamiltonian.n_infinite,
            dimension_names=self.hamiltonian.dimension_names,
            name="H",
        )


def block_diagonalize(
    hamiltonian: BlockSeries,
    embedding: Embedding,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """Run the standard recurrence over a structured embedding."""
    problem = _EmbeddingProblem(hamiltonian, embedding)
    outputs, _ = series_computation(
        {"H": problem.block_series()},
        algorithm=main,
        scope={
            "solve_sylvester": problem.solve_sylvester,
            "use_linear_operator": np.zeros((2, 2), dtype=bool),
            "two_block_optimized": True,
            "commuting_blocks": [True, True],
        },
        operator=multiply_projected,
    )
    return outputs["H_tilde"], outputs["U"], outputs["U†"]
