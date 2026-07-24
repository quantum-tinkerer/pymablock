"""Pymablock adapter for algebraic occupation-basis embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import sympy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock._combinatorics import BasisMap, NOFTransition
from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main
from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.operator_embedding import (
    Coordinate,
    FermionEmbedding,
    OperatorEmbedding,
)
from pymablock.operator_map import (
    AdjointOperatorMap,
    ModuleEndomorphism,
    OperatorMap,
    TargetOperator,
    multiply_projected,
)
from pymablock.series import BlockSeries, zero


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
    return not bool(value)


def _immutable(matrix: sympy.MatrixBase) -> sympy.ImmutableMatrix:
    if isinstance(matrix, sympy.ImmutableMatrix):
        return matrix
    return sympy.ImmutableMatrix(matrix)


@dataclass(frozen=True)
class _ComplementSpace:
    """The unenumerated complement fixed by one structured embedding."""

    embedding: OperatorEmbedding


class _EmbeddingBackend:
    """Compile a public embedding to generic weighted transitions."""

    def __init__(self, embedding: OperatorEmbedding):
        self.descriptor = embedding
        self.source_operators = tuple(embedding.operators)
        target_is_fermionic = isinstance(embedding, FermionEmbedding)
        if target_is_fermionic:
            self.coordinates = tuple(embedding.target.modes)
            self.target_operators = tuple(embedding.target.modes)
            self.target_is_nof = True
        else:
            self.coordinates = tuple(embedding.target.coordinates)
            self.target_is_nof = all(
                coordinate.values == (0, 1) for coordinate in self.coordinates
            )
            self.target_operators = (
                tuple(
                    SigmaMinus(sympy.Symbol(coordinate.name))
                    for coordinate in self.coordinates
                )
                if self.target_is_nof
                else ()
            )

        coordinate_symbols = tuple(
            sympy.Symbol(f"_target_{index}", integer=True, nonnegative=True)
            for index in range(len(self.coordinates))
        )
        if target_is_fermionic:
            target_occupations = dict(
                zip(self.target_operators, coordinate_symbols, strict=True)
            )
            source_occupations = tuple(
                sympy.sympify(
                    target_occupations[value] if isinstance(value, FermionOp) else value
                )
                for value in (
                    embedding._modes[operator] for operator in self.source_operators
                )
            )
            phase = sympy.prod(
                (1 - 2 * symbol) ** (embedding._fixed_before[operator] % 2)
                for operator, symbol in zip(
                    self.target_operators,
                    coordinate_symbols,
                    strict=True,
                )
            )
        else:
            values = dict(zip(self.coordinates, coordinate_symbols, strict=True))
            source_occupations = tuple(
                sympy.sympify(
                    values[expression]
                    if isinstance(expression, Coordinate)
                    else expression.function(values)
                )
                for expression in (
                    embedding._expressions[operator] for operator in self.source_operators
                )
            )
            phase = sympy.S.One

        source_identity = _one(self.source_operators)
        self.source_placeholders = tuple(source_identity._number_operator_placeholders)
        target_placeholders = (
            tuple(_one(self.target_operators)._number_operator_placeholders)
            if self.target_is_nof
            else ()
        )
        self.basis_map = BasisMap(
            source_occupations,
            coordinate_symbols,
            target_placeholders,
            phase,
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
        return (
            _one_term(self.target_operators, target_powers, amplitude) * transition.scalar
        )

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
            return result

        target = self.descriptor.target
        source_to_target = {
            self.descriptor.encode(state): index
            for index, state in enumerate(target.states)
        }
        matrix = sympy.MutableSparseMatrix(target.dimension, target.dimension, {})
        for column, state in enumerate(target.states):
            source_state = self.descriptor.encode(state)
            for transition in transitions:
                action = transition.apply(source_state)
                if action is None:
                    continue
                if (row := source_to_target.get(action.output_state)) is not None:
                    matrix[row, column] += action.weight * transition.scalar
        return _immutable(matrix)


class _EmbeddingProblem:
    """Lower one scalar Hamiltonian series to the formal P/Q block algebra."""

    def __init__(self, hamiltonian: BlockSeries, embedding: OperatorEmbedding):
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
        self.source_energy_placeholders = tuple(h0._number_operator_placeholders)

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
                    if denominator == 0:
                        if self._zero_denominator_is_projected_out(source, target_term):
                            continue
                        raise ZeroDivisionError(
                            "A virtual channel is degenerate with the retained space"
                        )
                    solved_terms.append(
                        (
                            source.form,
                            target_term
                            * self._target_reciprocal(denominator)
                            * source.scalar,
                        )
                    )
        column = OperatorMap(self.embedding, solved_terms).or_zero()
        return zero if column is zero else column.adjoint()

    def _solve_finite(self, value: AdjointOperatorMap):
        solved_terms = []
        for source_form, target in value.column.terms:
            for source in NOFTransition.from_form(source_form):
                for (row, column), target_coefficient in target.todok().items():
                    action = self.embedding.finite_transition_action(source, row)
                    if action is None:
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
            if not source:
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
    embedding: OperatorEmbedding,
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
