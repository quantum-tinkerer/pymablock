"""Perturbation theory for a selected set of occupation states."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import sympy

from pymablock._embedding.maps import (
    AdjointOperatorMap,
    ModuleEndomorphism,
    OperatorMap,
    _is_zero,
    multiply_projected,
)
from pymablock._embedding.selection import _EmbeddingBackend, _one_term
from pymablock._embedding.transitions import NOFTransition
from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main
from pymablock.series import BlockSeries, zero

if TYPE_CHECKING:
    from pymablock.number_ordered_form import NumberOrderedForm
    from pymablock.operator_embedding import Embedding


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

        # Selecting occupation states of diagonal H0 is automatically invariant.
        self.target_energy = self.source_energy.xreplace(
            dict(
                zip(
                    self.embedding.source_placeholders,
                    embedding.source_occupations,
                    strict=True,
                )
            )
        )
        self.finite_target_energy = (
            None
            if embedding.target_is_nof
            else tuple(
                self.target_energy.xreplace(
                    dict(zip(embedding.coordinate_symbols, state, strict=True))
                )
                for state in embedding.target.states
            )
        )

    @cache
    def _source_energy_at(self, state: tuple[int, ...]) -> sympy.Expr:
        return self.source_energy.xreplace(
            dict(
                zip(
                    self.embedding.source_placeholders,
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
        after_target = tuple(
            symbol - power
            for symbol, power in zip(
                self.embedding.coordinates, target.powers, strict=True
            )
        )
        encoded_after_target = tuple(
            occupation.xreplace(
                dict(zip(self.embedding.coordinates, after_target, strict=True))
            )
            for occupation in self.embedding.descriptor.source_occupations
        )
        source_output = tuple(
            occupation - power
            for occupation, power in zip(encoded_after_target, source.powers, strict=True)
        )
        source_value = self.source_energy.xreplace(
            dict(zip(self.embedding.source_placeholders, source_output, strict=True))
        )
        denominator = sympy.expand(self.target_energy - source_value)
        denominator = denominator.xreplace(
            self.embedding.support_substitutions(source, target.powers)
        )
        return denominator.xreplace(self.embedding.initial_to_middle(target.powers))

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
        return _is_zero(norm)

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
                    target_term = target.form
                    quotient = self._divide_channel(source, target_term, denominator)
                    if not _is_zero(quotient):
                        solved_terms.append((source.form, quotient))
        column = OperatorMap(self.embedding, solved_terms).or_zero()
        return zero if column is zero else column.adjoint()

    def _divide_channel(
        self, source: NOFTransition, target: NumberOrderedForm, denominator: sympy.Expr
    ) -> NumberOrderedForm:
        """Divide on binary sectors, discarding unsupported resonances first."""
        variables = tuple(
            symbol
            for symbol in self.embedding.target_placeholders
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
                        self.embedding.descriptor.target.dimension,
                        self.embedding.descriptor.target.dimension,
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
                            target_term / denominator,
                        )
                    )
        column = OperatorMap(self.embedding, solved_terms).or_zero()
        return zero if column is zero else column.adjoint()

    def block_series(self) -> BlockSeries:
        """Represent the retained and discarded Hamiltonian blocks."""
        embedding = self.embedding
        zero_order = (0,) * self.hamiltonian.n_infinite

        def evaluate(*index):
            row, column, *order = index
            source = self.hamiltonian[tuple(order)]
            if source is zero:
                return zero
            source = embedding.source_form(source)
            if _is_zero(source):
                return zero
            if row == column == 0:
                result = embedding.pullback(source)
                return zero if _is_zero(result) else result
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
