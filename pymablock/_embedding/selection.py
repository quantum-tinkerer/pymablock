"""Compile a source-state selection into operators on the retained states."""

from __future__ import annotations

from functools import cache, cached_property
from typing import TYPE_CHECKING

import sympy

from pymablock._embedding.transitions import NOFTransition, number_symbols
from pymablock.number_ordered_form import NumberOrderedForm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymablock._embedding.maps import TargetOperator
    from pymablock.operator_embedding import Embedding


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


class _EmbeddingBackend:
    """Evaluate source operators in the selected target occupation basis."""

    def __init__(self, embedding: Embedding):
        self.descriptor = embedding
        self.source_operators = tuple(embedding.operators)
        self.coordinates = embedding.coordinate_symbols
        self.target_operators = embedding.target.operators
        self.target_is_nof = embedding.target_is_nof

        self.source_placeholders = number_symbols(self.source_operators)
        self.target_placeholders = (
            number_symbols(self.target_operators) if self.target_is_nof else ()
        )
        if self.target_is_nof:
            self.target_identity = _one_term(
                self.target_operators, (0,) * len(self.target_operators), sympy.S.One
            )
            self.target_zero = NumberOrderedForm(
                self.target_operators, {}, validate=False
            )
        else:
            self.target_identity = sympy.ImmutableMatrix(
                sympy.eye(embedding.target.dimension)
            )
            self.target_zero = sympy.ImmutableMatrix(
                sympy.zeros(embedding.target.dimension)
            )

    @cache
    def target_shift(self, source_shift: tuple[int, ...]) -> tuple[int, ...] | None:
        """Find the binary target transition for a source occupation shift."""
        source = sympy.Matrix(source_shift)
        result = self.descriptor._occupation_left_inverse * source
        if self.descriptor._occupation_matrix * result != source:
            return None
        if any(not value.is_Integer or abs(value) > 1 for value in result):
            return None
        return tuple(map(int, result))

    def pullback_weight(
        self,
        transition: NOFTransition,
        target_shift: tuple[int, ...],
    ) -> sympy.Expr:
        """Return the target diagonal weight of a pulled-back transition."""
        source_action = transition.symbolic_action(self.descriptor.source_occupations)
        shifted = {
            symbol: symbol - power
            for symbol, power in zip(self.coordinates, target_shift, strict=True)
        }
        phase_ratio = self.descriptor.phase * sympy.conjugate(
            self.descriptor.phase.xreplace(shifted)
        )
        amplitude = source_action.weight * phase_ratio
        amplitude = amplitude.xreplace(self.transition_support(target_shift))
        return sympy.expand(amplitude.xreplace(self.initial_to_middle(target_shift)))

    def transition_support(
        self, target_shift: tuple[int, ...]
    ) -> dict[sympy.Symbol, sympy.Expr]:
        """Return support values forced by the retained transition."""
        return {
            symbol: sympy.S.One if power > 0 else sympy.S.Zero
            for symbol, power in zip(self.coordinates, target_shift, strict=True)
            if power
        }

    def initial_to_middle(
        self, target_shift: tuple[int, ...]
    ) -> dict[sympy.Symbol, sympy.Expr]:
        """Translate input coordinates to NOF middle coordinates."""
        support = self.transition_support(target_shift)
        return {
            symbol: placeholder + max(power, 0)
            for symbol, placeholder, power in zip(
                self.coordinates,
                self.target_placeholders,
                target_shift,
                strict=True,
            )
            if symbol not in support
        }

    @cache
    def support_substitutions(
        self,
        transition: NOFTransition,
        target_shift: tuple[int, ...],
    ) -> dict[sympy.Symbol, sympy.Expr]:
        """Infer Boolean coordinates fixed by a composed transition."""
        after_target = {
            symbol: symbol - power
            for symbol, power in zip(self.coordinates, target_shift, strict=True)
        }
        before_source = tuple(
            occupation.xreplace(after_target)
            for occupation in self.descriptor.source_occupations
        )
        equations = list(transition.support_equations(before_source))
        equations.extend(
            symbol - (1 if power > 0 else 0)
            for symbol, power in zip(self.coordinates, target_shift, strict=True)
            if power
        )
        return _boolean_solutions(equations, self.coordinates)

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
        target_powers = self.target_shift(transition.powers)
        if target_powers is None:
            return self.target_zero
        amplitude = self.pullback_weight(transition, target_powers)
        if amplitude == 0:
            return self.target_zero
        # The source action includes its Fock sign. A target fermion monomial
        # supplies a Fock sign of its own; remove it from the coefficient so
        # that evaluating the target operator does not count it twice.
        target_form = _one_term(self.target_operators, target_powers, sympy.S.One)
        (target_transition,) = NOFTransition.from_form(target_form)
        target_weight = target_transition.symbolic_action(self.coordinates).weight
        target_weight = target_weight.xreplace(
            self.transition_support(target_powers)
        ).xreplace(self.initial_to_middle(target_powers))
        amplitude *= target_weight  # Inverse of a sign.
        # Multiplication applies the public NOF binary-number normalization.
        return target_form * _one_term(
            self.target_operators, (0,) * len(target_powers), amplitude
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
                        sympy.conjugate(final_phase) * initial_phase * action.weight
                    )
        return sympy.ImmutableMatrix(matrix)


def _boolean_solutions(
    equations: Sequence[sympy.Expr],
    variables: tuple[sympy.Symbol, ...],
) -> dict[sympy.Symbol, sympy.Expr]:
    """Return Boolean variables uniquely fixed by a small equation system."""
    if not equations:
        return {}
    matrix, right_hand_side = sympy.linear_eq_to_matrix(equations, variables)
    reduced, pivots = matrix.row_join(right_hand_side).rref()
    substitutions = {}
    for row, pivot in enumerate(pivots):
        if pivot >= len(variables):
            return {}
        if any(
            reduced[row, column] != 0
            for column in range(len(variables))
            if column != pivot
        ):
            continue
        value = reduced[row, -1]
        if value in (sympy.S.Zero, sympy.S.One):
            substitutions[variables[pivot]] = value
    return substitutions
