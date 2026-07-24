"""Combinatorial views of number-ordered operators and basis embeddings.

The classes in this module do not define a new operator representation.  They
expose one NOF monomial as a weighted partial transition on occupation states,
and one basis embedding as a coordinate map

``W |target> = phase(target) |source(target)>``.

Projection, finite materialization, and diagonal energy denominators can then
share the same transition calculus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import TYPE_CHECKING

import sympy
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock._packed_binary import masks_from_monomial
from pymablock.number_ordered_form import LadderOp, NumberOrderedForm

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

OccupationState = tuple[int | sympy.Expr, ...]


@dataclass(frozen=True, slots=True)
class WeightedTransition:
    """One partial transition between occupation states."""

    input_state: OccupationState
    output_state: OccupationState
    weight: sympy.Expr


@dataclass(frozen=True)
class NOFTransition:
    """A coefficient-normalized weighted-transition view of one NOF term."""

    form: NumberOrderedForm
    powers: tuple[int, ...]
    scalar: sympy.Expr = sympy.S.One
    monomial: int | None = None

    @classmethod
    def from_form(cls, form: NumberOrderedForm) -> Iterable[NOFTransition]:
        """Decompose a NOF without expanding packed binary coefficients."""
        if form._n_inf_order == 0:
            for (infinite_powers, monomial), coefficient in form._packed_terms:
                if infinite_powers:  # pragma: no cover - guaranteed by metadata
                    raise AssertionError("A binary NOF has infinite powers")
                yield cls(
                    NumberOrderedForm._from_packed_terms(
                        form.operators,
                        ((((), monomial), sympy.S.One),),
                    ),
                    _powers_from_monomial(monomial, len(form.operators)),
                    coefficient,
                    monomial,
                )
            return

        for powers, coefficient in form.terms.items():
            powers = tuple(map(int, powers))
            yield cls(
                NumberOrderedForm(
                    form.operators,
                    {powers: coefficient},
                    validate=False,
                ),
                powers,
            )

    @cached_property
    def operators(self) -> tuple:
        """Return the ordered annihilation generators."""
        return tuple(self.form.operators)

    @cached_property
    def placeholders(self) -> tuple[sympy.Symbol, ...]:
        """Return the number-operator placeholders of the source algebra."""
        return tuple(self.form._number_operator_placeholders)

    @cached_property
    def fermion_indices(self) -> tuple[int, ...]:
        """Return source indices that contribute fermionic parity."""
        return tuple(
            index
            for index, operator in enumerate(self.operators)
            if isinstance(operator, FermionOp)
        )

    def apply(self, state: Sequence[int]) -> WeightedTransition | None:
        """Apply this term to a concrete occupation state."""
        ((_, coefficient),) = self.form.terms.items()
        initial = tuple(map(int, state))
        current = list(initial)
        amplitude = sympy.S.One

        for index, power in enumerate(self.powers):
            for _ in range(max(power, 0)):
                action = self._apply_generator(current, index, annihilate=True)
                if action is None:
                    return None
                amplitude *= action

        amplitude *= coefficient.xreplace(
            dict(zip(self.placeholders, map(sympy.Integer, current), strict=True))
        )
        if amplitude == 0:
            return None

        for index in reversed(range(len(self.powers))):
            for _ in range(max(-self.powers[index], 0)):
                action = self._apply_generator(current, index, annihilate=False)
                if action is None:
                    return None
                amplitude *= action

        amplitude = sympy.expand(amplitude)
        if amplitude == 0:
            return None
        return WeightedTransition(initial, tuple(current), amplitude)

    def symbolic_action(self, occupations: Sequence[sympy.Expr]) -> WeightedTransition:
        """Apply this term to symbolic occupations."""
        initial = tuple(map(sympy.sympify, occupations))
        if self.monomial is not None:
            return self._symbolic_binary_action(initial)

        ((_, coefficient),) = self.form.terms.items()
        current = list(initial)
        amplitude = sympy.S.One

        for index, power in enumerate(self.powers):
            for _ in range(max(power, 0)):
                amplitude *= self._symbolic_generator(current, index, annihilate=True)

        amplitude *= coefficient.xreplace(
            dict(zip(self.placeholders, current, strict=True))
        )

        for index in reversed(range(len(self.powers))):
            for _ in range(max(-self.powers[index], 0)):
                amplitude *= self._symbolic_generator(current, index, annihilate=False)

        return WeightedTransition(
            initial,
            tuple(current),
            sympy.expand(amplitude),
        )

    def support_equations(
        self, input_occupations: Sequence[sympy.Expr]
    ) -> tuple[sympy.Expr, ...]:
        """Return exact occupation constraints implied by this transition."""
        equations = []
        for occupation, operator, power in zip(
            input_occupations, self.operators, self.powers, strict=True
        ):
            if isinstance(operator, (FermionOp, SigmaMinus)) and power:
                equations.append(occupation - (1 if power > 0 else 0))
            elif isinstance(operator, BosonOp) and power > 0:
                # This exact constraint suffices for the finite/binary retained
                # coordinates currently compiled to an algebraic target.
                equations.append(occupation - power)
        return tuple(equations)

    def _apply_generator(
        self,
        state: list[int],
        index: int,
        *,
        annihilate: bool,
    ) -> sympy.Expr | None:
        operator = self.operators[index]
        occupation = state[index]
        if isinstance(operator, BosonOp):
            if annihilate and occupation == 0:
                return None
            factor = sympy.sqrt(occupation if annihilate else occupation + 1)
        elif isinstance(operator, LadderOp):
            factor = sympy.S.One
        elif isinstance(operator, (SigmaMinus, FermionOp)):
            if occupation != int(annihilate):
                return None
            factor = (
                (-1)
                ** sum(
                    state[earlier] for earlier in self.fermion_indices if earlier < index
                )
                if isinstance(operator, FermionOp)
                else sympy.S.One
            )
        else:  # pragma: no cover - guarded by NumberOrderedForm
            raise TypeError(f"Unsupported source operator: {operator!r}")
        state[index] += -1 if annihilate else 1
        return sympy.sympify(factor)

    def _symbolic_generator(
        self,
        state: list[sympy.Expr],
        index: int,
        *,
        annihilate: bool,
    ) -> sympy.Expr:
        operator = self.operators[index]
        occupation = state[index]
        if isinstance(operator, BosonOp):
            factor = sympy.sqrt(occupation if annihilate else occupation + 1)
        elif isinstance(operator, LadderOp):
            factor = sympy.S.One
        elif isinstance(operator, SigmaMinus):
            factor = occupation if annihilate else 1 - occupation
        elif isinstance(operator, FermionOp):
            factor = (occupation if annihilate else 1 - occupation) * sympy.prod(
                1 - 2 * state[earlier]
                for earlier in self.fermion_indices
                if earlier < index
            )
        else:  # pragma: no cover - guarded by NumberOrderedForm
            raise TypeError(f"Unsupported source operator: {operator!r}")
        state[index] += -1 if annihilate else 1
        return sympy.sympify(factor)

    def _symbolic_binary_action(self, initial: OccupationState) -> WeightedTransition:
        creators, numbers, annihilators = masks_from_monomial(
            self.monomial, num_modes=len(self.operators)
        )
        current = list(initial)
        amplitude = sympy.S.One
        for mode in reversed(range(len(self.operators))):
            active = 1 << mode
            if numbers & active:
                amplitude *= current[mode]
                continue
            if not (creators | annihilators) & active:
                continue
            parity = sympy.prod(
                1 - 2 * current[earlier]
                for earlier in self.fermion_indices
                if earlier < mode
            )
            if annihilators & active:
                amplitude *= current[mode] * parity
                current[mode] = sympy.S.Zero
            else:
                amplitude *= (1 - current[mode]) * parity
                current[mode] = sympy.S.One
        return WeightedTransition(initial, tuple(current), amplitude)


@dataclass(frozen=True)
class BasisMap:
    """A phase-decorated coordinate map between occupation bases."""

    source_occupations: tuple[sympy.Expr, ...]
    target_symbols: tuple[sympy.Symbol, ...]
    target_placeholders: tuple[sympy.Symbol, ...] = ()
    phase: sympy.Expr = sympy.S.One
    _occupation_matrix: sympy.Matrix | None = field(
        init=False, repr=False, compare=False, hash=False
    )
    _occupation_left_inverse: sympy.Matrix | None = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        occupation_matrix = sympy.Matrix(
            [
                [sympy.diff(occupation, symbol) for symbol in self.target_symbols]
                for occupation in self.source_occupations
            ]
        )
        affine = all(
            not entry.free_symbols.intersection(self.target_symbols)
            for entry in occupation_matrix
        )
        if affine and occupation_matrix.rank() == len(self.target_symbols):
            left_inverse = (
                occupation_matrix.T * occupation_matrix
            ).inv() * occupation_matrix.T
        else:
            occupation_matrix = left_inverse = None
        object.__setattr__(self, "_occupation_matrix", occupation_matrix)
        object.__setattr__(self, "_occupation_left_inverse", left_inverse)

    @cache
    def target_shift(self, source_shift: tuple[int, ...]) -> tuple[int, ...] | None:
        """Solve for a constant retained shift producing ``source_shift``."""
        if self._occupation_left_inverse is not None:
            source_vector = sympy.Matrix(source_shift)
            result = self._occupation_left_inverse * source_vector
            if self._occupation_matrix * result != source_vector:
                return None
            if any(not value.is_Integer for value in result):
                return None
            integer_result = tuple(map(int, result))
            return (
                integer_result
                if all(abs(value) <= 1 for value in integer_result)
                else None
            )

        shifts = sympy.symbols(f"_shift_0:{len(self.target_symbols)}", integer=True)
        shifted = {
            symbol: symbol - shift
            for symbol, shift in zip(self.target_symbols, shifts, strict=True)
        }
        equations = [
            sympy.expand(occupation.xreplace(shifted) - occupation + power)
            for occupation, power in zip(
                self.source_occupations, source_shift, strict=True
            )
        ]
        solution = sympy.solve(equations, shifts, dict=True)
        if len(solution) != 1 or any(shift not in solution[0] for shift in shifts):
            return None
        result = tuple(sympy.simplify(solution[0][shift]) for shift in shifts)
        if any(not value.is_Integer for value in result):
            return None
        integer_result = tuple(map(int, result))
        return (
            integer_result if all(abs(value) <= 1 for value in integer_result) else None
        )

    def pullback_weight(
        self,
        transition: NOFTransition,
        target_shift: tuple[int, ...],
    ) -> sympy.Expr:
        """Return the target diagonal weight of a pulled-back transition."""
        source_action = transition.symbolic_action(self.source_occupations)
        shifted = {
            symbol: symbol - power
            for symbol, power in zip(self.target_symbols, target_shift, strict=True)
        }
        phase_ratio = self.phase * sympy.conjugate(self.phase.xreplace(shifted))
        amplitude = source_action.weight * phase_ratio
        amplitude = amplitude.xreplace(self.transition_support(target_shift))
        return sympy.expand(amplitude.xreplace(self.initial_to_middle(target_shift)))

    def transition_support(
        self, target_shift: tuple[int, ...]
    ) -> dict[sympy.Symbol, sympy.Expr]:
        """Return support values forced by the retained transition."""
        return {
            symbol: sympy.S.One if power > 0 else sympy.S.Zero
            for symbol, power in zip(self.target_symbols, target_shift, strict=True)
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
                self.target_symbols,
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
            for symbol, power in zip(self.target_symbols, target_shift, strict=True)
        }
        before_source = tuple(
            occupation.xreplace(after_target) for occupation in self.source_occupations
        )
        equations = list(transition.support_equations(before_source))
        equations.extend(
            symbol - (1 if power > 0 else 0)
            for symbol, power in zip(self.target_symbols, target_shift, strict=True)
            if power
        )
        return _boolean_solutions(equations, self.target_symbols)

    def energy_denominator(
        self,
        transition: NOFTransition,
        target_shift: tuple[int, ...],
        *,
        source_energy: sympy.Expr,
        source_placeholders: tuple[sympy.Symbol, ...],
        target_energy: sympy.Expr,
    ) -> sympy.Expr:
        """Evaluate the diagonal Sylvester denominator on one channel."""
        after_target = tuple(
            symbol - power
            for symbol, power in zip(self.target_symbols, target_shift, strict=True)
        )
        encoded_after_target = tuple(
            occupation.xreplace(dict(zip(self.target_symbols, after_target, strict=True)))
            for occupation in self.source_occupations
        )
        source_output = tuple(
            occupation - power
            for occupation, power in zip(
                encoded_after_target, transition.powers, strict=True
            )
        )
        source_value = source_energy.xreplace(
            dict(zip(source_placeholders, source_output, strict=True))
        )
        denominator = sympy.expand(target_energy - source_value)
        denominator = denominator.xreplace(
            self.support_substitutions(transition, target_shift)
        )
        return denominator.xreplace(self.initial_to_middle(target_shift))


def _powers_from_monomial(monomial: int, num_modes: int) -> tuple[int, ...]:
    creators, _numbers, annihilators = masks_from_monomial(monomial, num_modes=num_modes)
    return tuple(
        -1 if creators & (1 << mode) else 1 if annihilators & (1 << mode) else 0
        for mode in range(num_modes)
    )


def _boolean_solutions(
    equations: Sequence[sympy.Expr],
    variables: tuple[sympy.Symbol, ...],
) -> dict[sympy.Symbol, sympy.Expr]:
    """Return Boolean variables uniquely fixed by a small equation system."""
    if not equations:
        return {}
    try:
        matrix, right_hand_side = sympy.linear_eq_to_matrix(equations, variables)
    except sympy.NonlinearError:
        solutions = sympy.solve(equations, variables, dict=True)
        if len(solutions) != 1:
            return {}
        return {
            symbol: value
            for symbol, value in solutions[0].items()
            if value in (sympy.S.Zero, sympy.S.One)
        }

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
