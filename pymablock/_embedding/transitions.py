"""Ladder amplitudes on source occupation states.

The classes in this module do not define a new operator representation.  They
expose one NOF monomial as a weighted partial transition on occupation states,
using the same ladder amplitudes for symbolic and concrete occupations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import TYPE_CHECKING

import sympy
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock.number_ordered_form import LadderOp, NumberOperator, NumberOrderedForm

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@cache
def number_symbols(operators: tuple) -> tuple[sympy.Symbol, ...]:
    """Obtain coefficient coordinates using public NOF term inspection.

    A number operator has one diagonal term whose coefficient is its occupation
    symbol. Query that term rather than depending on private placeholder names
    or metadata. This works with both plain and packed NOF storage.
    """
    powers = (0,) * len(operators)
    return tuple(
        NumberOrderedForm.from_expr(NumberOperator(op), operators=operators).terms[powers]
        for op in operators
    )


@dataclass(frozen=True, slots=True)
class WeightedTransition:
    """One partial transition between occupation states."""

    output_state: tuple[sympy.Expr, ...]
    weight: sympy.Expr


@dataclass(frozen=True)
class NOFTransition:
    """The occupation shift and amplitude of one NOF term."""

    form: NumberOrderedForm
    powers: tuple[int, ...]

    @classmethod
    def from_form(cls, form: NumberOrderedForm) -> Iterable[NOFTransition]:
        """Read the number-ordered term interface, independently of storage."""
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
        return number_symbols(self.operators)

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
        action = self.symbolic_action(state)
        return None if action.weight == 0 else action

    def symbolic_action(self, occupations: Sequence[sympy.Expr]) -> WeightedTransition:
        """Apply this term to symbolic occupations."""
        ((_, coefficient),) = self.form.terms.items()
        current = list(map(sympy.sympify, occupations))
        amplitude = sympy.S.One

        for index, power in enumerate(self.powers):
            for _ in range(max(power, 0)):
                factor = self._symbolic_generator(current, index, annihilate=True)
                if factor == 0:
                    return WeightedTransition(tuple(current), sympy.S.Zero)
                amplitude *= factor

        amplitude *= coefficient.xreplace(
            dict(zip(self.placeholders, current, strict=True))
        )

        for index in reversed(range(len(self.powers))):
            for _ in range(max(-self.powers[index], 0)):
                factor = self._symbolic_generator(current, index, annihilate=False)
                if factor == 0:
                    return WeightedTransition(tuple(current), sympy.S.Zero)
                amplitude *= factor

        return WeightedTransition(
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
            # Bosonic annihilation requires occupation >= power, not equality.
            # Its vanishing channels are handled by their transition weights.
        return tuple(equations)

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
