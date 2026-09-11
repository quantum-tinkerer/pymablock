"""Convert and assemble the packed terms stored by ``NumberOrderedForm``.

Public terms group creation operators, number factors, and annihilation
operators. Packed terms order fermion and spin factors by mode, with number
factors included in their keys. Conversion accounts for the fermionic sign
between these orderings and expands symbolic binary number factors.

SymPy storage holds one integer per binary monomial in two ``num_binary``-bit
planes: the low plane marks modes carrying a or N, the high plane modes
carrying a† or N. Arithmetic uses the three disjoint masks instead.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import sympy
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock._packed_binary import BinaryMonomial, _crossing_sign

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


PowerKey: TypeAlias = tuple[int | sympy.Expr, ...]
PackedKey: TypeAlias = tuple[PowerKey, BinaryMonomial]
PackedTerm: TypeAlias = tuple[PackedKey, sympy.Expr]


class Layout(NamedTuple):
    """Mode bookkeeping of an operator list.

    Boson and ladder modes come first and keep symbolic powers; the spin and
    fermion modes that follow are packed into binary monomials.
    """

    n_bosons: int
    n_boson_ladder: int
    num_binary: int
    fermion_mask: int
    # Number operator placeholder symbols, one per operator.
    placeholders: tuple[sympy.Symbol, ...]

    @classmethod
    def from_operators(
        cls, operators: Sequence, placeholders: tuple[sympy.Symbol, ...]
    ) -> Layout:
        binary = [isinstance(op, (FermionOp, SigmaMinus)) for op in operators]
        num_binary = sum(binary)
        n_boson_ladder = len(operators) - num_binary
        if any(binary[:n_boson_ladder]):
            raise ValueError(
                "Boson and ladder operators must precede spins and fermions."
            )
        fermion_mask = sum(
            1 << index
            for index, op in enumerate(operators[n_boson_ladder:])
            if isinstance(op, FermionOp)
        )
        return cls(
            n_bosons=sum(isinstance(op, BosonOp) for op in operators),
            n_boson_ladder=n_boson_ladder,
            num_binary=num_binary,
            fermion_mask=fermion_mask,
            placeholders=tuple(placeholders),
        )

    @property
    def boson_ladder_placeholders(self) -> tuple[sympy.Symbol, ...]:
        return self.placeholders[: self.n_boson_ladder]

    @property
    def binary_placeholders(self) -> tuple[sympy.Symbol, ...]:
        return self.placeholders[self.n_boson_ladder :]

    @property
    def zero_powers(self) -> tuple[sympy.Integer, ...]:
        return (sympy.S.Zero,) * self.n_boson_ladder


def _evaluate_binary(
    coefficient: sympy.Expr,
    variable: sympy.Symbol,
    value: sympy.Integer,
) -> sympy.Expr:
    """Substitute occupation 0 or 1, leaving singular results unevaluated."""
    evaluated = coefficient.xreplace({variable: value})
    if evaluated.is_finite is False or evaluated.has(
        sympy.zoo, sympy.nan, sympy.oo, -sympy.oo
    ):
        return sympy.Subs(coefficient, variable, value, evaluate=False)
    return evaluated


def boolean_coefficients(
    expression: sympy.Expr,
    variables: tuple[sympy.Symbol, ...],
) -> dict[int, sympy.Expr]:
    """Expand a function of occupations 0 or 1 in products of number variables.

    For each variable N, use f(N) = f(0) + N * (f(1) - f(0)). Bit i in a
    returned key means that the term contains variables[i].
    """
    result = {0: expression}
    for index, variable in enumerate(variables):
        if not any(
            variable in coefficient.free_symbols for coefficient in result.values()
        ):
            continue
        expanded = {}
        for mask, coefficient in result.items():
            at_zero = _evaluate_binary(coefficient, variable, sympy.S.Zero)
            difference = _evaluate_binary(coefficient, variable, sympy.S.One) - at_zero
            # Earlier masks contain only bits below index, so all new keys differ.
            if at_zero != 0:
                expanded[mask] = at_zero
            if difference != 0:
                expanded[mask | (1 << index)] = difference
        result = expanded
    return result


def _number_order_sign(monomial: BinaryMonomial, fermion_mask: int) -> int:
    """Return the fermionic sign between the packed and the public factor order.

    Packed factors follow increasing mode order. Public terms put creation
    operators first (increasing mode order) and annihilation operators last
    (decreasing mode order), with number factors in between.
    """
    creation, _, annihilation = monomial
    creation &= fermion_mask
    annihilation &= fermion_mask
    # Move creation operators before annihilation operators, then reverse the
    # order of the annihilation operators. Number factors contribute no sign.
    sign = _crossing_sign(creation, annihilation)
    k = annihilation.bit_count()
    return -sign if (k * (k - 1) // 2) & 1 else sign


def pack_ordered_factors(
    fermion_mask: int,
    creation_mask: int,
    number_mask: int,
    annihilation_mask: int,
) -> tuple[BinaryMonomial, int] | None:
    """Pack number-ordered factors and return the monomial and fermionic sign.

    Creation and annihilation masks must be disjoint. Creation operators are
    in increasing mode order, followed by number factors, then annihilation
    operators in decreasing mode order. Overlapping number factors give zero
    because a† N = N a = 0.
    """
    if number_mask & (creation_mask | annihilation_mask):
        return None
    monomial = (creation_mask, number_mask, annihilation_mask)
    return monomial, _number_order_sign(monomial, fermion_mask)


def encode_monomial(monomial: BinaryMonomial, num_modes: int) -> int:
    """Pack the monomial into the annihilation and creation presence planes."""
    creation, number, annihilation = monomial
    return (annihilation | number) | (creation | number) << num_modes


def decode_monomial(code: int, num_modes: int) -> BinaryMonomial:
    """Split the stored integer into disjoint creation, number, and annihilation masks."""
    annihilation_presence = code & (1 << num_modes) - 1
    creation_presence = code >> num_modes
    number = annihilation_presence & creation_presence
    return (creation_presence & ~number, number, annihilation_presence & ~number)


def pack_terms(
    layout: Layout, terms: Iterable[tuple[PowerKey, sympy.Expr]]
) -> sympy.Tuple:
    """Pack operator powers and occupation-dependent coefficients."""
    num_operators = layout.n_boson_ladder + layout.num_binary
    packed_terms: list[PackedTerm] = []

    for powers, coefficient in terms:
        if len(powers) != num_operators:
            raise ValueError(
                f"Powers tuple length ({len(powers)}) doesn't match "
                f"operators length ({num_operators})"
            )
        boson_ladder_powers = tuple(powers[: layout.n_boson_ladder])
        binary_powers = tuple(map(int, powers[layout.n_boson_ladder :]))
        if any(abs(power) > 1 for power in binary_powers):
            raise ValueError("Binary operator powers must be -1, 0, or 1")
        creation_mask = sum(
            (power < 0) << index for index, power in enumerate(binary_powers)
        )
        annihilation_mask = sum(
            (power > 0) << index for index, power in enumerate(binary_powers)
        )
        for number_mask, scalar in boolean_coefficients(
            coefficient, layout.binary_placeholders
        ).items():
            packed = pack_ordered_factors(
                layout.fermion_mask, creation_mask, number_mask, annihilation_mask
            )
            if packed is None:
                continue
            monomial, sign = packed
            packed_terms.append(((boson_ladder_powers, monomial), scalar * sign))

    return build_packed_terms(layout, packed_terms)


def build_packed_terms(layout: Layout, terms: Iterable[PackedTerm]) -> sympy.Tuple:
    """Combine equal packed keys and return immutable SymPy storage."""
    combined = defaultdict(lambda: sympy.S.Zero)
    for key, coefficient in terms:
        if coefficient != 0:
            combined[key] += coefficient
    encoded_terms = (
        ((powers, encode_monomial(monomial, layout.num_binary)), coefficient)
        for (powers, monomial), coefficient in combined.items()
        if coefficient != 0
    )
    ordered_terms = sorted(
        encoded_terms,
        key=lambda item: (tuple(map(sympy.default_sort_key, item[0][0])), item[0][1]),
    )
    return sympy.Tuple(
        *(
            sympy.Tuple(
                sympy.Tuple(sympy.Tuple(*powers), sympy.Integer(code)), coefficient
            )
            for (powers, code), coefficient in ordered_terms
        )
    )


def binary_powers(monomial: BinaryMonomial, num_modes: int) -> tuple[sympy.Integer, ...]:
    """Return the public powers, with zero for identity and number factors."""
    creation, _, annihilation = monomial
    return tuple(
        sympy.Integer(((annihilation >> index) & 1) - ((creation >> index) & 1))
        for index in range(num_modes)
    )


def unpack_terms(
    layout: Layout, packed_terms: Iterable[PackedTerm]
) -> dict[PowerKey, sympy.Expr]:
    """Group number factors into coefficients keyed by creation/annihilation powers."""
    numbers = layout.binary_placeholders
    result = defaultdict(lambda: sympy.S.Zero)

    for (boson_ladder_powers, monomial), coefficient in packed_terms:
        _, number_mask, _ = monomial
        number_factor = sympy.prod(
            number for index, number in enumerate(numbers) if number_mask >> index & 1
        )
        sign = _number_order_sign(monomial, layout.fermion_mask)
        powers = (*boson_ladder_powers, *binary_powers(monomial, layout.num_binary))
        result[powers] += sign * number_factor * coefficient

    return {
        powers: coefficient for powers, coefficient in result.items() if coefficient != 0
    }
