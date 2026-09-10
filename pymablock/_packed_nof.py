"""Low-level packed-binary keys used by :class:`NumberOrderedForm`."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import sympy
from sympy.physics.quantum.fermion import FermionOp

from pymablock._packed_binary import (
    masks_from_monomial,
    monomial_from_masks,
    multiply_monomials,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def anticommuting_mask(operators: Sequence, n_infinite: int) -> int:
    """Return binary-basis bits whose odd factors anticommute."""
    return sum(
        1 << index
        for index, operator in enumerate(operators[n_infinite:])
        if isinstance(operator, FermionOp)
    )


def _evaluate_binary(
    coefficient: sympy.Expr,
    variable: sympy.Symbol,
    value: sympy.Integer,
) -> sympy.Expr:
    """Evaluate a binary coefficient while holding singular samples."""
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
    """Return coefficients of the unique multilinear binary polynomial."""
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
            if at_zero != 0:
                expanded[mask] = expanded.get(mask, sympy.S.Zero) + at_zero
            if difference != 0:
                numbered = mask | (1 << index)
                expanded[numbered] = expanded.get(numbered, sympy.S.Zero) + difference
        result = {
            mask: coefficient
            for mask, coefficient in expanded.items()
            if coefficient != 0
        }
    return result


def _monomial(
    num_modes: int,
    *,
    creators: int = 0,
    numbers: int = 0,
    annihilators: int = 0,
) -> int:
    return monomial_from_masks(
        creators=creators,
        numbers=numbers,
        annihilators=annihilators,
        num_modes=num_modes,
    )


def canonical_monomial(
    num_modes: int,
    anticommuting_modes: int,
    creators: int,
    numbers: int,
    annihilators: int,
) -> tuple[int, int] | None:
    """Pack the creator-number-annihilator convention used by NOF."""
    polynomial = {_monomial(num_modes): 1}

    def multiply_right(factor: int) -> None:
        nonlocal polynomial
        polynomial = {
            product: left_integer * right_integer
            for left, left_integer in polynomial.items()
            for product, right_integer in multiply_monomials(
                left,
                factor,
                num_modes=num_modes,
                anticommuting_modes=anticommuting_modes,
            ).items()
        }

    for mode in range(num_modes):
        if creators & (1 << mode):
            multiply_right(_monomial(num_modes, creators=1 << mode))
    if numbers:
        multiply_right(_monomial(num_modes, numbers=numbers))
    for mode in reversed(range(num_modes)):
        if annihilators & (1 << mode):
            multiply_right(_monomial(num_modes, annihilators=1 << mode))

    if not polynomial:
        return None
    if len(polynomial) != 1:  # pragma: no cover - local factors are disjoint
        raise AssertionError("A canonical binary monomial unexpectedly branched")
    return next(iter(polynomial.items()))


def pack_terms(
    operators: Sequence,
    n_infinite: int,
    placeholders: tuple[sympy.Symbol, ...],
    terms: Iterable[tuple[Sequence[int], sympy.Expr]],
) -> sympy.Tuple:
    """Pack operator powers and occupation-dependent coefficients."""
    num_binary = len(operators) - n_infinite
    binary_numbers = placeholders[n_infinite:]
    fermion_mask = anticommuting_mask(operators, n_infinite)
    combined = defaultdict(lambda: sympy.S.Zero)

    for powers, coefficient in terms:
        infinite_powers = tuple(powers[:n_infinite])
        binary_powers = tuple(map(int, powers[n_infinite:]))
        if len(powers) != len(operators):
            raise ValueError(
                f"Powers tuple length ({len(powers)}) doesn't match "
                f"operators length ({len(operators)})"
            )
        if any(abs(power) > 1 for power in binary_powers):
            raise ValueError("Binary operator powers must be -1, 0, or 1")
        creators = sum((power < 0) << index for index, power in enumerate(binary_powers))
        annihilators = sum(
            (power > 0) << index for index, power in enumerate(binary_powers)
        )
        for number_mask, scalar in boolean_coefficients(
            coefficient, binary_numbers
        ).items():
            packed = canonical_monomial(
                num_binary,
                fermion_mask,
                creators,
                number_mask,
                annihilators,
            )
            if packed is None:
                continue
            monomial, normalization = packed
            combined[(infinite_powers, monomial)] += scalar * normalization

    return build_packed_terms(combined.items())


def build_packed_terms(
    terms: Iterable[tuple[tuple[tuple[int, ...], int], sympy.Expr]],
) -> sympy.Tuple:
    """Combine equal packed keys and return immutable SymPy storage."""
    combined = defaultdict(lambda: sympy.S.Zero)
    for key, coefficient in terms:
        if coefficient != 0:
            combined[key] += coefficient
    return sympy.Tuple(
        *(
            sympy.Tuple(
                sympy.Tuple(sympy.Tuple(*powers), sympy.Integer(monomial)),
                coefficient,
            )
            for (powers, monomial), coefficient in sorted(
                combined.items(),
                key=lambda item: (
                    tuple(map(sympy.default_sort_key, item[0][0])),
                    item[0][1],
                ),
            )
            if coefficient != 0
        )
    )


def remap_monomial(
    monomial: int,
    old_num_modes: int,
    old_to_new: Sequence[int],
    new_num_modes: int,
) -> int:
    """Move a packed monomial to a larger, reordered binary basis."""
    creators, numbers, annihilators = masks_from_monomial(
        monomial, num_modes=old_num_modes
    )

    def remap(mask: int) -> int:
        return sum(
            1 << new_index
            for old_index, new_index in enumerate(old_to_new)
            if mask & (1 << old_index)
        )

    return monomial_from_masks(
        creators=remap(creators),
        numbers=remap(numbers),
        annihilators=remap(annihilators),
        num_modes=new_num_modes,
    )


def unpack_terms(
    operators: Sequence,
    n_infinite: int,
    placeholders: tuple[sympy.Symbol, ...],
    packed_terms: Iterable,
) -> dict[tuple[int, ...], sympy.Expr]:
    """Decode packed keys into the public NOF power-key convention."""
    num_binary = len(operators) - n_infinite
    binary_numbers = placeholders[n_infinite:]
    fermion_mask = anticommuting_mask(operators, n_infinite)
    result = defaultdict(lambda: sympy.S.Zero)

    for key, coefficient in packed_terms:
        infinite_powers, monomial = key
        creators, numbers, annihilators = masks_from_monomial(
            int(monomial), num_modes=num_binary
        )
        canonical = canonical_monomial(
            num_binary,
            fermion_mask,
            creators,
            numbers,
            annihilators,
        )
        if canonical is None:  # pragma: no cover
            raise AssertionError("A stored packed monomial decodes to zero")
        canonical_code, normalization = canonical
        if canonical_code != monomial:  # pragma: no cover
            raise AssertionError("Packed and NOF canonical monomials disagree")
        binary_powers = tuple(
            -sympy.S.One
            if creators & (1 << index)
            else sympy.S.One
            if annihilators & (1 << index)
            else sympy.S.Zero
            for index in range(num_binary)
        )
        coefficient = (
            coefficient
            / normalization
            * sympy.prod(
                binary_numbers[index]
                for index in range(num_binary)
                if numbers & (1 << index)
            )
        )
        powers = (*infinite_powers, *binary_powers)
        result[powers] += coefficient

    return {
        powers: coefficient for powers, coefficient in result.items() if coefficient != 0
    }
