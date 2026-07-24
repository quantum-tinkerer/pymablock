"""Tests for the private packed hard-core monomial kernel."""

from __future__ import annotations

import random

import pytest

from pymablock._packed_binary import (
    masks_from_monomial,
    monomial_from_masks,
    multiply_monomials,
)


def _multiply_polynomials(
    left: dict[int, int],
    right: dict[int, int],
    *,
    num_modes: int,
    anticommuting_modes: int,
) -> dict[int, int]:
    result = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            for monomial, integer in multiply_monomials(
                left_monomial,
                right_monomial,
                num_modes=num_modes,
                anticommuting_modes=anticommuting_modes,
            ).items():
                result[monomial] = (
                    result.get(monomial, 0)
                    + left_coefficient * right_coefficient * integer
                )
    return {
        monomial: coefficient for monomial, coefficient in result.items() if coefficient
    }


def test_mask_roundtrip() -> None:
    rng = random.Random(0)
    for num_modes in range(9):
        for _ in range(100):
            factors = [rng.randrange(4) for _ in range(num_modes)]
            creators = sum((factor == 2) << mode for mode, factor in enumerate(factors))
            numbers = sum((factor == 3) << mode for mode, factor in enumerate(factors))
            annihilators = sum(
                (factor == 1) << mode for mode, factor in enumerate(factors)
            )
            monomial = monomial_from_masks(
                creators,
                numbers,
                annihilators,
                num_modes=num_modes,
            )
            assert masks_from_monomial(monomial, num_modes=num_modes) == (
                creators,
                numbers,
                annihilators,
            )


def test_selected_anticommuting_modes() -> None:
    spin_0 = monomial_from_masks(annihilators=0b0001, num_modes=4)
    spin_1 = monomial_from_masks(annihilators=0b0010, num_modes=4)
    fermion_0 = monomial_from_masks(annihilators=0b0100, num_modes=4)
    fermion_1 = monomial_from_masks(annihilators=0b1000, num_modes=4)
    fermionic_modes = 0b1100

    def multiply(left: int, right: int) -> dict[int, int]:
        return multiply_monomials(
            left,
            right,
            num_modes=4,
            anticommuting_modes=fermionic_modes,
        )

    assert multiply(spin_0, spin_1) == multiply(spin_1, spin_0)
    assert multiply(spin_0, fermion_0) == multiply(fermion_0, spin_0)
    assert multiply(fermion_0, fermion_1) == {
        monomial: -coefficient
        for monomial, coefficient in multiply(fermion_1, fermion_0).items()
    }


def test_local_hard_core_relation() -> None:
    annihilator = monomial_from_masks(annihilators=1, num_modes=1)
    creator = monomial_from_masks(creators=1, num_modes=1)
    number = monomial_from_masks(numbers=1, num_modes=1)

    assert multiply_monomials(annihilator, creator, num_modes=1) == {
        0: 1,
        number: -1,
    }
    assert multiply_monomials(creator, annihilator, num_modes=1) == {number: 1}
    assert multiply_monomials(creator, creator, num_modes=1) == {}


@pytest.mark.parametrize("anticommuting_modes", [0, 0b0011, 0b1111])
def test_multiplication_is_associative(anticommuting_modes: int) -> None:
    rng = random.Random(anticommuting_modes)
    num_modes = 4
    for _ in range(500):
        operands = [{rng.randrange(1 << (2 * num_modes)): 1} for _ in range(3)]
        left_associated = _multiply_polynomials(
            _multiply_polynomials(
                operands[0],
                operands[1],
                num_modes=num_modes,
                anticommuting_modes=anticommuting_modes,
            ),
            operands[2],
            num_modes=num_modes,
            anticommuting_modes=anticommuting_modes,
        )
        right_associated = _multiply_polynomials(
            operands[0],
            _multiply_polynomials(
                operands[1],
                operands[2],
                num_modes=num_modes,
                anticommuting_modes=anticommuting_modes,
            ),
            num_modes=num_modes,
            anticommuting_modes=anticommuting_modes,
        )
        assert left_associated == right_associated


def test_validation() -> None:
    with pytest.raises(ValueError, match="pairwise disjoint"):
        monomial_from_masks(creators=1, numbers=1)
    with pytest.raises(ValueError, match="too small"):
        masks_from_monomial(0b100, num_modes=1)
    with pytest.raises(ValueError, match="within the binary basis"):
        multiply_monomials(0, 0, num_modes=2, anticommuting_modes=0b100)
