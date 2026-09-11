"""Tests for packed fermion and spin monomial arithmetic."""

from __future__ import annotations

import random

import numpy as np
import pytest
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock._packed_binary import (
    IDENTITY,
    BinaryMonomial,
    _crossing_sign,
    adjoint_monomial,
    multiply_monomials,
    remap_monomial,
)
from pymablock._packed_terms import _number_order_sign, pack_ordered_factors

from .second_quantization_helpers import apply_local_factors, occupation_matrices

# Local factor codes used by apply_local_factors: 1 = a, 2 = a†, 3 = N.
A, C, N = 1, 2, 3


def encode(factors: list[tuple[int, int]]) -> BinaryMonomial:
    return (
        sum(1 << mode for mode, factor in factors if factor == C),
        sum(1 << mode for mode, factor in factors if factor == N),
        sum(1 << mode for mode, factor in factors if factor == A),
    )


def decode(monomial: BinaryMonomial) -> list[tuple[int, int]]:
    """Return ``(mode, factor)`` pairs in increasing mode order."""
    factors = {}
    for code, mask in zip((C, N, A), monomial):
        factors.update(
            (mode, code) for mode in range(mask.bit_length()) if mask >> mode & 1
        )
    return sorted(factors.items())


def all_monomials(num_modes: int):
    for code in range(1 << (2 * num_modes)):
        # Two bits per mode: identity, a, a†, N.
        yield encode(
            [
                (mode, (code >> (2 * mode)) & 3)
                for mode in range(num_modes)
                if (code >> (2 * mode)) & 3
            ]
        )


def inversion_parity(sequence) -> int:
    """Sign of the stable sort of a sequence: -1 for an odd number of inversions."""
    inversions = sum(
        1
        for i, left in enumerate(sequence)
        for right in sequence[i + 1 :]
        if left > right
    )
    return -1 if inversions & 1 else 1


def test_crossing_sign_is_inversion_parity() -> None:
    """The bit-parallel sign matches counting inversions of the mode sequence."""
    for num_modes in range(6):
        for left in range(1 << num_modes):
            for right in range(1 << num_modes):
                modes = [i for i in range(num_modes) if left >> i & 1] + [
                    i for i in range(num_modes) if right >> i & 1
                ]
                assert _crossing_sign(left, right) == inversion_parity(modes)
    # Masks wider than a machine word.
    left, right = 1 << 100 | 1 << 3, 1 << 70 | 1 << 2
    assert _crossing_sign(left, right) == inversion_parity([3, 100, 2, 70])


@pytest.mark.parametrize("num_modes", range(4))
def test_product_matches_occupation_matrices(num_modes):
    """Check every pair of monomials and fermion/spin assignment through three modes."""
    identity = np.eye(1 << num_modes)
    for fermion_mask in range(1 << num_modes):
        operators = [
            (FermionOp if fermion_mask & (1 << i) else SigmaMinus)(f"a{i}")
            for i in range(num_modes)
        ]
        matrices = occupation_matrices(operators, [(0, 1)] * num_modes)
        local = {}
        for mode, operator in enumerate(operators):
            lowering = matrices[operator].toarray()
            local[mode] = {A: lowering, C: lowering.T, N: lowering.T @ lowering}

        def matrix(monomial):
            result = identity
            for mode, factor in decode(monomial):
                result = result @ local[mode][factor]
            return result

        monomials = {monomial: matrix(monomial) for monomial in all_monomials(num_modes)}
        for left, left_matrix in monomials.items():
            adjoint, sign = adjoint_monomial(left, fermion_mask)
            np.testing.assert_array_equal(sign * monomials[adjoint], left_matrix.conj().T)
            for right, right_matrix in monomials.items():
                product = multiply_monomials(left, right, fermion_mask)
                expected = left_matrix @ right_matrix
                actual = sum(
                    (sign * monomials[monomial] for monomial, sign in product.items()),
                    np.zeros_like(identity),
                )
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("num_modes", [3, 16, 64, 128])
def test_long_ordered_factors_match_occupation_actions(num_modes):
    """Check conversion signs across 64-bit boundaries and mixed statistics."""
    rng = random.Random(num_modes)
    for _ in range(32):
        factors = [rng.randrange(4) for _ in range(num_modes)]
        fermion_mask = rng.getrandbits(num_modes)
        masks = [sum(1 << i for i, f in enumerate(factors) if f == k) for k in (C, N, A)]
        monomial, sign = pack_ordered_factors(fermion_mask, *masks)
        number_ordered_factors = (
            [(i, f) for i, f in enumerate(factors) if f == C]
            + [(i, f) for i, f in enumerate(factors) if f == N]
            + [(i, f) for i, f in reversed(list(enumerate(factors))) if f == A]
        )
        state = rng.getrandbits(num_modes)
        for i, f in number_ordered_factors:
            if f in (A, N):
                state |= 1 << i
            else:
                state &= ~(1 << i)
        for initial in (state, rng.getrandbits(num_modes)):
            expected = apply_local_factors(number_ordered_factors, initial, fermion_mask)
            final, amplitude = apply_local_factors(
                decode(monomial), initial, fermion_mask
            )
            assert (final, sign * amplitude) == expected


def test_ordered_number_factors_with_unpaired_operators():
    """The conversion discards a† N and N a before packing disjoint factors."""
    assert pack_ordered_factors((1 << 64) - 1, 1 << 40, 1 << 40, 0) is None
    assert pack_ordered_factors((1 << 64) - 1, 0, 1 << 40, 1 << 40) is None


def test_number_order_sign_is_inversion_parity() -> None:
    """Public order is creation ascending, then annihilation descending."""
    for num_modes in range(5):
        for monomial in all_monomials(num_modes):
            unpaired = [
                (mode, factor) for mode, factor in decode(monomial) if factor != N
            ]
            public = sorted(mode for mode, factor in unpaired if factor == C) + sorted(
                (mode for mode, factor in unpaired if factor == A), reverse=True
            )
            # Position in the packed (mode-ascending) order of each public factor.
            packed_positions = [mode for mode, _ in unpaired]
            sequence = [packed_positions.index(mode) for mode in public]
            fermion_mask = (1 << num_modes) - 1
            assert _number_order_sign(monomial, fermion_mask) == inversion_parity(
                sequence
            )


@pytest.mark.parametrize("length", [8, 16, 32, 64])
def test_long_products_and_adjoints_match_occupation_actions(length):
    """Exercise nonzero long products and cancellations among up to 256 terms."""
    num_modes = 2 * length
    rng = random.Random(length)

    for num_aa_dagger_pairs in (0, 1, 4, 8):
        active = sorted(rng.sample(range(num_modes), length))
        pairs = [(A, C)] * num_aa_dagger_pairs + [
            rng.choice([(A, N), (N, C), (C, A), (N, N)])
            for _ in range(length - num_aa_dagger_pairs)
        ]
        left = [(mode, pair[0]) for mode, pair in zip(active, pairs)]
        right = [(mode, pair[1]) for mode, pair in zip(active, pairs)]
        # Use all fermion modes, then a mixture of fermion and spin modes.
        for fermion_mask in ((1 << num_modes) - 1, rng.getrandbits(num_modes)):
            product = multiply_monomials(encode(left), encode(right), fermion_mask)
            assert len(product) == 2**num_aa_dagger_pairs
            # Choose occupations on which the product is nonzero, then flip
            # individual occupations to check zero actions and cancellations.
            nonzero_state = sum(1 << mode for mode, factor in right if factor in (A, N))
            states = [nonzero_state, rng.getrandbits(num_modes)] + [
                nonzero_state ^ (1 << mode) for mode in active[:4]
            ]
            for state in states:
                expected_state, expected_sign = apply_local_factors(
                    left + right, state, fermion_mask
                )
                actual = {}
                for monomial, coefficient in product.items():
                    final, sign = apply_local_factors(
                        decode(monomial), state, fermion_mask
                    )
                    if sign:
                        actual[final] = actual.get(final, 0) + coefficient * sign
                actual = {key: value for key, value in actual.items() if value}
                assert actual == (
                    {expected_state: expected_sign} if expected_sign else {}
                )

            adjoint, sign = adjoint_monomial(encode(left), fermion_mask)
            adjoint_factors = [
                (mode, N if factor == N else A + C - factor)
                for mode, factor in reversed(left)
            ]
            nonzero_state = sum(
                1 << mode for mode, factor in adjoint_factors if factor in (A, N)
            )
            for state in (nonzero_state, rng.getrandbits(num_modes)):
                expected = apply_local_factors(adjoint_factors, state, fermion_mask)
                final, amplitude = apply_local_factors(
                    decode(adjoint), state, fermion_mask
                )
                assert (final, sign * amplitude) == expected


def _multiply_polynomials(left, right, fermion_mask):
    result = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            for monomial, product_sign in multiply_monomials(
                left_monomial, right_monomial, fermion_mask
            ).items():
                result[monomial] = (
                    result.get(monomial, 0)
                    + left_coefficient * right_coefficient * product_sign
                )
    return {
        monomial: coefficient for monomial, coefficient in result.items() if coefficient
    }


def test_selected_fermion_modes() -> None:
    spin_0 = (0, 0, 0b0001)
    spin_1 = (0, 0, 0b0010)
    fermion_0 = (0, 0, 0b0100)
    fermion_1 = (0, 0, 0b1000)
    fermion_mask = 0b1100

    def multiply(left, right):
        return multiply_monomials(left, right, fermion_mask)

    assert multiply(spin_0, spin_1) == multiply(spin_1, spin_0)
    assert multiply(spin_0, fermion_0) == multiply(fermion_0, spin_0)
    assert multiply(fermion_0, fermion_1) == {
        monomial: -coefficient
        for monomial, coefficient in multiply(fermion_1, fermion_0).items()
    }


def test_annihilation_creation_identity() -> None:
    annihilation = (0, 0, 1)
    creation = (1, 0, 0)
    number = (0, 1, 0)

    assert multiply_monomials(annihilation, creation, 1) == {IDENTITY: 1, number: -1}
    assert multiply_monomials(creation, annihilation, 1) == {number: 1}
    assert multiply_monomials(creation, creation, 1) == {}
    assert adjoint_monomial(IDENTITY, 0) == (IDENTITY, 1)


@pytest.mark.parametrize("fermion_mask", [0, 0b0011, 0b1111])
def test_multiplication_is_associative(fermion_mask: int) -> None:
    rng = random.Random(fermion_mask)
    monomials = list(all_monomials(4))
    for _ in range(500):
        x, y, z = ({rng.choice(monomials): 1} for _ in range(3))
        assert _multiply_polynomials(
            _multiply_polynomials(x, y, fermion_mask), z, fermion_mask
        ) == _multiply_polynomials(
            x, _multiply_polynomials(y, z, fermion_mask), fermion_mask
        )


def test_remap_monomial_preserves_relative_order() -> None:
    monomial = (0b001, 0b010, 0b100)
    assert remap_monomial(monomial, [1, 3, 4]) == (0b00010, 0b01000, 0b10000)
    assert remap_monomial(IDENTITY, []) == IDENTITY
