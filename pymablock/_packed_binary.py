"""Integer arithmetic for packed fermion and spin monomials.

A monomial is a product with at most one factor per mode, ordered by increasing
mode index. Each factor is a, a†, or N = a†a, and three bit masks record which
modes carry which factor. ``fermion_mask`` marks the fermion modes: unpaired
operators (a or a†) on distinct fermion modes anticommute, while number factors
and spin operators commute with everything on other modes.

Every fermionic sign below is the parity of a permutation of the unpaired
fermion operators. For a product it is the parity of the stable sort by mode of
the left operators followed by the right ones: same-mode pairs become adjacent
and contribute no sign. For an adjoint it is the parity of reversing the
unpaired operators.
"""

from __future__ import annotations

from typing import TypeAlias

# Three disjoint masks, in creation, number, annihilation order.
BinaryMonomial: TypeAlias = tuple[int, int, int]

IDENTITY: BinaryMonomial = (0, 0, 0)
PackedPolynomial: TypeAlias = dict[BinaryMonomial, int]


def _prefix_parity(mask: int, num_bits: int) -> int:
    """Set bit i to the parity of input bits 0 through i, for i < num_bits."""
    shift = 1
    while shift < num_bits:
        mask ^= mask << shift
        shift <<= 1
    return mask


def _crossing_sign(left: int, right: int) -> int:
    """Return the sign of sorting the fermion operators in left, then right, by mode.

    Each pair with a left mode above a right mode is one exchange.
    """
    if not left or not right:
        return 1
    num_bits = max(left.bit_length(), right.bit_length())
    prefix = _prefix_parity(left, num_bits)
    # All pairs minus the pairs with left mode <= right mode, modulo two.
    parity = (left.bit_count() & right.bit_count() & 1) ^ (
        (prefix & right).bit_count() & 1
    )
    return -1 if parity else 1


def adjoint_monomial(
    monomial: BinaryMonomial, fermion_mask: int
) -> tuple[BinaryMonomial, int]:
    """Return the adjoint monomial and its fermionic sign."""
    creation, number, annihilation = monomial
    # Reversing k unpaired fermion operators makes k(k-1)/2 exchanges.
    k = ((creation | annihilation) & fermion_mask).bit_count()
    sign = -1 if (k * (k - 1) // 2) & 1 else 1
    return (annihilation, number, creation), sign


def multiply_monomials(
    left: BinaryMonomial, right: BinaryMonomial, fermion_mask: int
) -> PackedPolynomial:
    """Multiply two monomials, returning ``{monomial: ±1}``.

    On each mode the product of the left and right factors follows the table
    below, using ``a a† = 1 - N`` and ``a† a = N``. Every ``a a†`` pair
    doubles the number of terms.
    """
    left_creation, left_number, left_annihilation = left
    right_creation, right_number, right_annihilation = right
    # Exactly one of identity, a, a†, N holds on every mode.
    left_identity = ~(left_creation | left_number | left_annihilation)
    right_identity = ~(right_creation | right_number | right_annihilation)
    # Zero products:  a a,  a† a†,  a† N,  N a
    if (
        (left_annihilation & right_annihilation)
        | (left_creation & right_creation)
        | (left_creation & right_number)
        | (left_number & right_annihilation)
    ):
        return {}
    # a  =  1 a,  a 1,  a N
    annihilation = (
        (left_identity & right_annihilation)
        | (left_annihilation & right_identity)
        | (left_annihilation & right_number)
    )
    # a† =  1 a†,  a† 1,  N a†
    creation = (
        (left_identity & right_creation)
        | (left_creation & right_identity)
        | (left_number & right_creation)
    )
    # N  =  1 N,  N 1,  N N,  a† a
    number = (
        (left_identity & right_number)
        | (left_number & right_identity)
        | (left_number & right_number)
        | (left_creation & right_annihilation)
    )
    # a a† = 1 - N: each pair either disappears or becomes -N.
    pairs = left_annihilation & right_creation

    sign = _crossing_sign(
        (left_creation | left_annihilation) & fermion_mask,
        (right_creation | right_annihilation) & fermion_mask,
    )
    result = {}
    subset = pairs
    while True:
        result[creation, number | subset, annihilation] = (
            -sign if subset.bit_count() & 1 else sign
        )
        if not subset:
            return result
        subset = (subset - 1) & pairs


def remap_monomial(monomial: BinaryMonomial, old_to_new: list[int]) -> BinaryMonomial:
    """Move each mode bit to its new index, preserving relative mode order."""

    def remap(mask: int) -> int:
        return sum(
            1 << new_index
            for old_index, new_index in enumerate(old_to_new)
            if mask & (1 << old_index)
        )

    creation, number, annihilation = monomial
    return remap(creation), remap(number), remap(annihilation)
