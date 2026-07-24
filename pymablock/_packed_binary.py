"""Pure-Python packed monomials for fermions and hard-core modes.

Each mode occupies two bit planes: annihilation and creation presence.  Their
four combinations encode identity, annihilation, creation, and number.  The
``anticommuting_modes`` mask selects the modes with fermionic cross-mode
statistics; all remaining modes commute across modes while retaining the same
local hard-core algebra.
"""

from __future__ import annotations

from typing import TypeAlias

PackedMonomial: TypeAlias = int
PackedPolynomial: TypeAlias = dict[PackedMonomial, int]

_LOCAL_BITS = 2


def _mode_mask(num_modes: int) -> int:
    if num_modes < 0:
        raise ValueError(f"num_modes must be non-negative, got {num_modes}")
    return (1 << num_modes) - 1


def _prefix_parity(mask: int, num_modes: int) -> int:
    active_modes = _mode_mask(num_modes)
    parity = mask & active_modes
    shift = 1
    while shift < num_modes:
        parity ^= (parity << shift) & active_modes
        shift <<= 1
    return parity


def _crossing_sign(left_odd: int, right_odd: int, num_modes: int) -> int:
    """Return the sign from ordering odd factors by increasing mode."""
    if not left_odd or not right_odd:
        return 1
    prefix = _prefix_parity(left_odd, num_modes)
    parity = ((left_odd.bit_count() & 1) & (right_odd.bit_count() & 1)) ^ (
        (prefix & right_odd).bit_count() & 1
    )
    return -1 if parity else 1


def _pack_masks(
    annihilation_presence: int,
    creation_presence: int,
    num_modes: int,
) -> PackedMonomial:
    return annihilation_presence | (creation_presence << num_modes)


def _unpack_masks(code: PackedMonomial, num_modes: int) -> tuple[int, int]:
    active_modes = _mode_mask(num_modes)
    return code & active_modes, (code >> num_modes) & active_modes


def monomial_from_masks(
    creators: int = 0,
    numbers: int = 0,
    annihilators: int = 0,
    *,
    num_modes: int | None = None,
) -> PackedMonomial:
    """Build a packed monomial from disjoint local-factor masks."""
    if creators < 0 or numbers < 0 or annihilators < 0:
        raise ValueError("creator, number, and annihilator masks must be non-negative")
    if creators & numbers or creators & annihilators or numbers & annihilators:
        raise ValueError(
            "creator, number, and annihilator masks must be pairwise disjoint"
        )
    inferred_modes = max(
        creators.bit_length(),
        numbers.bit_length(),
        annihilators.bit_length(),
    )
    if num_modes is None:
        num_modes = inferred_modes
    elif inferred_modes > num_modes:
        raise ValueError(f"num_modes={num_modes} is too small for the supplied masks")
    return _pack_masks(annihilators | numbers, creators | numbers, num_modes)


def masks_from_monomial(
    code: PackedMonomial,
    *,
    num_modes: int,
) -> tuple[int, int, int]:
    """Return creator, number, and annihilator masks from a packed monomial."""
    if code < 0:
        raise ValueError(f"packed monomial must be non-negative, got {code}")
    _mode_mask(num_modes)
    if code.bit_length() > _LOCAL_BITS * num_modes:
        raise ValueError(f"num_modes={num_modes} is too small for packed monomial {code}")

    annihilation_presence, creation_presence = _unpack_masks(code, num_modes)
    numbers = annihilation_presence & creation_presence
    creators = creation_presence & ~annihilation_presence
    annihilators = annihilation_presence & ~creation_presence
    return creators, numbers, annihilators


def multiply_monomials(
    left: PackedMonomial,
    right: PackedMonomial,
    *,
    num_modes: int,
    anticommuting_modes: int | None = None,
) -> PackedPolynomial:
    """Multiply packed hard-core monomials in their canonical mode order.

    The local relation is ``a a† = 1 - a† a`` for every mode.  Odd factors on
    modes selected by ``anticommuting_modes`` mutually anticommute.  Other
    cross-mode pairs commute.
    """
    if left < 0 or right < 0:
        raise ValueError("packed monomials must be non-negative")
    active_modes = _mode_mask(num_modes)
    if max(left.bit_length(), right.bit_length()) > _LOCAL_BITS * num_modes:
        raise ValueError(f"num_modes={num_modes} is too small for the input monomials")
    if anticommuting_modes is None:
        anticommuting_modes = active_modes
    if anticommuting_modes < 0 or anticommuting_modes & ~active_modes:
        raise ValueError("anticommuting_modes must be a mask within the binary basis")

    left_annihilation, left_creation = _unpack_masks(left, num_modes)
    right_annihilation, right_creation = _unpack_masks(right, num_modes)
    left_annihilation_only = left_annihilation & ~left_creation
    left_creation_only = left_creation & ~left_annihilation
    right_annihilation_only = right_annihilation & ~right_creation
    right_creation_only = right_creation & ~right_annihilation

    # Same-mode a†a† and aa factors annihilate the product.
    if (left_creation_only & right_creation) | (
        left_annihilation & right_annihilation_only
    ):
        return {}
    branch_mask = left_annihilation_only & right_creation_only

    left_identity = active_modes & ~(left_annihilation | left_creation)
    left_number = left_annihilation & left_creation
    right_identity = active_modes & ~(right_annihilation | right_creation)
    right_number = right_annihilation & right_creation

    base_annihilation = (
        (left_identity & right_annihilation)
        | (left_annihilation & (right_identity | right_number))
        | (left_creation_only & right_annihilation_only)
    )
    base_creation = (
        (left_identity & right_creation)
        | (left_creation_only & (right_identity | right_annihilation_only))
        | (left_number & (active_modes ^ right_annihilation_only))
    )
    sign = _crossing_sign(
        (left_annihilation_only | left_creation_only) & anticommuting_modes,
        (right_annihilation_only | right_creation_only) & anticommuting_modes,
        num_modes,
    )
    if not branch_mask:
        return {_pack_masks(base_annihilation, base_creation, num_modes): sign}

    result = {}
    subset = branch_mask
    while True:
        result[
            _pack_masks(
                base_annihilation | subset,
                base_creation | subset,
                num_modes,
            )
        ] = -sign if subset.bit_count() & 1 else sign
        if not subset:
            return result
        subset = (subset - 1) & branch_mask
