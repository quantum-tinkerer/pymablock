"""Checks for candidate mixed-binary packed layouts."""

# ruff: noqa: D103

from __future__ import annotations

import random

from pymablock._packed_binary import monomial_from_masks

from .binary_storage import (
    JointBinaryOperator,
    SeparateBinaryOperator,
    _combine_monomials,
)


def _joint_factor(
    *,
    num_spins: int,
    num_fermions: int,
    spin_creators: int = 0,
    spin_numbers: int = 0,
    spin_annihilators: int = 0,
    fermion_creators: int = 0,
    fermion_numbers: int = 0,
    fermion_annihilators: int = 0,
) -> JointBinaryOperator:
    monomial = monomial_from_masks(
        creators=spin_creators | (fermion_creators << num_spins),
        numbers=spin_numbers | (fermion_numbers << num_spins),
        annihilators=spin_annihilators | (fermion_annihilators << num_spins),
        num_modes=num_spins + num_fermions,
    )
    return JointBinaryOperator.build(num_spins, num_fermions, ((monomial, 1),))


def test_joint_storage_implements_the_intended_statistics() -> None:
    kwargs = {"num_spins": 2, "num_fermions": 2}
    spin_0 = _joint_factor(**kwargs, spin_annihilators=0b01)
    spin_1 = _joint_factor(**kwargs, spin_annihilators=0b10)
    fermion_0 = _joint_factor(**kwargs, fermion_annihilators=0b01)
    fermion_1 = _joint_factor(**kwargs, fermion_annihilators=0b10)

    assert spin_0 * spin_1 == spin_1 * spin_0
    assert spin_0 * fermion_0 == fermion_0 * spin_0
    assert fermion_0 * fermion_1 == -(fermion_1 * fermion_0)


def test_local_hole_projector_is_statistics_independent() -> None:
    kwargs = {"num_spins": 1, "num_fermions": 1}
    spin_lowering = _joint_factor(**kwargs, spin_annihilators=1)
    spin_raising = _joint_factor(**kwargs, spin_creators=1)
    spin_number = _joint_factor(**kwargs, spin_numbers=1)
    identity = _joint_factor(**kwargs)

    assert spin_lowering * spin_raising == identity - spin_number


def test_joint_and_separate_storage_multiply_identically() -> None:
    rng = random.Random(0)
    num_spins = 4
    num_fermions = 4

    def random_separate() -> SeparateBinaryOperator:
        terms = []
        for _ in range(12):
            spin_creators = spin_numbers = spin_annihilators = 0
            fermion_creators = fermion_numbers = fermion_annihilators = 0
            for mode in range(num_spins):
                factor = rng.randrange(4)
                spin_annihilators |= (factor & 1) << mode
                spin_creators |= ((factor >> 1) & 1) << mode
            for mode in range(num_fermions):
                factor = rng.randrange(4)
                fermion_annihilators |= (factor & 1) << mode
                fermion_creators |= ((factor >> 1) & 1) << mode
            spin_numbers = spin_creators & spin_annihilators
            spin_creators &= ~spin_numbers
            spin_annihilators &= ~spin_numbers
            fermion_numbers = fermion_creators & fermion_annihilators
            fermion_creators &= ~fermion_numbers
            fermion_annihilators &= ~fermion_numbers
            terms.append(
                (
                    (
                        monomial_from_masks(
                            creators=spin_creators,
                            numbers=spin_numbers,
                            annihilators=spin_annihilators,
                            num_modes=num_spins,
                        ),
                        monomial_from_masks(
                            creators=fermion_creators,
                            numbers=fermion_numbers,
                            annihilators=fermion_annihilators,
                            num_modes=num_fermions,
                        ),
                    ),
                    rng.choice((-2, -1, 1, 2)),
                )
            )
        return SeparateBinaryOperator.build(num_spins, num_fermions, terms)

    for _ in range(20):
        left = random_separate()
        right = random_separate()
        assert (left * right).joint() == left.joint() * right.joint()


def test_separate_to_joint_key_is_lossless() -> None:
    spin = monomial_from_masks(creators=0b01, numbers=0b10, num_modes=2)
    fermion = monomial_from_masks(numbers=0b01, annihilators=0b10, num_modes=2)
    separate = SeparateBinaryOperator.build(2, 2, (((spin, fermion), 3),))

    assert separate.joint() == JointBinaryOperator.build(
        2,
        2,
        (
            (
                _combine_monomials(
                    spin,
                    fermion,
                    num_spins=2,
                    num_fermions=2,
                ),
                3,
            ),
        ),
    )
