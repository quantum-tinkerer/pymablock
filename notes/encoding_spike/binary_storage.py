"""Candidate packed storage layouts for mixed spin and fermion operators."""

# ruff: noqa: D102, D105

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pymablock._packed_binary import (
    masks_from_monomial,
    monomial_from_masks,
    multiply_monomials,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def multiply_binary_monomials(
    left: int,
    right: int,
    *,
    num_modes: int,
    anticommuting_modes: int,
) -> dict[int, int]:
    """Multiply hard-core monomials with signs only on selected modes.

    The packed kernel implements the local four-dimensional algebra for every
    mode, with the anticommuting mask selecting fermionic cross-mode signs.
    """
    return multiply_monomials(
        left,
        right,
        num_modes=num_modes,
        anticommuting_modes=anticommuting_modes,
    )


def _combine_monomials(
    spin_monomial: int,
    fermion_monomial: int,
    *,
    num_spins: int,
    num_fermions: int,
) -> int:
    spin_creators, spin_numbers, spin_annihilators = masks_from_monomial(
        spin_monomial, num_modes=num_spins
    )
    fermion_creators, fermion_numbers, fermion_annihilators = masks_from_monomial(
        fermion_monomial, num_modes=num_fermions
    )
    return monomial_from_masks(
        creators=spin_creators | (fermion_creators << num_spins),
        numbers=spin_numbers | (fermion_numbers << num_spins),
        annihilators=spin_annihilators | (fermion_annihilators << num_spins),
        num_modes=num_spins + num_fermions,
    )


@dataclass(frozen=True, slots=True)
class JointBinaryOperator:
    """One packed key with a basis-level anticommuting-mode mask."""

    num_spins: int
    num_fermions: int
    items: tuple[tuple[int, object], ...]

    @classmethod
    def build(
        cls,
        num_spins: int,
        num_fermions: int,
        terms: Iterable[tuple[int, object]],
    ) -> JointBinaryOperator:
        combined = {}
        for monomial, coefficient in terms:
            if not coefficient:
                continue
            combined[monomial] = combined.get(monomial, 0) + coefficient
            if not combined[monomial]:
                del combined[monomial]
        return cls(num_spins, num_fermions, tuple(sorted(combined.items())))

    @property
    def num_modes(self) -> int:
        return self.num_spins + self.num_fermions

    @property
    def anticommuting_modes(self) -> int:
        return ((1 << self.num_fermions) - 1) << self.num_spins

    def __add__(self, other: JointBinaryOperator) -> JointBinaryOperator:
        if (self.num_spins, self.num_fermions) != (
            other.num_spins,
            other.num_fermions,
        ):
            raise ValueError("Binary operators act on different bases")
        return type(self).build(
            self.num_spins,
            self.num_fermions,
            (*self.items, *other.items),
        )

    def __neg__(self) -> JointBinaryOperator:
        return type(self).build(
            self.num_spins,
            self.num_fermions,
            ((monomial, -coefficient) for monomial, coefficient in self.items),
        )

    def __sub__(self, other: JointBinaryOperator) -> JointBinaryOperator:
        return self + (-other)

    def __mul__(self, other: JointBinaryOperator) -> JointBinaryOperator:
        if (self.num_spins, self.num_fermions) != (
            other.num_spins,
            other.num_fermions,
        ):
            raise ValueError("Binary operators act on different bases")
        return type(self).build(
            self.num_spins,
            self.num_fermions,
            (
                (product, left_coefficient * right_coefficient * integer)
                for left, left_coefficient in self.items
                for right, right_coefficient in other.items
                for product, integer in multiply_binary_monomials(
                    left,
                    right,
                    num_modes=self.num_modes,
                    anticommuting_modes=self.anticommuting_modes,
                ).items()
            ),
        )


@dataclass(frozen=True, slots=True)
class SeparateBinaryOperator:
    """Independent packed spin and fermion factors."""

    num_spins: int
    num_fermions: int
    items: tuple[tuple[tuple[int, int], object], ...]

    @classmethod
    def build(
        cls,
        num_spins: int,
        num_fermions: int,
        terms: Iterable[tuple[tuple[int, int], object]],
    ) -> SeparateBinaryOperator:
        combined = {}
        for monomials, coefficient in terms:
            if not coefficient:
                continue
            combined[monomials] = combined.get(monomials, 0) + coefficient
            if not combined[monomials]:
                del combined[monomials]
        return cls(num_spins, num_fermions, tuple(sorted(combined.items())))

    def __mul__(self, other: SeparateBinaryOperator) -> SeparateBinaryOperator:
        if (self.num_spins, self.num_fermions) != (
            other.num_spins,
            other.num_fermions,
        ):
            raise ValueError("Binary operators act on different bases")
        return type(self).build(
            self.num_spins,
            self.num_fermions,
            (
                (
                    (spin_product, fermion_product),
                    left_coefficient * right_coefficient * spin_integer * fermion_integer,
                )
                for (left_spin, left_fermion), left_coefficient in self.items
                for (right_spin, right_fermion), right_coefficient in other.items
                for spin_product, spin_integer in multiply_binary_monomials(
                    left_spin,
                    right_spin,
                    num_modes=self.num_spins,
                    anticommuting_modes=0,
                ).items()
                for fermion_product, fermion_integer in multiply_binary_monomials(
                    left_fermion,
                    right_fermion,
                    num_modes=self.num_fermions,
                    anticommuting_modes=(1 << self.num_fermions) - 1,
                ).items()
            ),
        )

    def joint(self) -> JointBinaryOperator:
        return JointBinaryOperator.build(
            self.num_spins,
            self.num_fermions,
            (
                (
                    _combine_monomials(
                        spin,
                        fermion,
                        num_spins=self.num_spins,
                        num_fermions=self.num_fermions,
                    ),
                    coefficient,
                )
                for (spin, fermion), coefficient in self.items
            ),
        )
