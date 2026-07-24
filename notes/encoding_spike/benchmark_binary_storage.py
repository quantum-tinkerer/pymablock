"""Compare joint and factorized packed storage for spins and fermions."""

from __future__ import annotations

import random
import statistics
from timeit import repeat

from pymablock._packed_binary import monomial_from_masks

from .binary_storage import SeparateBinaryOperator


def _random_factor(
    rng: random.Random,
    num_modes: int,
    active_modes: int | None,
) -> int:
    creators = numbers = annihilators = 0
    modes = (
        range(num_modes)
        if active_modes is None
        else rng.sample(range(num_modes), min(active_modes, num_modes))
    )
    for mode in modes:
        factor = rng.randrange(1, 4) if active_modes is not None else rng.randrange(4)
        creators |= (factor == 1) << mode
        numbers |= (factor == 2) << mode
        annihilators |= (factor == 3) << mode
    return monomial_from_masks(
        creators=creators,
        numbers=numbers,
        annihilators=annihilators,
        num_modes=num_modes,
    )


def _random_operator(
    rng: random.Random,
    *,
    num_spins: int,
    num_fermions: int,
    num_terms: int,
    active_modes: int | None,
) -> SeparateBinaryOperator:
    return SeparateBinaryOperator.build(
        num_spins,
        num_fermions,
        (
            (
                (
                    _random_factor(rng, num_spins, active_modes),
                    _random_factor(rng, num_fermions, active_modes),
                ),
                rng.choice((-2, -1, 1, 2)),
            )
            for _ in range(num_terms)
        ),
    )


def _measure(
    *,
    num_spins: int,
    num_fermions: int,
    num_terms: int,
    active_modes: int | None,
) -> tuple[float, float]:
    rng = random.Random(1)
    left = _random_operator(
        rng,
        num_spins=num_spins,
        num_fermions=num_fermions,
        num_terms=num_terms,
        active_modes=active_modes,
    )
    right = _random_operator(
        rng,
        num_spins=num_spins,
        num_fermions=num_fermions,
        num_terms=num_terms,
        active_modes=active_modes,
    )
    joint_left, joint_right = left.joint(), right.joint()
    number = max(1, 2000 // num_terms**2)
    separate = (
        statistics.median(repeat(lambda: left * right, repeat=5, number=number)) / number
    )
    joint = (
        statistics.median(
            repeat(lambda: joint_left * joint_right, repeat=5, number=number)
        )
        / number
    )
    return separate, joint


def main() -> None:
    """Print representative PT-sparse and dense-key comparisons."""
    workloads = (
        ("sparse", 8, 8, 16, 2),
        ("sparse", 32, 32, 32, 3),
        ("dense", 8, 8, 16, None),
        ("dense", 32, 32, 32, None),
    )
    print("kind   modes  terms  separate    joint       speedup")
    for kind, num_spins, num_fermions, num_terms, active_modes in workloads:
        separate, joint = _measure(
            num_spins=num_spins,
            num_fermions=num_fermions,
            num_terms=num_terms,
            active_modes=active_modes,
        )
        print(
            f"{kind:6s} {num_spins + num_fermions:5d} {num_terms:6d} "
            f"{separate:10.6f} {joint:10.6f} {separate / joint:8.2f}x"
        )


if __name__ == "__main__":
    main()
