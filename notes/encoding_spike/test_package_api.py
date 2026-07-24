"""Physical acceptance tests through Pymablock's public embedding API."""

from __future__ import annotations

import sympy

from pymablock import block_diagonalize
from pymablock.second_quantization import levels, occupation_embedding

from .models import fermionic_ring_exchange, tunable_coupler


def test_package_api_coupler_reaches_fourth_order() -> None:
    """The public structured embedding handles the nonlinear boson model."""
    model = tunable_coupler()
    q1, q2 = levels("q1 q2", 2)
    a1, a2, coupler = model.encoding.operators
    embedding = occupation_embedding({a1: q1, a2: q2, coupler: 0})

    effective, *_ = block_diagonalize(
        [model.H0, model.V],
        subspace_eigenvectors=embedding,
    )

    assert len(effective[0, 0, 4].terms) == 5


def test_package_api_fermionic_ring_exchange() -> None:
    """The public structured embedding reproduces fourth-order ring exchange."""
    model = fermionic_ring_exchange()
    spins = levels("s0:4", 2)
    source_modes = {str(operator): operator for operator in model.encoding.operators}
    embedding = occupation_embedding(
        {source_modes[f"c{site}_up"]: spins[site] for site in range(4)}
        | {source_modes[f"c{site}_down"]: 1 - spins[site] for site in range(4)}
    )

    effective, *_ = block_diagonalize(
        [model.H0, model.V],
        subspace_eigenvectors=embedding,
    )

    ring = effective[0, 0, 4].terms[(1, -1, 1, -1)]
    assert sympy.factor(ring - 40 * model.t**4 / model.U**3) == 0
