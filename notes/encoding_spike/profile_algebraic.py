"""Sampling-profiler entry point for the algebraic embedding benchmarks."""

from __future__ import annotations

import os

from .algebraic import block_diagonalize_algebraic
from .models import fermionic_ring_exchange, tunable_coupler


def main() -> None:
    """Evaluate one fourth-order benchmark under an external profiler."""
    benchmark = os.environ.get("PYMABLOCK_EMBEDDING_BENCHMARK", "ring")
    if benchmark == "ring":
        model = fermionic_ring_exchange()
    elif benchmark == "coupler":
        model = tunable_coupler()
    else:
        raise ValueError(f"Unknown embedding benchmark: {benchmark}")
    effective, *_ = block_diagonalize_algebraic(
        [model.H0, model.V], encoding=model.encoding, to_order=4
    )
    effective[4]


if __name__ == "__main__":
    main()
