"""Cold-process comparison of finite, packed-algebraic, and map execution."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from time import perf_counter

from .algebraic import (
    block_diagonalize_algebraic,
    block_diagonalize_band_maps,
    block_diagonalize_factored_maps,
    block_diagonalize_maps,
)
from .encoding import block_diagonalize
from .models import (
    crepel_fu_two_star,
    fermionic_ring_exchange,
    synthetic_spin_floquet,
    tunable_coupler,
)

MODELS = {
    "coupler": (tunable_coupler, 4),
    "ring": (fermionic_ring_exchange, 4),
    "floquet": (synthetic_spin_floquet, 2),
    "crepel-fu": (crepel_fu_two_star, 4),
}
BACKENDS = {
    "finite": block_diagonalize,
    "algebraic": block_diagonalize_algebraic,
    "bands": block_diagonalize_band_maps,
    "factored": block_diagonalize_factored_maps,
    "maps": block_diagonalize_maps,
}


def run_one(model_name: str, backend_name: str) -> float:
    """Run and fully materialize one benchmark coefficient."""
    make_model, order = MODELS[model_name]
    model = make_model()
    start = perf_counter()
    effective, *_ = BACKENDS[backend_name](
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=order,
    )
    operator = effective[order]
    matrix = operator.matrix() if callable(operator.matrix) else operator.matrix
    if matrix.shape != (
        model.encoding.target.dimension,
        model.encoding.target.dimension,
    ):
        raise AssertionError("Effective operator has the wrong target dimension")
    return perf_counter() - start


def cold_sample(model_name: str, backend_name: str) -> float:
    """Measure one algorithm run in a fresh Python interpreter."""
    command = (
        sys.executable,
        "-m",
        "notes.encoding_spike.benchmark_maps",
        "--one",
        model_name,
        backend_name,
    )
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["seconds"])


def main() -> None:
    """Run one machine-readable sample or print the benchmark table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", nargs=2, metavar=("MODEL", "BACKEND"))
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()

    if arguments.one:
        model_name, backend_name = arguments.one
        print(json.dumps({"seconds": run_one(model_name, backend_name)}))
        return

    for model_name in MODELS:
        print(f"{model_name}:")
        for backend_name in BACKENDS:
            if model_name == "crepel-fu" and backend_name == "algebraic":
                continue
            samples = [
                cold_sample(model_name, backend_name) for _ in range(arguments.repeats)
            ]
            print(
                f"  {backend_name:9s} "
                f"{statistics.median(samples):.3f} s "
                f"({', '.join(f'{sample:.3f}' for sample in samples)})"
            )


if __name__ == "__main__":
    main()
