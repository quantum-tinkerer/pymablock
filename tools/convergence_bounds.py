"""Compare rigorous recurrence certificates with an exact two-level radius.

The grid search uses rational arithmetic. Every candidate is an admissible
majorant certificate; the search does not claim to find the global optimum.
Run with: python tools/convergence_bounds.py
"""

from fractions import Fraction as Q
from math import hypot, sqrt


def certificate(t: Q, d: Q, o: Q) -> Q:
    """Radius for unit gap and retained/eliminated norm slopes d and o."""
    assert 0 < t < Q(1, 2) and d >= 0 and o > 0
    return t * (1 - t / 2) * (1 - 2 * t) / (o * (1 - 2 * t * t) + 4 * d * t * (1 - t))


def verify(t: Q, d: Q, o: Q, radius: Q) -> None:
    """Independently check the original two coupled invariant inequalities."""
    d, o = radius * d, radius * o
    b = t * (o + 2 * d) / (1 - 2 * t)
    assert 2 * t * b + (o + 2 * d) * t <= b
    assert t * t / 2 + 2 * t * b + o + (2 * o + 4 * d) * t <= t


if __name__ == "__main__":
    assert certificate(Q(1, 5), Q(1), Q(1)) == Q(9, 130)
    assert certificate(Q(1, 4), Q(0), Q(1)) == Q(1, 8)
    print(
        "| a | g | Total-norm bound | Split bound | Best grid certificate | Exact radius |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for a, g in [(0, 1), (1, 1), (4, 1)]:
        candidates = [
            (certificate(Q(k, 1000), Q(a), Q(g)), Q(k, 1000)) for k in range(1, 500)
        ]
        for radius, t in candidates:
            verify(t, Q(a), Q(g), radius)
        best, _ = max(candidates)
        total = 9 / (130 * sqrt(2) * hypot(a, g))
        split = 7 / (8 * sqrt(2) * (7 * g + 6 * a))
        exact = 1 / (2 * hypot(a, g))
        assert (
            0 < total <= exact and 0 < split <= exact and float(best) / sqrt(2) <= exact
        )
        print(
            f"| {a} | {g} | {total:.6f} | {split:.6f} | {float(best) / sqrt(2):.6f} | {exact:.6f} |"
        )
