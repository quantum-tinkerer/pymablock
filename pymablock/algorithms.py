# type: ignore
"""Algorithms."""
# ruff: noqa: F821, D103, F841, E999, F811, F632


def main():
    with "H'_diag":
        start = 0
        if diagonal:
            "H"

    with "H'_offdiag":
        start = 0
        if offdiagonal:
            "H"

    with "U'":
        start = 0
        antihermitian

        if diagonal:
            "U'† @ U'" / -2
        if offdiagonal:
            # We can choose to query either "X" or "X".adj since X is Hermitian.
            # The choice for "X".adj is optimal for querying H_AA and "X" for H_BB.
            # We choose "X".adj because we follow the convention that H_AA is more important.
            -solve_sylvester("X".adj + "H'_diag @ U'" + "H'_diag @ U'".adj)

    with "U":
        start = 1
        "U'"

    with "U'†":
        if diagonal:
            "U'"
        if offdiagonal:
            -"U'"

    with "U†":
        start = 1
        "U'†"

    with "X":
        start = 0
        # X is hermitian, but (1, 0) is the only index we ever query.
        if upper:
            "B" + "H'_offdiag" + "H'_offdiag @ U'"

    with "B":
        start = 0
        if diagonal:
            ("U'† @ B" - "U'† @ B".adj + "H'_offdiag @ U'" + "H'_offdiag @ U'".adj) / -2

        if offdiagonal:
            -"U'† @ B"

    with "H_tilde":
        start = "H_0"
        if diagonal:
            (
                "H'_diag"
                + ("H'_offdiag @ U'" + "H'_offdiag @ U'".adj) / 2
                + ("U'† @ B" + "U'† @ B".adj) / -2
            )

    with "U'† @ U'":
        hermitian

    with "H'_diag @ U'":
        pass

    with "H'_offdiag @ U'":
        pass

    with "U'† @ B":
        pass

    return "H_tilde", "U", "U'"
