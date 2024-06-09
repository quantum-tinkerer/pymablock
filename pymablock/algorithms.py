# type: ignore
"""Algorithms definitions."""
# ruff: noqa: F821, D103, F841, E999, F811, F632, F401

from pymablock.algorithm_parsing import algorithm


@algorithm
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
            -solve_sylvester("X" + "H'_diag @ U'" + "H'_diag @ U'".adj)

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
        hermitian
        "B" + "H'_offdiag"

    with "C":
        # The offdiagonal blocks of C are hermitian, but we only query (1, 0).
        "U'† @ (H'_offdiag U' - B)" + "H'_offdiag @ U'"

    with "B":  # X - H'_offdiag
        start = 0
        hermitian
        if diagonal:
            ("C" - "C".adj) / 2
        if offdiagonal:
            # We can choose to query either "C" or "C".adj since C_offdiag is Hermitian.
            # The choice for "C".adj is optimal for querying H_AA and "C" for H_BB.
            # We choose "C".adj because we follow the convention that H_AA is more important.
            "C".adj

    with "H_tilde":
        start = "H_0"
        if diagonal:
            ("H'_diag" + ("C" + "C".adj) / 2)

    # We omit @ from the name to treat it as a series.
    with "(H'_offdiag U' - B)":
        "H'_offdiag @ U'" - "B"

    with "U'† @ (H'_offdiag U' - B)":
        pass

    with "U'† @ U'":
        hermitian

    with "H'_diag @ U'":
        pass

    with "H'_offdiag @ U'":
        pass

    with "U'† @ B":
        pass

    return "H_tilde", "U", "U†"
