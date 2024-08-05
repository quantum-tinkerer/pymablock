# type: ignore
"""Algorithms definitions."""
# ruff: noqa: F821, D103, F841, E999, F811, F632, F401

# The functions in this module are not usable directly. Their contents is used to
# specify the algorithms, as parsed by `pymablock.algorithm_parsing.parse_algorithm`.
# Check the documentation of that module for the DSL specification.


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

    return "H_tilde", "U", "U†"


def hermitian_antihermitian():
    with "H'_diag":
        start = 0
        if diagonal:
            "H"

    with "H'_offdiag":
        start = 0
        if offdiagonal:
            "H"

    with "V":
        start = 0
        antihermitian
        if offdiagonal:
            -solve_sylvester("Y" - "V @ H'_diag" - "V @ H'_diag".adj)

    with "W":
        start = 0
        hermitian
        "U'† @ U'" / -2

    with "U'":
        "W" + "V"

    with "U":
        start = 1
        "U'"

    with "U'†":
        "W" - "V"

    with "U†":
        start = 1
        "U'†"

    with "Z":
        start = 0
        antihermitian
        (-"U'† @ X" + "U'† @ X".adj) / 2

    with "Y":
        start = 0
        hermitian
        if offdiagonal:
            ("U† @ H'_offdiag @ U" - "U'† @ X" - "Z")

    with "X":
        start = 0
        "Y" + "Z"

    with "H_tilde":
        start = "H_0"
        if diagonal:
            "U† @ H @ U"

    with "V @ H'_diag":
        pass

    with "U'† @ U'":
        hermitian

    with "U'† @ X":
        pass

    with "U† @ H'_offdiag @ U":
        hermitian

    with "U† @ H @ U":
        hermitian

    return "H_tilde", "U", "U†"
