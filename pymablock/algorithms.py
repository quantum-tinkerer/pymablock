# type: ignore
"""Algorithms definitions."""
# ruff: noqa: F821, D103, F841, E999, F811, F632, F401

# The functions in this module are not usable directly. Their contents is used to
# specify the algorithms, as parsed by `pymablock.algorithm_parsing.parse_algorithm`.
# Check the documentation of that module for the DSL specification.


def main():
    # We use a notion of diagonal/offdiagonal that may not fully coincide with
    # the block structure. For that if the user provides a function for
    # `offdiag` in the scope, it may take a diagonal block of a series and
    # return its offdiagonal part. Similarly, the function `diag` (always
    # defined) takes a diagonal block and returns its diagonal part.
    #
    # To understand the algorithm faster by skipping boilerplate, look only at
    # the "if offdiag is None" branches.
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
            -solve_sylvester("Yadj".adj - "V @ H'_diag" - "V @ H'_diag".adj)

    with "W":
        start = 0
        hermitian
        if diagonal:
            "U'† @ U'" / -2
        if offdiagonal:
            zero if two_block_optimized else "U'† @ U'" / -2

    # We can choose to query either lower or upper block of Y since it is
    # Hermitian. The choice for lower block is optimal for querying H_AA and
    # upper for H_BB. We choose the lower because we follow the convention that
    # H_AA is more important. We enforce this by using the implementation
    # detail that hermitian matrices only compute their own upper blocks and
    # use conjugate to get the lower.
    with "Yadj":
        start = 0
        hermitian
        if offdiagonal:
            "X".adj if two_block_optimized else ("X".adj + "X") / 2

    with "U'":
        start = 0
        "W" + "V"

    with "U":
        start = 1
        "U'"

    with "U'†":
        "W" - "V"

    with "U†":
        start = 1
        "U'†"

    with "X":
        start = 0
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

    with "V @ H'_diag":
        pass

    return "H_tilde", "U", "U†"
