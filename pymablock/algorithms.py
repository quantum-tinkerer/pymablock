# type: ignore
"""Algorithms definitions."""
# ruff: noqa: F821, D103, F841, E999, F811, F632, F401

# The functions in this module are not usable directly. Their contents is used to
# specify the algorithms, as parsed by `pymablock.algorithm_parsing.parse_algorithm`.
# Check the documentation of that module for the DSL specification.


def main():
    # The algorithm implements two optional optimizations:
    # - two_block_optimized: if True, the algorithm assumes that the Hamiltonian
    #   has a 2x2 block structure and offdiagonal blocks are eliminated. This
    #   corresponds to the Schrieffer-Wolff transformation.
    # - hermitian_problem: if True, the algorithm uses the Hermitian/unitary
    #   simplifications. Otherwise it uses the inverse-series recurrence.
    # - commuting_blocks: a list of booleans indicating whether multiplying a
    #   diagonal matrix by an offdiagonal within that block gives an
    #   offdiagonal matrix. This property is broken in the context of selective
    #   diagonalization.
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
        antihermitian = hermitian_problem
        if offdiagonal:
            zero if not hermitian_problem else -solve_sylvester(
                "Yadj".adj - "V @ H'_diag" - "V @ H'_diag".adj
            )

    with "W":
        start = 0
        hermitian = hermitian_problem
        if diagonal:
            "U_inv' @ U'" / -2
        if offdiagonal:
            zero if (two_block_optimized or not hermitian_problem) else "U_inv' @ U'" / -2

    # We can choose to query either lower or upper block of Y since it is
    # Hermitian. The choice for lower block is optimal for querying H_AA and
    # upper for H_BB. We choose the lower because we follow the convention that
    # H_AA is more important. We enforce this by using the implementation
    # detail that hermitian matrices only compute their own upper blocks and
    # use conjugate to get the lower.
    with "Yadj":
        start = 0
        hermitian = hermitian_problem
        if offdiagonal:
            zero if not hermitian_problem else (
                "X".adj if two_block_optimized else ("X".adj + "X") / 2
            )
        if diagonal:
            zero if (commuting_blocks[index[0]] or not hermitian_problem) else (
                ("X".adj + "X") / 2
            )

    with "U'":
        start = 0
        ("W" + "V") if hermitian_problem else zero
        if diagonal:
            zero if hermitian_problem else "W"
        if offdiagonal:
            zero if hermitian_problem else solve_sylvester(
                "X" - "H'_diag @ U'" + "U' @ H'_diag"
            )

    with "U_inv'":
        start = 0
        if hermitian_problem:
            "W" - "V"
        else:
            -"U'" - "U_inv' @ U'"

    with "U":
        start = 1
        "U'"

    with "U†":
        start = 1
        "U_inv'"

    with "X":
        start = 0
        ("B" + "H'_offdiag" + "H'_offdiag @ U'") if hermitian_problem else zero
        if offdiagonal:
            zero if hermitian_problem else -(
                "H'_offdiag" + "H'_offdiag @ U'" + "U_inv' @ B"
            )
        if diagonal:
            zero if hermitian_problem else ("H'_diag @ U'" - "U' @ H'_diag")

    with "B":
        start = 0
        if diagonal:
            (
                (
                    "U_inv' @ B"
                    - "U_inv' @ B".adj
                    + "H'_offdiag @ U'"
                    + "H'_offdiag @ U'".adj
                )
                / -2
            ) if hermitian_problem else zero
        if diagonal:
            (
                zero if commuting_blocks[index[0]] else "V @ H'_diag" + "V @ H'_diag".adj
            ) if hermitian_problem else zero
        if offdiagonal:
            -"U_inv' @ B" if hermitian_problem else zero
        zero if hermitian_problem else "X" + "H'_offdiag" + "H'_offdiag @ U'"

    with "H_tilde":
        start = "H_0"
        if diagonal:
            (
                "H'_diag"
                + ("H'_offdiag @ U'" + "H'_offdiag @ U'".adj) / 2
                + ("U_inv' @ B" + "U_inv' @ B".adj) / -2
                - "Yadj"
            ) if hermitian_problem else ("H'_diag" + "B" + "U_inv' @ B")

    with "U_inv' @ U'":
        hermitian = hermitian_problem

    with "H'_diag @ U'":
        pass

    with "U' @ H'_diag":
        pass

    with "H'_offdiag @ U'":
        pass

    with "U_inv' @ B":
        pass

    with "V @ H'_diag":
        pass

    return "H_tilde", "U", "U†"
