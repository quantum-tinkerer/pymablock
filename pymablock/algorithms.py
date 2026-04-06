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
        if diagonal:
            zero if commuting_blocks[index[0]] else ("X".adj + "X") / 2

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
        if diagonal:
            zero if commuting_blocks[index[0]] else "V @ H'_diag" + "V @ H'_diag".adj

        if offdiagonal:
            -"U'† @ B"

    with "H_tilde":
        start = "H_0"
        if diagonal:
            (
                "H'_diag"
                + ("H'_offdiag @ U'" + "H'_offdiag @ U'".adj) / 2
                + ("U'† @ B" + "U'† @ B".adj) / -2
                - "Yadj"
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


def nonhermitian():
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
        if offdiagonal:
            -solve_sylvester(("Y" - "V @ H'_diag" + "H'_diag @ V"))

    with "W":
        start = 0
        "U_inv' @ U'" / -2

    with "U'":
        start = 0
        "W" + "V"

    with "U_inv'":
        start = 0
        "W" - "V"

    with "U":
        start = 1
        "U'"

    with "U†":
        start = 1
        "U_inv'"

    with "A":
        start = 0
        "H'_offdiag @ U'"

    with "Z":
        start = 0
        ("A" - "U_inv' @ H'_offdiag" - "U_inv' @ B" - "B_plus @ U_inv'") / 2

    with "B":
        start = 0
        if offdiagonal:
            -"U_inv' @ B"
        if diagonal:
            "V @ H'_diag" - "H'_diag @ V" + "Z" - "A"

    with "X":
        start = 0
        "B" + "H'_offdiag" + "A"

    with "Y":
        start = 0
        "X" - "Z"

    with "B_plus":
        start = 0
        "B" + "U_inv' @ B"

    with "H_tilde":
        start = "H_0"
        if diagonal:
            "H'_diag" - "B_plus"

    with "U_inv' @ U'":
        pass

    with "H'_diag @ V":
        pass

    with "V @ H'_diag":
        pass

    with "H'_offdiag @ U'":
        pass

    with "U_inv' @ B":
        pass

    with "B_plus @ U_inv'":
        pass

    with "U_inv' @ H'_offdiag":
        pass

    return "H_tilde", "U", "U†"
