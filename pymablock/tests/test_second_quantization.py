import numpy as np
import pytest
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm
from pymablock.second_quantization import (
    apply_mask_to_operator,
    find_operators,
    group_ordered,
    simplify_number_expression,
    solve_sylvester_bosonic,
)


def test_solve_sylvester_bosonic_simple():
    """Confirm that the index handling is correct."""
    # Create symbolic matrices with no bosonic operators
    a, b, c, d = sympy.symbols("a b c d", real=True)
    H_ii = sympy.Matrix([[a, 0], [0, b]])
    H_jj = sympy.Matrix([[c, 0], [0, d]])

    # Define eigenvalues for solve_sylvester_bosonic
    eigs = (((a, b)), ((c, d)))

    # Empty list of bosonic operators since we're not using any
    boson_operators = []

    # Get the solver function
    solve_sylvester = solve_sylvester_bosonic(eigs, boson_operators)

    # Create a test matrix Y
    Y = sympy.Matrix([[1, 2], [3, 4]])

    # Solve the Sylvester equation
    V = solve_sylvester(Y, index=(0, 1, 1))

    result = H_ii * V - V * H_jj

    # Check if the equation is satisfied
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            assert (result[i, j] - Y[i, j]).simplify() == 0


def test_solve_sylvester_bosonic_with_number_operator():
    """Test solving Sylvester equation with number operators in 1x1 matrices."""
    # Define a boson operator
    b = sympy.symbols("b", cls=BosonOp)
    nb = NumberOperator(b)

    omega, delta = sympy.symbols("omega delta", real=True)
    # Define the eigenvalues with number operators
    eigs = (((omega * nb),), ((delta * nb),))

    # List of boson operators
    boson_operators = [b]

    # Get the solver function
    solve_sylvester = solve_sylvester_bosonic(eigs, boson_operators)

    # Create a test matrix Y
    Y = sympy.Matrix([[b]])

    # Solve the Sylvester equation
    V = solve_sylvester(Y, index=(0, 1, 1))

    # Construct Hamiltonians
    H_ii = sympy.Matrix([[omega * nb]])
    H_jj = sympy.Matrix([[delta * nb]])

    # Verify the equation H_ii * V - V * H_jj = Y
    result = H_ii * V - V * H_jj

    # Check that the result matches Y after normal ordering
    assert NumberOrderedForm.from_expr(result[0, 0] - Y[0, 0]).simplify() == 0


def test_solve_sylvester_bosonic():
    a, b = sympy.symbols("a b", cls=BosonOp)
    n_a, n_b = NumberOperator(a), NumberOperator(b)
    g, J = sympy.symbols("g J", real=True)

    eigs = ((g * n_a**2, 2 + n_a), (J * n_a, 1 + n_b))
    boson_operators = [a, b]

    solve_sylvester = solve_sylvester_bosonic(eigs, boson_operators)

    Y = sympy.Matrix([[a, a * b], [Dagger(a), n_a]])
    V = solve_sylvester(Y, index=(0, 1, 1))

    H_ii = sympy.diag(*eigs[0])
    H_jj = sympy.diag(*eigs[1])

    Y_expected = H_ii * V - V * H_jj

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            result = NumberOrderedForm.from_expr(Y[i, j]) - Y_expected[i, j]
            assert result.simplify() == 0, f"Failed for Y[{i}, {j}]: {result.simplify()}"


def test_group_ordered_idempotence():
    """Test that grouping and reassembling terms preserves the expression.

    This tests the idempotence of the group_ordered when followed
    by the reassembly operation. Instead of using number_ordered_form,
    we directly create a properly number-ordered expression.
    """
    # Create boson operators
    a = sympy.symbols("a", cls=BosonOp)
    b = sympy.symbols("b", cls=BosonOp)

    # Create number operators
    Na = NumberOperator(a)
    Nb = NumberOperator(b)

    # Symbols for coefficients
    x, y, z = sympy.symbols("x y z", real=True)

    # Create a complex expression that's already in number-ordered form:
    # Number ordered means NO TERM can have both creation and annihilation
    # operators for the same particle
    expr = (
        # Terms with only number operators
        2 * Na
        + 3 * Nb
        + x * Na * Nb
        + y * Na**2
        # Pure creation operators
        + Dagger(a)
        + Dagger(a) ** 2
        + Dagger(a) * Dagger(b)
        # Pure annihilation operators
        + a
        + b**2
        + a * b
        # Mixed operators for DIFFERENT particles (a and b)
        + Dagger(a) * b
        + Dagger(b) * a
        + Dagger(a) * Nb * b
        + 3 * Dagger(b) * (Na + 1) * Na / Nb * a
        + z * Dagger(b) * Na * a
    )

    # Group the terms by powers of unmatched operators
    grouped_result = group_ordered(expr)

    # Reassemble the expression from the grouped result
    reassembled_expr = sympy.Add(
        *(i * value * j for (i, j), value in grouped_result.items())
    )

    # The reassembled expression should be equal to the original expression
    assert (reassembled_expr - expr).expand() == 0

    # Test with different combinations of creation and annihilation operators
    # but still following number-ordering rules (no a† and a in same term)
    expr2 = (
        # Pure creation or annihilation operators
        Dagger(a) ** 3
        + a**2
        + Dagger(b) ** 2
        + b
        # Mixed operators for DIFFERENT particles only
        + x * Dagger(a) * b**2
        + Dagger(b) ** 2 * a
        # With number operators
        + Dagger(a) * Nb
        + Na * b
    )

    grouped_result2 = group_ordered(expr2)
    reassembled_expr2 = sympy.Add(
        *(i * value * j for (i, j), value in grouped_result2.items())
    )

    assert (reassembled_expr2 - expr2).expand() == 0


def test_find_operators():
    """Test the find_operators function to identify bosonic operators in expressions."""
    a = BosonOp("a")
    b = BosonOp("b")

    # Simple expression with a single operator
    expr1 = a**2 + 1
    result1 = find_operators(expr1)
    assert len(result1) == 1
    assert result1[0] == a

    # Expression with multiple operators
    expr2 = a * b + Dagger(a) * Dagger(b)
    result2 = find_operators(expr2)
    assert len(result2) == 2
    assert set(result2) == {a, b}

    # Expression with no operators
    expr3 = sympy.Symbol("x") + 1
    result3 = find_operators(expr3)
    assert result3 == []

    # Expression with number operators - should find the original operators
    Na = NumberOperator(a)
    Nb = NumberOperator(b)
    expr4 = Na * Nb + Na**2
    result4 = find_operators(expr4)
    assert len(result4) == 2
    assert set(result4) == {a, b}

    # Expression with mixed number operators and original operators
    expr5 = Na * b + a * Nb
    result5 = find_operators(expr5)
    assert len(result5) == 2
    assert set(result5) == {a, b}

    # Expression with number operators inside more complex expressions
    expr6 = (Na + 1) * (Nb - 2) ** 2
    result6 = find_operators(expr6)
    assert len(result6) == 2
    assert set(result6) == {a, b}


def test_simplify_number_expression():
    """Test the simplify_number_expression function."""
    a = BosonOp("a")
    Na = NumberOperator(a)

    # Test normal case
    expr = Na - 2 * (Na)
    result = simplify_number_expression(expr)
    assert result == -Na

    # Test with multiple number operators
    b = BosonOp("b")
    Nb = NumberOperator(b)
    expr2 = Na * Nb - Nb * Na
    result2 = simplify_number_expression(expr2)
    assert result2 == 0

    # Test that it raises ValueError when given bosonic operators
    expr3 = Na + a
    with pytest.raises(ValueError):
        simplify_number_expression(expr3)


@pytest.mark.xfail(reason="There is a bug in the mask probably")
def test_hermitian_block_diagonalization():
    """Test that checks Hermiticity of the block-diagonalized Hamiltonian."""

    bosons = b_1, b_2 = sympy.symbols("b_1 b_2", cls=BosonOp)
    N_1, N_2 = [NumberOperator(boson) for boson in bosons]
    J = sympy.symbols("J", positive=True)

    # Define Hamiltonian
    H = sympy.Matrix([[J * (Dagger(b_1) * b_2 + Dagger(b_2) * b_1) + N_1**2]])

    # Block diagonalize
    H_tilde, *_ = block_diagonalize(H, symbols=[J])

    # Check hermiticity
    for order in range(5):
        H_order = H_tilde[0, 0, order][0, 0].subs(N_1, 2).subs(N_2, 0)

        # Calculate H_order - Dagger(H_order) which should be 0 if hermitian
        hermiticity_check = NumberOrderedForm.from_expr(
            H_order - Dagger(H_order)
        ).simplify()
        assert hermiticity_check == 0, f"H_tilde[0, 0, {order}] is not hermitian."


def test_apply_mask_to_operator():
    """Test the apply_mask_to_operator function with various mask configurations."""
    # Create boson operators
    b_1, b_2 = sympy.symbols("b_1 b_2", cls=BosonOp)

    # Test case 1: Basic mask that allows only number operators
    allowed_terms = [Dagger(b_1) * b_1]
    allowed_matrix = sympy.Matrix(
        [[sympy.Add(*allowed_terms) + Dagger(sympy.Add(*allowed_terms))]]
    )

    not_allowed_terms = [b_1, b_1**2, b_1 * b_2, b_1 * Dagger(b_2) * b_2]
    not_allowed_matrix = sympy.Matrix(
        [[sympy.Add(*not_allowed_terms) + Dagger(sympy.Add(*not_allowed_terms))]]
    )

    # Mask that only allows terms with matched creation/annihilation of b_1
    mask = ([b_1], [(([0], None), np.array([[True]]))])
    masked_expr = apply_mask_to_operator(allowed_matrix + not_allowed_matrix, mask)

    # The mask should filter out all terms except the allowed ones
    assert (
        NumberOrderedForm.from_expr(masked_expr[0, 0] - allowed_matrix[0, 0])
        .simplify()
        .as_expr()
        == 0
    )

    # Create number operators
    N_1, N_2 = [NumberOperator(boson) for boson in (b_1, b_2)]

    # Test case 2: Basic mask that allows only number operators, but with an operator
    # not contained in the mask
    allowed_terms = [N_1, N_1**2, (b_1 * N_1 * Dagger(b_1)) * b_2]
    allowed_matrix = sympy.Matrix(
        [[sympy.Add(*allowed_terms) + Dagger(sympy.Add(*allowed_terms))]]
    )

    not_allowed_terms = [(((N_1 - 1) ** (-2) * N_1) ** 4 - 1) * b_1]
    not_allowed_matrix = sympy.Matrix(
        [[sympy.Add(*not_allowed_terms) + Dagger(sympy.Add(*not_allowed_terms))]]
    )

    # Mask that only allows terms with matched creation/annihilation of b_1
    mask = ([b_1], [(([0], None), np.array([[True]]))])
    masked_expr = apply_mask_to_operator(allowed_matrix + not_allowed_matrix, mask)

    # The mask should filter out all terms except the allowed ones
    assert (
        NumberOrderedForm.from_expr(masked_expr[0, 0] - allowed_matrix[0, 0])
        .simplify()
        .as_expr()
        == 0
    )


def test_boson_operator_diagonalization():
    """Test that operator perturbation theory works the same as the matrix one."""
    a = BosonOp("a")
    n = NumberOperator(a)
    H_0 = sympy.Matrix([[n + n**2]])
    H_1 = sympy.Matrix([[n**3 + a + Dagger(a) + sympy.I * (a**2 - Dagger(a) ** 2)]])
    H_tilde, *_ = block_diagonalize([H_0, H_1])
    # Now finite matrix
    N = 7
    a_mat = sympy.zeros(N, N)
    for i in range(N - 1):
        a_mat[i, (i + 1)] = sympy.sqrt(i + 1)
    n_mat = sympy.diag(*[i for i in range(N)])
    H_0 = sympy.Matrix([[n_mat + n_mat**2]])
    H_1 = sympy.Matrix(
        [[n_mat**3 + a_mat + Dagger(a_mat) + sympy.I * (a_mat**2 - Dagger(a_mat) ** 2)]]
    )
    H_tilde_finite, *_ = block_diagonalize([H_0, H_1])
    # Compare the two
    assert H_tilde[0, 0, 1][0, 0].as_expr().subs({n: 2}) == H_tilde_finite[0, 0, 1][2, 2]

    # Now with two bosons
    b = BosonOp("b")
    n_b = NumberOperator(b)
    H_0 = sympy.Matrix([[n + n_b + n**2 / 3]])
    H_1 = sympy.Matrix([[a + Dagger(a) + b + Dagger(b) + a * b + Dagger(a) * Dagger(b)]])
    H_tilde, *_ = block_diagonalize([H_0, H_1])
    E_eff = H_tilde[0, 0, 2][0, 0].as_expr().subs({n: 0, n_b: 0})
    # Now finite matrix
    N = 3
    a_mat = sympy.zeros(N, N)
    for i in range(N - 1):
        a_mat[i, (i + 1)] = sympy.sqrt(i + 1)
    n_mat = sympy.diag(*[i for i in range(N)])
    b_mat = sympy.KroneckerProduct(a_mat, sympy.eye(N))
    a_mat = sympy.KroneckerProduct(sympy.eye(N), a_mat)
    n_b_mat = sympy.KroneckerProduct(n_mat, sympy.eye(N))
    n_mat = sympy.KroneckerProduct(sympy.eye(N), n_mat)
    H_0 = sympy.Matrix([[n_mat + n_b_mat + n_mat @ n_mat / 3]])
    H_1 = sympy.Matrix(
        [
            [
                a_mat
                + Dagger(a_mat)
                + b_mat
                + Dagger(b_mat)
                + a_mat @ b_mat
                + Dagger(a_mat) @ Dagger(b_mat)
            ]
        ]
    )
    H_tilde_finite, *_ = block_diagonalize([H_0, H_1])
    # Compare the two
    assert E_eff == H_tilde_finite[0, 0, 2][0, 0]
