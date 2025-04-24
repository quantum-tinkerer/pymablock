import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm
from pymablock.second_quantization import (
    apply_mask_to_operator,
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
    H_ii = sympy.Matrix([eigs[0]])
    H_jj = sympy.Matrix([eigs[1]])

    # Verify the equation H_ii * V - V * H_jj = Y
    result = H_ii * V - V * H_jj

    # Check that the result matches Y after normal ordering
    assert NumberOrderedForm.from_expr(result[0, 0] - Y[0, 0]).simplify().as_expr() == 0

    # Same for Dagger(b)
    Y = sympy.Matrix([[Dagger(b)]])

    # Solve the Sylvester equation
    V = solve_sylvester(Y, index=(0, 1, 1))

    result = H_ii * V - V * H_jj

    # Check that the result matches Y after normal ordering
    assert NumberOrderedForm.from_expr(result[0, 0] - Y[0, 0]).simplify().as_expr() == 0


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
            assert (
                result.simplify().as_expr() == 0
            ), f"Failed for Y[{i}, {j}]: {result.simplify()}"


def test_hermitian_block_diagonalization():
    """Test that checks Hermiticity of the block-diagonalized Hamiltonian."""

    bosons = b_1, b_2 = sympy.symbols("b_1 b_2", cls=BosonOp)
    N_1, N_2 = [NumberOperator(boson) for boson in bosons]
    J = sympy.symbols("J", positive=True)

    # Define Hamiltonian
    H = sympy.Matrix([[J * (Dagger(b_1) * b_2 + Dagger(b_2) * b_1) + N_1**2 + N_2 / 3]])

    # Block diagonalize
    H_tilde, *_ = block_diagonalize(H, symbols=[J])

    # Check hermiticity
    for order in range(5):
        H_order = H_tilde[0, 0, order][0, 0].subs(N_1, 2).subs(N_2, 0)

        # Calculate H_order - Dagger(H_order) which should be 0 if hermitian
        hermiticity_check = (
            NumberOrderedForm.from_expr(H_order - Dagger(H_order)).simplify().as_expr()
        )
        assert hermiticity_check == 0, f"H_tilde[0, 0, {order}] is not hermitian."


def test_apply_mask_to_operator():
    """Test the apply_mask_to_operator function with various mask configurations."""
    # Create boson operators
    b_1, b_2 = sympy.symbols("b_1 b_2", cls=BosonOp)
    n = sympy.symbols("n", integer=True, positive=True)

    # Test case 1: Basic mask that allows only number operators
    allowed_terms = [Dagger(b_1) * b_1]
    allowed_matrix = sympy.Matrix(
        [[sympy.Add(*allowed_terms) + Dagger(sympy.Add(*allowed_terms))]]
    )

    not_allowed_terms = [b_1, b_1**2, b_1 * b_2, b_1 * Dagger(b_2) * b_2]
    not_allowed_matrix = sympy.Matrix(
        [[sympy.Add(*not_allowed_terms) + Dagger(sympy.Add(*not_allowed_terms))]]
    )

    mask_nof = NumberOrderedForm.from_expr(1)
    mask = sympy.Matrix([[mask_nof]])

    input = (allowed_matrix + not_allowed_matrix).applyfunc(NumberOrderedForm.from_expr)
    masked_expr = apply_mask_to_operator(input, mask, keep=True)

    # The mask should filter out all terms except the allowed ones
    assert (
        NumberOrderedForm.from_expr(masked_expr[0, 0] - allowed_matrix[0, 0])
        .simplify()
        .as_expr()
        == 0
    )

    # Create number operators
    N_1, N_2 = [NumberOperator(boson) for boson in (b_1, b_2)]

    # Test case 2: Eliminate first order powers of b_1 times anything
    allowed_terms = [N_1, N_1**2, (b_1 * N_1 * Dagger(b_1)) * b_2]
    allowed_matrix = sympy.Matrix(
        [[sympy.Add(*allowed_terms) + Dagger(sympy.Add(*allowed_terms))]]
    )

    not_allowed_terms = [(((N_1 - 1) ** (-2) * N_1) ** 4 - 1) * b_1]
    not_allowed_matrix = sympy.Matrix(
        [[sympy.Add(*not_allowed_terms) + Dagger(sympy.Add(*not_allowed_terms))]]
    )

    # Create a mask matrix for test case 2
    # This mask filters out terms with unmatched b_1
    mask_nof = NumberOrderedForm.from_expr(b_1 * (b_2**n + Dagger(b_2) ** n + 1))
    mask = sympy.Matrix([[mask_nof + Dagger(mask_nof)]])

    input = (allowed_matrix + not_allowed_matrix).applyfunc(NumberOrderedForm.from_expr)
    masked_expr = apply_mask_to_operator(input, mask, keep=False)

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
    H_0 = sympy.Matrix([[n + n_b / 5 + n**2 / 3]])
    H_1 = sympy.Matrix([[a + Dagger(a) + b + Dagger(b) + a * b + Dagger(a) * Dagger(b)]])
    H_tilde, *_ = block_diagonalize([H_0, H_1])
    E_eff = H_tilde[0, 0, 2][0, 0].as_expr().subs({n: 0, n_b: 0})
    # Now finite matrix
    N = 4
    a_mat = sympy.zeros(N, N)
    for i in range(N - 1):
        a_mat[i, (i + 1)] = sympy.sqrt(i + 1)
    n_mat = sympy.diag(*[i for i in range(N)])
    b_mat = sympy.KroneckerProduct(a_mat, sympy.eye(N))
    a_mat = sympy.KroneckerProduct(sympy.eye(N), a_mat)
    n_b_mat = sympy.KroneckerProduct(n_mat, sympy.eye(N))
    n_mat = sympy.KroneckerProduct(sympy.eye(N), n_mat)
    H_0 = sympy.Matrix([[n_mat + n_b_mat / 5 + n_mat * n_mat / 3]])
    H_1 = sympy.Matrix(
        [
            [
                a_mat
                + Dagger(a_mat)
                + b_mat
                + Dagger(b_mat)
                + a_mat * b_mat
                + Dagger(a_mat) * Dagger(b_mat)
            ]
        ]
    )
    H_tilde_finite, *_ = block_diagonalize([H_0, H_1])
    # Compare the two
    assert E_eff == H_tilde_finite[0, 0, 2][0, 0]

    # One boson and a 2x2 matrix Hamiltonian (Jaynes-Cummings)
    a = BosonOp("a")
    n = NumberOperator(a)
    omega_r, omega_q, g = sympy.symbols("omega_r omega_q g", real=True)

    # Define the Jaynes-Cummings Hamiltonian in operator form
    H_0 = sympy.Matrix([[n * omega_r + omega_q / 2, 0], [0, n * omega_r - omega_q / 2]])
    H_1 = sympy.Matrix([[0, g * a], [g * Dagger(a), 0]])

    # Define separate subspaces (up and down) for the Jaynes-Cummings model
    subspace_indices = [0, 1]
    H_tilde, *_ = block_diagonalize(
        [H_0, H_1], subspace_indices=subspace_indices, symbols=[g]
    )

    # Compute the energy correction to the first level
    E_eff_up = H_tilde[0, 0, 4][0, 0].as_expr().subs({n: 0})
    E_eff_down = H_tilde[1, 1, 4][0, 0].as_expr().subs({n: 0})

    # Now finite matrix representation
    N = 5
    a_mat = sympy.zeros(N, N)
    for i in range(N - 1):
        a_mat[i, (i + 1)] = sympy.sqrt(i + 1)
    n_mat = sympy.diag(*[i for i in range(N)])

    # Construct the 2x2 block matrices with the bosonic operators
    H_0_up = n_mat * omega_r + omega_q / 2 * sympy.eye(N)
    H_0_down = n_mat * omega_r - omega_q / 2 * sympy.eye(N)
    H_0_mat = [[H_0_up, sympy.zeros(N, N)], [sympy.zeros(N, N), H_0_down]]

    H_1_up_down = g * a_mat
    H_1_down_up = g * Dagger(a_mat)
    H_1_mat = [[sympy.zeros(N, N), H_1_up_down], [H_1_down_up, sympy.zeros(N, N)]]

    # Block diagonalize the matrix representation
    H_tilde_finite, *_ = block_diagonalize([H_0_mat, H_1_mat])

    # Compare the effective energies from both approaches
    assert sympy.simplify(E_eff_up - H_tilde_finite[0, 0, 4][0, 0]) == 0
    assert sympy.simplify(E_eff_down - H_tilde_finite[1, 1, 4][0, 0]) == 0
