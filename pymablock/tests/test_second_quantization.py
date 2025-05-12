import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

from pymablock import block_diagonalize, operator_to_BlockSeries
from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm
from pymablock.second_quantization import (
    apply_mask_to_operator,
    solve_sylvester_2nd_quant,
)
from pymablock.series import cauchy_dot_product

from .test_block_diagonalization import compare_series, is_unitary


def test_solve_sylvester_2nd_quant_simple():
    """Confirm that the index handling is correct."""
    # Create symbolic matrices with no bosonic operators
    a, b, c, d = sympy.symbols("a b c d", real=True)
    H_ii = sympy.Matrix([[a, 0], [0, b]])
    H_jj = sympy.Matrix([[c, 0], [0, d]])

    # Define eigenvalues for solve_sylvester_2nd_quant
    eigs = (((a, b)), ((c, d)))

    # Get the solver function
    solve_sylvester = solve_sylvester_2nd_quant(eigs)

    # Create a test matrix Y
    Y = sympy.Matrix([[1, 2], [3, 4]])

    # Solve the Sylvester equation
    V = solve_sylvester(Y, index=(0, 1, 1))

    result = H_ii * V - V * H_jj

    # Check if the equation is satisfied
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            assert (result[i, j] - Y[i, j]).simplify() == 0


def test_solve_sylvester_2nd_quant_with_number_operator():
    """Test solving Sylvester equation with number operators in 1x1 matrices."""
    # Define a boson operator
    b = sympy.symbols("b", cls=BosonOp)
    nb = NumberOperator(b)

    omega, delta = sympy.symbols("omega delta", real=True)
    # Define the eigenvalues with number operators
    eigs = (((omega * nb),), ((delta * nb),))

    # Get the solver function
    solve_sylvester = solve_sylvester_2nd_quant(eigs)

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


def test_solve_sylvester_2nd_quant():
    a, b = sympy.symbols("a b", cls=BosonOp)
    n_a, n_b = NumberOperator(a), NumberOperator(b)
    g, J = sympy.symbols("g J", real=True)

    eigs = ((g * n_a**2, 2 + n_a), (J * n_a, 1 + n_b))

    solve_sylvester = solve_sylvester_2nd_quant(eigs)

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


def test_selective_block_diagonalization():
    """Test that various masks correctly filter specific terms in block diagonalization.

    This test verifies that different masks can be used with block_diagonalize to:
    1. Eliminate only first-order terms (a + a†)
    2. Apply rotating wave approximation (RWA) by filtering out counter-rotating terms
    3. Eliminate higher-order terms
    """
    # Define the Jaynes-Cummings model
    a = BosonOp("a")
    n = NumberOperator(a)
    k = sympy.symbols("k", integer=True, nonnegative=True)
    omega_r, omega_q, g = sympy.symbols("omega_r omega_q g", real=True)

    # Define the Jaynes-Cummings Hamiltonian in operator form
    # H_0 describes the resonator and qubit energies
    H_0 = sympy.Matrix([[n * omega_r + omega_q / 2, 0], [0, n * omega_r - omega_q / 2]])
    H_1 = sympy.Matrix([[0, g * (a + Dagger(a))], [g * (a + Dagger(a)), 0]])

    # Test case 1: Eliminate only 1st order terms (a + a†)
    mask = sympy.Matrix([[sympy.S.Zero, a + Dagger(a)], [a + Dagger(a), sympy.S.Zero]])
    result = sympy.Add(
        *block_diagonalize([H_0, H_1], symbols=[g], fully_diagonalize=mask)[0][
            0, 0, :4
        ].compressed()
    )

    # Confirm that masked terms are indeed zero
    assert not result[0, 1].terms.keys() & {(1,), (-1,)}

    # Test case 2: Apply high order RWA by filtering out creation operators in the upper triangle
    # and annihilation operators in the lower triangle
    mask = sympy.Matrix([[sympy.S.Zero, Dagger(a) ** k], [a**k, sympy.S.Zero]])

    result = sympy.Add(
        *block_diagonalize([H_0, H_1], symbols=[g], fully_diagonalize=mask)[0][
            0, 0, :4
        ].compressed()
    )

    # Confirm that all powers in result[0, 1] are positive (annihilation operators only)
    assert all(pow[0] >= 0 for pow in result[0, 1].terms)

    # Test case 3: Eliminate only a and Dagger(a) with powers above one
    mask = sympy.Matrix(
        [
            [sympy.S.Zero, a ** (k + 2) + Dagger(a) ** (k + 2)],
            [a ** (k + 2) + Dagger(a) ** (k + 2), sympy.S.Zero],
        ]
    )

    # Verify that higher-order terms are eliminated
    higher_order_terms = sympy.Add(
        *block_diagonalize([H_0, H_1], symbols=[g], fully_diagonalize=mask)[0][
            0, 0, 2:6
        ].compressed()
    )

    # Confirm that all higher-order terms are zero
    assert higher_order_terms.applyfunc(lambda x: x.as_expr()).is_zero_matrix


def test_solve_sylvester_2nd_quant_fermion_simple():
    """Test solve_sylvester_2nd_quant with fermion operators."""
    # Create fermion operators
    f = sympy.symbols("f", cls=FermionOp)
    n_f = NumberOperator(f)

    # Create Hamiltonian with diagonal H_0 and off-diagonal perturbation
    H_0 = sympy.Matrix([[n_f, 0], [0, 1]])
    H_1 = sympy.Matrix([[0, f], [Dagger(f), 0]])

    # Block diagonalize
    H_tilde, *_ = block_diagonalize([H_0, H_1])

    # Check that the off-diagonal terms are eliminated
    assert H_tilde[0, 0, 1][0, 1].as_expr() == 0
    assert H_tilde[0, 0, 1][1, 0].as_expr() == 0

    second_order_00 = H_tilde[0, 0, 2][0, 0].as_expr()
    second_order_11 = H_tilde[0, 0, 2][1, 1].as_expr()

    expected_00 = -(1 - n_f)
    expected_11 = n_f

    assert (
        NumberOrderedForm.from_expr(second_order_00 - expected_00).simplify().as_expr()
        == 0
    )
    assert (
        NumberOrderedForm.from_expr(second_order_11 - expected_11).simplify().as_expr()
        == 0
    )


def test_solve_sylvester_2nd_quant_fermion_complex():
    """Test solving Sylvester equation with more complex fermion expressions."""
    # Create fermion operators
    f, g = sympy.symbols("f g", cls=FermionOp)
    n_f, n_g = NumberOperator(f), NumberOperator(g)

    # Create symbolic parameters
    alpha, beta = sympy.symbols("alpha beta", real=True)

    # Define eigenvalues with number operators and parameters
    eigs = ((alpha * n_f, beta * (1 - n_f)), (alpha * n_g, beta * (2 - n_g)))

    # Get the solver function
    solve_sylvester = solve_sylvester_2nd_quant(eigs)

    # Create a test matrix Y with various fermion operators
    Y = sympy.Matrix([[f, f * g], [Dagger(f), n_f * (1 - n_g)]])

    # Solve the Sylvester equation
    V = solve_sylvester(Y, index=(0, 1, 1))

    # Construct Hamiltonians
    H_ii = sympy.diag(*eigs[0])
    H_jj = sympy.diag(*eigs[1])

    # Expected result from equation H_ii * V - V * H_jj = Y
    expected = H_ii * V - V * H_jj

    # Check each matrix element
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            result = NumberOrderedForm.from_expr(Y[i, j] - expected[i, j]).simplify()
            assert result.as_expr() == 0, f"Failed for Y[{i}, {j}]: {result}"


def test_block_diagonalize_fermion_simple():
    """Test block_diagonalize with fermion operators in a simple case."""
    # Create fermion operators
    f = sympy.symbols("f", cls=FermionOp)
    n_f = NumberOperator(f)

    # Define parameters
    omega, delta = sympy.symbols("omega delta", real=True)

    # Create Hamiltonian with diagonal H_0 and off-diagonal perturbation
    H_0 = sympy.Matrix([[n_f * omega, 0], [0, (2 - n_f) * omega]])
    H_1 = sympy.Matrix([[0, delta * f], [delta * Dagger(f), 0]])
    H = operator_to_BlockSeries([H_0, H_1])

    # Block diagonalize
    H_tilde, U, U_dagger = block_diagonalize(H)
    for order in range(4):
        assert H_tilde[0, 0, order].is_diagonal()

    is_unitary(U, U_dagger, wanted_orders=(4,))

    H_reconstructed = cauchy_dot_product(U, cauchy_dot_product(H_tilde, U_dagger))

    compare_series(H, H_reconstructed, wanted_orders=(4,))


def test_block_diagonalize_fermion_complex():
    """Test block_diagonalize with more complex fermion system (two fermion modes)."""
    # Create fermion operators for two modes
    f, g = sympy.symbols("f g", cls=FermionOp)
    n_f, n_g = NumberOperator(f), NumberOperator(g)

    # Define parameters
    omega_f, omega_g, J = sympy.symbols("omega_f omega_g J", real=True)

    # Create unperturbed Hamiltonian with diagonal terms
    H_0 = sympy.Matrix(
        [
            [n_f * omega_f + n_g * omega_g, 0],
            [0, (2 - n_f) * omega_f + (1 - n_g) * omega_g],
        ]
    )

    # Create perturbation with off-diagonal hopping terms
    H_1 = sympy.Matrix(
        [
            [0, J * (Dagger(f) * g + Dagger(g) * f)],
            [J * (Dagger(g) * f + Dagger(f) * g), 0],
        ]
    )
    H = operator_to_BlockSeries([H_0, H_1])

    # Block diagonalize
    H_tilde, U, U_dagger = block_diagonalize(H)

    # Verify block-diagonalization
    for order in range(4):
        assert H_tilde[0, 0, order].is_diagonal()

    # Check unitarity of the transformation
    is_unitary(U, U_dagger, wanted_orders=(4,))

    # Verify that the transformation preserves the original Hamiltonian
    H_reconstructed = cauchy_dot_product(U, cauchy_dot_product(H_tilde, U_dagger))
    compare_series(H, H_reconstructed, wanted_orders=(4,))


def test_fermion_interaction_model():
    """Test block_diagonalize with a fermion interaction model."""
    # Create fermion operators for a multi-mode system
    f1, f2, f3 = sympy.symbols("f1 f2 f3", cls=FermionOp)
    n1, n2, n3 = NumberOperator(f1), NumberOperator(f2), NumberOperator(f3)

    # Define parameters
    e1, e2, e3, U12, U23, t = sympy.symbols("e1 e2 e3 U12 U23 t", real=True)

    # Create Hubbard-like Hamiltonian:
    # - Single-particle energies
    # - Two-site interactions
    # - Hopping between sites
    H_0 = sympy.Matrix(
        [
            [
                e1 * n1
                + e2 * n2
                + e3 * n3  # Single-particle energies
                + U12 * n1 * n2
                + U23 * n2 * n3  # Two-site interactions
            ]
        ]
    )

    # Perturbation: hopping between sites 1-2 and 2-3
    H_1 = sympy.Matrix(
        [
            [
                t * (Dagger(f1) * f2 + Dagger(f2) * f1)  # Hopping between sites 1-2
                + t * (Dagger(f2) * f3 + Dagger(f3) * f2)  # Hopping between sites 2-3
            ]
        ]
    )

    # Block diagonalize
    H_tilde, *_ = block_diagonalize([H_0, H_1], symbols=[t])

    # Test case 1: Check second order correction for n1=1, n2=0, n3=0
    # This state can hop to n1=0, n2=1, n3=0 and back
    energy_100 = H_tilde[0, 0, 2][0, 0].subs({n1: 1, n2: 0, n3: 0}).as_expr()
    expected_100 = -(t**2) / (e2 - e1)
    assert sympy.simplify(energy_100 - expected_100) == 0

    # Test case 2: Check second order correction for n1=0, n2=0, n3=1
    # This state can hop to n1=0, n2=1, n3=0 and back
    energy_001 = H_tilde[0, 0, 2][0, 0].subs({n1: 0, n2: 0, n3: 1}).as_expr()
    expected_001 = -(t**2) / (e2 - e3)
    assert sympy.simplify(energy_001 - expected_001) == 0

    # Test case 3: Check second order correction for n1=1, n2=1, n3=0
    # The hopping is now affected by the interaction U12
    energy_110 = H_tilde[0, 0, 2][0, 0].subs({n1: 1, n2: 1, n3: 0}).as_expr()
    # Expected: second order hopping from site 2 to 3 and back
    expected_110 = -(t**2) / (e3 - e2 - U12)
    assert sympy.simplify(energy_110 - expected_110) == 0


def test_mixed_fermion_boson_sylvester():
    """Test solving Sylvester equation with mixed fermion and boson operators."""
    # Create boson and fermion operators
    a = sympy.symbols("a", cls=BosonOp)
    f = sympy.symbols("f", cls=FermionOp)

    n_a = NumberOperator(a)
    n_f = NumberOperator(f)

    # Define eigenvalues with number operators
    omega_a, omega_f = sympy.symbols("omega_a omega_f", real=True)
    eigs = ((omega_a * n_a,), (omega_f * n_f,))

    # Get the solver function
    solve_sylvester = solve_sylvester_2nd_quant(eigs)

    # Test case 1: Fermion operator in Y
    Y1 = sympy.Matrix([[f]])
    V1 = solve_sylvester(Y1, index=(0, 1, 1))

    # Construct Hamiltonians
    H_ii = sympy.Matrix([eigs[0]])
    H_jj = sympy.Matrix([eigs[1]])

    # Verify the equation H_ii * V1 - V1 * H_jj = Y1
    result1 = H_ii * V1 - V1 * H_jj
    assert NumberOrderedForm.from_expr(result1[0, 0] - Y1[0, 0]).simplify().as_expr() == 0

    # Test case 2: Boson operator in Y
    Y2 = sympy.Matrix([[a]])
    V2 = solve_sylvester(Y2, index=(0, 1, 1))

    # Verify the equation H_ii * V2 - V2 * H_jj = Y2
    result2 = H_ii * V2 - V2 * H_jj
    assert NumberOrderedForm.from_expr(result2[0, 0] - Y2[0, 0]).simplify().as_expr() == 0

    # Test case 3: Mixed boson-fermion operator in Y
    Y3 = sympy.Matrix([[a * f]])
    V3 = solve_sylvester(Y3, index=(0, 1, 1))

    # Verify the equation H_ii * V3 - V3 * H_jj = Y3
    result3 = H_ii * V3 - V3 * H_jj
    assert NumberOrderedForm.from_expr(result3[0, 0] - Y3[0, 0]).simplify().as_expr() == 0


def test_mixed_fermion_boson_diagonalization():
    """Test block diagonalization with mixed fermion-boson Hamiltonian."""
    # Create operators
    a = sympy.symbols("a", cls=BosonOp)
    f = sympy.symbols("f", cls=FermionOp)

    n_a = NumberOperator(a)
    n_f = NumberOperator(f)

    # Define parameters
    omega_a, omega_f, g = sympy.symbols("omega_a omega_f g", real=True)

    # Create Hamiltonian representing a fermion-boson coupled system
    # (similar to a Jaynes-Cummings model with fermions)
    H_0 = sympy.Matrix(
        [[omega_a * n_a + omega_f * n_f, 0], [0, omega_a * n_a + omega_f * (1 - n_f)]]
    )

    # Coupling term: fermion can flip state by emitting or absorbing a boson
    H_1 = sympy.Matrix([[0, g * a * f], [g * Dagger(a) * Dagger(f), 0]])

    # Convert to BlockSeries
    H = operator_to_BlockSeries([H_0, H_1])

    # Block diagonalize
    H_tilde, U, U_dagger = block_diagonalize(H)

    # Verify block-diagonalization (H_tilde is diagonal at each order)
    for order in range(4):
        assert H_tilde[0, 0, order].is_diagonal()

    # Check unitarity of the transformation
    is_unitary(U, U_dagger, wanted_orders=(4,))

    # Verify that the transformation preserves the original Hamiltonian
    H_reconstructed = cauchy_dot_product(U, cauchy_dot_product(H_tilde, U_dagger))
    compare_series(H, H_reconstructed, wanted_orders=(4,))


def test_holstein_model():
    """Test a Holstein-like model with electron-phonon coupling."""
    # Create operators
    a = sympy.symbols("a", cls=BosonOp)  # Phonon operator
    f = sympy.symbols("f", cls=FermionOp)  # Electron operator

    n_a = NumberOperator(a)
    n_f = NumberOperator(f)

    # Define parameters
    omega_0, e_0, g = sympy.symbols("omega_0 e_0 g", real=True)

    # Create Holstein Hamiltonian
    # H = ω₀ a†a + e₀ f†f + g(a + a†)f†f
    H_0 = sympy.Matrix([[omega_0 * n_a + e_0 * n_f]])  # Uncoupled phonon and electron
    H_1 = sympy.Matrix([[g * (a + Dagger(a)) * n_f]])  # Electron-phonon coupling

    # Block diagonalize
    H_tilde, *_ = block_diagonalize([H_0, H_1], symbols=[g])

    # In the Holstein model, we expect a polaron shift in the electron energy
    # For second-order, calculate energy for n_f = 1 and various phonon numbers
    # Let's verify the polaron shift formula

    # Get the second order correction with n_f = 1
    E_2_with_electron = H_tilde[0, 0, 2][0, 0].subs({n_f: 1}).as_expr()

    # The expected polaron shift is -g²/ω₀
    expected_shift = -(g**2) / omega_0

    # For the Holstein model, the second-order correction should be
    # independent of the phonon number
    assert sympy.simplify(E_2_with_electron - expected_shift) == 0

    # Check that without an electron (n_f = 0), there's no correction
    E_2_no_electron = H_tilde[0, 0, 2][0, 0].subs({n_f: 0}).as_expr()
    assert E_2_no_electron == 0
