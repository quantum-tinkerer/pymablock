import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

from pymablock.second_quantization import (
    NumberOperator,
    multiply_b,
    multiply_daggered_b,
    multiply_fn,
    number_ordered_form,
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

    result = -(H_ii * V - V * H_jj)

    # Check if the equation is satisfied
    assert (sympy.simplify(result) - Y) == sympy.zeros(2, 2)


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

    # Verify the equation H_ii * V - V * H_jj = -Y
    result = -(H_ii * V - V * H_jj)

    # Check that the result matches Y after normal ordering
    assert number_ordered_form(result[0, 0] - Y[0, 0], simplify=True) == 0


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

    Y_expected = -(H_ii * V - V * H_jj)

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            assert number_ordered_form(Y[i, j] - Y_expected[i, j], simplify=True) == 0


def test_multiply_b():
    b = BosonOp("b")
    Nb = NumberOperator(b)

    # Test case 1: Simple expression with only annihilation operators
    expr1 = b**2
    result1 = multiply_b(expr1, b)
    expected1 = b**3
    assert (result1 - expected1).expand() == 0

    # Test case 2: Expression with creation operators
    expr2 = Dagger(b) * b
    result2 = normal_ordered_form(multiply_b(expr2, b).doit().expand())
    expected2 = normal_ordered_form((b * Dagger(b) * b).doit().expand())
    assert (result2 - expected2).expand() == 0

    # Test case 3: Complex expression with multiple terms
    expr3 = Nb + Nb**2 * b + Dagger(b) * Nb
    result3 = normal_ordered_form(multiply_b(expr3, b).doit().expand())
    expected3 = normal_ordered_form((expr3 * b).doit().expand())
    assert (result3 - expected3).expand() == 0


def test_multiply_daggered_b():
    b = BosonOp("b")
    Nb = NumberOperator(b)

    # Test case 1: Simple expression with number operator
    expr1 = Nb
    result1 = normal_ordered_form(multiply_daggered_b(expr1, Dagger(b)).doit().expand())
    expected1 = normal_ordered_form((Nb * Dagger(b)).doit().expand())
    assert (result1 - expected1).expand() == 0

    # Test case 2: Expression with annihilation operators
    expr2 = b
    result2 = normal_ordered_form(multiply_daggered_b(expr2, Dagger(b)).doit().expand())
    expected2 = normal_ordered_form((b * Dagger(b)).doit().expand())
    assert (result2 - expected2).expand() == 0


def test_multiply_fn():
    b = BosonOp("b")
    c = BosonOp("c")
    Nb = NumberOperator(b)
    Nc = NumberOperator(c)

    # Test case: Expression with multiple operators
    fn = (Nb**2 + 1) * Nb * (Nc + 1)
    expr = b**2 * c

    result = normal_ordered_form(multiply_fn(expr, fn).doit().expand(), independent=True)
    expected = normal_ordered_form((expr * fn).doit().expand(), independent=True)

    assert (result - expected).expand() == 0


def test_number_ordered_form():
    b = BosonOp("b")
    c = BosonOp("c")
    Nb = NumberOperator(b)
    Nc = NumberOperator(c)

    # Test case: Complex expression with multiple operators
    expr = (
        b**2 * Dagger(b) * (c + 1) ** 2
        + (Nc + 1) ** (1) * (b + 1) * (c + Nb * Dagger(c)) ** 2
    )

    result = normal_ordered_form(
        number_ordered_form(expr).doit().expand(), independent=True
    )
    expected = normal_ordered_form(expr.doit().expand(), independent=True)

    assert (result - expected).expand() == 0
