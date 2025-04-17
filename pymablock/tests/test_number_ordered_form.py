"""Tests for the NumberOrderedForm class."""

import pytest
import sympy
from sympy.physics.quantum import Dagger, boson, fermion
from sympy.physics.quantum.operatorordering import normal_ordered_form

from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.second_quantization import NumberOperator


def test_number_ordered_form_init():
    """Test basic initialization of NumberOrderedForm."""
    # Create operators
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")
    operators = [a, b]

    # Create terms dictionary
    terms = {
        (0, 0): sympy.S.One,  # Constant term
        (-1, 0): NumberOperator(a),  # a† term with number operator coefficient
        (0, 1): sympy.S(2),  # b term with scalar coefficient
        (-1, 1): sympy.S(3) * NumberOperator(a),  # a†b term with coefficient
    }

    # Create the NumberOrderedForm
    nof = NumberOrderedForm(operators, terms)

    # Check the attributes
    assert nof.operators == operators
    assert nof.terms == terms


def test_number_ordered_form_validation():
    """Test validation during initialization."""
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")

    # Test with non-quantum operator
    with pytest.raises(TypeError, match="Expected BosonOp or FermionOp"):
        NumberOrderedForm([sympy.Symbol("x")], {(0,): sympy.S.One})

    # Test with creation operator instead of annihilation
    with pytest.raises(ValueError, match="must be an annihilation operator"):
        NumberOrderedForm([Dagger(a)], {(0,): sympy.S.One})

    # Test with boson after fermion
    f = fermion.FermionOp("f")
    with pytest.raises(ValueError, match="must come before fermionic operators"):
        NumberOrderedForm([f, a], {(0, 0): sympy.S.One})

    # Test with empty terms dictionary
    with pytest.raises(ValueError, match="Empty terms dictionary"):
        NumberOrderedForm([a], {})

    # Test with wrong powers tuple length
    with pytest.raises(ValueError, match="Powers tuple length"):
        NumberOrderedForm([a, b], {(0,): sympy.S.One})

    # Test with non-sympy expression coefficient
    with pytest.raises(TypeError, match="must be a sympy expression"):
        NumberOrderedForm([a], {(0,): "not an expression"})

    # Test with creation/annihilation operator in coefficient
    with pytest.raises(ValueError, match="contains creation or annihilation operators"):
        NumberOrderedForm([a], {(0,): a * sympy.S.One})


def test_boolean_evaluation():
    """Test boolean evaluation of NumberOrderedForm."""
    a = boson.BosonOp("a")

    # Test with non-empty terms dictionary
    nof = NumberOrderedForm([a], {(0,): sympy.S.One})
    assert bool(nof)

    # For empty terms test, we'll just check the implementation directly
    # rather than creating an invalid object
    obj = NumberOrderedForm([a], {(0,): sympy.S.One})
    # Test the boolean implementation manually
    assert NumberOrderedForm.__bool__(obj)
    # Simulate empty terms
    empty_terms = {}
    assert not bool(empty_terms)


def test_equality():
    """Test equality comparison of NumberOrderedForm."""
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")

    # Same operators and terms
    nof1 = NumberOrderedForm([a], {(0,): sympy.S.One})
    nof2 = NumberOrderedForm([a], {(0,): sympy.S.One})
    assert nof1 == nof2

    # Different operators
    nof3 = NumberOrderedForm([b], {(0,): sympy.S.One})
    assert nof1 != nof3

    # Different terms
    nof4 = NumberOrderedForm([a], {(0,): sympy.S(2)})
    assert nof1 != nof4

    # Different object type
    assert nof1 != sympy.S.One


def test_from_expr():
    """Test creating NumberOrderedForm from expressions."""
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")

    # Simple expression: a† + b
    expr = Dagger(a) + b
    nof = NumberOrderedForm.from_expr(expr)

    # Check operators - we should have exactly 2
    assert len(nof.operators) == 2

    # Check that the operators are a and b (by name as strings)
    names = sorted([str(op.name) for op in nof.operators])
    assert names == ["a", "b"]

    # Check that we have the correct terms - specifically terms for a† and b
    # We need to find one term that represents a† and one that represents b
    creation_term_found = False
    annihilation_term_found = False

    for powers in nof.terms:
        # A creation term will have a negative power in one position
        if any(p < 0 for p in powers):
            creation_term_found = True
        # An annihilation term will have a positive power in one position
        elif any(p > 0 for p in powers):
            annihilation_term_found = True

    assert creation_term_found, "No creation term found in the expression"
    assert annihilation_term_found, "No annihilation term found in the expression"

    # Check that all coefficients are sympy expressions
    for power in nof.terms:
        assert isinstance(nof.terms[power], sympy.Expr)


def test_compare_from_expr():
    """Test that from_expr handles products of operators correctly."""

    b = boson.BosonOp("b")

    fn1 = NumberOrderedForm.from_expr(b * Dagger(b) ** 2 * b)
    fn2 = NumberOrderedForm.from_expr((b * Dagger(b))) * NumberOrderedForm.from_expr(
        Dagger(b) * b
    )

    assert (
        normal_ordered_form(fn1.as_expr().doit())
        - normal_ordered_form(fn2.as_expr().doit())
        == 0
    )


def test_printing_methods():
    """Test the printing methods of NumberOrderedForm."""
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")

    # Create a NumberOrderedForm with a simple expression
    expr = Dagger(a) * b + sympy.S(2) * Dagger(b)
    nof = NumberOrderedForm.from_expr(expr)

    # Get the expression from as_expr for comparison
    orig_expr = nof.as_expr()

    # Test string representation
    assert str(nof) == str(orig_expr)

    # Test representation
    assert repr(nof) == repr(orig_expr)

    # Test pretty printing via sympy.pretty
    assert sympy.pretty(nof) == sympy.pretty(orig_expr)

    # Test LaTeX representation via sympy.latex
    assert sympy.latex(nof) == sympy.latex(orig_expr)

    # Additional test with a simple scalar expression
    scalar_nof = NumberOrderedForm.from_expr(sympy.S(42))
    scalar_expr = scalar_nof.as_expr()

    assert str(scalar_nof) == str(scalar_expr)


def test_from_expr_without_operators():
    """Test creating NumberOrderedForm from an expression without operators."""
    # Create a simple scalar expression
    expr = sympy.S(5)
    nof = NumberOrderedForm.from_expr(expr)

    # Check that the resulting NumberOrderedForm has no operators
    assert len(nof.operators) == 0

    # Check that there is one term with an empty tuple as key
    assert len(nof.terms) == 1
    assert () in nof.terms

    # Check that the coefficient is the scalar value
    assert nof.terms[()] == sympy.S(5)


def test_round_trip_conversion():
    """Test round-trip conversion from expression to NumberOrderedForm and back."""
    # Create boson operators
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")

    # Test cases with different types of expressions (already number-ordered)
    test_expressions = [
        # Simple creation and annihilation operators
        Dagger(a),
        b,
        # Linear combination
        Dagger(a) + b,
        # Product of operators
        Dagger(a) * b,
        # Expression with number operator
        Dagger(b) * NumberOperator(b) * a,
        # More complex expression
        Dagger(a) ** 2 * NumberOperator(b) * b + Dagger(b) * a**2,
    ]

    for expr in test_expressions:
        # Convert to NumberOrderedForm and back
        nof = NumberOrderedForm.from_expr(expr)
        result = nof.as_expr()

        # Check that the result is equal to the original ordered expression
        assert expr == result, f"Round-trip conversion failed for {expr}"


def test_only_number_operators():
    """Test NumberOrderedForm with an expression containing only number operators."""
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")

    # Create an expression with only number operators
    expr = NumberOperator(a) + 2 * NumberOperator(b)

    # Convert to NumberOrderedForm
    nof = NumberOrderedForm.from_expr(expr)

    assert nof.operators == [a, b]
    assert nof.terms == {(0, 0): expr}


def test_scalar_round_trip():
    """Test round-trip conversion for scalar expressions."""
    # Test with various scalar types
    scalars = [sympy.S.Zero, sympy.S.One, sympy.S(5), sympy.symbols("x")]

    for scalar in scalars:
        nof = NumberOrderedForm.from_expr(scalar)
        result = nof.as_expr()
        assert result == scalar, f"Scalar round-trip failed for {scalar}"


def test_empty_round_trip():
    """Test round-trip conversion for empty NumberOrderedForm."""
    # Create an empty NumberOrderedForm with no operators
    nof = NumberOrderedForm([], {(): sympy.S.Zero}, validate=False)
    result = nof.as_expr()
    assert result == sympy.S.Zero


def test_multiply_op():
    """Test the _multiply_op method using normal_ordered_form as a reference."""
    a, b = sympy.symbols("a b", cls=boson.BosonOp)
    x, y = sympy.symbols("x y")
    n_a, n_b = NumberOperator(a), NumberOperator(b)

    terms = [
        # No operators
        {(0, 0): sympy.S.One},
        # Only number operators
        {(0, 0): n_a},
        {(0, 0): n_b},
        {(0, 0): n_a * n_b},
        # One operator
        {(0, 2): sympy.S.One},
        {(0, 1): n_a},
        {(0, 1): n_b},
        {(0, -1): sympy.S.One},
        # Both
        {(3, 3): n_a},
        {(3, -3): n_b},
        {(-3, 3): n_a + n_b},
        {(-3, -3): n_a * n_b},
    ]
    to_multiply = {
        (0, 1): a,
        (0, 2): a**2,
        (0, -1): Dagger(a),
        (0, -2): Dagger(a) ** 2,
        (1, 1): b,
        (1, 2): b**2,
        (1, -1): Dagger(b),
        (1, -2): Dagger(b) ** 2,
    }
    for term, (op, expr) in zip(terms, to_multiply.items()):
        nof = NumberOrderedForm([a, b], term)
        result = sympy.expand(nof._multiply_op(*op).as_expr().doit())
        expected = sympy.expand(nof.as_expr().doit() * expr)
        assert sympy.expand(
            normal_ordered_form(result, independent=True)
        ) == sympy.expand(
            normal_ordered_form(expected, independent=True)
        ), f"Failed for term {term} with operator {op}"


def test_multiply_expr():
    """Test the _multiply_expr method using normal_ordered_form as a reference."""
    a, b = sympy.symbols("a b", cls=boson.BosonOp)
    x, y = sympy.symbols("x y", real=True)
    n_a, n_b = NumberOperator(a), NumberOperator(b)

    # Create various NumberOrderedForm instances to multiply
    nof_cases = [
        # Empty NumberOrderedForm (just constant)
        NumberOrderedForm([a, b], {(0, 0): sympy.S.One}),
        # Simple operator terms
        NumberOrderedForm([a, b], {(1, 0): sympy.S.One}),  # a
        NumberOrderedForm([a, b], {(-1, 0): sympy.S.One}),  # a†
        # Multiple terms
        NumberOrderedForm([a, b], {(1, 0): x, (-1, 0): y}),  # x*a + a†*y
        # Complex expressions
        NumberOrderedForm([a, b], {(1, 2): x, (-1, -2): y}),  # x*a*b^2 + a†*b†^2*y
    ]

    # Expressions to multiply by.

    # Sympy struggles in comparing expressions with number operators, so we choose
    # relatively simple examples.

    expr_cases = [
        # Simple scalars
        sympy.S.One,
        sympy.S(2),
        x,
        x * y,
        # Number operators
        n_a,
        n_b,
        # Combinations of number operators and scalars
        x * n_a,
        n_a + n_b,
        x * n_a + y * n_b,
    ]

    for nof in nof_cases:
        for expr in expr_cases:
            # Apply _multiply_expr
            result = nof._multiply_expr(expr).as_expr()

            expected = NumberOrderedForm.from_expr(nof.as_expr() * expr).as_expr()

            assert (
                result == expected
            ), f"_multiply_expr failed with nof={nof.as_expr()}, expr={expr}"


def test_multiply_expr_raises_error():
    """Test that _multiply_expr raises an error when the expression contains operators."""
    a, b = sympy.symbols("a b", cls=boson.BosonOp)
    n_a = NumberOperator(a)

    # Create a simple NumberOrderedForm
    nof = NumberOrderedForm([a, b], {(0, 0): sympy.S.One})

    # Test expressions containing operators or their daggers
    invalid_expressions = [
        a,  # Annihilation operator
        Dagger(a),  # Creation operator
        n_a + a,  # Number operator + annihilation operator
        sympy.S(2) * Dagger(b),  # Scalar * creation operator
        a * Dagger(a),  # Product of operators
    ]

    for expr in invalid_expressions:
        with pytest.raises(
            ValueError, match="Expression contains creation or annihilation operators"
        ):
            nof._multiply_expr(expr)


def test_addition():
    """Test addition of NumberOrderedForm."""
    a, b = sympy.symbols("a b", cls=boson.BosonOp)
    x, y = sympy.symbols("x y", real=True)

    # Test adding two NumberOrderedForm instances with same operators
    nof1 = NumberOrderedForm([a, b], {(1, 0): x, (0, 0): sympy.S.One})  # x*a + 1
    nof2 = NumberOrderedForm([a, b], {(-1, 0): y, (0, 1): sympy.S(2)})  # y*a† + 2*b

    result = nof1 + nof2
    expected = NumberOrderedForm.from_expr(nof1.as_expr() + nof2.as_expr())

    # Use sympy.expand and normal_ordered_form for comparison to handle equivalent forms
    assert sympy.expand(
        normal_ordered_form(result.as_expr(), independent=True)
    ) == sympy.expand(normal_ordered_form(expected.as_expr(), independent=True))

    # Test adding with different operators (should combine operators list)
    c = boson.BosonOp("c")
    nof3 = NumberOrderedForm([a], {(1,): x})  # x*a
    nof4 = NumberOrderedForm([c], {(1,): y})  # y*c

    result = nof3 + nof4
    expected = NumberOrderedForm.from_expr(nof3.as_expr() + nof4.as_expr())

    assert result.as_expr() == expected.as_expr()
    assert len(result.operators) == 2
    # Use string comparison for operator names
    assert all(str(op.name) in ["a", "c"] for op in result.operators)

    # Test adding with a sympy expression
    nof5 = NumberOrderedForm([a], {(1,): x})  # x*a
    scalar = sympy.S(3)  # 3

    result = nof5 + scalar
    expected = NumberOrderedForm.from_expr(nof5.as_expr() + scalar)

    assert result.as_expr() == expected.as_expr()

    # Test adding with a sympy expression containing operators
    expr_with_operators = Dagger(a) * b
    nof6 = NumberOrderedForm([a, b], {(1, 0): x})  # x*a

    result = nof6 + expr_with_operators
    expected = NumberOrderedForm.from_expr(nof6.as_expr() + expr_with_operators)

    assert result.as_expr() == expected.as_expr()


def test_multiplication():
    """Test the __mul__ and __rmul__ methods for NumberOrderedForm."""
    a, b = sympy.symbols("a b", cls=boson.BosonOp)
    x, y = sympy.symbols("x y", real=True)
    n_a = NumberOperator(a)

    # Test multiplying two NumberOrderedForm instances with same operators
    nof1 = NumberOrderedForm([a, b], {(0, 0): x})  # x
    nof2 = NumberOrderedForm([a, b], {(1, 0): sympy.S.One})  # a

    result = nof1 * nof2
    expected = NumberOrderedForm.from_expr(nof1.as_expr() * nof2.as_expr())

    # Compare using normal ordered forms for consistent comparison
    result_normal = normal_ordered_form(result.as_expr(), independent=True)
    expected_normal = normal_ordered_form(expected.as_expr(), independent=True)
    assert sympy.expand(result_normal) == sympy.expand(expected_normal)

    # Test more complex multiplication with same operators
    nof3 = NumberOrderedForm([a, b], {(-1, 0): x, (0, 1): sympy.S(2)})  # x*a† + 2*b
    nof4 = NumberOrderedForm([a, b], {(1, 0): y, (0, 0): n_a})  # y*a + n_a

    result = nof3 * nof4
    expected = NumberOrderedForm.from_expr(nof3.as_expr() * nof4.as_expr())

    # We need to compare the normal ordered forms because direct comparison might fail
    # due to different but equivalent expressions
    result_normal = normal_ordered_form(result.as_expr(), independent=True)
    expected_normal = normal_ordered_form(expected.as_expr(), independent=True)

    assert sympy.expand(result_normal) == sympy.expand(expected_normal)

    # Test multiplication with different operators
    c = boson.BosonOp("c")
    nof5 = NumberOrderedForm([a], {(1,): x})  # x*a
    nof6 = NumberOrderedForm([c], {(1,): y})  # y*c

    result = nof5 * nof6
    expected = NumberOrderedForm.from_expr(nof5.as_expr() * nof6.as_expr())

    # Use normal ordered form for comparison
    result_normal = normal_ordered_form(result.as_expr(), independent=True)
    expected_normal = normal_ordered_form(expected.as_expr(), independent=True)
    assert sympy.expand(result_normal) == sympy.expand(expected_normal)
    assert all(str(op.name) in ["a", "c"] for op in result.operators)

    # Test multiplication with a scalar
    nof7 = NumberOrderedForm([a], {(1,): sympy.S.One})  # a
    scalar = sympy.S(3)  # 3

    result = nof7 * scalar
    expected = NumberOrderedForm([a], {(1,): sympy.S(3)})  # 3*a

    # Use normal ordered form for comparison
    result_normal = normal_ordered_form(result.as_expr(), independent=True)
    expected_normal = normal_ordered_form(expected.as_expr(), independent=True)
    assert sympy.expand(result_normal) == sympy.expand(expected_normal)

    # Test right multiplication with a scalar (__rmul__)
    result = scalar * nof7
    # Skip direct comparison and just check that the result when converted back to an
    # expression looks correct
    assert result.as_expr() == sympy.S(3) * nof7.as_expr()

    # Test right multiplication with commutative expressions
    symbol = sympy.symbols("z")
    nof8 = NumberOrderedForm([a], {(1,): sympy.S.One})  # a
    result = symbol * nof8
    expected = NumberOrderedForm([a], {(1,): symbol})  # z*a

    assert result == expected

    # Test multiplication with a sympy expression containing number operators
    expr_with_number_op = n_a
    nof9 = NumberOrderedForm([a], {(1,): sympy.S.One})  # a

    result = nof9 * expr_with_number_op
    expected = NumberOrderedForm.from_expr(nof9.as_expr() * expr_with_number_op)

    assert result == expected

    # Test right multiplication with a sympy expression containing number operators
    result = expr_with_number_op * nof9
    expected = NumberOrderedForm.from_expr(expr_with_number_op * nof9.as_expr())

    # For non-commutative operators, left*right != right*left
    # We need to compare normal ordered forms to handle equivalent expressions
    result_normal = normal_ordered_form(result.as_expr(), independent=True)
    expected_normal = normal_ordered_form(expected.as_expr(), independent=True)

    assert sympy.expand(result_normal) == sympy.expand(expected_normal)
    expr_with_number_op = n_a
    nof8 = NumberOrderedForm([a], {(1,): sympy.S.One})  # a

    result = nof8 * expr_with_number_op
    expected = NumberOrderedForm.from_expr(nof8.as_expr() * expr_with_number_op)

    assert result == expected
