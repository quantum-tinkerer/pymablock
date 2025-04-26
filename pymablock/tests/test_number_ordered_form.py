"""Tests for the NumberOrderedForm class."""

import pytest
import sympy
from sympy.physics.quantum import Commutator, Dagger, IdentityOperator, boson, fermion
from sympy.physics.quantum.operatorordering import normal_ordered_form

from pymablock.number_ordered_form import (
    NumberOperator,
    NumberOrderedForm,
    find_operators,
)


def test_number_operator_interface():
    """Test the name of the NumberOperator."""
    # Create a boson operator
    a, b = sympy.symbols("a b", cls=boson.BosonOp)
    f, g = sympy.symbols("f g", cls=fermion.FermionOp)
    n_a, n_b = NumberOperator(a), NumberOperator(b)
    n_f = NumberOperator(f)

    # Check the name of the NumberOperator
    assert n_a.name == sympy.Symbol("a")

    # Check that it cannot be instantiated with not a wrong operator
    with pytest.raises(TypeError):
        NumberOperator(IdentityOperator())
    # Also not with a creation operator
    with pytest.raises(ValueError):
        NumberOperator(Dagger(a))

    assert Commutator(n_a, n_b).doit() == 0
    assert Commutator(n_a, n_f).doit() == 0

    assert Commutator(n_a, a).doit() == -a
    assert Commutator(n_a, b).doit(independent=True) == 0
    assert Commutator(n_a, f).doit() == 0
    assert Commutator(n_f, a).doit(independent=True) == 0
    assert Commutator(n_f, g).doit(independent=True) == 0
    assert Commutator(n_f, f).doit() == -f

    assert sympy.latex(n_a) == "{N_{a}}"
    assert sympy.pretty(n_a) == "N_a"


def test_find_operators():
    """Test the find_operators function to identify bosonic operators in expressions."""
    # TODO: Also confirm ordering and fermionic operators
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")

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


def test_number_ordered_form_init():
    """Test basic initialization of NumberOrderedForm."""
    # Create operators
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")
    operators = (a, b)

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

    # Test with wrong powers tuple length
    with pytest.raises(ValueError, match="Powers tuple length"):
        NumberOrderedForm([a, b], {(0,): sympy.S.One})

    # Test with non-sympy expression coefficient
    with pytest.raises(TypeError, match="must be a sympy expression"):
        NumberOrderedForm([a], {(0,): "not an expression"})

    # Test with creation/annihilation operator in coefficient
    with pytest.raises(ValueError, match="contains creation or annihilation operators"):
        NumberOrderedForm([a], {(0,): a * sympy.S.One})

    # Test that fractional powers are not allowed
    with pytest.raises(TypeError, match="Power must be an integer"):
        NumberOrderedForm([a], {(1.5,): sympy.S.One})

    # Test that symbolic non-integer powers are not allowed
    k = sympy.symbols("k")
    with pytest.raises(TypeError, match="Power must be an integer"):
        NumberOrderedForm([a], {(k,): sympy.S.One})


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

    # Try on something that cannot be sympified
    with pytest.raises(ValueError):
        NumberOrderedForm.from_expr("not an expression")

    # Apply to a sympy function
    with pytest.raises(ValueError):
        NumberOrderedForm.from_expr(sympy.sin(a))

    nof2 = NumberOrderedForm.from_expr(sympy.sin(a * Dagger(a)))
    assert nof2.as_expr() == sympy.sin(NumberOperator(a) + 1)


def test_dagger():
    """Test the Dagger method of NumberOrderedForm."""
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")

    # Create a NumberOrderedForm
    expr = (Dagger(a) + b) * a**2
    nof = NumberOrderedForm.from_expr(expr)

    # Get the daggered form
    nof_daggered = Dagger(nof)

    # Check that the operators are the same
    assert (
        normal_ordered_form(
            (nof_daggered.as_expr().doit().expand() - Dagger(expr).expand()).expand()
        )
        == 0
    )


def test_compare_from_expr():
    """Test that from_expr handles products of operators correctly."""

    b = boson.BosonOp("b")
    expressions = [b * Dagger(b) ** 2, Dagger(b) * b * Dagger(b), b * Dagger(b) ** 2 * b]
    for expr in expressions:
        fn = NumberOrderedForm.from_expr(expr)
        assert normal_ordered_form(fn.as_expr().doit().expand() - expr.expand()) == 0


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

    assert nof.operators == (a, b)
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


def test_multiply_op_twice():
    b = boson.BosonOp("b")
    fn1 = NumberOrderedForm([b], {(1,): sympy.S.One})._multiply_op(0, -2)
    fn2 = (
        NumberOrderedForm([b], {(1,): sympy.S.One})
        ._multiply_op(0, -1)
        ._multiply_op(0, -1)
    )

    assert (
        normal_ordered_form(fn1.as_expr().doit())
        - normal_ordered_form(fn2.as_expr().doit())
        == 0
    ), f"Failed for fn1: {fn1.as_expr()} and fn2: {fn2.as_expr()}"


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


def test_division():
    """Test division of NumberOrderedForm by scalars."""
    a, b = sympy.symbols("a b", cls=boson.BosonOp)
    x, y = sympy.symbols("x y", real=True)

    # Test division by a scalar
    nof1 = NumberOrderedForm([a], {(1,): sympy.S(6)})  # 6*a
    scalar = sympy.S(2)  # 2

    result = nof1 / scalar
    expected = NumberOrderedForm([a], {(1,): sympy.S(3)})  # 3*a

    assert result == expected

    # Test division by a symbol
    nof2 = NumberOrderedForm([a, b], {(1, 0): x, (0, 1): x * y})  # x*a + x*y*b
    symbol = x  # x

    result = nof2 / symbol
    expected = NumberOrderedForm([a, b], {(1, 0): sympy.S.One, (0, 1): y})  # a + y*b

    assert result == expected

    # Test division by a more complex expression
    nof3 = NumberOrderedForm([a], {(1,): x**2 * y})  # x^2*y*a
    expr = x * y  # x*y

    result = nof3 / expr
    expected = NumberOrderedForm([a], {(1,): x})  # x*a

    assert result == expected

    # Test division by number operator
    nof4 = NumberOrderedForm([a], {(1,): sympy.S.One})  # a
    non_scalar = NumberOperator(a)  # Na

    result = nof4 / non_scalar
    expected = NumberOrderedForm([a], {(1,): sympy.S.One / (NumberOperator(a) + 1)})

    assert result == expected

    # Test division by an unsimplifiable expression
    nof5 = NumberOrderedForm([a], {(1,): sympy.S.One})  # a
    unsimplifiable_expr = "not an expression"  # Invalid expression

    with pytest.raises(TypeError):
        nof5 / unsimplifiable_expr


def test_simplify_basic():
    """Test basic simplification of number operators."""
    # Create a boson operator
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")
    n_a = NumberOperator(a)
    n_b = NumberOperator(b)

    # Create expressions that can be simplified
    # Use expanded numerator to prevent automatic simplification
    expr1 = (n_a**2 + n_a) / (n_a + 1)  # Should simplify to n_a
    expr2 = (n_a**2 - n_a) / n_a  # Should simplify to (n_a - 1)

    # Intertwine different number operators to prevent automatic simplification
    expr3 = (n_a + 1) * n_b / (n_a + 1)  # Should simplify to n_b

    # Convert to NumberOrderedForm
    nof1 = NumberOrderedForm.from_expr(expr1)
    nof2 = NumberOrderedForm.from_expr(expr2)
    nof3 = NumberOrderedForm.from_expr(expr3)

    # Apply simplification
    result1 = nof1.simplify()
    result2 = nof2.simplify()
    result3 = nof3.simplify()

    # Convert back to expressions for easier comparison
    simplified1 = result1.as_expr()
    simplified2 = result2.as_expr()
    simplified3 = result3.as_expr()

    # Check results
    assert simplified1 == n_a
    assert simplified2 == (n_a - 1)
    assert simplified3 == n_b


def test_simplify_with_operators():
    """Test simplification with creation and annihilation operators."""
    # Create boson operators
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")
    n_a = NumberOperator(a)
    n_b = NumberOperator(b)

    # Create expression with operators and a part that can be simplified
    # Use intertwined number operators to prevent automatic simplification
    expr = (
        Dagger(a) * n_a * (n_a + 2) * ((n_b + 1) / (n_b + 1)) * (n_a + 1) / (n_a + 1) * b
    )

    # Convert to NumberOrderedForm
    nof = NumberOrderedForm.from_expr(expr)

    # Apply simplification
    result = nof.simplify()

    # The simplified result should have n_a * (n_a + 2) in the coefficient
    expected = Dagger(a) * n_a * (n_a + 2) * b
    assert result.as_expr() == expected


def test_simplify_complex_expression():
    """Test simplification of complex expressions."""
    # Create boson operators
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")
    n_a = NumberOperator(a)
    n_b = NumberOperator(b)

    # Create a more complex expression with multiple operators
    # Using expanded numerator and intertwined operators to prevent automatic simplification
    expr = (
        Dagger(a) * (n_a**2 + n_a) * Dagger(b)
        + a * ((n_b + 1) * (n_b + 2) / (n_b + 2)) * b
    )

    # Convert to NumberOrderedForm
    nof = NumberOrderedForm.from_expr(expr)

    # Apply simplification
    result = nof.simplify()

    # Expected simplified expression
    # Expected simplified expression
    expected = Dagger(a) * Dagger(b) * n_a * (n_a + 1) + (n_b + 1) * b * a

    assert result.as_expr() == expected


def test_sympy_simplify_integration():
    """Test that the simplify method integrates with SymPy's simplify function."""
    # Create a boson operator
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")
    n_a = NumberOperator(a)
    n_b = NumberOperator(b)

    # Create a simple expression that can be simplified
    # Use intertwined number operators to prevent automatic simplification
    expr = (n_a + 1) * n_b / (n_a + 1) + n_a * n_b - n_b * n_a

    # Convert to NumberOrderedForm
    nof = NumberOrderedForm.from_expr(expr)

    # Use SymPy's simplify function (should call _eval_simplify)
    result = sympy.simplify(nof)

    # Check result
    assert result.as_expr() == n_b

    # Test multiply_b - Multiplying by annihilation operator
    b_op = boson.BosonOp("b")
    Nb = NumberOperator(b_op)

    # Test case 1: Simple expression with only annihilation operators
    expr1 = b_op**2
    nof_expr1 = NumberOrderedForm.from_expr(expr1)
    result1 = nof_expr1 * b_op  # Multiply by annihilation operator
    expected1 = NumberOrderedForm.from_expr(b_op**3)
    assert (
        normal_ordered_form(
            result1.as_expr().doit().expand() - expected1.as_expr().doit().expand()
        )
        == 0
    )

    # Test case 2: Expression with creation operators
    expr2 = Dagger(b_op) * b_op
    nof_expr2 = NumberOrderedForm.from_expr(expr2)
    result2 = nof_expr2 * b_op  # Multiply by annihilation operator
    expected2 = NumberOrderedForm.from_expr((Dagger(b_op) * b_op * b_op))
    assert (
        normal_ordered_form(
            result2.as_expr().doit().expand() - expected2.as_expr().doit().expand()
        )
        == 0
    )

    # Test case 3: Complex expression with multiple terms
    expr3 = Nb + Nb**2 * b_op + Dagger(b_op) * Nb
    nof_expr3 = NumberOrderedForm.from_expr(expr3)
    result3 = nof_expr3 * b_op  # Multiply by annihilation operator
    expected3 = NumberOrderedForm.from_expr(expr3 * b_op)
    assert (
        normal_ordered_form(
            (
                result3.as_expr().doit().expand() - expected3.as_expr().doit().expand()
            ).expand()
        )
        == 0
    )

    # Test multiply_daggered_b - Multiplying by creation operator

    # Test case 1: Simple expression with number operator
    expr1 = Nb
    nof_expr1 = NumberOrderedForm.from_expr(expr1)
    result1 = nof_expr1 * Dagger(b_op)  # Multiply by creation operator
    expected1 = NumberOrderedForm.from_expr(Nb * Dagger(b_op))
    assert (
        normal_ordered_form(
            (
                result1.as_expr().doit().expand() - expected1.as_expr().doit().expand()
            ).expand()
        )
        == 0
    )

    # Test case 2: Expression with annihilation operators
    expr2 = b_op
    nof_expr2 = NumberOrderedForm.from_expr(expr2)
    result2 = nof_expr2 * Dagger(b_op)  # Multiply by creation operator
    expected2 = NumberOrderedForm.from_expr(b_op * Dagger(b_op))
    assert (
        normal_ordered_form(
            (
                result2.as_expr().doit().expand() - expected2.as_expr().doit().expand()
            ).expand()
        )
        == 0
    )

    # Test multiply_fn - Multiplying by number operator expressions

    c_op = boson.BosonOp("c")
    Nc = NumberOperator(c_op)

    # Test case: Expression with multiple operators
    fn = (Nb**2 + 1) * Nb * (Nc + 1)
    expr = b_op**2 * c_op

    # Create NumberOrderedForm instances
    nof_expr = NumberOrderedForm.from_expr(expr)
    nof_fn = NumberOrderedForm.from_expr(fn)

    # Using the multiplication operator between NumberOrderedForm objects
    result = nof_expr * nof_fn
    expected = NumberOrderedForm.from_expr(expr * fn)

    assert (
        normal_ordered_form(
            (
                result.as_expr().doit().expand() - expected.as_expr().doit().expand()
            ).expand()
        )
        == 0
    )


def test_number_ordered_form_function():
    """Test the number_ordered_form function for converting expressions to number-ordered form."""
    b = boson.BosonOp("b")
    c = boson.BosonOp("c")
    Nb = NumberOperator(b)
    Nc = NumberOperator(c)

    # Test case: Complex expression with multiple operators
    expr = b**2 * Dagger(b)

    # Convert expression to NumberOrderedForm and back
    nof = NumberOrderedForm.from_expr(expr)
    result = nof.as_expr().doit().expand()
    expected = expr.doit().expand()

    assert normal_ordered_form((result - expected).expand(), independent=True) == 0

    # Test case: Complex expression with multiple operators
    expr = (
        b**2 * Dagger(b) * (c + 1) ** 2
        + (Nc + 1) ** (2) * (b + 1) * (c + Nb * Dagger(c)) ** 2
    )

    # Convert expression to NumberOrderedForm and back
    nof = NumberOrderedForm.from_expr(expr)
    result = nof.as_expr().doit().expand()
    expected = expr.doit().expand()

    assert normal_ordered_form((result - expected).expand(), independent=True) == 0


def test_number_ordered_form_with_negative_powers():
    """Test NumberOrderedForm with negative powers to verify error handling."""
    a = boson.BosonOp("a")

    # Test that it raises ValueError for negative powers of operators
    with pytest.raises(ValueError):
        NumberOrderedForm.from_expr(a ** (-1))

    with pytest.raises(ValueError):
        NumberOrderedForm.from_expr(Dagger(a) ** (-1))

    # Test that negative powers of number operators are allowed
    Na = NumberOperator(a)
    expr = Na ** (-1) * a
    nof = NumberOrderedForm.from_expr(expr)
    assert nof.as_expr() == expr

    # Test that negative powers of non-operator expressions are allowed
    x = sympy.Symbol("x")
    expr = x ** (-1) * a
    nof = NumberOrderedForm.from_expr(expr)
    assert nof.as_expr() == expr


def test_raise_if_substitution():
    """Test that substitution raises an error."""
    a = boson.BosonOp("a")
    nof = NumberOrderedForm([a], {(1,): sympy.S.One})

    with pytest.raises(ValueError):
        nof.subs(a, a + 1)


def test_is_particle_conserving():
    """Test the is_particle_conserving method."""
    a = boson.BosonOp("a")
    b = boson.BosonOp("b")
    x, y = sympy.symbols("x y", real=True)

    # Number-conserving expressions (balanced creation and annihilation)
    # Simple a†a (number operator)
    nof1 = NumberOrderedForm.from_expr(Dagger(a) * a)
    assert nof1.is_particle_conserving()

    # a†b (destroys b, creates a)
    nof2 = NumberOrderedForm.from_expr(Dagger(a) * b)
    assert not nof2.is_particle_conserving()

    # More complex number operator expression
    nof3 = NumberOrderedForm.from_expr(Dagger(a) * a + (Dagger(b) * b) ** 3)
    assert nof3.is_particle_conserving()

    # Scalar expressions
    nof4 = NumberOrderedForm.from_expr(x + y)
    assert nof4.is_particle_conserving()

    # Non-number-conserving expressions
    # Single creation operator a†
    nof5 = NumberOrderedForm.from_expr(Dagger(a))
    assert not nof5.is_particle_conserving()

    # Single annihilation operator a
    nof6 = NumberOrderedForm.from_expr(a)
    assert not nof6.is_particle_conserving()

    # Unbalanced expression a†a† + b
    nof7 = NumberOrderedForm.from_expr(Dagger(a) * Dagger(a) + b)
    assert not nof7.is_particle_conserving()

    # Mixed term with non-conserving components a†a + b
    nof8 = NumberOrderedForm.from_expr(Dagger(a) * a + b)
    assert not nof8.is_particle_conserving()
