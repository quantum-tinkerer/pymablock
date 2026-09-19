import pytest
import sympy as s
from sympy.core.assumptions import _assume_defined
from sympy.core.cache import clear_cache
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm
from pymablock.tests.second_quantization_helpers import nof_matrix


class Unknown(s.Expr):
    is_commutative = None


def condition(kind):
    n = s.Symbol("N", commutative=False)
    x = s.Symbol("x", real=True)
    a = s.MatrixSymbol("A", 2, 2)
    return {
        "operator": s.Eq(n, 0),
        "scalar": x > 0,
        "trace": s.Eq(s.Trace(a), 0),
        "determinant": s.Eq(s.Determinant(a), 0),
        "and": s.And(s.Eq(n, 0), x > 0),
        "or": s.Or(s.Eq(n, 0), x > 0),
        "predicate": s.Q.zero(n),
        "unknown": s.Eq(Unknown(), 0, evaluate=False),
    }[kind]


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("operator", False),
        ("scalar", True),
        ("trace", True),
        ("determinant", True),
        ("and", False),
        ("or", False),
        ("predicate", False),
        ("unknown", None),
    ],
)
@pytest.mark.parametrize("first", sorted(_assume_defined))
def test_query_order(kind, expected, first):
    clear_cache()
    p = s.Piecewise((1, condition(kind)), (0, True))
    getattr(p, "is_" + first)
    assert p.is_commutative is expected
    if expected is False:
        assert p.is_integer is False
        assert p.is_real is False
    elif expected is True:
        assert p.is_integer is True
        assert p.is_nonnegative is True


def test_arithmetic_and_substitution():
    n, a = s.symbols("N a", commutative=False)
    p = s.Piecewise((1, s.Eq(n, 0)), (0, True))
    assert a * p - p * a != 0
    assert p.subs(n, 0) == 1
    assert p.subs(n, 1) == 0
    scalar = p.subs(n, s.Symbol("x"))
    assert scalar.is_commutative is True
    assert a * scalar - scalar * a == 0


def test_noncommutative_branch():
    x = s.Symbol("x", real=True)
    a = s.Symbol("a", commutative=False, finite=True)
    p = s.Piecewise((a, x > 0), (2 * a, True))
    assert p.is_commutative is False
    assert p.is_finite is True


def test_nof_piecewise_roundtrip():
    a = BosonOp("a")
    n = NumberOperator(a)
    p = NumberOrderedForm.from_expr(s.Piecewise((1, s.Eq(n, 0)), (0, True)), [a])
    for form in (p, a * p, p * a, a * p - p * a):
        assert NumberOrderedForm.from_expr(form.as_expr(), [a]) == form
    assert a * p - p * a != 0
    assert nof_matrix(p, [range(4)]) == s.diag(1, 0, 0, 0)
    assert nof_matrix(a * p, [range(4)]).is_zero_matrix
    assert nof_matrix(p * a, [range(4)])[0, 1] == 1
    with pytest.raises(ValueError, match="diagonal"):
        NumberOrderedForm.from_expr(s.Piecewise((a, s.Eq(n, 0)), (0, True)))
