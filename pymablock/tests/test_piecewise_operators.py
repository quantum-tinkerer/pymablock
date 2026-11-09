import pytest
import sympy as s
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOperator,
    NumberOrderedForm,
    _equal_value_indicator,
    _simplify_on_fixed_numbers,
)
from pymablock.tests.second_quantization_helpers import nof_matrix


def test_calculated_vacuum_projector_roundtrip():
    """An embedding calculation exports an ordered operator-valued Piecewise."""
    from pymablock import block_diagonalize
    from pymablock.operator_embedding import Embedding

    a = BosonOp("a")
    embedding = Embedding({}, reference={a: 0})
    blocks, *_ = block_diagonalize(
        [NumberOperator(a) + 1, a + a.adjoint()], subspace_eigenvectors=embedding
    )
    # The complement block contains the calculated vacuum projector.
    value = blocks[1, 1, 1]
    if isinstance(value, s.MatrixBase):
        value = value[0, 0]
    assert value.has(s.Piecewise)
    assert NumberOrderedForm.from_expr(value.as_expr()) == value
    projector = s.Piecewise((1, s.Eq(NumberOperator(a), 0)), (0, True))
    direct = NumberOrderedForm.from_expr(a) * NumberOrderedForm.from_expr(projector)
    assert NumberOrderedForm.from_expr(a * projector) == direct


def test_patch_keeps_scalar_piecewise_commutative():
    """The global patch orders operator-valued conditions only."""
    from pymablock.number_ordered_form import _install_piecewise_patch

    _install_piecewise_patch()
    _install_piecewise_patch()
    n, a = s.symbols("N a", commutative=False)
    p = s.Piecewise((1, s.Eq(n, 0)), (0, True))
    assert a * p - p * a != 0
    assert p.subs(n, 0) == 1
    assert p.subs(n, 1) == 0
    scalar = p.subs(n, s.Symbol("x"))
    assert scalar.is_commutative is True
    assert a * scalar - scalar * a == 0

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


def test_projector_coefficient_arithmetic():
    a, b = BosonOp("a"), BosonOp("b")
    n, m = NumberOperator(a), NumberOperator(b)
    p0, p2 = (
        NumberOrderedForm.from_expr(s.Piecewise((1, s.Eq(n, k)), (0, True)), [a, b])
        for k in (0, 2)
    )
    spectator = NumberOrderedForm.from_expr((1 + m) ** 8, [a, b])
    assert (p0 + p2) * (n + 1) == p0 + 3 * p2
    assert not p0 * p2
    assert not a * p0
    assert p0 * a
    # Reduction preserves both an unprojected background and unrelated factors.
    expression = spectator + (n + 1) * (p0 + p2) * spectator
    assert expression == spectator + (p0 + 3 * p2) * spectator
    assert nof_matrix(expression, [range(3), range(2)]) == s.diag(2, 512, 1, 256, 4, 1024)


@pytest.mark.parametrize("operator", [BosonOp("a"), LadderOp("l")])
def test_projector_respects_occupation_domain(operator):
    n = NumberOperator(operator)
    p = NumberOrderedForm.from_expr(s.Piecewise((1, s.Eq(n + 2, 1)), (0, True)))
    if isinstance(operator, BosonOp):
        assert not p
    else:
        assert p * (n + 3) == 2 * p
        assert nof_matrix(p, [range(-2, 2)]) == s.diag(0, 1, 0, 0)


@pytest.mark.parametrize("power", [1, 2])
def test_fixed_number_sums_stay_factored(power):
    numbers = s.symbols("n:8", integer=True, nonnegative=True)
    coefficient = s.prod(
        _equal_value_indicator(n, 0) + _equal_value_indicator(n, 1) for n in numbers
    )
    result = _simplify_on_fixed_numbers(coefficient**power, numbers, numbers)
    assert result == coefficient
    assert s.count_ops(result) < 8 * 8


def test_affine_fixed_number_sum_cancels():
    a = BosonOp("a")
    n = NumberOperator(a)
    g = s.Symbol("g")
    p0, p1 = (_equal_value_indicator(n, v) for v in (0, 1))
    expression = g * (p0 + p1) - g * p0 - g * p1
    assert expression != 0  # SymPy leaves this scalar sum undistributed.
    result = NumberOrderedForm.from_expr(expression)
    assert result == 0
    assert result.is_zero is True


@pytest.mark.parametrize("pole", [s.zoo, s.oo])
def test_affine_fixed_number_pole_stays_in_its_sector(pole):
    n = s.Symbol("n", integer=True, nonnegative=True)
    coefficient = pole * _equal_value_indicator(n, 0)
    assert _simplify_on_fixed_numbers(coefficient, (n,), (n,)) == coefficient


def test_fixed_number_background_indicators_cancel():
    n = s.Symbol("n", integer=True, nonnegative=True)
    g = s.Symbol("g")
    indicators = [_equal_value_indicator(n, v) for v in (0, 1)]
    nested = [
        s.Piecewise((1 + delta, s.Eq(n, v)), (0, True))
        for v, delta in enumerate(indicators)
    ]
    # Setting the inner indicators to zero creates indicators in the background.
    coefficient = sum(g * delta for delta in nested) - 2 * g * sum(indicators)
    assert _simplify_on_fixed_numbers(coefficient, (n,), (n,)) == 0
