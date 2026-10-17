"""Rectangular map contracts beyond square number-ordered algebra."""

import pickle

import pytest
import sympy as sp
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock.number_ordered_form import NumberOrderedForm as F
from pymablock.operator_embedding import Embedding

from .second_quantization_helpers import nof_matrix


@pytest.mark.parametrize("adjoint", [False, True])
def test_scalars_preserve_map_target(adjoint):
    """Scalar arithmetic cannot introduce indicators outside the map's domain."""
    a, q = BosonOp("a"), SigmaMinus("q")
    embedding = Embedding({q: a}, reference={a: 0})
    w = embedding._retained_frame(1)[0, 0]
    value = F.from_expr(a + a.adjoint()) * w
    if adjoint:
        value = value.adjoint()
    c = sp.Symbol("c")
    for scalar in (2, sp.I, c, F.from_expr(2)):
        for result in (value * scalar, scalar * value):
            assert result.target == value.target * scalar
            assert result.embedding == embedding
            assert result.side == value.side
    assert (value / 2).target == value.target / 2
    assert value * 0 == 0
    assert (value - value).is_zero
    assert value.subs(c, 0) == value
    assert not isinstance(value, F)
    with pytest.raises(TypeError, match="square"):
        nof_matrix(value)


@pytest.mark.parametrize("adjoint", [False, True])
def test_map_expression_contracts(adjoint):
    """Symbolic reconstruction and printers retain W and its orientation."""
    a, q = BosonOp("a"), SigmaMinus("q")
    embedding = Embedding({q: a}, reference={a: 0})
    w = embedding._retained_frame(1)[0, 0]
    c = sp.Symbol("c", real=True)
    value = F.from_expr(c * a) * w
    if adjoint:
        value = value.adjoint()
    for restored in (value.func(*value.args), pickle.loads(pickle.dumps(value))):
        assert restored == value
        assert hash(restored) == hash(value)
    assert str(value) == str(value.as_expr())
    assert sp.pretty(value) == sp.pretty(value.as_expr())
    assert sp.latex(value) == sp.latex(value.as_expr())
    assert value.adjoint().adjoint() == value
    assert value == value.as_expr()
    assert hash(value) == hash(value.as_expr())
    assert value.subs(c, 0) == 0
    assert value.xreplace({c: 0}) == 0
    assert value.xreplace({q: SigmaMinus("z")}).as_expr() == value.as_expr().xreplace(
        {q: SigmaMinus("z")}
    )
    with pytest.raises(ValueError, match="opposite matching"):
        value * value
