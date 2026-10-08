"""Exact finite-basis contracts for embedding frames and structural lifting."""

from itertools import product

import pytest
import sympy as sp
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock.number_ordered_form import LadderOp
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.number_ordered_form import NumberOrderedForm as F
from pymablock.operator_embedding import Embedding
from pymablock.tests.second_quantization_helpers import nof_matrix


def assert_matrix_equal(actual, expected):
    assert (actual - expected).applyfunc(sp.simplify) == sp.zeros(*expected.shape)


@pytest.mark.parametrize("reference", [0, 1])
def test_normalized_generator_lift_and_attachment_contracts(reference):
    a, s = BosonOp("a"), SigmaMinus("s")
    e = Embedding({s: a**2 / sp.sqrt(N(a) * (N(a) - 1))}, reference={a: reference})
    w = sp.eye(6)[:, [reference, reference + 2]]
    attached = e._attach(1, 1)

    def target(x):
        return nof_matrix(x, [range(6)])

    source = nof_matrix
    assert_matrix_equal(source(attached.adjoint() * attached), sp.eye(2))
    assert_matrix_equal(target(attached * attached.adjoint()), w * w.T)
    for S in (1, s, s.adjoint(), N(s), s + s.adjoint()):
        value = F.from_expr(S, operators=(s,))
        assert_matrix_equal(target(e._lift(value)), w * source(value) * w.T)
        assert_matrix_equal(target((attached * value).target) * w, w * source(value))
    for X in (a, a.adjoint(), a**2, a.adjoint() * a**2, N(a), sp.sqrt(N(a) + 1)):
        x = e._convert_operator(X)
        assert_matrix_equal(source(e.restrict(x)), w.T * target(x) * w)
        for Y in (1, a, a.adjoint()):
            y = e._convert_operator(Y)
            assert_matrix_equal(
                target((x * attached) * (attached.adjoint() * y)),
                target(x) * w * w.T * target(y),
            )
            assert_matrix_equal(
                source((attached.adjoint() * x) * (y * attached)),
                w.T * target(x * y) * w,
            )


def test_spectator_transfer_and_matrix_source_contracts():
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    e = Embedding({s: a}, reference=[{a: 0, b: 0}, {a: 0, b: 1}])
    states = list(product(range(4), range(3)))
    basis = sp.eye(len(states))
    columns = [basis[:, [states.index((q, r)) for q in (0, 1)]] for r in (0, 1)]
    w = columns[0].row_join(columns[1])
    frame = e._frame_columns(1)

    def target(x):
        return nof_matrix(x, [range(4), range(3)])

    def source(matrix):
        return sp.BlockMatrix(
            [
                [nof_matrix(x) if isinstance(x, F) else x * sp.eye(2) for x in row]
                for row in matrix.tolist()
            ]
        ).as_explicit()

    assert_matrix_equal(target(e._transfers[1]) * columns[0], columns[1])
    assert_matrix_equal(source(frame.adjoint() * frame), sp.eye(4))
    assert_matrix_equal(target((frame * frame.adjoint())[0, 0]), w * w.T)
    for X in (a, b, b.adjoint(), a.adjoint() * b, N(a), N(b)):
        x = e._convert_operator(X)
        assert_matrix_equal(source(e.restrict(x)), w.T * target(x) * w)
    for i, j in product(range(2), repeat=2):
        for S in (1, s, s.adjoint(), N(s)):
            matrix = sp.zeros(2)
            matrix[i, j] = F.from_expr(S, operators=(s,))
            result = frame * matrix
            actual = sp.BlockMatrix(
                [
                    [
                        sp.zeros(len(states), 2)
                        if x == 0
                        else target(x.target) * columns[0]
                        for x in result.tolist()[0]
                    ]
                ]
            ).as_explicit()
            assert_matrix_equal(actual, w * source(matrix))


def test_lift_preserves_fermion_order_and_number_coefficients():
    c, d, f, g = map(FermionOp, ("c", "d", "f", "g"))
    e = Embedding({f: d, g: c}, reference={c: 0, d: 0})
    # f† g† maps to d† c†, so the doubly occupied column has a minus sign.
    w = sp.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]])
    for S in (
        f.adjoint() * g,
        g.adjoint() * f,
        f.adjoint() * g.adjoint(),
        f * g,
        f.adjoint() * (N(g) + sp.I * N(f)) * g,
        N(f) + 2 * N(g),
    ):
        value = F.from_expr(S, operators=(f, g))
        assert_matrix_equal(nof_matrix(e._lift(value)), w * nof_matrix(value) * w.T)


def test_lift_bilateral_numbers_and_symbolic_powers():
    a, b = LadderOp("a"), LadderOp("b")
    e = Embedding({b: a, N(b): N(a) + 2}, reference={a: -2})
    for S in (b, b.adjoint(), N(b), b.adjoint() * N(b) * b):
        value = F.from_expr(S, operators=(b,))
        assert_matrix_equal(
            nof_matrix(e._lift(value), [range(-4, 3)]),
            nof_matrix(value, [range(-2, 5)]),
        )
    for mode_type in (BosonOp, LadderOp):
        target, source = mode_type("target"), mode_type("source")
        generators = {source: target}
        if mode_type is LadderOp:
            generators[N(source)] = N(target)
        embedding = Embedding(generators, reference={target: 0})
        power = sp.Symbol("power", integer=True, positive=True)
        for shift in (power, -power):
            value = F((source,), {(shift,): sp.S.One}, validate=False)
            assert embedding._lift(value).terms == {(shift,): sp.S.One}


def test_existing_nof_conversion_is_structural(monkeypatch):
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    e = Embedding({s: a}, reference={a: 0, b: 0})
    value = F.from_expr(sp.sqrt(N(a) + 1) * a.adjoint(), operators=(a,))
    monkeypatch.setattr(F, "as_expr", lambda _self: pytest.fail("NOF round trip"))
    result = e._convert_operator(value)
    assert result.operators == (a, b)
    assert result.terms == {(-1, 0): next(iter(value.terms.values()))}
