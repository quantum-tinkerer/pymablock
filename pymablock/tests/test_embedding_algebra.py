"""Algebra and SymPy reconstruction of rectangular embedding attachments."""

import pickle

import pytest
import sympy as s
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.operator_embedding import Embedding


def attach(value, e, side):
    return (
        NumberOrderedForm.from_expr(value) * e
        if side == 1
        else s.adjoint(e) * NumberOrderedForm.from_expr(value)
    )


def test_attachment_is_structural():
    a = BosonOp("a")
    e = Embedding(reference=[{a: 0}])
    ket = attach(1, e, 1)
    bra = ket.adjoint()
    plain = NumberOrderedForm.from_expr(1, [a])
    assert ket != plain and plain != ket and ket != bra
    assert len({ket, bra, plain}) == 3
    for value in (ket, bra):
        assert value.func(*value.args) == value
        assert pickle.loads(pickle.dumps(value)) == value
        assert value.adjoint().adjoint() == value
        assert NumberOrderedForm.from_expr(value.as_expr()) == value
    assert NumberOrderedForm((a,), {}) + ket == ket


def test_common_reference_matrix():
    a = BosonOp("a")
    e = Embedding(reference=[{a: 0}])
    w = s.ImmutableMatrix(
        [
            [attach(1, e, 1), 0, attach(a.adjoint() ** 3 / s.sqrt(6), e, 1)],
            [0, attach(a.adjoint() ** 2 / s.sqrt(2), e, 1), 0],
        ]
    )
    assert w.adjoint() * w == s.eye(3)
    target = s.ImmutableMatrix([[1, 2, s.I], [3, 0, 1], [2, -s.I, 4]])
    assert w.adjoint() * (w * target) == target
    assert all(
        not isinstance(x, NumberOrderedForm) or x.embedding is None
        for x in w * w.adjoint()
    )


def test_coefficients_survive_substitution():
    a = BosonOp("a")
    g = s.Symbol("g")
    e = Embedding(reference=[{a: 0}])
    ket = attach(g * a.adjoint(), e, 1)
    assert ket.subs(g, 2) == attach(2 * a.adjoint(), e, 1)
    assert ket.xreplace({g: 2}) == attach(2 * a.adjoint(), e, 1)


def test_generator_attachment_composition():
    from sympy.physics.quantum.pauli import SigmaMinus

    from pymablock.number_ordered_form import NumberOperator as N

    a, q = BosonOp("a"), SigmaMinus("q")
    e = Embedding({q: a}, reference={a: 0})
    w = NumberOrderedForm.from_expr(e)
    x = NumberOrderedForm.from_expr(a + a.adjoint() + N(a))
    target = NumberOrderedForm.from_expr(q + q.adjoint())
    assert (w.adjoint() * x * w - e.restrict(x)).is_zero
    assert (w.adjoint() * (w * target) - target).is_zero
    assert (w.adjoint() * w).embedding is None
    for value in (x * w, w.adjoint() * x):
        assert NumberOrderedForm.from_expr(value.as_expr()) == value
        assert value.func(*value.args) == value
    with pytest.raises(ValueError, match="power"):
        (NumberOrderedForm.from_expr(q) * w) ** 2


@pytest.mark.parametrize("generator", [False, True])
def test_cancellation_before_resonance_check(generator):
    from sympy.physics.quantum.pauli import SigmaMinus

    from pymablock import block_diagonalize
    from pymablock.number_ordered_form import NumberOperator as N

    if generator:
        a, b, c, t = map(SigmaMinus, ("a", "b", "c", "t"))
        h0 = N(a) + 3 * N(c)
        v = (a + a.adjoint()) * (1 - 2 * N(b) + b + b.adjoint())
        e = Embedding({t: c}, reference={a: 0, b: 0, c: 0})
    else:
        h0 = s.diag(0, 0, 1, 1)
        v = s.Matrix([[0, 0, 1, 1], [0, 0, 1, -1], [1, 1, 0, 0], [1, -1, 0, 0]])
        e = Embedding(reference=[(0, {})])
    eff, *_ = block_diagonalize([h0, v], subspace_eigenvectors=e)
    for order, expected in ((2, -2), (3, 0), (4, 4)):
        value = eff[0, 0, order]
        from pymablock.series import zero

        actual = (
            s.S.Zero if value is zero else value.as_expr() if generator else value[0, 0]
        )
        assert actual == expected
    if not generator:
        v[1, 3] = v[3, 1] = 0
        eff, *_ = block_diagonalize([h0, v], subspace_eigenvectors=e)
        with pytest.raises(ZeroDivisionError):
            _ = eff[0, 0, 4]
