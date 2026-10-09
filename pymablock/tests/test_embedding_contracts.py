"""Exact finite-basis contracts for embedding frames and structural lifting."""

import pickle
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
    attached = e._retained_frame(1)[0, 0]

    def target(x):
        return nof_matrix(x, [range(6)])

    def source(x):
        return nof_matrix(x) if isinstance(x, F) else x * sp.eye(2)

    assert_matrix_equal(source(attached.adjoint() * attached), sp.eye(2))
    assert_matrix_equal(target(attached * attached.adjoint()), w * w.T)
    for S in (1, s, s.adjoint(), N(s), s + s.adjoint()):
        value = F.from_expr(S, operators=(s,))
        assert_matrix_equal(target(e._first_lattice.lift(value)), w * source(value) * w.T)
        assert_matrix_equal(target((attached * value).target) * w, w * source(value))
    for X in (a, a.adjoint(), a**2, a.adjoint() * a**2, N(a), sp.sqrt(N(a) + 1)):
        x = e._first_lattice._parse_target(X)
        assert_matrix_equal(source(e.restrict(x)), w.T * target(x) * w)
        for Y in (1, a, a.adjoint()):
            y = e._first_lattice._parse_target(Y)
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
    frame = e._retained_frame(1)

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
        x = e._first_lattice._parse_target(X)
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
        assert_matrix_equal(
            nof_matrix(e._first_lattice.lift(value)), w * nof_matrix(value) * w.T
        )


def test_lift_bilateral_numbers_and_symbolic_powers():
    a, b = LadderOp("a"), LadderOp("b")
    e = Embedding({b: a, N(b): N(a) + 2}, reference={a: -2})
    for S in (b, b.adjoint(), N(b), b.adjoint() * N(b) * b):
        value = F.from_expr(S, operators=(b,))
        assert_matrix_equal(
            nof_matrix(e._first_lattice.lift(value), [range(-4, 3)]),
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
            occupations = [range(-2, 4)] if mode_type is LadderOp else [range(4)]
            for exponent in (1, 2, 3):
                assert_matrix_equal(
                    nof_matrix(
                        embedding._first_lattice.lift(value).xreplace({power: exponent}),
                        occupations,
                    ),
                    nof_matrix(value.xreplace({power: exponent}), occupations),
                )


def test_existing_nof_conversion_preserves_values():
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    e = Embedding({s: a}, reference={a: 0, b: 0})
    value = F.from_expr(sp.sqrt(N(a) + 1) * a.adjoint(), operators=(a,))
    result = e._first_lattice._parse_target(value)
    assert_matrix_equal(
        nof_matrix(result, [range(4), range(2)]),
        sp.kronecker_product(nof_matrix(value, [range(4)]), sp.eye(2)),
    )


@pytest.mark.parametrize("mode_type", [BosonOp, LadderOp])
@pytest.mark.parametrize("sign", [-1, 1])
def test_restrict_symbolic_ladder_power(mode_type, sign):
    a, b = mode_type("a"), mode_type("b")
    generators = {b: a}
    if mode_type is LadderOp:
        generators[N(b)] = N(a)
    embedding = Embedding(generators, reference={a: 0})
    k = sp.Symbol("k", integer=True, positive=True)
    target, source = (a, b) if sign == 1 else (a.adjoint(), b.adjoint())
    for coefficient in (1, N(a) + 1):
        expression = coefficient * target**k
        result = embedding.restrict(expression)
        assert result == F.from_expr(
            sp.sympify(coefficient).xreplace({N(a): N(b)}) * source**k
        )
        for exponent in (1, 2, 3):
            assert result.xreplace({k: exponent}) == embedding.restrict(
                expression.xreplace({k: exponent})
            )


def test_symbolic_shift_into_finite_source_is_not_silently_zero():
    a, s = BosonOp("a"), SigmaMinus("s")
    embedding = Embedding({s: a}, reference={a: 0})
    k = sp.Symbol("k", integer=True, positive=True)
    with pytest.raises(NotImplementedError, match="finite source shift"):
        embedding.restrict(a**k)


def test_symbolic_shift_may_leave_lattice():
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    embedding = Embedding({s: a * b}, reference={a: 0, b: 0})
    k, m = sp.symbols("k m", integer=True, positive=True)
    # Equal shifts preserve the lattice, whereas unequal shifts leave it.
    with pytest.raises(NotImplementedError, match="leaves the lattice"):
        embedding.restrict(a**k * b**m)


@pytest.mark.parametrize("reference_list", [False, True])
def test_reconstructed_lattices_preserve_frame_matrix_elements(reference_list):
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    references = [{a: 0, b: 1}, {a: 0, b: 2}]
    embedding = Embedding(
        {s: sp.I * a}, reference=references if reference_list else references[0]
    )
    operator = a + a.adjoint() + b + b.adjoint() + N(a) * N(b)
    # W|1,r> = -i|1,r>: the a transition has phase -i; b connects r=1,2
    # with amplitude sqrt(2), independently of the source spin occupation.
    spin_flip = -sp.I * s + sp.I * s.adjoint()
    expected = sp.Matrix(
        [[spin_flip + N(s), sp.sqrt(2)], [sp.sqrt(2), spin_flip + 2 * N(s)]]
    )
    if not reference_list:
        expected = expected[:1, :1]

    def expressions(matrix):
        return matrix.applyfunc(
            lambda value: value.as_expr() if isinstance(value, F) else value
        )

    # Populate both frame and lattice caches before reconstruction.
    embedding.restrict(operator)
    frame = embedding._retained_frame(1)
    for restored in (
        embedding.func(*embedding.args),
        pickle.loads(pickle.dumps(embedding)),
    ):
        result = restored.restrict(operator)
        if not reference_list:
            result = sp.Matrix([[result]])
        assert_matrix_equal(expressions(result), expected)
    restored_frame = pickle.loads(pickle.dumps(frame))
    assert_matrix_equal(
        expressions(
            restored_frame.adjoint() * embedding._target_matrix(operator) * restored_frame
        ),
        expected,
    )
