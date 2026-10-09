"""Exact checks of every output block, including mixed perturbation orders."""

from itertools import product

import pytest
import sympy as s
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.operator_embedding import Embedding
from pymablock.series import BlockSeries, one, zero
from pymablock.tests.second_quantization_helpers import nof_matrix


def test_correlated_boson_projector():
    """The conserved number difference selects equal target occupations."""
    a, b, q = map(BosonOp, ("a", "b", "q"))
    embedding = Embedding({q: (N(a) + 1) ** (-s.S.Half) * a * b}, reference={a: 0, b: 0})
    actual = nof_matrix(embedding._first_lattice.projector, [range(3)] * 2)
    assert actual == s.diag(*(int(i == j) for i, j in product(range(3), repeat=2)))


def test_floquet_projector_selects_integer_sublattice():
    """A double shift retains exactly the even ladder occupations."""
    from pymablock.number_ordered_form import LadderOp

    target, source = LadderOp("target"), LadderOp("source")
    embedding = Embedding(
        {source: target**2, N(source): N(target) / 2}, reference={target: 0}
    )
    assert nof_matrix(embedding._first_lattice.projector, [range(-3, 4)]) == s.diag(
        0, 1, 0, 1, 0, 1, 0
    )


def test_correlated_boson_sylvester():
    """Displacing one oscillator shifts every retained energy by -g**2/omega."""
    a, b, q = map(BosonOp, ("a", "b", "q"))
    omega, g = s.symbols("omega g", positive=True)
    embedding = Embedding({q: (N(a) + 1) ** (-s.S.Half) * a * b}, reference={a: 0, b: 0})
    h, *_ = block_diagonalize(
        [omega * (N(a) + 2 * N(b)), g * (a + a.adjoint())],
        subspace_eigenvectors=embedding,
    )
    assert (h[0, 0, 2] + g**2 / omega).applyfunc(s.cancel).is_zero


@pytest.mark.parametrize("dimensions", [1, 2])
def test_complete_rotation(dimensions):
    origin = (0,) * dimensions
    h0 = s.diag(0, 2, 7, 11)
    v = s.ImmutableMatrix(
        [
            [1, 1 + s.I, 2, 1],
            [1 - s.I, -1, 1, 2 * s.I],
            [2, 1, 2, 3],
            [1, -2 * s.I, 3, -2],
        ]
    )
    data = {origin: h0, (1, *origin[1:]): v}
    if dimensions == 2:
        data[(0, 1)] = s.ImmutableMatrix(
            [[0, 2, 1, 0], [2, 3, 0, s.I], [1, 0, 1, 2], [0, -s.I, 2, -1]]
        )
        data[(1, 1)] = s.diag(1, -1, 2, 0)
    h = BlockSeries(data=data, n_infinite=dimensions)
    embedding = Embedding({}, reference=[{}, {Embedding.row: 1}])
    outputs = block_diagonalize(h, subspace_eigenvectors=embedding)
    reference = block_diagonalize(h, subspace_indices=[0, 0, 1, 1])
    q = s.eye(4)[:, 2:]

    def lower(value, i, j):
        if value is zero:
            return s.zeros(2)
        if value is one:
            return s.eye(2)
        if i == j == 0:
            return value
        target = value.applyfunc(
            lambda x: x.target.as_expr() if hasattr(x, "target") else x
        )
        if i == 1:
            target = q.adjoint() * target
        if j == 1:
            target = target * q
        return target

    orders = sorted(
        (n for n in product(range(5), repeat=dimensions) if sum(n) <= 4),
        key=lambda n: (sum(n), n),
    )
    for n in orders:
        for k in range(3):
            blocks = [
                [lower(outputs[k][i, j, *n], i, j) for j in range(2)] for i in range(2)
            ]
            for i, j in product(range(2), repeat=2):
                difference = blocks[i][j] - (
                    s.zeros(2)
                    if reference[k][i, j, *n] is zero
                    else s.eye(2)
                    if reference[k][i, j, *n] is one
                    else reference[k][i, j, *n]
                )
                assert difference.applyfunc(s.simplify).is_zero_matrix, (k, n, i, j)


def test_bosonic_projector_output_blocks():
    from sympy.physics.quantum.boson import BosonOp
    from sympy.physics.quantum.pauli import SigmaMinus

    a, q = BosonOp("a"), SigmaMinus("q")
    cutoff = 9
    lowering = s.zeros(cutoff)
    for n in range(1, cutoff):
        lowering[n - 1, n] = s.sqrt(n)
    target = [3 * N(a) + N(a) * (N(a) - 1) / 5, (1 + s.I) * a + (1 - s.I) * a.adjoint()]
    e = Embedding({q: a**2 / s.sqrt(2)}, reference={a: 0})
    actual = block_diagonalize(
        BlockSeries(data={(0,): target[0], (1,): target[1]}), subspace_eigenvectors=e
    )
    h0 = s.diag(*(3 * n + s.Rational(n * (n - 1), 5) for n in range(cutoff)))
    v = (1 + s.I) * lowering + (1 - s.I) * lowering.T
    labels = [0 if n in (0, 2) else 1 for n in range(cutoff)]
    reference = block_diagonalize([h0, v], subspace_indices=labels)
    selected = ((0, 2), (1, 3))
    for k, n, i, j in product(range(3), range(5), range(2), range(2)):
        entry = actual[k][i, j, n]
        if entry is one:
            matrix = s.eye(2)
        elif entry is zero or entry == 0:
            matrix = s.zeros(2)
        elif i == j == 0:
            matrix = nof_matrix(entry)
        else:
            matrix = nof_matrix(entry.target, [range(cutoff)]).extract(
                selected[i], selected[j]
            )
        expected = reference[k][i, j, n]
        if expected is one:
            expected = s.eye(2)
        elif expected is zero:
            expected = s.zeros(2)
        else:
            expected = expected[:2, :2]
        assert (matrix - expected).applyfunc(s.simplify).is_zero_matrix, (k, n, i, j)


@pytest.mark.parametrize("generator", [False, True])
def test_cancellation_before_resonance_check(generator):
    if generator:
        a, b, c, t = map(SigmaMinus, ("a", "b", "c", "t"))
        h0 = N(a) + 3 * N(c)
        v = (a + a.adjoint()) * (1 - 2 * N(b) + b + b.adjoint())
        e = Embedding({t: c}, reference={a: 0, b: 0, c: 0})
    else:
        h0 = s.diag(0, 0, 1, 1)
        v = s.Matrix([[0, 0, 1, 1], [0, 0, 1, -1], [1, 1, 0, 0], [1, -1, 0, 0]])
        e = Embedding({}, reference=[{}])
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


@pytest.mark.parametrize(
    "case",
    [
        "point-zero",
        "point-resonance",
        "ladder-zero",
        "nonlinear",
        "binary-zero",
        "binary-pole",
        "explicit-zero",
    ],
)
def test_division_policy(case):
    a, b, q = map(BosonOp, ("a", "b", "q"))
    n = N(a)
    point = s.Piecewise((1, s.Eq(n, 1)), (0, True))
    energy, coupling, expected = {
        "point-zero": (n - 1, 1 - point, s.diag(1, 0, -1)),
        "point-resonance": (n - 1, point, None),
        "ladder-zero": (n + 2, a, s.diag(0, -1, -1)),
        "nonlinear": (
            3,
            s.Piecewise((1, s.Eq(n**2 + n, 2)), (0, True)),
            s.diag(0, -s.Rational(1, 3), 0),
        ),
        "binary-zero": (1 - n, 1 - n, s.diag(-1, 0)),
        "binary-pole": (1 - n, 1, None),
        "explicit-zero": (0, 1, None),
    }[case]
    if case.startswith("binary"):
        q = SigmaMinus("q")
    if case == "explicit-zero":
        x = s.Symbol("x")
        energy = (x**2 - 1) / (x - 1) - x - 1
    h, *_ = block_diagonalize(
        [n + energy * N(b), coupling * b.adjoint() + s.adjoint(coupling) * b],
        subspace_eigenvectors=Embedding({q: a}, reference={a: 0, b: 0}),
    )
    if case in ("point-resonance", "explicit-zero"):
        with pytest.raises(ZeroDivisionError):
            _ = h[0, 0, 2]
    elif case == "binary-pole":
        expression = h[0, 0, 2].as_expr()
        assert expression.subs(N(q), 0) == -1
        assert expression.subs(N(q), 1).has(s.zoo, s.nan)
    else:
        assert nof_matrix(h[0, 0, 2], [range(expected.rows)]) == expected
