"""Exact checks of every output block, including mixed perturbation orders."""

from itertools import product

import pytest
import sympy as s

from pymablock import block_diagonalize
from pymablock.operator_embedding import Embedding
from pymablock.series import BlockSeries, one, zero
from pymablock.tests.second_quantization_helpers import nof_matrix


def test_correlated_boson_projector():
    """The conserved number difference selects equal source occupations."""
    from sympy.physics.quantum.boson import BosonOp

    from pymablock.number_ordered_form import NumberOperator as N

    a, b, q = map(BosonOp, ("a", "b", "q"))
    embedding = Embedding({q: (N(a) + 1) ** (-s.S.Half) * a * b}, reference={a: 0, b: 0})
    actual = nof_matrix(embedding._projector, [range(3)] * 2)
    assert actual == s.diag(*(int(i == j) for i, j in product(range(3), repeat=2)))


def test_floquet_projector_selects_integer_sublattice():
    """A double shift retains exactly the even ladder occupations."""
    from pymablock.number_ordered_form import LadderOp
    from pymablock.number_ordered_form import NumberOperator as N

    source, target = LadderOp("source"), LadderOp("target")
    embedding = Embedding(
        {target: source**2, N(target): N(source) / 2}, reference={source: 0}
    )
    assert nof_matrix(embedding._projector, [range(-3, 4)]) == s.diag(0, 1, 0, 1, 0, 1, 0)


@pytest.mark.parametrize("coupled", [False, True])
def test_gap_checks_ignore_terms_annihilating_the_embedding(coupled):
    """A zero gap can leave a spectator whose coupling vanishes on both levels."""
    from sympy.physics.quantum.boson import BosonOp
    from sympy.physics.quantum.pauli import SigmaMinus

    from pymablock.number_ordered_form import NumberOperator as N

    a, b, c = map(BosonOp, ("a", "b", "c"))
    x, y = SigmaMinus("x"), SigmaMinus("y")
    embedding = Embedding({x: a, y: b}, reference={a: 0, b: 0, c: 0})
    h0 = N(a) + 2 * N(b) + N(b) * N(c)
    coupling = 1 - N(a) if coupled else N(a) * (1 - N(a))
    h, *_ = block_diagonalize(
        [h0, coupling * (c + c.adjoint())], subspace_eigenvectors=embedding
    )
    if coupled:
        with pytest.raises(ZeroDivisionError, match="degenerate"):
            _ = h[0, 0, 2]
    else:
        assert h[0, 0, 2].is_zero


def test_sylvester_recognizes_algebraically_zero_gap():
    x = s.Symbol("x")
    gap = (x**2 - 1) / (x - 1) - x - 1
    h, *_ = block_diagonalize(
        [s.diag(0, gap), s.Matrix([[0, 1], [1, 0]])],
        subspace_eigenvectors=Embedding(reference=[(0, {})]),
    )
    with pytest.raises(ZeroDivisionError, match="degenerate"):
        _ = h[0, 0, 2]


@pytest.mark.parametrize("coupled", [False, True])
def test_bosonic_point_support_at_a_zero_gap(coupled):
    """Resolve a resonant occupation without dividing an inactive branch by zero."""
    from sympy.physics.quantum.boson import BosonOp

    from pymablock.number_ordered_form import NumberOperator as N

    a, b, q = map(BosonOp, ("a", "b", "q"))
    point = s.Piecewise((1, s.Eq(N(a), 1)), (0, True))
    coupling = point if coupled else 1 - point
    h, *_ = block_diagonalize(
        [N(a) + (N(a) - 1) * N(b), coupling * (b + b.adjoint())],
        subspace_eigenvectors=Embedding({q: a}, reference={a: 0, b: 0}),
    )
    if coupled:
        with pytest.raises(ZeroDivisionError, match="degenerate"):
            _ = h[0, 0, 2]
    else:
        assert nof_matrix(h[0, 0, 2], [range(3)]) == s.diag(1, 0, -1)


def test_correlated_boson_sylvester():
    """Displacing one oscillator shifts every retained energy by -g**2/omega."""
    from sympy.physics.quantum.boson import BosonOp

    from pymablock.number_ordered_form import NumberOperator as N

    a, b, q = map(BosonOp, ("a", "b", "q"))
    omega, g = s.symbols("omega g", positive=True)
    embedding = Embedding({q: (N(a) + 1) ** (-s.S.Half) * a * b}, reference={a: 0, b: 0})
    h, *_ = block_diagonalize(
        [omega * (N(a) + 2 * N(b)), g * (a + a.adjoint())],
        subspace_eigenvectors=embedding,
    )
    assert (h[0, 0, 2] + g**2 / omega).applyfunc(s.cancel).is_zero


def test_nonlinear_occupation_condition_is_not_a_point_substitution():
    from sympy.physics.quantum.boson import BosonOp

    from pymablock.number_ordered_form import NumberOperator as N

    a, b, q = map(BosonOp, ("a", "b", "q"))
    coupling = s.Piecewise((1, s.Eq(N(a) ** 2 + N(a), 2)), (0, True))
    h, *_ = block_diagonalize(
        [N(a) + 3 * N(b), coupling * (b + b.adjoint())],
        subspace_eigenvectors=Embedding({q: a}, reference={a: 0, b: 0}),
    )
    assert nof_matrix(h[0, 0, 2], [range(3)]) == s.diag(0, -s.Rational(1, 3), 0)


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
    embedding = Embedding(reference=[(0, {}), (1, {})])
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
        source = value.applyfunc(
            lambda x: x.source.as_expr() if hasattr(x, "source") else x
        )
        if i == 1:
            source = q.adjoint() * source
        if j == 1:
            source = source * q
        return source

    orders = sorted(
        (n for n in product(range(5), repeat=dimensions) if sum(n) <= 4),
        key=lambda n: (sum(n), n),
    )
    full = [{}, {}, {}]
    for n in orders:
        for k in range(3):
            blocks = [
                [lower(outputs[k][i, j, *n], i, j) for j in range(2)] for i in range(2)
            ]
            full[k][n] = s.BlockMatrix(blocks).as_explicit()
            for i, j in product(range(2), repeat=2):
                difference = blocks[i][j] - (
                    s.zeros(2)
                    if reference[k][i, j, *n] is zero
                    else s.eye(2)
                    if reference[k][i, j, *n] is one
                    else reference[k][i, j, *n]
                )
                assert difference.applyfunc(s.simplify).is_zero_matrix, (k, n, i, j)

        unit = s.zeros(4)
        transformed = s.zeros(4)
        for i in product(*(range(x + 1) for x in n)):
            j = tuple(a - b for a, b in zip(n, i))
            unit += full[2][i] * full[1][j]
            for degree, coefficient in data.items():
                rest = tuple(a - b for a, b in zip(j, degree))
                if min(rest) >= 0:
                    transformed += full[2][i] * coefficient * full[1][rest]
        assert (
            (unit - (s.eye(4) if n == origin else s.zeros(4)))
            .applyfunc(s.simplify)
            .is_zero_matrix
        )
        assert (transformed - full[0][n]).applyfunc(s.simplify).is_zero_matrix


def test_fermion_result_after_cancellation():
    from sympy.physics.quantum import Dagger
    from sympy.physics.quantum.fermion import FermionOp

    from pymablock.number_ordered_form import NumberOperator as N
    from pymablock.number_ordered_form import NumberOrderedForm

    a, b, f = map(FermionOp, ("source", "virtual", "target"))
    ea, eb, g = s.symbols(
        "source_energy virtual_energy coupling", real=True, nonzero=True
    )
    h = BlockSeries(
        data={(0,): ea * N(a) + eb * N(b), (1,): g * (Dagger(b) * a + Dagger(a) * b)}
    )
    embedding = Embedding({f: a}, reference={a: 0, b: 0})
    actual = block_diagonalize(h, subspace_eigenvectors=embedding)[0][0, 0, 2]
    expected = NumberOrderedForm.from_expr(g**2 * N(f) / (ea - eb), operators=(f,))
    assert (actual - expected).applyfunc(s.cancel).is_zero


def test_bosonic_projector_output_blocks():
    from sympy.physics.quantum.boson import BosonOp
    from sympy.physics.quantum.pauli import SigmaMinus

    from pymablock.number_ordered_form import NumberOperator as N

    a, q = BosonOp("a"), SigmaMinus("q")
    cutoff = 9
    lowering = s.zeros(cutoff)
    for n in range(1, cutoff):
        lowering[n - 1, n] = s.sqrt(n)
    source = [3 * N(a) + N(a) * (N(a) - 1) / 5, (1 + s.I) * a + (1 - s.I) * a.adjoint()]
    e = Embedding({q: a**2 / s.sqrt(2)}, reference={a: 0})
    actual = block_diagonalize(
        BlockSeries(data={(0,): source[0], (1,): source[1]}), subspace_eigenvectors=e
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
            matrix = nof_matrix(entry.source, [range(cutoff)]).extract(
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


def test_bosonic_occupation_boundary_is_not_silently_dropped():
    from sympy.physics.quantum.boson import BosonOp

    from pymablock.number_ordered_form import NumberOperator as N

    a, b = BosonOp("a"), BosonOp("b")
    embedding = Embedding({b: a * s.sqrt((N(a) - 1) / N(a))}, reference={a: 1})
    with pytest.raises(NotImplementedError, match="occupation inequality"):
        block_diagonalize([N(a), a + a.adjoint()], subspace_eigenvectors=embedding)
