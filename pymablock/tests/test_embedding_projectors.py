"""Exact checks of every output block, including mixed perturbation orders."""

from itertools import product

import pytest
import sympy as s

from pymablock import block_diagonalize
from pymablock._operator_embedding import block_diagonalize as graph
from pymablock.operator_embedding import Embedding
from pymablock.series import BlockSeries, one, zero


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
    outputs = graph(h, embedding)
    reference = block_diagonalize(h, subspace_indices=[0, 0, 1, 1])
    w, q = s.eye(4)[:, :2], s.eye(4)[:, 2:]

    def lower(value, i, j):
        if value is zero:
            return s.zeros(2)
        if value is one:
            return s.eye(2)
        if i == j == 0:
            return value
        source = value.applyfunc(lambda x: x.as_expr() if hasattr(x, "as_expr") else x)
        frames = (w, q)
        return frames[i].adjoint() * source * frames[j]

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
    actual = graph(h, embedding)[0][0, 0, 2]
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
    actual = graph(BlockSeries(data={(0,): source[0], (1,): source[1]}), e)
    h0 = s.diag(*(3 * n + s.Rational(n * (n - 1), 5) for n in range(cutoff)))
    v = (1 + s.I) * lowering + (1 - s.I) * lowering.T
    labels = [0 if n in (0, 2) else 1 for n in range(cutoff)]
    reference = block_diagonalize([h0, v], subspace_indices=labels)
    selected = ((0, 2), (1, 3))
    for k, n, i, j in product(range(3), range(5), range(2), range(2)):
        entry = actual[k][i, j, n]
        if entry is one:
            matrix = s.eye(2)
        elif entry is zero:
            matrix = s.zeros(2)
        elif i == j == 0:
            matrix = entry.to_matrix()
        else:
            matrix = entry.to_matrix([range(cutoff)]).extract(selected[i], selected[j])
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
