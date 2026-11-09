"""Embedding results contracts and finite-matrix checks."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import sympy
import sympy as s
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize, operator_to_BlockSeries
from pymablock.number_ordered_form import (
    LadderOp,
    NumberOrderedForm,
)
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.operator_embedding import Embedding
from pymablock.second_quantization import solve_scalar
from pymablock.series import BlockSeries, cauchy_dot_product, one, zero
from pymablock.tests.second_quantization_helpers import (
    nof_matrix,
    occupation_matrices,
    operator_matrix,
)


@pytest.mark.parametrize("finite", [False, True])
def test_mixed_source_second_order(finite):
    """A spin-dependent hopping amplitude gives -N(f) N(s)**2 / 3."""
    a, v = FermionOp("a"), FermionOp("v")
    b, f = BosonOp("b"), FermionOp("f")
    s = SigmaMinus("s")
    embedding = (
        Embedding(
            {f: a, s: b},
            reference={a: 0, v: 0, b: 0},
        )
        if not finite
        else Embedding(
            {},
            reference=[{a: n, b: k, v: 0} for n in range(2) for k in range(3)],
        )
    )
    h, *_ = block_diagonalize(
        [N(a) + 4 * N(v) + 10 * N(b), N(b) * (Dagger(v) * a + Dagger(a) * v)],
        subspace_eigenvectors=embedding,
    )
    if finite:
        expected = sympy.diag(
            *[-sympy.Rational(n * k**2, 3) for n in range(2) for k in range(3)]
        )
        assert h[0, 0, 2] == expected
    else:
        expected = NumberOrderedForm.from_expr(
            -N(f) * N(s) ** 2 / 3, operators=embedding._first_lattice._source_operators
        )
        assert (h[0, 0, 2] - expected).applyfunc(sympy.simplify).is_zero


def finite_matrix(value, occupations=()):
    """Evaluate source coefficients, including matrices and series sentinels."""
    from pymablock.series import zero

    size = int(np.prod([len(x) for x in occupations]))
    if value is zero:
        return np.zeros((size, size))
    if isinstance(value, sympy.MatrixBase):
        return np.block(
            [[finite_matrix(x, occupations) for x in row] for row in value.tolist()]
        )
    if isinstance(value, NumberOrderedForm):
        return np.asarray(nof_matrix(value, occupations), dtype=complex)
    return complex(value) * np.eye(size)


@pytest.mark.parametrize(
    "model",
    [
        "spin",
        "finite",
        "degenerate",
        "fermion",
        "hole",
        "boson",
        "ladder",
        "rows",
        "spectator",
    ],
)
def test_perturbation_against_finite_matrix(model):
    """One matrix oracle checks restriction and orders 0..4 in each source algebra.

    Infinite-mode samples lie at least four transitions from the artificial upper
    boundary; the bilateral sample also stays four steps above the lower boundary.
    """
    a, b = BosonOp("a"), BosonOp("b")
    q, s = SigmaMinus("q"), SigmaMinus("s")
    source_domain = [range(2)]
    occupations = [range(8)]
    operators = (a,)
    h0, v = 3 * N(a) + N(a) * (N(a) - 1) / 5, a + a.adjoint()
    e = Embedding({s: a}, reference={a: 0})
    kept = [0, 1]
    sample = [0, 1]
    if model == "finite":
        e = Embedding({}, reference=[{a: 2}, {a: 0}])
        source_domain, kept = [], [2, 0]
    elif model == "degenerate":
        h0 = sympy.diag(0, 0, 7)
        v = sympy.Matrix([[0, 2, 1], [2, 0, sympy.I], [1, -sympy.I, 0]])
        e = Embedding({}, reference=[{Embedding.row: 1}, {}])
        operators, occupations, source_domain, kept = (), [], [], [1, 0]
    elif model in ("fermion", "hole"):
        a, b, f = map(FermionOp, ("a", "b", "f"))
        operators, occupations = (a, b), [range(2)] * 2
        h0, v = 2 * N(a) + 5 * N(b), a.adjoint() * b + b.adjoint() * a
        hole = model == "hole"
        e = Embedding({f: a.adjoint() if hole else a}, reference={a: int(hole), b: 0})
        kept = [2, 0] if hole else [0, 2]
    elif model in ("boson", "ladder"):
        if model == "ladder":
            a, b = LadderOp("a"), LadderOp("b")
            occupations = [range(-6, 7), range(2)]
            source_domain = [range(-6, 7)]
            generators = {b: a, N(b): N(a)}
            sample = list(range(4, 9))
        else:
            occupations = [range(9), range(2)]
            source_domain = [range(9)]
            generators = {b: a}
            sample = list(range(4))
        operators = (a, q)
        h0 = 3 * N(a) + N(a) * (N(a) - 1) / 5 + (8 + N(a) / 7) * N(q)
        v = (a + a.adjoint()) * (q + q.adjoint()) + sympy.Rational(2, 7) * (
            a + a.adjoint()
        )
        e = Embedding(generators, reference={a: 0, q: 0})
        kept = list(range(0, 2 * len(occupations[0]), 2))
    elif model == "rows":
        h0 = sympy.diag(h0, 2 + h0)
        v = sympy.Matrix(
            [
                [v, 1 + 2 * a + sympy.I * a.adjoint()],
                [1 + 2 * a.adjoint() - sympy.I * a, 2 * v],
            ]
        )
        e = Embedding({s: a}, reference=[{Embedding.row: 1, a: 0}, {a: 0}])
        kept, sample = [8, 9, 0, 1], list(range(4))
    elif model == "spectator":
        operators, occupations = (a, b), [range(6), range(6)]
        h0 = 3 * N(a) + N(a) * (N(a) - 1) / 5 + 7 * N(b)
        v = a + a.adjoint() + b + b.adjoint()
        e = Embedding({s: a}, reference=[{a: 0, b: 1}, {a: 0, b: 0}])
        kept, sample = [1, 7, 0, 6], list(range(4))
    matrices = occupation_matrices(operators, occupations)
    full_h0, full_v = (operator_matrix(x, matrices).toarray() for x in (h0, v))
    basis = np.eye(len(full_h0))
    complement = [i for i in range(len(basis)) if i not in kept]
    oracle = block_diagonalize(
        [full_h0, full_v], subspace_eigenvectors=[basis[:, kept], basis[:, complement]]
    )[0]
    effective = block_diagonalize([h0, v], subspace_eigenvectors=e)[0]
    for expression, matrix in ((h0, full_h0), (v, full_v)):
        np.testing.assert_allclose(
            finite_matrix(e.restrict(expression), source_domain),
            matrix[np.ix_(kept, kept)],
            atol=1e-12,
        )
    for order in range(5):
        actual = (
            np.zeros((len(kept), len(kept)))
            if effective[0, 0, order] is zero
            else finite_matrix(effective[0, 0, order], source_domain)
        )
        expected = (
            finite_matrix(oracle[0, 0, order])
            if oracle[0, 0, order] is zero
            else np.asarray(oracle[0, 0, order])
        )
        if expected.shape == (1, 1) and not expected.any():
            expected = np.zeros_like(actual)
        np.testing.assert_allclose(
            actual[np.ix_(sample, sample)],
            expected[np.ix_(sample, sample)],
            atol=1e-11,
            rtol=1e-10,
        )


@pytest.mark.parametrize("particle_hole", [False, True])
def test_hubbard_dimer_fourth_order(particle_hole):
    u = [FermionOp(f"c{i}u") for i in range(2)]
    d = [FermionOp(f"c{i}d") for i in range(2)]
    spins = [SigmaMinus(f"s{i}") for i in range(2)]
    t, interaction = s.symbols("t U", positive=True)
    down = d if particle_hole else [op.adjoint() for op in d]
    generators = {sp: dn * up for sp, dn, up in zip(spins, down, u)}
    references = [
        {**dict(zip(u, q)), **dict(zip(d, q if particle_hole else [1 - n for n in q]))}
        for q in product((0, 1), repeat=2)
    ]
    h0 = interaction * sum(
        N(up) * (1 - N(dn) if particle_hole else N(dn)) for up, dn in zip(u, d)
    )
    hopping = -t * sum(
        sign * (ops[0].adjoint() * ops[1] + ops[1].adjoint() * ops[0])
        for ops, sign in ((u, 1), (d, -1 if particle_hole else 1))
    )
    embedded, *_ = block_diagonalize(
        [h0, hopping],
        subspace_eigenvectors=Embedding(generators, reference=references[0]),
        symbols=[t],
    )
    reference, *_ = block_diagonalize(
        [h0, hopping],
        subspace_eigenvectors=Embedding({}, reference=references),
        symbols=[t],
    )
    # Site-major fermion ordering gives the same phases as the spin generators.
    actual = nof_matrix(embedded[0, 0, 4])
    expected = (
        8
        * t**4
        / interaction**3
        * s.Matrix([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])
    )
    assert (actual - reference[0, 0, 4]).applyfunc(s.simplify) == s.zeros(4)
    assert (actual - expected).applyfunc(s.simplify) == s.zeros(4)


@pytest.mark.parametrize("embedded", [False, True])
def test_binary_poles_preserve_nondegenerate_sectors(embedded):
    f, g, source_f, source_g = map(FermionOp, ("f", "g", "F", "G"))
    q = SigmaMinus("q")
    u = s.Symbol("U", positive=True)
    if embedded:
        embedding = Embedding({source_f: f, source_g: g}, reference={q: 0, f: 0, g: 0})
        h, *_ = block_diagonalize(
            [u * N(f) * (1 - N(q)) + u * N(g) * N(q), q + q.adjoint()],
            subspace_eigenvectors=embedding,
        )
    else:
        source_f, source_g = f, g
        h, *_ = block_diagonalize(
            [s.diag(u * N(f), u * N(g)), s.Matrix([[0, 1], [1, 0]])],
            subspace_indices=[0, 1],
        )
    for order, coefficient in ((2, 1 / u), (4, -1 / u**3), (6, 2 / u**5)):
        value = h[0, 0, order] if embedded else h[0, 0, order][0, 0]
        for nf, ng, sign in ((1, 0, 1), (0, 1, -1)):
            actual = value.as_expr().subs({N(source_f): nf, N(source_g): ng})
            assert s.cancel(actual - sign * coefficient) == 0


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("selective", [False, True])
def test_diagonalize_retained_oscillator_levels(finite, selective):
    """Both retained levels obey the independent oscillator characteristic equation."""
    a, q = BosonOp("a"), SigmaMinus("q")
    embedding = (
        Embedding({}, reference=[{a: 0}, {a: 1}])
        if finite
        else Embedding({q: a}, reference={a: 0})
    )
    mask = s.Matrix([[0, 1], [1, 0]]) if finite else q + q.adjoint()
    h, *_ = block_diagonalize(
        [3 * N(a) + N(a) * (N(a) - 1), a + a.adjoint()],
        subspace_eigenvectors=embedding,
        fully_diagonalize={0: mask} if selective else (0,),
    )
    coefficients = []
    for order in range(5):
        value = h[0, 0, order]
        if value is zero or value == 0:
            matrix = s.zeros(2)
        elif isinstance(value, NumberOrderedForm):
            matrix = nof_matrix(value)
        else:
            matrix = value
        assert matrix.is_diagonal()
        coefficients.append(matrix)
    assert coefficients[2] == s.diag(-s.Rational(1, 3), -s.Rational(1, 15))

    # Four oscillator levels include every path of length four from levels 0,1.
    # Construct their Hamiltonian directly, without NOF or embedding conversion.
    g, energy = s.symbols("g energy")
    target = s.diag(0, 3, 8, 15)
    for n in range(3):
        target[n, n + 1] = target[n + 1, n] = g * s.sqrt(n + 1)
    characteristic = target.charpoly(energy).as_expr()
    for row in range(2):
        effective_energy = sum(
            c[row, row] * g**order for order, c in enumerate(coefficients)
        )
        residual = s.Poly(characteristic.subs(energy, effective_energy), g)
        assert all(residual.nth(order) == 0 for order in range(5))


def test_diagonalize_finite_matrix_embedding():
    """A retained matrix block includes internal and virtual energy corrections."""
    h, *_ = block_diagonalize(
        [s.diag(0, 2, 5), s.Matrix([[0, 1, 1], [1, 0, 2], [1, 2, 0]])],
        subspace_eigenvectors=Embedding({}, reference=[{}, {Embedding.row: 1}]),
        fully_diagonalize=(0,),
    )
    assert h[0, 0, 1].is_zero_matrix
    # Ordinary second-order perturbation theory sums over both other levels.
    assert h[0, 0, 2] == s.diag(-s.Rational(7, 10), -s.Rational(5, 6))
    assert h[0, 0, 3] == s.diag(s.Rational(2, 5), -s.Rational(2, 3))


@pytest.mark.parametrize("selective", [False, True])
def test_diagonalize_embedded_infinite_oscillators(selective):
    """Displacing independent oscillators gives their exact energy corrections."""
    a, b, c, f, g = map(BosonOp, ("a", "b", "c", "f", "g"))
    h, *_ = block_diagonalize(
        [3 * N(a) + 5 * N(b) + 11 * N(c), sum(op + op.adjoint() for op in (a, b, c))],
        subspace_eigenvectors=Embedding({f: a, g: b}, reference={a: 0, b: 0, c: 0}),
        fully_diagonalize={0: f + f.adjoint()} if selective else (0, 1),
    )
    if selective:
        assert h[0, 0, 1] == NumberOrderedForm.from_expr(g + g.adjoint())
    else:
        assert h[0, 0, 1] is zero or h[0, 0, 1] == 0
    expected = -s.Rational(1, 3) - s.Rational(1, 11)
    if not selective:
        expected -= s.Rational(1, 5)
        assert h[1, 1, 1] is zero or h[1, 1, 1] == 0
        complement = nof_matrix(h[1, 1, 2], [range(2)] * 3)
        assert complement == s.diag(0, expected, 0, expected, 0, expected, 0, expected)
    assert h[0, 0, 2] == expected


def test_bilateral_zero_rhs_convention():
    """The negative ladder site is physical and selects the zero solution."""
    ell = LadderOp("ell")
    result = solve_scalar(N(ell) + 1, N(ell) + 1, 0)
    assert nof_matrix(result, [range(-2, 2)]) == s.diag(1, 0, 1, 1)


@pytest.mark.parametrize("finite", [False, True])
def test_dressed_observable_conversion(finite):
    """Dressing N in a driven oscillator adds the vacuum population g²/omega²."""
    a, q = BosonOp("a"), SigmaMinus("q")
    embedding = (
        Embedding({}, reference=[{a: 0}])
        if finite
        else Embedding({q: a}, reference={a: 0})
    )
    _, transform, inverse = block_diagonalize(
        [3 * N(a), a + a.adjoint()], subspace_eigenvectors=embedding
    )
    observable = operator_to_BlockSeries({(0,): N(a)}, subspace_eigenvectors=embedding)
    # The same reusable path accepts a non-diagonal zeroth-order observable.
    cross = operator_to_BlockSeries(
        {(0,): a + a.adjoint()}, subspace_eigenvectors=embedding
    )
    assert cross[0, 1, 0] is not zero
    dressed = cauchy_dot_product(
        inverse, observable, transform, operator=lambda a, b: a * b
    )
    value = dressed[0, 0, 2]
    if finite:
        assert value == s.Matrix([[s.Rational(1, 9)]])
    else:
        # Generator selection retains levels 0,1, so only level 1 is dressed by Q.
        assert nof_matrix(value) == s.diag(0, s.Rational(2, 9))


@pytest.mark.parametrize("preconverted", [False, True])
def test_float_fourth_order_matches_exact_values(preconverted):
    """Floating inputs agree numerically with their exact values through fourth order."""
    a, b = BosonOp("a"), BosonOp("b")
    embedding = Embedding({}, reference=[{a: 0, b: 0}, {a: 1, b: 0}])
    h0 = s.Float(1.1) * N(a) + s.Float(2.3) * N(b)
    v = s.Float(0.2) * (a + a.adjoint()) + s.Float(0.3) * (
        a.adjoint() * b + b.adjoint() * a
    )

    def exact(expr):
        return expr.xreplace({v: s.Rational(v) for v in expr.atoms(s.Float)})

    coefficients = [h0, v]
    if preconverted:
        coefficients = [NumberOrderedForm.from_expr(x, (a, b)) for x in coefficients]
    floating, *_ = block_diagonalize(coefficients, subspace_eigenvectors=embedding)
    rational, *_ = block_diagonalize(
        [exact(h0), exact(v)], subspace_eigenvectors=embedding
    )
    for order in (2, 4):
        np.testing.assert_allclose(
            np.asarray(floating[0, 0, order], dtype=complex),
            np.asarray(rational[0, 0, order], dtype=complex),
            rtol=1e-12,
        )


def test_zero_source_amplitude_at_degenerate_gap():
    """A vanishing retained amplitude must be discarded before gap division."""
    a, q = BosonOp("a"), SigmaMinus("q")
    omega, g = s.symbols("omega g", positive=True)
    h, *_ = block_diagonalize(
        omega * N(a) + g * (N(a) * (N(a) - 1) + a + a.adjoint()),
        subspace_eigenvectors=Embedding({q: a}, reference={a: 0}),
        symbols=[g],
    )
    assert (
        s.simplify(h[0, 0, 3] - g**3 / omega**2 * (4 * N(q) - (q + q.adjoint()) / 2)) == 0
    )


def test_correlated_boson_nonnegative_gap_stays_compact():
    a, b, q = map(BosonOp, ("a", "b", "q"))
    g = s.Symbol("g", positive=True)
    h, *_ = block_diagonalize(
        [N(a) + 2 * N(b) + N(a) * N(b) / 7, g * (a + a.adjoint())],
        subspace_eigenvectors=Embedding(
            {q: (N(a) + 1) ** (-s.S.Half) * a * b}, reference={a: 0, b: 0}
        ),
    )
    assert s.count_ops(h[0, 0, 4]) < 10


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
            target_form = entry if i == j else entry.target
            matrix = nof_matrix(target_form, [range(cutoff)]).extract(
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
        with pytest.raises(ValueError):
            _ = eff[0, 0, 4]


@pytest.mark.parametrize(
    ("case", "representation"),
    [
        *(
            (case, "generator")
            for case in (
                "point-zero",
                "point-resonance",
                "ladder-zero",
                "nonlinear",
                "binary-zero",
                "binary-pole",
                "explicit-zero",
            )
        ),
        # A finite reference list only represents the truncated-boson cases.
        ("binary-zero", "list"),
        ("binary-pole", "list"),
    ],
)
def test_division_policy(case, representation):
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
    embedding = Embedding({q: a}, reference={a: 0, b: 0})
    if representation == "list":
        embedding = Embedding({}, reference=[{a: 0, b: 0}, {a: 1, b: 0}])
    h, *_ = block_diagonalize(
        [n + energy * N(b), coupling * b.adjoint() + s.adjoint(coupling) * b],
        subspace_eigenvectors=embedding,
    )
    if case in ("point-resonance", "explicit-zero", "binary-pole"):
        with pytest.raises(ValueError):
            _ = h[0, 0, 2]
    elif representation == "list":
        assert h[0, 0, 2] == expected
    else:
        assert nof_matrix(h[0, 0, 2], [range(expected.rows)]) == expected
