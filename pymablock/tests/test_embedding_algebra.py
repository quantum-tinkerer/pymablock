"""Embedding algebra contracts and finite-matrix checks."""

from __future__ import annotations

import pickle
from itertools import product

import numpy as np
import pytest
import sympy
import sympy as s
import sympy as sp
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize
from pymablock.number_ordered_form import (
    LadderOp,
    NumberOrderedForm,
    _equal_value_indicator,
)
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.number_ordered_form import NumberOrderedForm as F
from pymablock.operator_embedding import Embedding
from pymablock.second_quantization import _divide_by_energy_gap, solve_scalar
from pymablock.tests.second_quantization_helpers import (
    nof_matrix,
    occupation_matrices,
    operator_matrix,
)


@pytest.mark.parametrize("annihilate", [False, True])
def test_conversion_preserves_factored_spectators(annihilate):
    """One term stays compact as independent spectator modes are added."""
    target = tuple(SigmaMinus(f"t{i:02}") for i in range(16))
    source = tuple(SigmaMinus(f"s{i:02}") for i in range(16))
    embedding = Embedding(dict(zip(source, target)), reference=dict.fromkeys(target, 0))
    powers = (int(annihilate),) + (0,) * 15
    coefficient = sympy.prod(2 + N(op) for op in target[1:])
    expression = NumberOrderedForm(target, {powers: coefficient})
    expected = NumberOrderedForm(
        source, {powers: sympy.prod(2 + N(op) for op in source[1:])}
    )
    assert embedding.restrict(expression) == expected


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("finite", [False, True])
def test_mixed_spin_and_fermions_against_fock_matrices(reverse, interleave, finite):
    """Prepare encoded columns using creation matrices, independently of phases."""
    modes = tuple(FermionOp(name) for name in ("a", "b", "c", "d"))
    up, down, left, right = (
        modes[i] for i in ((0, 2, 1, 3) if interleave else (0, 1, 2, 3))
    )
    f, g = FermionOp("f"), FermionOp("g")
    s = SigmaMinus("s")
    first, second = (g, f) if reverse else (f, g)
    backend = Embedding(
        {s: Dagger(down) * up, first: left, second: right},
        reference={up: 0, down: 1, left: 0, right: 0},
    )
    matrices = occupation_matrices(modes, [(0, 1)] * 4)
    target_states = list(product((0, 1), repeat=4))
    columns = []
    direct = {first: left, second: right}
    for state in product((0, 1), repeat=len(backend._first_lattice._source_operators)):
        values = dict(zip(backend._first_lattice._source_operators, state, strict=True))
        baseline = tuple(int(op == down) for op in modes)
        column = np.eye(16)[:, target_states.index(baseline)]
        for source in reversed(backend._first_lattice._source_operators):
            if values[source]:
                raising = (
                    matrices[Dagger(up)] @ matrices[down]
                    if source == s
                    else matrices[Dagger(direct[source])]
                )
                column = raising @ column
        columns.append(column)
    w = np.column_stack(columns)
    np.testing.assert_allclose(w.T @ w, np.eye(8))
    references = [
        dict(zip(modes, state, strict=True))
        for state in target_states
        if state[modes.index(up)] + state[modes.index(down)] == 1
    ]
    finite_embedding = Embedding({}, reference=references)
    selected = [target_states.index(tuple(ref[op] for op in modes)) for ref in references]
    source_matrices = occupation_matrices(
        backend._first_lattice._source_operators,
        [(0, 1)] * 3,
    )
    for expression in (
        left,
        right,
        Dagger(left) * right,
        Dagger(up) * down,
        N(up),
        (Dagger(up) * down) * left,
        up * down,
    ):
        result = backend.restrict(expression)
        actual = operator_matrix(result.simplify(), source_matrices).toarray()
        full = operator_matrix(expression, matrices).toarray()
        expected = w.T @ full @ w
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        if finite:
            np.testing.assert_allclose(
                np.array(finite_embedding.restrict(expression), dtype=complex),
                full[np.ix_(selected, selected)],
                atol=1e-14,
            )
    # The chosen columns preserve each directly retained fermion generator.
    for target, source in ((left, first), (right, second)):
        np.testing.assert_allclose(
            w.T @ matrices[target] @ w, source_matrices[source].toarray()
        )


def test_transfer_fermion_sign_and_generator_phase():
    """A spectator before the moving fermion changes its generator-defined phase."""
    spectator, target, source = map(FermionOp, ("a", "b", "f"))
    embedding = Embedding(
        {source: sympy.I * target},
        reference=[
            {spectator: 0, target: 0},
            {spectator: 1, target: 0},
        ],
    )
    source_matrices = occupation_matrices((source,), [(0, 1)])
    matrices = occupation_matrices((spectator, target), [(0, 1)] * 2)
    w = np.diag([1, -1j, 1, 1j])
    for expression in (spectator, target, Dagger(spectator) * target):
        actual = operator_matrix(
            embedding.restrict(expression), source_matrices
        ).toarray()
        full = operator_matrix(expression, matrices).toarray()
        np.testing.assert_allclose(actual, w.conj().T @ full @ w, atol=1e-14)


def test_bilateral_reference_transfers_have_no_vacuum():
    from pymablock.number_ordered_form import LadderOp

    ell = LadderOp("ell")
    embedding = Embedding({}, reference=[{ell: -2}, {ell: 3}])
    assert embedding.restrict(ell**5) == sympy.Matrix([[0, 1], [0, 0]])
    assert embedding.restrict(N(ell)) == sympy.diag(-2, 3)


def test_restriction_does_not_require_a_complement_projector():
    """A shifted boson ladder can be compressed before inequality projectors exist."""
    a, b = BosonOp("a"), BosonOp("b")
    embedding = Embedding({b: a * sympy.sqrt((N(a) - 3) / N(a))}, reference={a: 3})
    assert embedding.restrict(N(a)) == NumberOrderedForm.from_expr(3 + N(b))
    with pytest.raises(NotImplementedError):
        block_diagonalize([N(a)], subspace_eigenvectors=embedding)


def assert_matrix_equal(actual, expected):
    assert (actual - expected).applyfunc(sp.simplify) == sp.zeros(*expected.shape)


@pytest.mark.parametrize("reference", [0, 1])
def test_normalized_generator_lift_and_map_contracts(reference):
    a, s = BosonOp("a"), SigmaMinus("s")
    e = Embedding({s: a**2 / sp.sqrt(N(a) * (N(a) - 1))}, reference={a: reference})
    w = sp.eye(6)[:, [reference, reference + 2]]
    w_map = e._retained_frame(1)[0, 0]

    def target(x):
        return nof_matrix(x, [range(6)])

    def source(x):
        return nof_matrix(x) if isinstance(x, F) else x * sp.eye(2)

    assert_matrix_equal(source(w_map.adjoint() * w_map), sp.eye(2))
    assert_matrix_equal(target(w_map * w_map.adjoint()), w * w.T)
    for S in (1, s, s.adjoint(), N(s), s + s.adjoint()):
        value = F.from_expr(S, operators=(s,))
        assert_matrix_equal(target(e._first_lattice.lift(value)), w * source(value) * w.T)
        assert_matrix_equal(target((w_map * value).target) * w, w * source(value))
    for X in (a, a.adjoint(), a**2, a.adjoint() * a**2, N(a), sp.sqrt(N(a) + 1)):
        x = e._first_lattice.parse_target(X)
        assert_matrix_equal(source(e.restrict(x)), w.T * target(x) * w)
        for Y in (1, a, a.adjoint()):
            y = e._first_lattice.parse_target(Y)
            assert_matrix_equal(
                target((x * w_map) * (w_map.adjoint() * y)),
                target(x) * w * w.T * target(y),
            )
            assert_matrix_equal(
                source((w_map.adjoint() * x) * (y * w_map)),
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
        x = e._first_lattice.parse_target(X)
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


def test_lift_bilateral_numbers():
    a, b = LadderOp("a"), LadderOp("b")
    e = Embedding({b: a, N(b): N(a) + 2}, reference={a: -2})
    for S in (b, b.adjoint(), N(b), b.adjoint() * N(b) * b):
        value = F.from_expr(S, operators=(b,))
        assert_matrix_equal(
            nof_matrix(e._first_lattice.lift(value), [range(-4, 3)]),
            nof_matrix(value, [range(-2, 5)]),
        )


def test_existing_nof_conversion_preserves_values():
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    e = Embedding({s: a}, reference={a: 0, b: 0})
    value = F.from_expr(sp.sqrt(N(a) + 1) * a.adjoint(), operators=(a,))
    result = e._first_lattice.parse_target(value)
    assert_matrix_equal(
        nof_matrix(result, [range(4), range(2)]),
        sp.kronecker_product(nof_matrix(value, [range(4)]), sp.eye(2)),
    )

    # A source name may also name a frozen target: roles follow the mapping.
    roles = Embedding({a: b}, reference={a: 0, b: 0})
    assert nof_matrix(roles.restrict(N(b)), [range(3)]) == sp.diag(0, 1, 2)
    assert roles.restrict(N(a)) == 0


@pytest.mark.parametrize("mode_type", [BosonOp, LadderOp])
@pytest.mark.parametrize("sign", [-1, 1])
def test_restrict_symbolic_ladder_power(mode_type, sign):
    a, b = mode_type("a"), mode_type("b")
    generators = {b: a}
    if mode_type is LadderOp:
        generators[N(b)] = N(a)
    embedding = Embedding(generators, reference={a: 0})
    k = sp.Symbol("k", integer=True, positive=True)
    target = a if sign == 1 else a.adjoint()
    with pytest.raises(NotImplementedError, match="concrete integers"):
        embedding.restrict(target**k)


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


@pytest.mark.parametrize("encoding", ["spin", "pair", "frozen", "hole"])
def test_fermion_encoding_matrix_elements(encoding):
    a, b, f = map(FermionOp, ("a", "b", "f"))
    q = SigmaMinus("q")
    generators, reference, indices, phases = {
        "spin": ({q: b.adjoint() * a}, {a: 0, b: 1}, [1, 2], [1, 1]),
        "pair": ({q: b * a}, {a: 0, b: 0}, [0, 3], [1, 1]),
        "frozen": ({f: b}, {a: 1, b: 0}, [2, 3], [1, -1]),
        "hole": ({f: b.adjoint()}, {a: 0, b: 1}, [1, 0], [1, 1]),
    }[encoding]
    embedding = Embedding(generators, reference=reference)
    w = sp.eye(4)[:, indices] * sp.diag(*phases)
    for expression in (a, b, b * a, b.adjoint() * a, N(a), N(b), a * a.adjoint()):
        target = nof_matrix(F.from_expr(expression, (a, b)))
        actual = F.from_expr(embedding.restrict(expression), tuple(generators))
        assert_matrix_equal(nof_matrix(actual), w.T * target * w)


@pytest.mark.parametrize("mode_type", [BosonOp, FermionOp])
def test_map_substitution_and_reconstruction(mode_type):
    a, b, z = map(mode_type, ("a", "b", "0z"))
    q = mode_type("q")
    g = s.Symbol("g", real=True)
    e = Embedding({q: a}, reference={a: 0, b: 1})
    renamed = Embedding({q: a}, reference={a: 0, z: 1})
    w = e._retained_frame(1)[0, 0]
    x = NumberOrderedForm.from_expr(g * N(b) + a.adjoint() * b)
    expected = (
        NumberOrderedForm.from_expr(2 * N(z) + a.adjoint() * z)
        * renamed._retained_frame(1)[0, 0]
    )
    for value in (x * w, (x * w).adjoint()):
        for restored in (
            value.func(*value.args),
            pickle.loads(pickle.dumps(value)),
        ):
            assert restored == value
        assert value.adjoint().adjoint() == value
    for method in ("subs", "xreplace"):
        replaced = getattr(x * w, method)({g: 2})
        actual = replaced.xreplace({b: z, b.adjoint(): z.adjoint()})
        assert actual == expected
        assert replaced.adjoint().xreplace({b: z}) == expected.adjoint()
    assert (x * w) ** 1 == x * w
    with pytest.raises(ValueError):
        (x * w) ** 2


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
    assert value.subs(c, 0) == 0
    assert value.xreplace({c: 0}) == 0
    assert value.xreplace({q: SigmaMinus("z")}).as_expr() == value.as_expr().xreplace(
        {q: SigmaMinus("z")}
    )
    with pytest.raises(ValueError, match="opposite matching"):
        value * value


def test_zero_maps_on_retained_domain():
    from pymablock.number_ordered_form import NumberOperator as N

    a, q = BosonOp("a"), SigmaMinus("q")
    embedding = Embedding({q: a}, reference={a: 0})
    w = embedding._retained_frame(1)[0, 0]
    complement = embedding._complement_frame(1)[0, 0]
    for value in (
        w.adjoint() * F.from_expr(N(a)) * complement,
        complement * w,
        F.from_expr(N(a) * (N(a) - 1)) * w,
    ):
        assert value.is_zero is True
    assert (F.from_expr(a.adjoint()) * w).is_zero is not True


def test_nof_defers_to_map_priority():
    a, q = BosonOp("a"), SigmaMinus("q")
    w = Embedding({q: a}, reference={a: 0})._retained_frame(1)[0, 0]
    value = F.from_expr(a)
    assert value.__mul__(w) is NotImplemented
    assert (value * w).target == value


@pytest.mark.parametrize("generic_gap", [False, True])
def test_target_division_excludes_unselected_poles(generic_gap):
    """The target-coordinate solve stays finite on and outside a truncated boson."""

    a, q = BosonOp("a"), SigmaMinus("q")
    e = Embedding({q: a}, reference={a: 0})
    lattice = e._first_lattice
    omega, alpha = sp.symbols("omega alpha", nonzero=True)
    gap = omega + alpha * N(a) if generic_gap else N(a) - 2
    divided = solve_scalar(lattice.projector, gap, 0) * lattice.projector
    expected = (
        sp.diag(1 / omega, 1 / (omega + alpha), 0, 0)
        if generic_gap
        else sp.diag(-sp.Rational(1, 2), -1, 0, 0)
    )
    assert_matrix_equal(nof_matrix(divided, [range(4)]), expected)
    left = lattice.parse_target(a + a.adjoint())
    assert_matrix_equal(
        nof_matrix(left * divided, [range(4)]), nof_matrix(left, [range(4)]) * expected
    )


@pytest.mark.xfail(
    strict=True,
    reason="Indicator algebra does not infer exclusivity of different floor conditions",
)
def test_mutually_exclusive_floor_indicators():
    """Even and odd sublattice supports have an empty intersection."""

    n = sp.Symbol("n", integer=True)
    even = _equal_value_indicator(n / 2, sp.floor(n / 2))
    odd = _equal_value_indicator((n - 1) / 2, sp.floor((n - 1) / 2))
    assert _divide_by_energy_gap(even * odd, sp.S.Zero, (n,)) == 0


def test_lattice_consumers_normalize_equal_nofs():
    """Inject the same mode-list substitution that SymPy's equality cache permits."""
    a, b, q = BosonOp("a"), BosonOp("b"), SigmaMinus("q")
    lattice = Embedding({q: a}, reference={a: 0})._first_lattice
    plain = F((), {(): sp.Integer(2)})
    cached = F((a, b), {(0, 0): sp.Integer(2)})
    assert plain == cached and hash(plain) == hash(cached)
    assert lattice.restrict(cached) == lattice.restrict(plain) == 2
    assert lattice.lift(cached) == lattice.lift(plain) == 2 * lattice.projector
