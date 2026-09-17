"""Occupation selections, target algebras, and projected perturbation operations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pytest
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus
from sympy.physics.quantum.spin import JminusOp, JzOp

from pymablock import block_diagonalize
from pymablock._operator_embedding import ModuleEndomorphism, OperatorMap
from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.operator_embedding import _NOFTransition
from pymablock.second_quantization import Embedding
from pymablock.series import zero
from pymablock.tests.second_quantization_helpers import (
    occupation_matrices,
    operator_matrix,
)


def test_fermion_embedding_returns_target_nof() -> None:
    source, virtual = FermionOp("source"), FermionOp("virtual")
    target = FermionOp("target")
    source_energy, virtual_energy, coupling = sympy.symbols(
        "source_energy virtual_energy coupling",
        nonzero=True,
        real=True,
    )
    h_0 = source_energy * NumberOperator(source) + virtual_energy * NumberOperator(
        virtual
    )
    perturbation = coupling * (Dagger(virtual) * source + Dagger(source) * virtual)
    embedding = Embedding(
        target=(target,), occupations={source: NumberOperator(target), virtual: 0}
    )

    effective, *_ = block_diagonalize(
        [h_0, perturbation],
        subspace_eigenvectors=embedding,
    )

    assert effective[0, 0, 2] == NumberOrderedForm.from_expr(
        coupling**2 * NumberOperator(target) / (source_energy - virtual_energy),
        operators=(target,),
    )


def test_frozen_fermion_phase_is_internal() -> None:
    fixed, source = FermionOp("a_fixed"), FermionOp("b_source")
    target, virtual = FermionOp("target"), FermionOp("virtual")
    source_energy, virtual_energy, coupling = sympy.symbols(
        "source_energy virtual_energy coupling",
        nonzero=True,
        real=True,
    )
    h_0 = source_energy * NumberOperator(source) + virtual_energy * NumberOperator(
        virtual
    )
    perturbation = coupling * (Dagger(virtual) * source + Dagger(source) * virtual)

    effective, *_ = block_diagonalize(
        [h_0, perturbation],
        subspace_eigenvectors=Embedding(
            target=(target,),
            occupations={fixed: 1, source: NumberOperator(target), virtual: 0},
        ),
    )

    assert effective[0, 0, 2].terms[(0,)] == (
        coupling**2
        * NumberOrderedForm.from_expr(NumberOperator(target), operators=(target,)).terms[
            (0,)
        ]
        / (source_energy - virtual_energy)
    )


def test_nonbinary_target_uses_package_matrix_interface() -> None:
    source = BosonOp("source")
    retained_spin = JminusOp("S")
    frequency, coupling = sympy.symbols(
        "frequency coupling",
        nonzero=True,
        real=True,
    )
    embedding = Embedding(target={retained_spin: 3}, occupations={source: JzOp("S") + 1})

    effective, *_ = block_diagonalize(
        [
            frequency * NumberOperator(source),
            coupling * (source + Dagger(source)),
        ],
        subspace_eigenvectors=embedding,
    )

    assert isinstance(effective[0, 0, 2], sympy.MatrixBase)
    assert effective[0, 0, 2] == sympy.diag(
        0,
        0,
        -3 * coupling**2 / frequency,
    )


def test_bosonic_excursion_is_not_a_product_of_compressions() -> None:
    """The spin target retains the virtual second boson level in products."""

    a, s = BosonOp("a"), SigmaMinus("s")
    backend = Embedding(target=(s,), occupations={a: NumberOperator(s)})

    def _pullback(expr):
        return backend._pullback(backend._source_form(expr))

    def expected(expr):
        return NumberOrderedForm.from_expr(expr, operators=(s,))

    assert _pullback(a) == expected(s)
    assert _pullback(a * Dagger(a)) == expected(1 + NumberOperator(s))
    assert _pullback(a) * _pullback(Dagger(a)) == expected(1 - NumberOperator(s))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("frozen", [0, 1])
def test_retained_fermions_preserve_car_and_mode_correspondence(reverse, frozen) -> None:
    """Frozen particles and permutations must not change the target CAR."""

    a, fixed, b = (FermionOp(name) for name in ("a", "m", "z"))
    f, g = FermionOp("f"), FermionOp("g")
    left, right = (g, f) if reverse else (f, g)
    backend = Embedding(
        target=(g, f),
        occupations={
            a: NumberOperator(left),
            fixed: frozen,
            b: NumberOperator(right),
        },
    )
    assert backend.target.operators == (f, g)
    for source, target in (
        (a, left),
        (b, right),
        (Dagger(a), Dagger(left)),
        (Dagger(b), Dagger(right)),
        (Dagger(a) * b, Dagger(left) * right),
        (a * b, left * right),
        (b * Dagger(b), right * Dagger(right)),
    ):
        result = backend._pullback(backend._source_form(source))
        assert result == NumberOrderedForm.from_expr(target, operators=(f, g))


def test_spin_in_two_fermions_uses_single_occupancy() -> None:
    """The target is one spin, with charge-changing source actions projected out."""

    up, down = FermionOp("up"), FermionOp("down")
    s = SigmaMinus("s")
    backend = Embedding(
        target=(s,), occupations={up: NumberOperator(s), down: 1 - NumberOperator(s)}
    )

    def expected(expr):
        return NumberOrderedForm.from_expr(expr, operators=(s,))

    assert backend._pullback(backend._source_form(NumberOperator(up))) == expected(
        NumberOperator(s)
    )
    assert backend._pullback(backend._source_form(NumberOperator(down))) == expected(
        1 - NumberOperator(s)
    )
    assert not backend._pullback(backend._source_form(up))
    assert not backend._pullback(
        backend._source_form(NumberOperator(up) * NumberOperator(down))
    )
    # In canonical source order (down, up), this bilinear takes down to up.
    assert backend._pullback(backend._source_form(Dagger(up) * down)) == expected(
        Dagger(s)
    )


@pytest.mark.parametrize(
    "rule, error",
    [
        (lambda _s, _t: {BosonOp("a"): 0}, "injective"),
        (lambda _s, t: {BosonOp("a"): NumberOperator(t)}, "declared target"),
        (lambda s, _t: {BosonOp("a"): s}, "declared target"),
        (lambda s, _t: {BosonOp("a"): NumberOperator(s) / 2}, "integer affine"),
        (lambda s, _t: {BosonOp("a"): NumberOperator(s) - 1}, "nonnegative"),
        (lambda s, _t: {FermionOp("a"): 2 * NumberOperator(s)}, "zero or one"),
    ],
)
def test_invalid_occupation_rules_are_rejected(rule, error) -> None:
    s, t = SigmaMinus("s"), SigmaMinus("t")
    with pytest.raises(ValueError, match=error):
        Embedding(target=(s,), occupations=rule(s, t))


def test_invalid_target_declarations_are_rejected() -> None:
    s = SigmaMinus("s")
    a = BosonOp("a")
    with pytest.raises(ValueError, match="distinct"):
        Embedding(target=(s, s), occupations={a: NumberOperator(s)})
    with pytest.raises(ValueError, match="explicit dimension"):
        Embedding(target=(JminusOp("S"),), occupations={a: 0})
    with pytest.raises(ValueError, match="dimension two"):
        Embedding(target={s: 3}, occupations={a: NumberOperator(s)})


def test_fermion_targets_reject_non_direct_encodings() -> None:
    f = FermionOp("f")
    a, b = FermionOp("a"), FermionOp("b")
    with pytest.raises(ValueError, match="direct"):
        Embedding(target=(f,), occupations={a: 1 - NumberOperator(f)})
    with pytest.raises(ValueError, match="exactly one"):
        Embedding(target=(f,), occupations={a: NumberOperator(f), b: NumberOperator(f)})


def test_binary_validation_does_not_enumerate_target(monkeypatch) -> None:
    from pymablock.operator_embedding import _TargetSpace

    def forbidden(_self):
        raise AssertionError("Target enumeration is not needed")

    monkeypatch.setattr(_TargetSpace, "states", property(forbidden))
    spins = tuple(SigmaMinus(f"s{i}") for i in range(20))
    embedding = Embedding(
        target=spins,
        occupations={BosonOp(f"a{i}"): NumberOperator(s) for i, s in enumerate(spins)},
    )
    assert embedding.target.dimension == 2**20


def test_finite_virtual_resonance_still_raises() -> None:
    """Dropping transitions within P must not hide a resonant state in Q."""
    a = BosonOp("a")
    n = NumberOperator(a)
    embedding = Embedding(target={JminusOp("S"): 3}, occupations={a: JzOp("S") + 1})
    # Retained n=2 and excluded n=3 both have energy -6.
    effective, *_ = block_diagonalize(
        [n * (n - 5), a + Dagger(a)], subspace_eigenvectors=embedding
    )
    with pytest.raises(ZeroDivisionError, match="degenerate"):
        _ = effective[0, 0, 2]


@pytest.mark.parametrize("finite", [False, True])
def test_boson_annihilation_preserves_occupation_dependent_denominator(finite):
    """Intermediate energies are 10 and 11, versus retained energies 1 and 4."""
    a, b = BosonOp("a"), BosonOp("b")
    s = JminusOp("S") if finite else SigmaMinus("s")
    n = JzOp("S") + sympy.S.Half if finite else NumberOperator(s)
    embedding = Embedding(target={s: 2}, occupations={a: 1 + n, b: 0})
    effective, *_ = block_diagonalize(
        [NumberOperator(a) ** 2 + 10 * NumberOperator(b), Dagger(b) * a + Dagger(a) * b],
        subspace_eigenvectors=embedding,
    )
    expected = (
        sympy.diag(-sympy.Rational(1, 9), -sympy.Rational(2, 7))
        if finite
        else NumberOrderedForm.from_expr(-(1 - n) / 9 - 2 * n / 7, operators=(s,))
    )
    assert effective[0, 0, 2] == expected


@pytest.mark.parametrize("coupled", [False, True])
def test_sector_resonance_requires_nonzero_virtual_channel(coupled):
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    n = NumberOperator(a)
    effective, *_ = block_diagonalize(
        [n + (1 - n) * NumberOperator(b), (1 if coupled else 1 - n) * (b + Dagger(b))],
        subspace_eigenvectors=Embedding(
            target=(s,), occupations={a: NumberOperator(s), b: 0}
        ),
    )
    if coupled:
        with pytest.raises(ZeroDivisionError, match="degenerate"):
            _ = effective[0, 0, 2]
    else:
        assert effective[0, 0, 2] == NumberOrderedForm.from_expr(
            NumberOperator(s) - 1, operators=(s,)
        )


def test_empty_boson_channel_does_not_create_a_resonance():
    """The nominal zero gap at n=0 has zero annihilation amplitude."""
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    effective, *_ = block_diagonalize(
        [NumberOperator(a) ** 2 - NumberOperator(b), Dagger(b) * a + Dagger(a) * b],
        subspace_eigenvectors=Embedding(
            target=(s,), occupations={a: NumberOperator(s), b: 0}
        ),
    )
    assert effective[0, 0, 2] == NumberOrderedForm.from_expr(
        NumberOperator(s) / 2, operators=(s,)
    )


@dataclass(frozen=True)
class MatrixEmbedding:
    """Small exact embedding used only to lower formal maps in tests."""

    name: str = "W"

    @property
    def bridge(self):
        return sympy.ImmutableMatrix([[1, 0], [0, 1], [0, 0]])

    @property
    def projector(self):
        bridge = self.bridge
        return sympy.eye(3) - bridge * bridge.adjoint()

    @property
    def _target_zero(self):
        return sympy.ImmutableMatrix(sympy.zeros(2))

    @property
    def _target_identity(self):
        return sympy.ImmutableMatrix(sympy.eye(2))

    def _pullback(self, source):
        return self.bridge.adjoint() * source * self.bridge


def _lower(operator_map):
    embedding = operator_map.embedding
    result = sympy.zeros(3, 2)
    for source, target in operator_map.terms:
        result += embedding.projector * source * embedding.bridge * target
    return sympy.ImmutableMatrix(result)


def test_storage_combines_equal_source_factors() -> None:
    embedding = MatrixEmbedding()
    source = sympy.ImmutableMatrix([[0, 0, 0], [0, 0, 0], [1, 2, 0]])
    first = sympy.ImmutableMatrix([[1, 2], [0, 0]])
    second = sympy.ImmutableMatrix([[0, -2], [3, 0]])

    operator_map = OperatorMap(
        embedding,
        (
            (source, first),
            (source, second),
            (source, sympy.zeros(2)),
        ),
    )

    assert operator_map.terms == ((source, first + second),)
    assert not OperatorMap(embedding, ())
    assert operator_map + (-operator_map) is zero


def test_left_and_right_actions_match_explicit_projection() -> None:
    embedding = MatrixEmbedding()
    source = sympy.ImmutableMatrix([[0, 0, 0], [0, 0, 0], [1, 2, 0]])
    left = sympy.ImmutableMatrix([[1, 0, 1], [0, 2, 0], [3, 0, 4]])
    right = sympy.ImmutableMatrix([[1, 2], [3, 4]])
    operator_map = OperatorMap.from_source(embedding, source)

    assert _lower(operator_map.left(left)) == (
        embedding.projector * left * _lower(operator_map)
    )
    assert _lower(operator_map.right(right)) == _lower(operator_map) * right


def test_inner_product_matches_explicit_maps() -> None:
    embedding = MatrixEmbedding()
    x = sympy.ImmutableMatrix([[0, 0, 1], [0, 0, 2], [1, 3, 0]])
    y = sympy.ImmutableMatrix([[1, 0, 0], [0, 1, 0], [4, 5, 0]])
    a = sympy.ImmutableMatrix([[1, 2], [0, 1]])
    b = sympy.ImmutableMatrix([[2, 0], [3, 1]])
    left = OperatorMap(embedding, ((x, a),))
    right = OperatorMap(embedding, ((y, b),))

    assert left.inner(right) == _lower(left).adjoint() * _lower(right)


def test_arithmetic_requires_one_embedding() -> None:
    first_embedding = MatrixEmbedding("first")
    second_embedding = MatrixEmbedding("second")
    source = sympy.ImmutableMatrix([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    first = OperatorMap.from_source(first_embedding, source)
    second = OperatorMap.from_source(second_embedding, source)

    assert _lower(3 * first / 2) == sympy.Rational(3, 2) * _lower(first)
    with pytest.raises(ValueError, match="same embedding"):
        _ = first + second
    with pytest.raises(ValueError, match="same embedding"):
        first.inner(second)


def test_lazy_complement_endomorphisms_match_explicit_projection() -> None:
    embedding = MatrixEmbedding()
    source = sympy.ImmutableMatrix([[0, 0, 1], [0, 0, 2], [1, 3, 0]])
    left = sympy.ImmutableMatrix([[1, 0, 1], [0, 2, 0], [3, 0, 4]])
    column = OperatorMap.from_source(embedding, source)

    source_action = ModuleEndomorphism.source(embedding, left)
    assert _lower(source_action.apply(column)) == (
        embedding.projector * left * embedding.projector * _lower(column)
    )

    rank_one = ModuleEndomorphism.rank_one(column, column)
    assert _lower(rank_one.apply(column)) == (
        _lower(column) * _lower(column).adjoint() * _lower(column)
    )


def test_adjoint_preserves_declared_basis_after_cached_equal_expression():
    """Equal identities can have different declared, unused generators."""
    from sympy.physics.quantum.pauli import SigmaMinus

    from pymablock._operator_embedding import _adjoint
    from pymablock.number_ordered_form import NumberOrderedForm

    first = (SigmaMinus("a"),)
    second = (SigmaMinus("x"), SigmaMinus("y"))
    a = NumberOrderedForm.from_expr(sympy.S.One, operators=first)
    b = NumberOrderedForm.from_expr(sympy.S.One, operators=second)
    _ = a.adjoint()
    result = _adjoint(b)
    assert tuple(result.operators) == second
    assert result.as_expr() == 1


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
    s = JminusOp("S") if finite else SigmaMinus("s")
    ns = JzOp("S") + sympy.S.Half if finite else N(s)
    first, second = (g, f) if reverse else (f, g)
    backend = Embedding(
        target={s: 2, g: 2, f: 2},
        occupations={up: ns, down: 1 - ns, left: N(first), right: N(second)},
    )
    matrices = occupation_matrices(modes, [(0, 1)] * 4)
    source_states = list(product((0, 1), repeat=4))
    columns = []
    direct = {first: left, second: right}
    for state in backend.target.states:
        values = dict(zip(backend.target.operators, state, strict=True))
        baseline = tuple(
            values[s] if op == up else 1 - values[s] if op == down else 0 for op in modes
        )
        column = np.eye(16)[:, source_states.index(baseline)]
        for target in (g, f):
            if values[target]:
                column = matrices[Dagger(direct[target])] @ column
        columns.append(column)
    w = np.column_stack(columns)
    np.testing.assert_allclose(w.T @ w, np.eye(8))
    target_matrices = occupation_matrices(
        tuple(
            SigmaMinus("reference_spin") if op == s and finite else op
            for op in backend.target.operators
        ),
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
        result = backend._pullback(backend._source_form(expression))
        actual = (
            np.asarray(result, dtype=complex)
            if finite
            else operator_matrix(result, target_matrices).toarray()
        )
        expected = w.T @ operator_matrix(expression, matrices) @ w
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    # The chosen columns preserve each directly retained fermion generator.
    for source, target in ((left, first), (right, second)):
        np.testing.assert_allclose(
            w.T @ matrices[source] @ w, target_matrices[target].toarray()
        )


@pytest.mark.parametrize("finite", [False, True])
def test_mixed_target_second_order(finite):
    """A spin-dependent hopping amplitude gives -N(f) N(s)**2 / 3."""
    a, v = FermionOp("a"), FermionOp("v")
    b, f = BosonOp("b"), FermionOp("f")
    s = JminusOp("S") if finite else SigmaMinus("s")
    ns = JzOp("S") + 1 if finite else N(s)
    embedding = Embedding(
        target={f: 2, s: 3 if finite else 2}, occupations={a: N(f), v: 0, b: ns}
    )
    h, *_ = block_diagonalize(
        [N(a) + 4 * N(v) + 10 * N(b), N(b) * (Dagger(v) * a + Dagger(a) * v)],
        subspace_eigenvectors=embedding,
    )
    if finite:
        assert h[0, 0, 2] == sympy.diag(
            0, 0, 0, 0, -sympy.Rational(1, 3), -sympy.Rational(4, 3)
        )
    else:
        assert h[0, 0, 2] == NumberOrderedForm.from_expr(
            -N(f) * N(s) / 3, operators=embedding.target.operators
        )


def _only_transition(expression, operators):
    form = NumberOrderedForm.from_expr(expression, operators=operators)
    (transition,) = tuple(_NOFTransition.from_form(form))
    return transition


def test_boson_transition_contains_kinematic_weight() -> None:
    """A NOF transition owns both its occupation shift and matrix element."""
    boson = BosonOp("a")

    lowering = _only_transition(boson, (boson,)).apply((3,))
    raising = _only_transition(Dagger(boson), (boson,)).apply((3,))

    assert lowering.output_state == (2,)
    assert lowering.weight == sympy.sqrt(3)
    assert raising.output_state == (4,)
    assert raising.weight == 2


def test_fermion_transition_contains_cross_mode_parity() -> None:
    """Fermion transitions retain their Jordan-Wigner parity."""
    first, second = FermionOp("first"), FermionOp("second")
    lowering = _only_transition(second, (first, second))

    assert lowering.apply((0, 1)).weight == 1
    assert lowering.apply((1, 1)).weight == -1


def test_state_selection_solves_integer_target_shift() -> None:
    """Selecting even boson occupations makes a two-step source shift binary."""
    a, s = BosonOp("a"), SigmaMinus("s")
    backend = Embedding(target=(s,), occupations={a: 2 * N(s)})
    assert backend._target_shift((2,)) == (1,)
    assert backend._target_shift((-2,)) == (-1,)
    assert backend._target_shift((1,)) is None


def test_frozen_particle_sets_retained_fermion_phase() -> None:
    """The target definition absorbs the sign from an earlier occupied mode."""
    fixed, source, target = (FermionOp(name) for name in ("a", "b", "f"))
    backend = Embedding(target=(target,), occupations={fixed: 1, source: N(target)})
    result = backend._pullback(backend._source_form(source))
    assert result == NumberOrderedForm.from_expr(target, operators=(target,))


def test_multiple_boson_annihilations_stop_at_vacuum():
    """The common symbolic action also handles forbidden concrete transitions."""
    a = BosonOp("a")
    transition = _only_transition(a**2, (a,))
    assert transition.apply((0,)) is None
    assert transition.apply((1,)) is None
    assert transition.apply((3,)).weight == sympy.sqrt(6)


def test_forbidden_creation_ignores_coefficient_pole():
    """A zero ladder amplitude excludes a sector before its coefficient is used."""
    from pymablock.operator_embedding import _number_symbols

    f = FermionOp("f")
    (n,) = _number_symbols((f,))
    form = NumberOrderedForm((f,), {(-1,): 1 / (1 - n)}, validate=False)
    (transition,) = _NOFTransition.from_form(form)
    assert transition.apply((1,)) is None
    assert transition.apply((0,)).weight == 1
