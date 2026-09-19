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

from pymablock import block_diagonalize
from pymablock._operator_embedding import _ComplementBlock, _CouplingBlock
from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.operator_embedding import _NOFTransition, _number_symbols
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
    embedding = Embedding({target: source}, reference={source: 0, virtual: 0})

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
            {target: source},
            reference={fixed: 1, source: 0, virtual: 0},
        ),
    )

    assert effective[0, 0, 2].terms[(0,)] == (
        coupling**2
        * NumberOrderedForm.from_expr(NumberOperator(target), operators=(target,)).terms[
            (0,)
        ]
        / (source_energy - virtual_energy)
    )


def test_reference_list_returns_matrix() -> None:
    source = BosonOp("source")
    frequency, coupling = sympy.symbols(
        "frequency coupling",
        nonzero=True,
        real=True,
    )
    embedding = Embedding(reference=[{source: n} for n in range(3)])

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
    backend = Embedding({s: a}, reference={a: 0})

    def _pullback(expr):
        return backend.restrict(expr)

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
        {left: a, right: b},
        reference={a: 0, fixed: frozen, b: 0},
    )
    assert backend._basis._target_operators == (f, g)
    for source, target in (
        (a, left),
        (b, right),
        (Dagger(a), Dagger(left)),
        (Dagger(b), Dagger(right)),
        (Dagger(a) * b, Dagger(left) * right),
        (a * b, left * right),
        (b * Dagger(b), right * Dagger(right)),
    ):
        result = backend.restrict(source)
        assert result == NumberOrderedForm.from_expr(target, operators=(f, g))


def test_spin_in_two_fermions_uses_single_occupancy() -> None:
    """The target is one spin, with charge-changing source actions projected out."""

    up, down = FermionOp("up"), FermionOp("down")
    s = SigmaMinus("s")
    backend = Embedding({s: Dagger(down) * up}, reference={up: 0, down: 1})

    def expected(expr):
        return NumberOrderedForm.from_expr(expr, operators=(s,))

    assert backend.restrict(NumberOperator(up)) == expected(NumberOperator(s))
    assert backend.restrict(NumberOperator(down)) == expected(1 - NumberOperator(s))
    assert not backend.restrict(up)
    assert not backend.restrict(NumberOperator(up) * NumberOperator(down))
    # In canonical source order (down, up), this bilinear takes down to up.
    assert backend.restrict(Dagger(up) * down) == expected(Dagger(s))


def test_invalid_generator_definitions():
    s, t = SigmaMinus("s"), SigmaMinus("t")
    a, b = BosonOp("a"), BosonOp("b")
    f = FermionOp("f")
    with pytest.raises(ValueError, match="reference"):
        Embedding({s: a}, reference={b: 0})
    with pytest.raises(ValueError, match="normalized"):
        Embedding({s: 2 * a}, reference={a: 0})
    with pytest.raises(ValueError, match="independent"):
        Embedding({s: a, t: a}, reference={a: 0})
    with pytest.raises(ValueError, match="parity"):
        Embedding({f: a}, reference={a: 0})
    with pytest.raises(ValueError, match="normalized"):
        Embedding({s: a + b}, reference={a: 0, b: 0})


def test_particle_hole_and_pair_encodings():
    f, a, b = (FermionOp(name) for name in ("f", "a", "b"))
    hole = Embedding({f: Dagger(a)}, reference={a: 1})
    assert hole.restrict(N(a)) == NumberOrderedForm.from_expr(1 - N(f), operators=(f,))
    assert hole.restrict(a) == NumberOrderedForm.from_expr(Dagger(f), operators=(f,))
    s = SigmaMinus("s")
    pair = Embedding({s: b * a}, reference={a: 0, b: 0})
    assert pair.restrict(b * a) == NumberOrderedForm.from_expr(s, operators=(s,))
    assert pair.restrict(N(a) * N(b)) == NumberOrderedForm.from_expr(N(s), operators=(s,))


def test_binary_validation_does_not_enumerate_target(monkeypatch) -> None:
    from pymablock.operator_embedding import _ReferenceBasis

    def forbidden(_self, _source):
        raise AssertionError("Target enumeration is not needed")

    monkeypatch.setattr(_ReferenceBasis, "_actions", forbidden)
    spins = tuple(SigmaMinus(f"s{i}") for i in range(20))
    embedding = Embedding(
        {s: BosonOp(f"a{i}") for i, s in enumerate(spins)},
        reference={BosonOp(f"a{i}"): 0 for i in range(len(spins))},
    )
    assert set(embedding.restrict(1).operators) == set(spins)


def test_finite_virtual_resonance_still_raises() -> None:
    """Dropping transitions within P must not hide a resonant state in Q."""
    a = BosonOp("a")
    n = NumberOperator(a)
    embedding = Embedding(reference=[{a: n} for n in range(3)])
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
    s = SigmaMinus("s")
    n = NumberOperator(s)
    embedding = (
        Embedding(
            {s: a / sympy.sqrt(2)},
            reference={a: 1, b: 0},
        )
        if not finite
        else Embedding(reference=[{a: n, b: 0} for n in (1, 2)])
    )
    effective, *_ = block_diagonalize(
        [NumberOperator(a) ** 2 + 10 * NumberOperator(b), Dagger(b) * a + Dagger(a) * b],
        subspace_eigenvectors=embedding,
    )
    expected = NumberOrderedForm.from_expr(-(1 - n) / 9 - 2 * n / 7, operators=(s,))
    assert effective[0, 0, 2] == (expected.to_matrix() if finite else expected)


@pytest.mark.parametrize("coupled", [False, True])
def test_sector_resonance_requires_nonzero_virtual_channel(coupled):
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    n = NumberOperator(a)
    effective, *_ = block_diagonalize(
        [n + (1 - n) * NumberOperator(b), (1 if coupled else 1 - n) * (b + Dagger(b))],
        subspace_eigenvectors=Embedding({s: a}, reference={a: 0, b: 0}),
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
        subspace_eigenvectors=Embedding({s: a}, reference={a: 0, b: 0}),
    )
    assert effective[0, 0, 2] == NumberOrderedForm.from_expr(
        NumberOperator(s) / 2, operators=(s,)
    )


@dataclass(frozen=True)
class _MatrixBasis:
    """Explicit source matrices used to check the implicit discarded-space blocks."""

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
    embedding = operator_map.basis
    result = sympy.zeros(3, 2)
    for source, target in operator_map.terms:
        result += embedding.projector * source * embedding.bridge * target
    return sympy.ImmutableMatrix(result)


def test_storage_combines_equal_source_factors() -> None:
    embedding = _MatrixBasis()
    source = sympy.ImmutableMatrix([[0, 0, 0], [0, 0, 0], [1, 2, 0]])
    first = sympy.ImmutableMatrix([[1, 2], [0, 0]])
    second = sympy.ImmutableMatrix([[0, -2], [3, 0]])

    operator_map = _CouplingBlock(
        embedding,
        (
            (source, first),
            (source, second),
            (source, sympy.zeros(2)),
        ),
    )

    assert operator_map.terms == ((source, first + second),)
    assert not _CouplingBlock(embedding, ())
    assert operator_map + (-operator_map) is zero


def test_left_and_right_actions_match_explicit_projection() -> None:
    embedding = _MatrixBasis()
    source = sympy.ImmutableMatrix([[0, 0, 0], [0, 0, 0], [1, 2, 0]])
    left = sympy.ImmutableMatrix([[1, 0, 1], [0, 2, 0], [3, 0, 4]])
    right = sympy.ImmutableMatrix([[1, 2], [3, 4]])
    operator_map = _CouplingBlock.from_source(embedding, source)

    assert _lower(operator_map.left(left)) == (
        embedding.projector * left * _lower(operator_map)
    )
    assert _lower(operator_map.right(right)) == _lower(operator_map) * right


def test_inner_product_matches_explicit_maps() -> None:
    embedding = _MatrixBasis()
    x = sympy.ImmutableMatrix([[0, 0, 1], [0, 0, 2], [1, 3, 0]])
    y = sympy.ImmutableMatrix([[1, 0, 0], [0, 1, 0], [4, 5, 0]])
    a = sympy.ImmutableMatrix([[1, 2], [0, 1]])
    b = sympy.ImmutableMatrix([[2, 0], [3, 1]])
    left = _CouplingBlock(embedding, ((x, a),))
    right = _CouplingBlock(embedding, ((y, b),))

    assert left.inner(right) == _lower(left).adjoint() * _lower(right)


def test_arithmetic_requires_one_embedding() -> None:
    first_embedding = _MatrixBasis("first")
    second_embedding = _MatrixBasis("second")
    source = sympy.ImmutableMatrix([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    first = _CouplingBlock.from_source(first_embedding, source)
    second = _CouplingBlock.from_source(second_embedding, source)

    assert _lower(3 * first / 2) == sympy.Rational(3, 2) * _lower(first)
    with pytest.raises(ValueError, match="same embedding"):
        _ = first + second
    with pytest.raises(ValueError, match="same embedding"):
        first.inner(second)


def test_lazy_complement_endomorphisms_match_explicit_projection() -> None:
    embedding = _MatrixBasis()
    source = sympy.ImmutableMatrix([[0, 0, 1], [0, 0, 2], [1, 3, 0]])
    left = sympy.ImmutableMatrix([[1, 0, 1], [0, 2, 0], [3, 0, 4]])
    column = _CouplingBlock.from_source(embedding, source)

    source_action = _ComplementBlock.source(left)
    assert _lower(source_action.apply(column)) == (
        embedding.projector * left * embedding.projector * _lower(column)
    )

    outer_action = _ComplementBlock.outer(column, column)
    assert _lower(outer_action.apply(column)) == (
        _lower(column) * _lower(column).adjoint() * _lower(column)
    )

    # Composition reverses under adjoint; complex scales must conjugate.
    action = source_action.compose(outer_action) / sympy.I + outer_action.compose(
        source_action
    )
    q_left = embedding.projector * left * embedding.projector
    outer = _lower(column) * _lower(column).adjoint()
    explicit = q_left * outer / sympy.I + outer * q_left
    assert _lower(action.apply(column)) == explicit * _lower(column)
    assert _lower(action.adjoint().apply(column)) == explicit.adjoint() * _lower(column)
    # Dividing an adjoint block conjugates the divisor in its stored column.
    divided_row = column.adjoint() / sympy.I
    assert _lower(divided_row.adjoint()).adjoint() == _lower(column).adjoint() / sympy.I
    assert action.apply(zero) is zero
    assert source_action.compose(outer_action - outer_action).apply(column) is zero


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
    s = SigmaMinus("s")
    first, second = (g, f) if reverse else (f, g)
    backend = Embedding(
        {s: Dagger(down) * up, first: left, second: right},
        reference={up: 0, down: 1, left: 0, right: 0},
    )
    matrices = occupation_matrices(modes, [(0, 1)] * 4)
    source_states = list(product((0, 1), repeat=4))
    columns = []
    direct = {first: left, second: right}
    for state in product((0, 1), repeat=len(backend._basis._target_operators)):
        values = dict(zip(backend._basis._target_operators, state, strict=True))
        baseline = tuple(int(op == down) for op in modes)
        column = np.eye(16)[:, source_states.index(baseline)]
        for target in reversed(backend._basis._target_operators):
            if values[target]:
                raising = (
                    matrices[Dagger(up)] @ matrices[down]
                    if target == s
                    else matrices[Dagger(direct[target])]
                )
                column = raising @ column
        columns.append(column)
    w = np.column_stack(columns)
    np.testing.assert_allclose(w.T @ w, np.eye(8))
    references = [
        dict(zip(modes, state, strict=True))
        for state in source_states
        if state[modes.index(up)] + state[modes.index(down)] == 1
    ]
    finite_embedding = Embedding(reference=references)
    selected = [source_states.index(tuple(ref[op] for op in modes)) for ref in references]
    target_matrices = occupation_matrices(
        backend._basis._target_operators,
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
        actual = operator_matrix(result, target_matrices).toarray()
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
    for source, target in ((left, first), (right, second)):
        np.testing.assert_allclose(
            w.T @ matrices[source] @ w, target_matrices[target].toarray()
        )


@pytest.mark.parametrize("finite", [False, True])
def test_mixed_target_second_order(finite):
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
        else Embedding(reference=[{a: n, b: k, v: 0} for n in range(2) for k in range(3)])
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
            -N(f) * N(s) ** 2 / 3, operators=embedding._basis._target_operators
        )
        assert (h[0, 0, 2] - expected).applyfunc(sympy.simplify).is_zero


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
    backend = Embedding({s: a**2 / sympy.sqrt(2)}, reference={a: 0})
    expected = NumberOrderedForm.from_expr(sympy.sqrt(2) * s)
    assert backend.restrict(a**2) == expected
    assert backend.restrict(Dagger(a) ** 2) == expected.adjoint()
    assert backend.restrict(a).is_zero


def test_frozen_particle_sets_retained_fermion_phase() -> None:
    """The target definition absorbs the sign from an earlier occupied mode."""
    fixed, source, target = (FermionOp(name) for name in ("a", "b", "f"))
    backend = Embedding({target: source}, reference={fixed: 1, source: 0})
    result = backend.restrict(source)
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


def test_reference_fixes_complex_phases_and_cross_relations():
    a, b = BosonOp("a"), BosonOp("b")
    s, t = SigmaMinus("s"), SigmaMinus("t")
    theta = sympy.Symbol("theta", real=True)
    phase = sympy.exp(sympy.I * theta)
    embedding = Embedding({s: phase * a}, reference={a: 0})
    result = embedding.restrict(a).as_expr()
    assert sympy.simplify(result - s / phase) == 0
    with pytest.raises(ValueError, match="target algebra"):
        Embedding({s: a, t: (1 - 2 * N(a)) * b}, reference={a: 0, b: 0})


def test_retained_infinite_modes_and_ladder_reference():
    from pymablock.number_ordered_form import LadderOp

    a, b = BosonOp("a"), BosonOp("b")
    source, ell = LadderOp("source"), LadderOp("ell")
    embedding = Embedding(
        {b: a, ell: source, N(ell): N(source) - 3},
        reference={a: 0, source: 3},
    )
    for expression, expected in (
        (a**3, b**3),
        (Dagger(a) ** 2 * (N(a) + 3) * a, Dagger(b) ** 2 * (N(b) + 3) * b),
        (source**4, ell**4),
        (N(source), N(ell) + 3),
        (a * Dagger(a), 1 + N(b)),
    ):
        difference = embedding.restrict(expression) - NumberOrderedForm.from_expr(
            expected, operators=(b, ell)
        )
        assert all(sympy.simplify(value) == 0 for value in difference.terms.values())
    with pytest.raises(ValueError, match="independent number"):
        Embedding({ell: source}, reference={source: 0})
    with pytest.raises(ValueError, match="reference index zero"):
        Embedding({ell: source, N(ell): N(source)}, reference={source: 3})


def test_boson_generator_normalization_is_not_renormalized_silently():
    a, b = BosonOp("a"), BosonOp("b")
    with pytest.raises(ValueError, match="normalized"):
        Embedding({b: 2 * a}, reference={a: 0})
    with pytest.raises(ValueError, match="physical source"):
        Embedding({b: Dagger(a)}, reference={a: 0})


def test_retained_boson_virtual_ancilla_against_matrix_formula():
    """Keep the full oscillator; compare non-diagonal second-order elements."""
    a, b = BosonOp("a"), BosonOp("b")
    q = SigmaMinus("q")
    source = (a, q)
    h0 = 3 * N(a) + N(a) * (N(a) - 1) / 5 + (8 + N(a) / 7) * N(q)
    v = (a + Dagger(a)) * (q + Dagger(q))
    embedding = Embedding({b: a}, reference={a: 0, q: 0})
    h, *_ = block_diagonalize([h0, v], subspace_eigenvectors=embedding)
    second = h[0, 0, 2]
    occupations = (range(8), range(2))
    matrices = occupation_matrices(source, occupations)
    energy = operator_matrix(h0, matrices).diagonal().real
    vmat = operator_matrix(v, matrices).toarray()
    retained, complement = np.arange(0, 16, 2), np.arange(1, 16, 2)
    coupling = vmat[np.ix_(retained, complement)]
    gap = energy[retained, None] - energy[None, complement]
    reference = (
        (coupling / gap) @ coupling.conj().T + coupling @ (coupling / gap).conj().T
    ) / 2
    # Evaluate rational functions of the target number by their spectral values.
    actual = np.zeros((6, 6), dtype=complex)
    n = N(b)
    for powers, coefficient in second.terms.items():
        coefficient = coefficient.xreplace(
            dict(zip(_number_symbols((b,)), (n,), strict=True))
        )
        p = int(powers[0])
        for column in range(6):
            row = column - p
            if not 0 <= row < 6:
                continue
            middle = column - max(p, 0)
            weight = sympy.sqrt(
                sympy.factorial(max(row, column)) / sympy.factorial(min(row, column))
            )
            actual[row, column] = complex((coefficient.subs(n, middle) * weight).evalf())
    np.testing.assert_allclose(actual, reference[:6, :6], atol=1e-12, rtol=1e-12)
    assert np.max(np.abs(actual - np.diag(np.diag(actual)))) > 0.01


def test_retained_boson_with_drive_through_fourth_order():
    """Higher orders exercise shifted target denominators and vanishing leakage."""
    a, b = BosonOp("a"), BosonOp("b")
    q = SigmaMinus("q")
    h0 = 3 * N(a) + N(a) * (N(a) - 1) / 5 + (8 + N(a) / 7) * N(q)
    v = (a + Dagger(a)) * (q + Dagger(q)) + sympy.Rational(2, 7) * (a + Dagger(a))
    embedding = Embedding({b: a}, reference={a: 0, q: 0})
    effective = block_diagonalize([h0, v], subspace_eigenvectors=embedding)[0]
    matrices = occupation_matrices((a, q), (range(10), range(2)))
    h0_matrix, v_matrix = (operator_matrix(expr, matrices).toarray() for expr in (h0, v))
    reference = block_diagonalize(
        [h0_matrix, v_matrix], subspace_indices=np.tile([0, 1], 10)
    )[0]
    (n,) = _number_symbols((b,))
    for order in (3, 4):
        actual = np.zeros((4, 4), dtype=complex)
        for powers, coefficient in effective[0, 0, order].terms.items():
            p = int(powers[0])
            for column in range(4):
                row = column - p
                if 0 <= row < 4:
                    weight = sympy.sqrt(
                        sympy.factorial(max(row, column))
                        / sympy.factorial(min(row, column))
                    )
                    actual[row, column] = complex(
                        (coefficient.subs(n, column - max(p, 0)) * weight).evalf()
                    )
        np.testing.assert_allclose(
            actual, reference[0, 0, order][:4, :4], atol=1e-11, rtol=1e-10
        )


@pytest.mark.parametrize("kind", [BosonOp, FermionOp])
@pytest.mark.parametrize("complex_mixing", [False, True])
def test_linear_mode_mixing_against_fock_matrices(kind, complex_mixing):
    """Build the retained columns directly from the supplied creation operator."""
    a, b, c = (kind(name) for name in ("a", "b", "c"))
    f = kind("f")
    phase = sympy.I if complex_mixing else sympy.S.One
    image = (a + phase * b) / sympy.sqrt(2)
    embedding = Embedding({f: image}, reference={a: 0, b: 0, c: 0})
    size = 4 if kind is BosonOp else 2
    matrices = occupation_matrices((a, b, c), [range(size)] * 3)
    creator = operator_matrix(Dagger(image).expand(), matrices).toarray()
    column = np.eye(size**3)[:, 0]
    columns = [column]
    for n in range(1, size):
        column = creator @ column / np.sqrt(n)
        columns.append(column)
    w = np.column_stack(columns)
    np.testing.assert_allclose(w.conj().T @ w, np.eye(size), atol=1e-14)
    target_matrices = occupation_matrices((f,), [range(size)])
    for expression in (a, b, Dagger(a) * b, N(a) * N(b), a * Dagger(b)):
        actual = operator_matrix(
            embedding.restrict(expression), target_matrices
        ).toarray()
        expected = w.conj().T @ operator_matrix(expression, matrices) @ w
        np.testing.assert_allclose(actual, expected, atol=1e-13)
    h, *_ = block_diagonalize(
        [2 * (N(a) + N(b)) + 7 * N(c), Dagger(c) * image + Dagger(image) * c],
        subspace_eigenvectors=embedding,
    )
    assert (h[0, 0, 2] + N(f) / 5).applyfunc(sympy.simplify).is_zero


def test_symbolic_rotation_and_cross_relations():
    a, b, f, g = (FermionOp(name) for name in ("a", "b", "f", "g"))
    angle = sympy.Symbol("theta", real=True)
    first = sympy.cos(angle) * a + sympy.sin(angle) * b
    second = -sympy.sin(angle) * a + sympy.cos(angle) * b
    for generators in ({f: first}, {f: first, g: second}):
        embedding = Embedding(generators, reference={a: 0, b: 0})
        for target, expression in generators.items():
            assert (
                (embedding.restrict(expression) - target)
                .applyfunc(sympy.simplify)
                .is_zero
            )
    with pytest.raises(ValueError, match="orthogonal"):
        Embedding({f: first, g: first}, reference={a: 0, b: 0})
    with pytest.raises(NotImplementedError, match="empty reference"):
        Embedding({f: first}, reference={a: 1, b: 0})
    amplitude = sympy.Symbol("z")
    with pytest.raises(NotImplementedError, match=r"Cannot establish.*normalized"):
        Embedding({f: amplitude * a}, reference={a: 0})


def test_three_mode_rotation_with_frozen_fermionic_spectator():
    a, b, c, fixed = (FermionOp(name) for name in ("a", "b", "c", "0_fixed"))
    f, g = FermionOp("f"), FermionOp("g")
    images = {f: (a + b + c) / sympy.sqrt(3), g: (a - b) / sympy.sqrt(2)}
    embedding = Embedding(images, reference={a: 0, b: 0, c: 0, fixed: 1})
    matrices = occupation_matrices((fixed, a, b, c), [range(2)] * 4)
    vacuum = np.eye(16)[:, 8]
    columns = []
    for occupations in product(range(2), repeat=2):
        column = vacuum
        for mode, n in reversed(tuple(zip((f, g), occupations, strict=True))):
            if n:
                column = operator_matrix(Dagger(images[mode]).expand(), matrices) @ column
        columns.append(column)
    w = np.column_stack(columns)
    np.testing.assert_allclose(w.conj().T @ w, np.eye(4), atol=1e-14)
    target_matrices = occupation_matrices((f, g), [range(2)] * 2)
    for expression in (a, c, Dagger(b) * a, N(a) * N(c), fixed * Dagger(fixed)):
        actual = operator_matrix(
            embedding.restrict(expression), target_matrices
        ).toarray()
        expected = w.conj().T @ operator_matrix(expression, matrices) @ w
        np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("occupations", [(0, 0), (1, 2)])
def test_matrix_with_operator_entries_against_full_matrix(reverse, occupations):
    """Nondegenerate retained levels and complex couplings through fourth order."""
    b = BosonOp("b")
    h0 = sympy.diag(5 * N(b), 2 + 5 * N(b))
    v = sympy.Matrix(
        [
            [b + Dagger(b), 2 * b + 3 * sympy.I * Dagger(b) + 1],
            [2 * Dagger(b) - 3 * sympy.I * b + 1, 2 * (b + Dagger(b))],
        ]
    )
    components = (1, 0) if reverse else (0, 1)
    embedding = Embedding(reference=[(i, {b: occupations[i]}) for i in components])
    effective, *_ = block_diagonalize([h0, v], subspace_eigenvectors=embedding)
    # A returning four-step path rises by at most two levels. Both cutoffs close it.
    for cutoff in (5, 6):
        lowering = sympy.zeros(cutoff)
        for n in range(1, cutoff):
            lowering[n - 1, n] = sympy.sqrt(n)
        # Explicitly assemble component blocks, without NOF/embedding arithmetic.
        full_h0 = sympy.diag(*range(0, 5 * cutoff, 5), *range(2, 2 + 5 * cutoff, 5))
        full_v = sympy.BlockMatrix(
            [
                [
                    lowering + lowering.T,
                    2 * lowering + 3 * sympy.I * lowering.T + sympy.eye(cutoff),
                ],
                [
                    2 * lowering.T - 3 * sympy.I * lowering + sympy.eye(cutoff),
                    2 * (lowering + lowering.T),
                ],
            ]
        ).as_explicit()
        kept = [i * cutoff + occupations[i] for i in components]
        complement = [i for i in range(2 * cutoff) if i not in kept]
        basis = sympy.eye(2 * cutoff)
        reference, *_ = block_diagonalize(
            [full_h0, full_v],
            subspace_eigenvectors=[basis[:, kept], basis[:, complement]],
        )
        for order in range(5):
            assert (effective[0, 0, order] - reference[0, 0, order]).applyfunc(
                sympy.simplify
            ) == sympy.zeros(2)


def test_reference_list_compresses_products_before_selecting_states():
    b = BosonOp("b")
    embedding = Embedding(reference=[{b: 2}, {b: 0}])
    assert embedding.restrict(b) == sympy.zeros(2)
    assert embedding.restrict(b * Dagger(b)) == sympy.diag(3, 1)
    assert embedding.restrict(b**2) == sympy.Matrix([[0, 0], [sympy.sqrt(2), 0]])


def test_pure_matrix_source_retains_degenerate_internal_transitions():
    embedding = Embedding(reference=[(1, {}), (0, {})])
    h0 = sympy.diag(0, 0, 7)
    v = sympy.Matrix([[0, 2, 1], [2, 0, sympy.I], [1, -sympy.I, 0]])
    h, *_ = block_diagonalize([h0, v], subspace_eigenvectors=embedding)
    assert h[0, 0, 1] == sympy.Matrix([[0, 2], [2, 0]])
    assert h[0, 0, 2] == -sympy.Matrix([[1, sympy.I], [-sympy.I, 1]]) / 7


@pytest.mark.parametrize(
    "reference, message",
    [
        ([], "at least one"),
        ([{}, {}], "distinct"),
        ([(1.5, {})], "indices"),
        ([{BosonOp("b"): -1}], "occupations"),
        ([{FermionOp("f"): 2}], "occupations"),
        ([{}, {BosonOp("b"): 0}], "same source"),
    ],
)
def test_invalid_reference_lists(reference, message):
    with pytest.raises(ValueError, match=message):
        Embedding(reference=reference)


def test_matrix_source_validation():
    embedding = Embedding(reference=[(1, {})])
    with pytest.raises(ValueError, match="matrix source"):
        embedding.restrict(1)
    with pytest.raises(ValueError, match="outside"):
        embedding.restrict(sympy.eye(1))
    with pytest.raises(ValueError, match="square"):
        embedding.restrict(sympy.zeros(2, 3))
    with pytest.raises(ValueError, match="diagonal H0"):
        block_diagonalize(
            [sympy.Matrix([[0, 1], [1, 2]])], subspace_eigenvectors=embedding
        )
    h, *_ = block_diagonalize(
        [sympy.diag(1, 2), sympy.eye(3)], subspace_eigenvectors=embedding
    )
    with pytest.raises(ValueError, match="same source matrix shape"):
        _ = h[0, 0, 1]
