"""Occupation selections, source algebras, and projected perturbation operations."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize
from pymablock.number_ordered_form import (
    NumberOperator,
    NumberOrderedForm,
    _number_operator_to_placeholder,
)
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding
from pymablock.tests.second_quantization_helpers import (
    nof_matrix,
    occupation_matrices,
    operator_matrix,
)


def test_qubit_projector_selects_retained_occupations():
    targets = [BosonOp(f"a{i}") for i in range(8)]
    sources = [SigmaMinus(f"q{i}") for i in range(8)]
    embedding = Embedding(
        dict(zip(sources, targets)), reference=dict.fromkeys(targets, 0)
    )
    for state, expected in [((0,) * 8, 1), ((1,) * 8, 1), ((2,) + (0,) * 7, 0)]:
        actions = embedding._first_lattice._projector.act(state)
        assert sum(weight for _, weight in actions.values()) == expected


def test_fermion_embedding_returns_source_nof() -> None:
    target, virtual = FermionOp("target"), FermionOp("virtual")
    source = FermionOp("source")
    target_energy, virtual_energy, coupling = sympy.symbols(
        "target_energy virtual_energy coupling",
        nonzero=True,
        real=True,
    )
    h_0 = target_energy * NumberOperator(target) + virtual_energy * NumberOperator(
        virtual
    )
    perturbation = coupling * (Dagger(virtual) * target + Dagger(target) * virtual)
    embedding = Embedding({source: target}, reference={target: 0, virtual: 0})

    effective, *_ = block_diagonalize(
        [h_0, perturbation],
        subspace_eigenvectors=embedding,
    )

    expected = NumberOrderedForm.from_expr(
        coupling**2 * NumberOperator(source) / (target_energy - virtual_energy),
        operators=(source,),
    )

    assert (effective[0, 0, 2] - expected).applyfunc(sympy.cancel).is_zero


def test_frozen_fermion_phase_is_internal() -> None:
    fixed, target = FermionOp("a_fixed"), FermionOp("b_target")
    source, virtual = FermionOp("source"), FermionOp("virtual")
    target_energy, virtual_energy, coupling = sympy.symbols(
        "target_energy virtual_energy coupling",
        nonzero=True,
        real=True,
    )
    h_0 = target_energy * NumberOperator(target) + virtual_energy * NumberOperator(
        virtual
    )
    perturbation = coupling * (Dagger(virtual) * target + Dagger(target) * virtual)

    effective, *_ = block_diagonalize(
        [h_0, perturbation],
        subspace_eigenvectors=Embedding(
            {source: target},
            reference={fixed: 1, target: 0, virtual: 0},
        ),
    )

    expected = (
        coupling**2
        * NumberOrderedForm.from_expr(NumberOperator(source), operators=(source,)).terms[
            (0,)
        ]
        / (target_energy - virtual_energy)
    )

    assert sympy.cancel(effective[0, 0, 2].terms[(0,)] - expected) == 0


def test_reference_list_returns_matrix() -> None:
    target = BosonOp("target")
    frequency, coupling = sympy.symbols(
        "frequency coupling",
        nonzero=True,
        real=True,
    )
    embedding = Embedding({}, reference=[{target: n} for n in range(3)])

    effective, *_ = block_diagonalize(
        [
            frequency * NumberOperator(target),
            coupling * (target + Dagger(target)),
        ],
        subspace_eigenvectors=embedding,
    )

    assert effective[0, 0, 2] == sympy.diag(
        0,
        0,
        -3 * coupling**2 / frequency,
    )


def test_bosonic_excursion_is_not_a_product_of_compressions() -> None:
    """The source spin retains the virtual second boson level in products."""

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
    """Frozen particles and permutations must not change the source CAR."""

    a, fixed, b = (FermionOp(name) for name in ("a", "m", "z"))
    f, g = FermionOp("f"), FermionOp("g")
    left, right = (g, f) if reverse else (f, g)
    backend = Embedding(
        {left: a, right: b},
        reference={a: 0, fixed: frozen, b: 0},
    )
    for target, source in (
        (a, left),
        (b, right),
        (Dagger(a), Dagger(left)),
        (Dagger(b), Dagger(right)),
        (Dagger(a) * b, Dagger(left) * right),
        (a * b, left * right),
        (b * Dagger(b), right * Dagger(right)),
    ):
        result = backend.restrict(target)
        assert result.simplify() == NumberOrderedForm.from_expr(source, operators=(f, g))


def test_spin_in_two_fermions_uses_single_occupancy() -> None:
    """The source is one spin, with charge-changing target actions projected out."""

    up, down = FermionOp("up"), FermionOp("down")
    s = SigmaMinus("s")
    backend = Embedding({s: Dagger(down) * up}, reference={up: 0, down: 1})

    def expected(expr):
        return NumberOrderedForm.from_expr(expr, operators=(s,))

    assert backend.restrict(NumberOperator(up)) == expected(NumberOperator(s))
    assert backend.restrict(NumberOperator(down)) == expected(1 - NumberOperator(s))
    assert not backend.restrict(up)
    assert not backend.restrict(NumberOperator(up) * NumberOperator(down)).simplify()
    # In canonical target order (down, up), this bilinear takes down to up.
    assert backend.restrict(Dagger(up) * down) == expected(Dagger(s))


def test_invalid_generator_definitions():
    s, t = SigmaMinus("s"), SigmaMinus("t")
    a, b = BosonOp("a"), BosonOp("b")
    f = FermionOp("f")
    with pytest.raises(ValueError):
        Embedding({s: a}, reference={b: 0})
    with pytest.raises(ValueError):
        Embedding({s: 2 * a}, reference={a: 0})
    with pytest.raises(ValueError):
        Embedding({s: a, t: a}, reference={a: 0})
    with pytest.raises(ValueError):
        Embedding({f: a}, reference={a: 0})
    with pytest.raises(ValueError):
        Embedding({s: a + b}, reference={a: 0, b: 0})


def test_particle_hole_and_pair_encodings():
    f, a, b = (FermionOp(name) for name in ("f", "a", "b"))
    hole = Embedding({f: Dagger(a)}, reference={a: 1})
    assert hole.restrict(N(a)) == NumberOrderedForm.from_expr(1 - N(f), operators=(f,))
    assert hole.restrict(a) == NumberOrderedForm.from_expr(Dagger(f), operators=(f,))
    s = SigmaMinus("s")
    pair = Embedding({s: b * a}, reference={a: 0, b: 0})
    assert pair.restrict(b * a) == NumberOrderedForm.from_expr(s, operators=(s,))
    assert pair.restrict(N(a) * N(b)).simplify() == NumberOrderedForm.from_expr(
        N(s), operators=(s,)
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


def test_binary_validation_does_not_enumerate_source() -> None:
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
    embedding = Embedding({}, reference=[{a: n} for n in range(3)])
    # Retained n=2 and excluded n=3 both have energy -6.
    effective, *_ = block_diagonalize(
        [n * (n - 5), a + Dagger(a)], subspace_eigenvectors=embedding
    )
    with pytest.raises(ZeroDivisionError):
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
        else Embedding({}, reference=[{a: n, b: 0} for n in (1, 2)])
    )
    effective, *_ = block_diagonalize(
        [NumberOperator(a) ** 2 + 10 * NumberOperator(b), Dagger(b) * a + Dagger(a) * b],
        subspace_eigenvectors=embedding,
    )
    expected = NumberOrderedForm.from_expr(-(1 - n) / 9 - 2 * n / 7, operators=(s,))
    assert effective[0, 0, 2] == (nof_matrix(expected) if finite else expected)


@pytest.mark.parametrize("coupled", [False, True])
def test_sector_poles_respect_virtual_channel_support(coupled):
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    n = NumberOperator(a)
    effective, *_ = block_diagonalize(
        [n + (1 - n) * NumberOperator(b), (1 if coupled else 1 - n) * (b + Dagger(b))],
        subspace_eigenvectors=Embedding({s: a}, reference={a: 0, b: 0}),
    )
    if coupled:
        correction = effective[0, 0, 2].as_expr()
        assert correction.subs(NumberOperator(s), 0) == -1
        assert correction.subs(NumberOperator(s), 1).has(sympy.zoo, sympy.nan)
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


def test_state_selection_solves_integer_source_shift() -> None:
    """Selecting even boson occupations makes a two-step target shift binary."""
    a, s = BosonOp("a"), SigmaMinus("s")
    backend = Embedding({s: a**2 / sympy.sqrt(2)}, reference={a: 0})
    expected = NumberOrderedForm.from_expr(sympy.sqrt(2) * s)
    assert backend.restrict(a**2) == expected
    assert backend.restrict(Dagger(a) ** 2) == expected.adjoint()
    assert backend.restrict(a).is_zero


def test_frozen_particle_sets_retained_fermion_phase() -> None:
    """The source definition absorbs the sign from an earlier occupied mode."""
    fixed, target, source = (FermionOp(name) for name in ("a", "b", "f"))
    backend = Embedding({source: target}, reference={fixed: 1, target: 0})
    result = backend.restrict(target)
    assert result == NumberOrderedForm.from_expr(source, operators=(source,))


def test_reference_fixes_complex_phases_and_cross_relations():
    a, b = BosonOp("a"), BosonOp("b")
    s, t = SigmaMinus("s"), SigmaMinus("t")
    theta = sympy.Symbol("theta", real=True)
    phase = sympy.exp(sympy.I * theta)
    embedding = Embedding({s: phase * a}, reference={a: 0})
    result = embedding.restrict(a).as_expr()
    assert sympy.simplify(result - s / phase) == 0
    with pytest.raises(ValueError):
        Embedding({s: a, t: (1 - 2 * N(a)) * b}, reference={a: 0, b: 0})


@pytest.mark.parametrize("operator_type", [SigmaMinus, FermionOp])
def test_generator_relations_with_shared_occupation_phase(operator_type):
    """A controlled phase preserves the algebra only when both images transform."""
    a, b = operator_type("a"), operator_type("b")
    f, g = operator_type("f"), operator_type("g")
    image = (1 - 2 * N(a)) * b
    embedding = Embedding({f: (1 - 2 * N(b)) * a, g: image}, reference={a: 0, b: 0})
    target = occupation_matrices((a, b), [(0, 1)] * 2)
    source = occupation_matrices((f, g), [(0, 1)] * 2)
    w = np.diag([1, 1, 1, -1])
    for expression in (a, b, Dagger(a) * b, a * Dagger(b)):
        actual = operator_matrix(embedding.restrict(expression).simplify(), source)
        expected = w @ operator_matrix(expression, target).toarray() @ w
        np.testing.assert_array_equal(actual.toarray(), expected)
    with pytest.raises(ValueError):
        Embedding({f: a, g: image}, reference={a: 0, b: 0})


def test_retained_infinite_modes_and_ladder_reference():
    from pymablock.number_ordered_form import LadderOp

    a, b = BosonOp("a"), BosonOp("b")
    target, ell = LadderOp("target"), LadderOp("ell")
    embedding = Embedding(
        {b: a, ell: target, N(ell): N(target) - 3},
        reference={a: 0, target: 3},
    )
    for expression, expected in (
        (a**3, b**3),
        (Dagger(a) ** 2 * (N(a) + 3) * a, Dagger(b) ** 2 * (N(b) + 3) * b),
        (target**4, ell**4),
        (N(target), N(ell) + 3),
        (a * Dagger(a), 1 + N(b)),
    ):
        difference = embedding.restrict(expression) - NumberOrderedForm.from_expr(
            expected, operators=(b, ell)
        )
        assert all(sympy.simplify(value) == 0 for value in difference.terms.values())
    with pytest.raises(ValueError):
        Embedding({ell: target}, reference={target: 0})
    with pytest.raises(ValueError):
        Embedding({ell: target, N(ell): N(target)}, reference={target: 3})


def test_boson_generator_normalization_is_not_renormalized_silently():
    a, b = BosonOp("a"), BosonOp("b")
    with pytest.raises(ValueError):
        Embedding({b: 2 * a}, reference={a: 0})
    with pytest.raises(ValueError):
        Embedding({b: Dagger(a)}, reference={a: 0})


def test_retained_boson_virtual_ancilla_against_matrix_formula():
    """Keep the full oscillator; compare non-diagonal second-order elements."""
    a, b = BosonOp("a"), BosonOp("b")
    q = SigmaMinus("q")
    target = (a, q)
    h0 = 3 * N(a) + N(a) * (N(a) - 1) / 5 + (8 + N(a) / 7) * N(q)
    v = (a + Dagger(a)) * (q + Dagger(q))
    embedding = Embedding({b: a}, reference={a: 0, q: 0})
    h, *_ = block_diagonalize([h0, v], subspace_eigenvectors=embedding)
    second = h[0, 0, 2]
    occupations = (range(8), range(2))
    matrices = occupation_matrices(target, occupations)
    energy = operator_matrix(h0, matrices).diagonal().real
    vmat = operator_matrix(v, matrices).toarray()
    retained, complement = np.arange(0, 16, 2), np.arange(1, 16, 2)
    coupling = vmat[np.ix_(retained, complement)]
    gap = energy[retained, None] - energy[None, complement]
    reference = (
        (coupling / gap) @ coupling.conj().T + coupling @ (coupling / gap).conj().T
    ) / 2
    # Evaluate rational functions of the source number by their spectral values.
    actual = np.zeros((6, 6), dtype=complex)
    n = N(b)
    for powers, coefficient in second.terms.items():
        coefficient = coefficient.xreplace({_number_operator_to_placeholder(n): n})
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
    """Higher orders exercise shifted source denominators and vanishing leakage."""
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
    n = _number_operator_to_placeholder(N(b))
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


def test_generator_images_are_single_shifts():
    """Linear combinations of target modes are rejected, not rotated."""
    a, b, f = (FermionOp(name) for name in ("a", "b", "f"))
    with pytest.raises(ValueError):
        Embedding({f: (a + b) / sympy.sqrt(2)}, reference={a: 0, b: 0})
    amplitude = sympy.Symbol("z")
    with pytest.raises(NotImplementedError):
        Embedding({f: amplitude * a}, reference={a: 0})


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
    embedding = Embedding(
        {}, reference=[{Embedding.row: i, b: occupations[i]} for i in components]
    )
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
    embedding = Embedding({}, reference=[{b: 2}, {b: 0}])
    assert embedding.restrict(b) == sympy.zeros(2)
    assert embedding.restrict(b * Dagger(b)) == sympy.diag(3, 1)
    assert embedding.restrict(b**2) == sympy.Matrix([[0, 0], [sympy.sqrt(2), 0]])


def test_pure_matrix_target_retains_degenerate_internal_transitions():
    embedding = Embedding({}, reference=[{Embedding.row: 1}, {}])
    h0 = sympy.diag(0, 0, 7)
    v = sympy.Matrix([[0, 2, 1], [2, 0, sympy.I], [1, -sympy.I, 0]])
    h, *_ = block_diagonalize([h0, v], subspace_eigenvectors=embedding)
    assert h[0, 0, 1] == sympy.Matrix([[0, 2], [2, 0]])
    assert h[0, 0, 2] == -sympy.Matrix([[1, sympy.I], [-sympy.I, 1]]) / 7


@pytest.mark.parametrize(
    "reference",
    [
        [],
        [{}, {}],
        [{Embedding.row: 1.5}],
        [{Embedding.row: -1}],
        [{BosonOp("b"): -1}],
        [{FermionOp("f"): 2}],
        [{}, {BosonOp("b"): 0}],
    ],
)
def test_invalid_reference_lists(reference):
    with pytest.raises(ValueError):
        Embedding({}, reference=reference)


def test_matrix_target_validation():
    embedding = Embedding({}, reference=[{Embedding.row: 1}])
    with pytest.raises(ValueError):
        embedding.restrict(1)
    with pytest.raises(ValueError):
        embedding.restrict(sympy.eye(1))
    with pytest.raises(ValueError):
        embedding.restrict(sympy.zeros(2, 3))
    with pytest.raises(ValueError):
        block_diagonalize(
            [sympy.Matrix([[0, 1], [1, 2]])], subspace_eigenvectors=embedding
        )
    h, *_ = block_diagonalize(
        [sympy.diag(1, 2), sympy.eye(3)], subspace_eigenvectors=embedding
    )
    with pytest.raises(ValueError):
        _ = h[0, 0, 1]


def test_generator_lattices_in_matrix_rows_through_fourth_order():
    """Two ancilla branches each retain a qubit, with virtual oscillator levels."""
    a, s = BosonOp("a"), SigmaMinus("s")
    energy = 5 * N(a) + N(a) * (N(a) - 1)
    h0 = sympy.diag(energy, 2 + energy)
    v = sympy.Matrix(
        [
            [a + Dagger(a), 1 + 2 * a + Dagger(a)],
            [1 + 2 * Dagger(a) + a, 2 * (a + Dagger(a))],
        ]
    )
    embedding = Embedding({s: a}, reference=[{a: 0}, {Embedding.row: 1, a: 0}])
    effective, *_ = block_diagonalize([h0, v], subspace_eigenvectors=embedding)
    source_matrices = occupation_matrices((s,), [(0, 1)])
    coefficients = [
        operator_matrix(effective[0, 0, order], source_matrices).toarray()
        for order in range(5)
    ]
    for cutoff in (4, 5):
        target_matrices = occupation_matrices((a,), [range(cutoff)])
        full_h0, full_v = (operator_matrix(x, target_matrices).toarray() for x in (h0, v))
        kept = [0, 1, cutoff, cutoff + 1]
        complement = [i for i in range(2 * cutoff) if i not in kept]
        basis = np.eye(2 * cutoff)
        reference, *_ = block_diagonalize(
            [full_h0, full_v],
            subspace_eigenvectors=[basis[:, kept], basis[:, complement]],
        )
        for order, coefficient in enumerate(coefficients):
            np.testing.assert_allclose(coefficient, reference[0, 0, order], atol=1e-11)
        errors = []
        for coupling in (0.02, 0.04):
            approximation = sum(coupling**n * x for n, x in enumerate(coefficients))
            exact = np.linalg.eigvalsh(full_h0 + coupling * full_v)[:4]
            errors.append(np.max(np.abs(np.linalg.eigvalsh(approximation) - exact)))
        assert errors[1] < 1e-6
        assert errors[1] > 20 * errors[0]


def test_lattices_separated_by_spectator_offset():
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    embedding = Embedding({s: a}, reference=[{a: 0, b: 2}, {a: 0, b: 1}])
    actual = embedding.restrict(b + Dagger(b))
    identity = NumberOrderedForm.from_expr(1, operators=(s,))
    assert actual == sympy.Matrix(
        [[0, sympy.sqrt(2) * identity], [sympy.sqrt(2) * identity, 0]]
    )
    assert embedding.restrict(N(a)) == sympy.diag(
        *(embedding._first_embedding.restrict(N(a)),) * 2
    )
    w = embedding._retained_frame(1)
    assert (
        (w.adjoint() * w - sympy.eye(2))
        .applyfunc(lambda x: x.simplify() if isinstance(x, NumberOrderedForm) else x)
        .is_zero_matrix
    )
    # Full target products include excursions beyond either spectator level.
    assert embedding.restrict(b * Dagger(b)) == sympy.diag(3 * identity, 2 * identity)


def test_transfers_along_moving_modes_are_rejected():
    """Disjoint even/odd lattices would require occupation-dependent boson norms."""
    a, s = BosonOp("a"), SigmaMinus("s")
    generator = a**2 / sympy.sqrt(N(a) * (N(a) - 1))
    with pytest.raises(NotImplementedError):
        Embedding({s: generator}, reference=[{a: 0}, {a: 1}])


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


def test_reference_lattices_must_be_disjoint():
    a, s = BosonOp("a"), SigmaMinus("s")
    with pytest.raises(ValueError):
        Embedding(
            {s: a / sympy.sqrt(N(a))},
            reference=[{a: 0}, {a: 1}],
        )


def test_invalid_transfer_checks_every_lattice():
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    with pytest.raises(ValueError):
        Embedding(
            {s: (1 + N(b)) * a},
            reference=[{a: 0, b: 0}, {a: 0, b: 1}],
        )


@pytest.mark.parametrize("source_type", [BosonOp, FermionOp])
def test_list_reconstruction_substitution_and_printing(source_type):
    import pickle

    a, z = map(BosonOp, ("a", "z"))
    f, target = source_type("f"), source_type("target")
    embedding = Embedding(
        {f: target},
        reference=[
            {target: 0, a: 1},
            {Embedding.row: 1, target: 0, a: 2},
        ],
    )
    assert Embedding.row != sympy.Symbol("matrix_index")
    assert Embedding.row.func(*Embedding.row.args) == Embedding.row
    assert Embedding({}, reference=[{}]) == Embedding({}, reference=[{Embedding.row: 0}])
    assert embedding.func(*embedding.args) == embedding
    assert pickle.loads(pickle.dumps(embedding)) == embedding
    assert (
        eval(
            str(embedding),
            {
                "Embedding": Embedding,
                "f": f,
                "target": target,
                "a": a,
            },
        )
        == embedding
    )
    renamed = Embedding(
        {f: target},
        reference=[
            {target: 0, z: 1},
            {Embedding.row: 1, target: 0, z: 2},
        ],
    )
    w = embedding._retained_frame(2)
    expected = renamed._retained_frame(2)
    for method in ("subs", "xreplace"):
        assert getattr(embedding, method)({a: z}) == renamed
    assert w.xreplace({a: z}) == expected
    phase = sympy.Symbol("phase", real=True)
    phased = Embedding(
        {f: sympy.exp(sympy.I * phase) * target}, reference=embedding.args[1]
    )
    for method in ("subs", "xreplace"):
        assert getattr(phased._retained_frame(2), method)({phase: 0}) == w


def test_bilateral_reference_transfers_have_no_vacuum():
    from pymablock.number_ordered_form import LadderOp

    ell = LadderOp("ell")
    embedding = Embedding({}, reference=[{ell: -2}, {ell: 3}])
    assert embedding.restrict(ell**5) == sympy.Matrix([[0, 1], [0, 0]])
    assert embedding.restrict(N(ell)) == sympy.diag(-2, 3)


def test_point_reference_matrix_elements():
    embedding = Embedding({}, reference={})
    for value in (0, 1, 2):
        assert embedding.restrict(value) == value


def test_restriction_does_not_require_a_complement_projector():
    """A shifted boson ladder can be compressed before inequality projectors exist."""
    a, b = BosonOp("a"), BosonOp("b")
    embedding = Embedding({b: a * sympy.sqrt((N(a) - 3) / N(a))}, reference={a: 3})
    assert embedding.restrict(N(a)) == NumberOrderedForm.from_expr(3 + N(b))
    with pytest.raises(NotImplementedError):
        block_diagonalize([N(a)], subspace_eigenvectors=embedding)


@pytest.mark.parametrize("generators", [False, True])
def test_indexed_reference_values(generators):
    a, f = BosonOp("a"), BosonOp("f")
    embedding = Embedding(
        {f: a} if generators else {},
        reference=[{Embedding.row: 1, a: 0}, {a: 0}],
    )
    actual = embedding.restrict(sympy.diag(N(a), 3 + N(a)))
    if generators:
        actual = operator_matrix(actual, occupation_matrices((f,), [(0, 1)])).toarray()
        np.testing.assert_array_equal(actual, np.diag([3, 4, 0, 1]))
    else:
        assert actual == sympy.diag(3, 0)


def test_mapping_reference_rejects_nonzero_matrix_row():
    with pytest.raises(ValueError, match="require a reference list"):
        Embedding({}, reference={Embedding.row: 2})
    assert Embedding({}, reference={Embedding.row: 0}) == Embedding({}, reference={})
    embedding = Embedding({}, reference=[{Embedding.row: 2}])
    assert embedding.restrict(sympy.diag(3, 5, 7)) == sympy.Matrix([[7]])
