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
    LadderOp,
    NumberOrderedForm,
)
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding
from pymablock.series import zero
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


def test_binary_validation_does_not_enumerate_source() -> None:
    spins = tuple(SigmaMinus(f"s{i}") for i in range(20))
    embedding = Embedding(
        {s: BosonOp(f"a{i}") for i, s in enumerate(spins)},
        reference={BosonOp(f"a{i}"): 0 for i in range(len(spins))},
    )
    assert set(embedding.restrict(1).operators) == set(spins)


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


@pytest.mark.parametrize("source_type", [BosonOp, FermionOp])
def test_list_substitution_and_printing(source_type):
    a, z = map(BosonOp, ("a", "z"))
    f, target = source_type("f"), source_type("target")
    embedding = Embedding(
        {f: target},
        reference=[
            {target: 0, a: 1},
            {Embedding.row: 1, target: 0, a: 2},
        ],
    )
    assert (
        eval(str(embedding), {"Embedding": Embedding, "f": f, "target": target, "a": a})
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


def test_restriction_does_not_require_a_complement_projector():
    """A shifted boson ladder can be compressed before inequality projectors exist."""
    a, b = BosonOp("a"), BosonOp("b")
    embedding = Embedding({b: a * sympy.sqrt((N(a) - 3) / N(a))}, reference={a: 3})
    assert embedding.restrict(N(a)) == NumberOrderedForm.from_expr(3 + N(b))
    with pytest.raises(NotImplementedError):
        block_diagonalize([N(a)], subspace_eigenvectors=embedding)


# Each case tests an unsupported public input, without pinning error wording.
a, b = BosonOp("a"), BosonOp("b")
s, t = SigmaMinus("s"), SigmaMinus("t")
f = FermionOp("f")
ell, target_ell = LadderOp("ell"), LadderOp("target_ell")


@pytest.mark.parametrize(
    "generators, reference, error",
    [
        ({s: a}, {b: 0}, ValueError),
        ({s: 2 * a}, {a: 0}, ValueError),
        ({s: a, t: a}, {a: 0}, ValueError),
        ({f: a}, {a: 0}, ValueError),
        ({s: a + b}, {a: 0, b: 0}, ValueError),
        ({b: a.adjoint()}, {a: 0}, ValueError),
        ({b: 2 * a}, {a: 0}, ValueError),
        ({f: sympy.Symbol("z") * f}, {f: 0}, NotImplementedError),
        ({ell: target_ell}, {target_ell: 0}, ValueError),
        ({ell: target_ell, N(ell): N(target_ell)}, {target_ell: 3}, ValueError),
        ({s: a / sympy.sqrt(N(a))}, [{a: 0}, {a: 1}], ValueError),
        (
            {s: a**2 / sympy.sqrt(N(a) * (N(a) - 1))},
            [{a: 0}, {a: 1}],
            NotImplementedError,
        ),
        ({s: (1 + N(b)) * a}, [{a: 0, b: 0}, {a: 0, b: 1}], ValueError),
        ({s: (1 + N(t)) * s, t: t}, {s: 0, t: 0}, ValueError),
        ({}, None, TypeError),
        ({}, [], ValueError),
        ({}, [{}, {}], ValueError),
        ({}, [{Embedding.row: 1.5}], ValueError),
        ({}, [{Embedding.row: -1}], ValueError),
        ({}, [{a: -1}], ValueError),
        ({}, [{f: 2}], ValueError),
        ({}, [{}, {a: 0}], ValueError),
        ({}, {Embedding.row: 2}, ValueError),
    ],
)
def test_invalid_embedding_input(generators, reference, error):
    with pytest.raises(error):
        Embedding(generators, reference=reference)


@pytest.mark.parametrize(
    "operation",
    [
        lambda e: e.restrict(1),
        lambda e: e.restrict(sympy.eye(1)),
        lambda e: e.restrict(sympy.zeros(2, 3)),
        lambda e: block_diagonalize(
            [sympy.Matrix([[0, 1], [1, 2]])], subspace_eigenvectors=e
        ),
        lambda e: block_diagonalize(
            [sympy.diag(1, 2), sympy.eye(3)], subspace_eigenvectors=e
        )[0][0, 0, 1],
    ],
)
def test_invalid_matrix_target(operation):
    with pytest.raises(ValueError):
        operation(Embedding({}, reference=[{Embedding.row: 1}]))


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
