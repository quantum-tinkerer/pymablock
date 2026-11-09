"""Embedding construction contracts and finite-matrix checks."""

from __future__ import annotations

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
)
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.operator_embedding import Embedding
from pymablock.tests.second_quantization_helpers import (
    occupation_matrices,
    operator_matrix,
)

a, b = BosonOp("a"), BosonOp("b")
s, t = SigmaMinus("s"), SigmaMinus("t")
f = FermionOp("f")
ell, target_ell = LadderOp("ell"), LadderOp("target_ell")


def test_binary_validation_does_not_enumerate_source() -> None:
    spins = tuple(SigmaMinus(f"s{i}") for i in range(20))
    embedding = Embedding(
        {s: BosonOp(f"a{i}") for i, s in enumerate(spins)},
        reference={BosonOp(f"a{i}"): 0 for i in range(len(spins))},
    )
    assert set(embedding.restrict(1).operators) == set(spins)


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
