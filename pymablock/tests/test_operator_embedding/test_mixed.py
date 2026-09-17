"""Mixed target algebras checked against independent occupation matrices."""

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
from pymablock._embedding.solver import _EmbeddingBackend
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.second_quantization import Embedding
from pymablock.tests.second_quantization_helpers import (
    occupation_matrices,
    operator_matrix,
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
    s = JminusOp("S") if finite else SigmaMinus("s")
    ns = JzOp("S") + sympy.S.Half if finite else N(s)
    first, second = (g, f) if reverse else (f, g)
    backend = _EmbeddingBackend(
        Embedding(
            target={s: 2, g: 2, f: 2},
            occupations={up: ns, down: 1 - ns, left: N(first), right: N(second)},
        )
    )
    matrices = occupation_matrices(modes, [(0, 1)] * 4)
    source_states = list(product((0, 1), repeat=4))
    columns = []
    direct = {first: left, second: right}
    for state in backend.descriptor.target.states:
        values = dict(zip(backend.target_operators, state, strict=True))
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
            for op in backend.target_operators
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
        result = backend.pullback(backend.source_form(expression))
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
