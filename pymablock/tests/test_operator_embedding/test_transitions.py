"""Tests for weighted occupation transitions and basis maps."""

from __future__ import annotations

import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock._embedding.selection import _EmbeddingBackend
from pymablock._embedding.transitions import NOFTransition
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.second_quantization import Embedding


def _only_transition(expression, operators):
    form = NumberOrderedForm.from_expr(expression, operators=operators)
    (transition,) = tuple(NOFTransition.from_form(form))
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
    backend = _EmbeddingBackend(Embedding(target=(s,), occupations={a: 2 * N(s)}))
    assert backend.target_shift((2,)) == (1,)
    assert backend.target_shift((-2,)) == (-1,)
    assert backend.target_shift((1,)) is None


def test_frozen_particle_sets_retained_fermion_phase() -> None:
    """The target definition absorbs the sign from an earlier occupied mode."""
    fixed, source, target = (FermionOp(name) for name in ("a", "b", "f"))
    backend = _EmbeddingBackend(
        Embedding(target=(target,), occupations={fixed: 1, source: N(target)})
    )
    result = backend.pullback(backend.source_form(source))
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
    from pymablock._embedding.transitions import number_symbols

    f = FermionOp("f")
    (n,) = number_symbols((f,))
    form = NumberOrderedForm((f,), {(-1,): 1 / (1 - n)}, validate=False)
    (transition,) = NOFTransition.from_form(form)
    assert transition.apply((1,)) is None
    assert transition.apply((0,)).weight == 1
