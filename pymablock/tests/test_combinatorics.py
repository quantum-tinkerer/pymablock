"""Tests for weighted occupation transitions and basis maps."""

from __future__ import annotations

import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

from pymablock._combinatorics import BasisMap, NOFTransition
from pymablock.number_ordered_form import NumberOrderedForm


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
    """Packed fermion transitions retain their Jordan-Wigner parity."""
    first, second = FermionOp("first"), FermionOp("second")
    lowering = _only_transition(second, (first, second))

    assert lowering.apply((0, 1)).weight == 1
    assert lowering.apply((1, 1)).weight == -1


def test_affine_basis_map_solves_integer_target_shift() -> None:
    """The coordinate map, rather than the embedding backend, solves shifts."""
    retained = sympy.Symbol("retained", integer=True, nonnegative=True)
    basis_map = BasisMap((2 * retained,), (retained,))

    assert basis_map.target_shift((2,)) == (1,)
    assert basis_map.target_shift((-2,)) == (-1,)
    assert basis_map.target_shift((1,)) is None


def test_basis_phase_is_part_of_transition_pullback() -> None:
    """A phase-decorated basis map supplies the pullback phase ratio."""
    source = FermionOp("source")
    retained = sympy.Symbol("retained", integer=True, nonnegative=True)
    placeholder = sympy.Symbol("number", integer=True)
    transition = _only_transition(source, (source,))
    basis_map = BasisMap(
        (retained,),
        (retained,),
        (placeholder,),
        1 - 2 * retained,
    )

    assert basis_map.pullback_weight(transition, (1,)) == -1
