"""Checks for the finite spin-1 Floquet benchmark."""

# ruff: noqa: D103

from __future__ import annotations

import sympy

from pymablock.number_ordered_form import NumberOrderedForm

from .algebraic import block_diagonalize_algebraic
from .encoding import block_diagonalize
from .models import synthetic_spin_floquet


def _simplify_matrix(matrix: sympy.MatrixBase) -> sympy.ImmutableMatrix:
    return sympy.ImmutableMatrix(
        matrix.applyfunc(lambda value: sympy.trigsimp(sympy.expand_complex(value)))
    )


def test_one_rotation_keeps_all_floquet_indices() -> None:
    model = synthetic_spin_floquet()
    perturbation = NumberOrderedForm.from_expr(
        model.V, operators=model.encoding.operators
    )
    floquet_index = model.encoding.operators.index(model.floquet)
    floquet_shifts = {int(powers[floquet_index]) for powers in perturbation.terms}

    assert model.spin == 1
    assert model.encoding.target.states == ((0,), (1,), (2,))
    assert min(floquet_shifts) < 0 < max(floquet_shifts)


def test_finite_and_algebraic_floquet_results_agree() -> None:
    model = synthetic_spin_floquet()
    reference, *_ = block_diagonalize(
        [model.H0, model.V], encoding=model.encoding, to_order=2
    )
    algebraic, *_ = block_diagonalize_algebraic(
        [model.H0, model.V], encoding=model.encoding, to_order=2
    )

    first_order = algebraic[1]
    assert isinstance(first_order.form, sympy.MatrixBase)
    assert first_order.matrix().shape == (3, 3)
    assert _simplify_matrix(reference[1].matrix - model.expected_first_order()) == (
        sympy.zeros(3)
    )
    assert _simplify_matrix(first_order.matrix() - model.expected_first_order()) == (
        sympy.zeros(3)
    )

    # A generic exact point checks the nontrivial virtual correction without asking
    # SymPy to canonicalize the full symbolic rational expressions.
    substitutions = {
        model.phases[0]: 0,
        model.phases[1]: sympy.pi / 3,
        model.phases[2]: sympy.pi / 2,
        model.cavity_phase: sympy.pi / 7,
        model.chi: 11,
        model.Omega: 3,
        model.epsilon: 1,
    }
    difference = (algebraic[2].matrix() - reference[2].matrix).subs(substitutions)
    assert difference.applyfunc(sympy.simplify) == sympy.zeros(3)
