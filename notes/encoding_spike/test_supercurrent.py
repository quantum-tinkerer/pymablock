"""Tests for the single-NOF supercurrent spike."""

import sympy

from pymablock.number_ordered_form import NumberOrderedForm

from .supercurrent import (
    benchmark_supercurrent,
)


def test_single_nof_supercurrent_matches_finite_reference():
    """The NOF result agrees with both finite retained-space evaluations."""
    benchmark = benchmark_supercurrent()
    model = benchmark["model"]
    energies = benchmark["energies"]
    current = benchmark["current"]
    effective_dot = benchmark["effective_dot"]

    assert isinstance(model.H0, NumberOrderedForm)
    assert isinstance(model.V, NumberOrderedForm)
    assert isinstance(current, NumberOrderedForm)
    assert all(
        isinstance(operator, NumberOrderedForm) for operator in effective_dot.values()
    )
    assert all(operator.operators == model.dot for operator in effective_dot.values())
    assert current.operators == model.dot
    assert current.is_particle_conserving()
    assert current != 0
    assert sympy.simplify(energies[1, 0] - energies[0, 1]) == 0
    assert max(benchmark["finite_residuals"].values()) < 1e-15
    assert max(benchmark["target_residuals"].values()) < 1e-15
