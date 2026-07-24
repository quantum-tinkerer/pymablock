"""Unit checks for the packed binary algebra used by the embedding spike."""

# ruff: noqa: D103

from __future__ import annotations

import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock.number_ordered_form import NumberOrderedForm

from .packed import (
    BooleanPolynomial,
    nof_to_packed_fermions,
    packed_to_fermion_nof,
    packed_to_spin_nof,
    spin_term_from_boolean,
)


def test_boolean_polynomial_reduces_projectors_and_inverts_pointwise() -> None:
    n0, n1 = sympy.symbols("n0 n1")
    reduced = BooleanPolynomial.from_expr(n0 * (1 - n1) + n0 * n1, (n0, n1))

    assert reduced.items == ((1, sympy.S.One),)

    inverse = BooleanPolynomial.from_expr(2 + n0 + n1, (n0, n1)).reciprocal()
    identity = BooleanPolynomial.from_expr((2 + n0 + n1) * inverse.as_expr(), (n0, n1))
    assert identity.items == ((0, sympy.S.One),)


def test_spin_jordan_wigner_round_trip_preserves_number_ordered_term() -> None:
    operators = tuple(SigmaMinus(sympy.Symbol(f"s{index}")) for index in range(4))
    identity = NumberOrderedForm(operators, {(0, 0, 0, 0): sympy.S.One}, validate=False)
    placeholders = tuple(identity._number_operator_placeholders)
    powers = (1, -1, 0, 0)
    coefficient = BooleanPolynomial.from_expr(3 - 2 * placeholders[2], placeholders)

    packed = spin_term_from_boolean(
        tuple(f"s{index}" for index in range(4)), powers, coefficient
    )
    round_trip = packed_to_spin_nof(packed, operators)

    assert round_trip == NumberOrderedForm(
        operators,
        {powers: 3 - 2 * placeholders[2]},
        validate=False,
    )


def test_packed_fermion_product_and_adjoint_match_number_ordered_form() -> None:
    f0, f1, f2 = (FermionOp(f"f{index}") for index in range(3))
    operators = (f0, f1, f2)
    left = NumberOrderedForm.from_expr(
        Dagger(f0) * f1 + 2 * Dagger(f2) * f0, operators=operators
    )
    right = NumberOrderedForm.from_expr(
        Dagger(f1) * f2 - Dagger(f0) * f1, operators=operators
    )
    packed_left = nof_to_packed_fermions(left)
    packed_right = nof_to_packed_fermions(right)

    assert packed_left * packed_right == nof_to_packed_fermions(left * right)
    assert packed_left.adjoint() == nof_to_packed_fermions(Dagger(left))
    assert (packed_left * packed_right).adjoint() == (
        packed_right.adjoint() * packed_left.adjoint()
    )


def test_packed_fermion_readout_reuses_nof_storage() -> None:
    f0, f1, f2 = (FermionOp(f"f{index}") for index in range(3))
    operators = (f0, f1, f2)
    form = NumberOrderedForm.from_expr(
        Dagger(f0) * Dagger(f2) + f2 * f0 + 3 * Dagger(f1) * f0 + Dagger(f2) * f2,
        operators=operators,
    )

    assert (
        packed_to_fermion_nof(
            nof_to_packed_fermions(form),
            operators,
        )
        == form
    )
