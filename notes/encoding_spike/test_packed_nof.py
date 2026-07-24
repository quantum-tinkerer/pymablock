"""Checks for packed binary storage inside a number-ordered form."""

# ruff: noqa: D103

from __future__ import annotations

import random

import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm


def _canonical(form: NumberOrderedForm) -> dict:
    return {
        powers: sympy.expand(coefficient)
        for powers, coefficient in form.terms.items()
        if coefficient != 0
    }


def test_boolean_coefficients_become_structural_number_masks() -> None:
    fermion = FermionOp("f")
    number = NumberOperator(fermion)
    delta, interaction = sympy.symbols("Delta U", nonzero=True)
    form = NumberOrderedForm.from_expr(
        1 / (delta + interaction * number),
        operators=(fermion,),
    )

    assert len(form.args[1]) == 2
    placeholder = form._number_operator_placeholders[0]
    assert all(
        len(key) == 2 and tuple(key[0]) == () and not coefficient.has(placeholder)
        for key, coefficient in form.args[1]
    )
    assert _canonical(NumberOrderedForm(operators=(fermion,), terms=form.terms)) == (
        _canonical(form)
    )


def test_mixed_binary_statistics_and_adjoint_match_nof() -> None:
    spin_0, spin_1 = SigmaMinus("s0"), SigmaMinus("s1")
    fermion_0, fermion_1 = FermionOp("f0"), FermionOp("f1")
    operators = (spin_0, spin_1, fermion_0, fermion_1)
    expressions = (
        spin_0 * spin_1,
        spin_0 * fermion_0,
        fermion_0 * fermion_1,
        Dagger(spin_0) * spin_1 + Dagger(fermion_0) * fermion_1,
    )

    for expression in expressions:
        form = NumberOrderedForm.from_expr(expression, operators=operators)
        assert _canonical(NumberOrderedForm(operators, form.terms)) == _canonical(form)
        assert _canonical(form.adjoint()) == _canonical(Dagger(form))


def test_packed_storage_canonicalizes_numbers_on_active_binary_modes() -> None:
    spin = SigmaMinus("s")
    number = NumberOperator(spin)
    placeholder = NumberOrderedForm((spin,), {(0,): 1})._number_operator_placeholders[0]
    left = NumberOrderedForm(
        (spin,),
        {(1,): -placeholder - 1},
        validate=False,
    )
    right = NumberOrderedForm.from_expr(2 * Dagger(spin), operators=(spin,))

    product = left * right
    assert _canonical(product) == _canonical(
        NumberOrderedForm.from_expr(-2 * (1 - number), operators=(spin,))
    )
    assert all(not coefficient.has(placeholder) for _, coefficient in product.args[1])


def test_random_products_match_current_nof() -> None:
    rng = random.Random(4)
    boson = BosonOp("a")
    spins = (SigmaMinus("s0"), SigmaMinus("s1"))
    fermions = (FermionOp("f0"), FermionOp("f1"))
    operators = (boson, *spins, *fermions)
    placeholders = NumberOrderedForm(
        operators,
        {(0,) * len(operators): 1},
    )._number_operator_placeholders

    def random_form() -> NumberOrderedForm:
        terms = {}
        for _ in range(5):
            powers = (
                rng.randint(-2, 2),
                *(rng.randint(-1, 1) for _ in range(4)),
            )
            coefficient = rng.choice((-2, -1, 1, 2))
            if rng.random() < 0.7:
                coefficient *= 1 + rng.choice(placeholders)
            terms[powers] = terms.get(powers, 0) + coefficient
        form = NumberOrderedForm(operators, terms, validate=False)
        return NumberOrderedForm(operators, form.terms, validate=False)

    for _ in range(30):
        left = random_form()
        right = random_form()
        product = left * right
        assert _canonical(NumberOrderedForm(operators, product.terms)) == (
            _canonical(product)
        )
