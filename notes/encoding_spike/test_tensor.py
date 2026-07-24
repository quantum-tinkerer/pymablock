"""Checks for the native-factor tensor operator."""

# ruff: noqa: D103

from __future__ import annotations

import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

from pymablock.number_ordered_form import NumberOrderedForm

from .algebraic import MapComponent, TargetBlock
from .maps import AffineBand
from .packed import PackedForm, nof_to_packed_fermions
from .tensor import TensorOperator


def test_tensor_operator_preserves_native_factor_algebras() -> None:
    boson = BosonOp("a")
    fermion = FermionOp("f")
    boson_lowering = NumberOrderedForm.from_expr(boson)
    fermion_lowering = nof_to_packed_fermions(NumberOrderedForm.from_expr(fermion))
    operator = TensorOperator.from_factors((boson_lowering, fermion_lowering))

    product = operator.adjoint() * operator
    factors, coefficient = product.items[0]

    assert coefficient == 1
    assert factors[0] == Dagger(boson_lowering) * boson_lowering
    assert factors[1] == fermion_lowering.adjoint() * fermion_lowering


def test_tensor_operator_represents_sums_without_common_monomial_storage() -> None:
    boson = BosonOp("a")
    boson_lowering = NumberOrderedForm.from_expr(boson)
    boson_identity = NumberOrderedForm(
        boson_lowering.operators,
        {(0,): sympy.S.One},
    )
    packed_identity = PackedForm.identity(("f",))
    packed_lowering = nof_to_packed_fermions(NumberOrderedForm.from_expr(FermionOp("f")))
    identities = (boson_identity, packed_identity)

    operator = TensorOperator.build(
        identities,
        (
            ((boson_lowering, packed_identity), 1),
            ((boson_identity, packed_lowering), 2),
        ),
    )

    assert len(operator.items) == 2
    assert isinstance(operator.items[0][0][0], NumberOrderedForm)
    assert isinstance(operator.items[1][0][1], PackedForm)


def test_bosonic_target_block_and_map_action_remain_nof() -> None:
    source = BosonOp("a")
    target = BosonOp("b")
    source_action = TensorOperator.from_factor(NumberOrderedForm.from_expr(source))
    target_hamiltonian = TensorOperator.from_factor(
        NumberOrderedForm.from_expr(Dagger(target) * target)
    )
    parameter = sympy.Symbol("m", integer=True, nonnegative=True)
    bridge = AffineBand(
        "source",
        "target",
        (parameter,),
        (2 * parameter,),
        (parameter,),
        sympy.S.One,
        ((0,), (1,), (2,)),
    )
    component = MapComponent(source_action, bridge, target_hamiltonian)
    block = TargetBlock.build("target", target_hamiltonian)

    assert isinstance(component.right_action.native, NumberOrderedForm)
    assert (block * block).form.native == (
        target_hamiltonian.native * target_hamiltonian.native
    )
