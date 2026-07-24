"""Public-API tests for algebraic second-quantized embeddings."""

from __future__ import annotations

import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm
from pymablock.second_quantization import (
    fermion_embedding,
    levels,
    occupation_embedding,
)


def test_fermion_embedding_returns_target_nof() -> None:
    source, virtual = FermionOp("source"), FermionOp("virtual")
    target = FermionOp("target")
    source_energy, virtual_energy, coupling = sympy.symbols(
        "source_energy virtual_energy coupling",
        nonzero=True,
        real=True,
    )
    h_0 = source_energy * NumberOperator(source) + virtual_energy * NumberOperator(
        virtual
    )
    perturbation = coupling * (Dagger(virtual) * source + Dagger(source) * virtual)
    embedding = fermion_embedding({source: target, virtual: 0})

    effective, *_ = block_diagonalize(
        [h_0, perturbation],
        subspace_eigenvectors=embedding,
    )

    assert effective[0, 0, 2] == NumberOrderedForm.from_expr(
        coupling**2 * NumberOperator(target) / (source_energy - virtual_energy),
        operators=(target,),
    )


def test_frozen_fermion_phase_is_internal() -> None:
    fixed, source = FermionOp("a_fixed"), FermionOp("b_source")
    target, virtual = FermionOp("target"), FermionOp("virtual")
    source_energy, virtual_energy, coupling = sympy.symbols(
        "source_energy virtual_energy coupling",
        nonzero=True,
        real=True,
    )
    h_0 = source_energy * NumberOperator(source) + virtual_energy * NumberOperator(
        virtual
    )
    perturbation = coupling * (Dagger(virtual) * source + Dagger(source) * virtual)

    effective, *_ = block_diagonalize(
        [h_0, perturbation],
        subspace_eigenvectors=fermion_embedding({fixed: 1, source: target, virtual: 0}),
    )

    assert effective[0, 0, 2].terms[(0,)] == (
        coupling**2
        * effective[0, 0, 2]._number_operator_placeholders[0]
        / (source_energy - virtual_energy)
    )


def test_nonbinary_target_uses_package_matrix_interface() -> None:
    source = BosonOp("source")
    retained_level = levels("level", 3)
    frequency, coupling = sympy.symbols(
        "frequency coupling",
        nonzero=True,
        real=True,
    )
    embedding = occupation_embedding({source: retained_level})

    effective, *_ = block_diagonalize(
        [
            frequency * NumberOperator(source),
            coupling * (source + Dagger(source)),
        ],
        subspace_eigenvectors=embedding,
    )

    assert isinstance(effective[0, 0, 2], sympy.MatrixBase)
    assert effective[0, 0, 2] == sympy.diag(
        0,
        0,
        -3 * coupling**2 / frequency,
    )
