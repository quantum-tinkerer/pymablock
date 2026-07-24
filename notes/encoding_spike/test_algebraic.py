"""Cross-checks for the non-enumerating embedding algebra."""

# ruff: noqa: D103

from __future__ import annotations

import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from pymablock.number_ordered_form import LadderOp, NumberOrderedForm

from .algebraic import (
    AlgebraicEmbedding,
    AlgebraicProblem,
    OperatorMap,
    block_diagonalize_algebraic,
)
from .encoding import block_diagonalize, fermion_embedding, levels, occupation_map
from .models import fermionic_ring_exchange, tunable_coupler
from .packed import packed_to_spin_nof


def test_ladder_source_uses_number_ordered_hybrid_path() -> None:
    ladder = LadderOp("m")
    level = levels("level", 2)
    embedding = AlgebraicEmbedding(occupation_map({ladder: level}))

    lowering = embedding.to_target(embedding.source_form(ladder))
    raising = embedding.to_target(embedding.source_form(Dagger(ladder)))

    assert packed_to_spin_nof(lowering.native, embedding.target_operators).terms == {
        (1,): sympy.S.One
    }
    assert packed_to_spin_nof(raising.native, embedding.target_operators).terms == {
        (-1,): sympy.S.One
    }


def test_algebraic_coupler_matches_reference_through_fourth_order() -> None:
    model = tunable_coupler()
    problem = AlgebraicProblem([model.H0, model.V], model.encoding)
    assert type(problem.block_series()[1, 0, 1]) is OperatorMap

    algebraic, *_ = block_diagonalize_algebraic(
        [model.H0, model.V], encoding=model.encoding, to_order=4
    )

    # Evaluating the algebraic result does not enumerate target or source states.
    algebraic_fourth = algebraic[4]
    assert "states" not in model.encoding.target.__dict__

    reference, *_ = block_diagonalize(
        [model.H0, model.V], encoding=model.encoding, to_order=4
    )
    substitutions = dict(
        zip(
            (*model.frequencies, *model.anharmonicities, *model.couplings),
            map(sympy.Rational, (5, 11, 17, -2, -3, -5, 1, 2, 3)),
            strict=True,
        )
    )
    assert algebraic_fourth.matrix().subs(substitutions) == reference[4].matrix.subs(
        substitutions
    )


def test_algebraic_ring_exchange_without_hilbert_enumeration() -> None:
    model = fermionic_ring_exchange()
    algebraic, *_ = block_diagonalize_algebraic(
        [model.H0, model.V], encoding=model.encoding, to_order=4
    )
    fourth_order = algebraic[4]

    assert "states" not in model.encoding.target.__dict__
    assert (
        sympy.factor(
            fourth_order.form.terms[(1, -1, 1, -1)] - 40 * model.t**4 / model.U**3
        )
        == 0
    )


def test_algebraic_fermion_embedding_returns_target_nof() -> None:
    source, virtual = FermionOp("source"), FermionOp("virtual")
    target = FermionOp("target")
    source_energy, virtual_energy, coupling = sympy.symbols(
        "source_energy virtual_energy coupling",
        nonzero=True,
    )
    H0 = (
        source_energy * Dagger(source) * source
        + virtual_energy * Dagger(virtual) * virtual
    )
    V = coupling * (Dagger(virtual) * source + Dagger(source) * virtual)
    encoding = fermion_embedding({source: target, virtual: 0})

    effective, *_ = block_diagonalize_algebraic(
        [H0, V],
        encoding=encoding,
        to_order=2,
    )
    second_order = effective[2]

    assert "states" not in encoding.target.__dict__
    assert second_order.generators == (target,)
    assert second_order.form == NumberOrderedForm.from_expr(
        coupling**2 * Dagger(target) * target / (source_energy - virtual_energy),
        operators=(target,),
    )
