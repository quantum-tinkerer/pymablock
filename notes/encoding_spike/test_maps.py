"""Checks for sparse weighted-map execution."""

# ruff: noqa: D103

from __future__ import annotations

import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

from pymablock.series import zero

from .algebraic import (
    AdjointOffDiagonalMap,
    BandMap,
    OffDiagonalMap,
    RuleMap,
    TargetBlock,
    block_diagonalize_band_maps,
    block_diagonalize_factored_maps,
    block_diagonalize_maps,
)
from .encoding import block_diagonalize, levels, occupation_map
from .maps import (
    ExplicitOperatorMap,
    MapEmbedding,
    MapKey,
    MapRule,
    OffDiagonalMapKey,
    matrix_unit,
    state_matrix,
)
from .models import (
    crepel_fu_two_star,
    fermionic_ring_exchange,
    synthetic_spin_floquet,
    tunable_coupler,
)


def test_state_maps_are_native_sympy_matrices() -> None:
    level = levels("m", 3)
    target = occupation_map({BosonOp("a"): level}).target
    x = sympy.Symbol("x")
    left = state_matrix(
        target,
        (
            (MapKey((1,), (2,)), x),
            (MapKey((0,), (1,)), 2),
        ),
    )
    right = state_matrix(
        target,
        (
            (MapKey((0,), (1,)), 3),
            (MapKey((1,), (2,)), 5),
        ),
    )

    assert left * right == sympy.ImmutableSparseMatrix(
        [[0, 0, 0], [0, 0, 0], [3 * x, 0, 0]]
    )
    assert left.adjoint() == sympy.ImmutableSparseMatrix(
        [[0, 2, 0], [0, 0, sympy.conjugate(x)], [0, 0, 0]]
    )


def test_hilbert_hotel_style_embedding_is_a_map() -> None:
    source = BosonOp("a")
    level = levels("m", 4)
    embedding = MapEmbedding(occupation_map({source: 2 * level}))

    assert len(embedding.map.items) == 1
    occupation_map_key, coefficient = embedding.map.items[0]
    assert coefficient == 1
    assert occupation_map_key.output_expressions == (
        2 * occupation_map_key.input_symbols[0],
    )
    assert tuple(
        occupation_map_key.evaluate(state) for state in embedding.target.states
    ) == ((0,), (2,), (4,), (6,))
    assert embedding.bridge.left_space == embedding.source_space
    assert embedding.bridge.right_space == embedding.target
    assert embedding.bridge.items == tuple(
        (
            MapRule(state, (2 * state[0],)),
            sympy.S.One,
        )
        for state in embedding.target.states
    )
    assert embedding.bridge.adjoint().items == tuple(
        (
            MapRule((2 * state[0],), state),
            sympy.S.One,
        )
        for state in embedding.target.states
    )
    target_identity = embedding.bridge.adjoint() @ embedding.bridge
    even_source_projector = embedding.bridge @ embedding.bridge.adjoint()
    assert target_identity.is_identity
    assert target_identity.fixed_shift == (0,)
    assert not even_source_projector.is_identity
    assert even_source_projector.fixed_shift == (0,)
    assert tuple(key.input_state for key, _ in even_source_projector.items) == (
        (0,),
        (2,),
        (4,),
        (6,),
    )
    assert embedding.to_target(embedding.source_form(source)) == sympy.zeros(
        embedding.target.dimension
    )

    lowering_by_two = embedding.to_target(embedding.source_form(source**2))
    expected = state_matrix(
        embedding.target,
        {
            MapKey((1,), (0,)): sympy.sqrt(2),
            MapKey((2,), (1,)): 2 * sympy.sqrt(3),
            MapKey((3,), (2,)): sympy.sqrt(30),
        }.items(),
    )
    assert lowering_by_two == expected
    assert (
        embedding.to_target(embedding.source_form(Dagger(source) ** 2))
        == expected.adjoint()
    )


def test_offdiagonal_block_is_a_canonical_map_without_eager_projection() -> None:
    source = BosonOp("a")
    level = levels("m", 3)
    embedding = MapEmbedding(occupation_map({source: level}))
    number = embedding.source_form(Dagger(source) * source)

    offdiagonal = OffDiagonalMap.from_source(embedding, number)

    assert isinstance(offdiagonal, OffDiagonalMap)
    assert isinstance(offdiagonal.adjoint(), AdjointOffDiagonalMap)
    assert offdiagonal + (-offdiagonal) is zero
    assert offdiagonal.items == (
        (
            OffDiagonalMapKey((0,), MapKey((1,), (1,))),
            sympy.S.One,
        ),
        (
            OffDiagonalMapKey((0,), MapKey((2,), (2,))),
            sympy.Integer(2),
        ),
    )
    # Q n W is zero, but the map keeps Q formal instead of eagerly computing PHP.
    assert offdiagonal.inner(offdiagonal) is zero


def test_composed_operator_map_has_explicit_spaces_and_no_embedding() -> None:
    source = BosonOp("a")
    level = levels("m", 4)
    context = MapEmbedding(occupation_map({source: 2 * level}))

    rule_map = RuleMap.from_source(context, context.source_form(source))

    assert rule_map.right_space == context.target
    assert rule_map.left_space == context.complement_space
    assert isinstance(rule_map.map, ExplicitOperatorMap)
    assert not hasattr(rule_map, "embedding")
    assert rule_map.items == (
        (((1,), (1,)), sympy.sqrt(2)),
        (((2,), (3,)), 2),
        (((3,), (5,)), sympy.sqrt(6)),
    )
    assert RuleMap.from_source(context, context.source_form(source**2)) is zero


def test_band_map_compresses_equal_factors_and_lowers_to_rules() -> None:
    source = BosonOp("a")
    level = levels("m", 4)
    context = MapEmbedding(occupation_map({source: 2 * level}))
    first = matrix_unit(context.target, 0, 0, 1)
    second = matrix_unit(context.target, 1, 1, 1)
    lowering = context.source_form(source)
    raising = context.source_form(Dagger(source))

    band_map = BandMap(
        context,
        (
            (lowering, first),
            (lowering, second),
            (raising, first + second),
        ),
    )

    assert band_map.compression == (3, 1)
    assert band_map.components[0].left_action == lowering + raising
    assert band_map.components[0].bridge == context.bridge
    assert band_map.components[0].right_action == first + second

    expected = RuleMap.from_source(context, lowering + raising).right(
        TargetBlock(context, first + second)
    )
    assert band_map.lower().items == expected.items


def test_map_coupler_matches_finite_reference() -> None:
    model = tunable_coupler()
    mapped, *_ = block_diagonalize_maps(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=4,
    )
    factored, *_ = block_diagonalize_factored_maps(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=4,
    )
    reference, *_ = block_diagonalize(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=4,
    )
    substitutions = dict(
        zip(
            (*model.frequencies, *model.anharmonicities, *model.couplings),
            map(sympy.Rational, (5, 11, 17, -2, -3, -5, 1, 2, 3)),
            strict=True,
        )
    )
    assert mapped[4].matrix().subs(substitutions) == reference[4].matrix.subs(
        substitutions
    )
    assert mapped[4].matrix().subs(substitutions) == factored[4].matrix().subs(
        substitutions
    )
    banded, *_ = block_diagonalize_band_maps(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=4,
    )
    assert banded[4].matrix().subs(substitutions) == reference[4].matrix.subs(
        substitutions
    )


def test_map_ring_exchange_matches_every_target_sector() -> None:
    model = fermionic_ring_exchange()
    mapped, *_ = block_diagonalize_maps(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=4,
    )
    reference, *_ = block_diagonalize(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=4,
    )
    substitutions = {model.U: 7, model.t: 2}
    assert mapped[4].matrix().subs(substitutions) == reference[4].matrix.subs(
        substitutions
    )
    banded, *_ = block_diagonalize_band_maps(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=4,
    )
    assert banded[4].matrix().subs(substitutions) == reference[4].matrix.subs(
        substitutions
    )
    assert (
        sympy.factor(
            mapped[4].matrix_element((0, 1, 0, 1), (1, 0, 1, 0))
            - 40 * model.t**4 / model.U**3
        )
        == 0
    )


def test_map_floquet_matches_first_and_second_order() -> None:
    model = synthetic_spin_floquet()
    mapped, *_ = block_diagonalize_maps(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=2,
    )
    reference, *_ = block_diagonalize(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=2,
    )
    first_difference = mapped[1].matrix() - model.expected_first_order()
    assert first_difference.applyfunc(
        lambda value: sympy.trigsimp(sympy.expand_complex(value))
    ) == sympy.zeros(3)

    substitutions = {
        model.phases[0]: 0,
        model.phases[1]: sympy.pi / 3,
        model.phases[2]: sympy.pi / 2,
        model.cavity_phase: sympy.pi / 7,
        model.chi: 11,
        model.Omega: 3,
        model.epsilon: 1,
    }
    difference = (mapped[2].matrix() - reference[2].matrix).subs(substitutions)
    assert difference.applyfunc(sympy.simplify) == sympy.zeros(3)
    banded, *_ = block_diagonalize_band_maps(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=2,
    )
    difference = (banded[2].matrix() - reference[2].matrix).subs(substitutions)
    assert difference.applyfunc(sympy.simplify) == sympy.zeros(3)


def test_crepel_fu_fourth_order_uses_two_a_stars() -> None:
    model = crepel_fu_two_star()
    mapped, *_ = block_diagonalize_maps(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=4,
    )
    f0, _, _, f3, _ = model.target_fermions
    state = model.encoding.target.state
    range_two_second = mapped[2].matrix_element(state([f3]), state([f0]))
    range_two_fourth = mapped[4].matrix_element(state([f3]), state([f0]))
    expected_fourth = (
        -(model.t0**4)
        * (2 * model.Delta**2 + 4 * model.Delta * model.V0 + model.V0**2)
        / (2 * (model.Delta + model.V0) ** 3 * (model.Delta + 2 * model.V0) ** 2)
    )
    assert len(model.encoding.operators) == 14
    assert model.encoding.target.dimension == 16
    assert range_two_second == 0
    assert sympy.factor(range_two_fourth - expected_fourth) == 0

    banded, *_ = block_diagonalize_band_maps(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=4,
    )
    reference, *_ = block_diagonalize(
        [model.H0, model.V],
        encoding=model.encoding,
        to_order=4,
    )
    substitutions = {
        model.Delta: 10,
        model.V0: 2,
        model.UA: 7,
        model.UB: 11,
        model.t0: 1,
    }
    assert mapped[4].generators == model.target_fermions
    assert reference.info.retained_states == 16
    assert reference.info.virtual_states == 555
    expected_matrix = reference[4].matrix.subs(substitutions)
    assert mapped[4].matrix().subs(substitutions) == expected_matrix
    assert banded[4].matrix().subs(substitutions) == expected_matrix
