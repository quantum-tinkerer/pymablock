"""Projection-aware second-quantization spike.

This package is deliberately isolated from :mod:`pymablock`.  It exercises a possible
public interface while reusing Pymablock's established perturbative recurrence.
"""

from .algebraic import (
    AdjointBandMap,
    AdjointOffDiagonalMap,
    AdjointRuleMap,
    AlgebraicTargetOperator,
    BandMap,
    MapComponent,
    OffDiagonalMap,
    RuleMap,
    block_diagonalize_algebraic,
    block_diagonalize_band_maps,
    block_diagonalize_factored_maps,
    block_diagonalize_maps,
)
from .encoding import (
    EffectiveSeries,
    FermionEmbedding,
    FermionOperator,
    FermionSpace,
    OccupationEncoding,
    PauliPolynomial,
    TargetOperator,
    block_diagonalize,
    fermion_embedding,
    levels,
    occupation_map,
)
from .maps import (
    AffineBand,
    ComplementSpace,
    EmbeddingMapForm,
    MapEmbedding,
    MapKey,
    MapRule,
    OccupationMapKey,
    OffDiagonalMapKey,
    SourceSpace,
    state_matrix,
)
from .tensor import TensorOperator

__all__ = [
    "AdjointBandMap",
    "AdjointOffDiagonalMap",
    "AdjointRuleMap",
    "AffineBand",
    "AlgebraicTargetOperator",
    "BandMap",
    "ComplementSpace",
    "EffectiveSeries",
    "EmbeddingMapForm",
    "FermionEmbedding",
    "FermionOperator",
    "FermionSpace",
    "MapComponent",
    "MapEmbedding",
    "MapKey",
    "MapRule",
    "OccupationEncoding",
    "OccupationMapKey",
    "OffDiagonalMap",
    "OffDiagonalMapKey",
    "PauliPolynomial",
    "RuleMap",
    "SourceSpace",
    "TargetOperator",
    "TensorOperator",
    "block_diagonalize",
    "block_diagonalize_algebraic",
    "block_diagonalize_band_maps",
    "block_diagonalize_factored_maps",
    "block_diagonalize_maps",
    "fermion_embedding",
    "levels",
    "occupation_map",
    "state_matrix",
]
