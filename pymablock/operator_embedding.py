"""Structured embeddings with explicitly declared target operator algebras."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from math import prod
from typing import TYPE_CHECKING

import sympy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus
from sympy.physics.quantum.spin import JminusOp, JzOp

from pymablock.number_ordered_form import LadderOp, NumberOperator, generator_types

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Embedding"]


def _operator_sort_key(operator) -> tuple[int, str]:
    if isinstance(operator, JminusOp):
        return len(generator_types), str(operator.name)
    return generator_types.index(type(operator)), str(operator.name)


@dataclass(frozen=True)
class _TargetSpace:
    """Declared generators and their finite occupation bases."""

    operators: tuple
    dimensions: tuple[int, ...]

    @cached_property
    def states(self) -> tuple[tuple[int, ...], ...]:
        return tuple(product(*(range(size) for size in self.dimensions)))

    @cached_property
    def index(self) -> dict[tuple[int, ...], int]:
        return {state: index for index, state in enumerate(self.states)}

    @property
    def dimension(self) -> int:
        return prod(self.dimensions)


class Embedding:
    """Embed a declared target algebra through source occupation rules.

    Parameters
    ----------
    target : sequence of operators or mapping of operators to dimensions
        Target ``SigmaMinus`` or ``FermionOp`` annihilation generators. Their
        dimension is two. For higher spins, pass e.g. ``{JminusOp("S"): 3}``;
        occupations may then depend on ``JzOp("S")`` in units of hbar=1.
        Generators are ordered canonically by type and name. A target containing
        a higher spin uses finite matrices ordered by increasing occupations
        (increasing magnetic quantum number for higher spins).
    occupations : mapping
        Source annihilation generators mapped to integer affine expressions in
        target ``NumberOperator`` objects, or target ``JzOp`` objects. These are
        occupation constraints, not substitutions for source operators. Source
        modes must be listed even when their occupation is fixed.

    Notes
    -----
    Spin targets use the source occupation-basis phase convention. Fermionic
    targets currently require a direct one-to-one assignment of each retained
    fermion to a source fermion, with remaining source modes fixed empty or full.
    The fermionic phase accounts for frozen particles and mode permutations so
    that retained generators map to their declared source generators. Mixing
    target spins and fermions is not currently supported.

    The map must be injective and have physical integer source occupations.
    Only full-rank affine occupation rules are supported. This restricted
    contract permits validation without enumerating a source or target basis.
    Arbitrary superpositions require a prior source-basis rotation or explicit
    subspace eigenvector matrices; this constructor does not perform rotations.

    Examples
    --------
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.pauli import SigmaMinus
    >>> from pymablock.number_ordered_form import NumberOperator as N
    >>> a, s = BosonOp("a"), SigmaMinus("s")
    >>> embedding = Embedding(target=(s,), occupations={a: N(s)})
    >>> embedding.encode((1,))
    (1,)

    """

    def __init__(
        self,
        *,
        target: Sequence | Mapping,
        occupations: Mapping,
    ):
        """Compile and validate the target-to-source occupation map."""
        if not occupations:
            raise ValueError("An embedding must specify source occupations")
        dimensions = dict(target) if isinstance(target, Mapping) else None
        operators = tuple(target)
        if len(operators) != len(set(operators)):
            raise ValueError("Target generators must be distinct")
        if not all(isinstance(op, (SigmaMinus, FermionOp, JminusOp)) for op in operators):
            raise TypeError("Targets must be spin or fermion lowering operators")
        if any(isinstance(op, FermionOp) and not op.is_annihilation for op in operators):
            raise ValueError("Target fermions must be annihilation operators")
        operators = tuple(sorted(operators, key=_operator_sort_key))
        sizes = []
        for op in operators:
            size = dimensions[op] if dimensions is not None else 2
            if isinstance(op, JminusOp) and dimensions is None:
                raise ValueError("A higher-spin target requires an explicit dimension")
            size = sympy.sympify(size)
            if not size.is_Integer or size < 2:
                raise ValueError("Target dimensions must be integers of at least two")
            if not isinstance(op, JminusOp) and size != 2:
                raise ValueError("SigmaMinus and fermion targets have dimension two")
            sizes.append(int(size))
        self.target = _TargetSpace(operators, tuple(sizes))
        fermions = tuple(op for op in operators if isinstance(op, FermionOp))
        if fermions and len(fermions) != len(operators):
            raise ValueError("Mixed spin and fermion targets are not supported")
        self.target_is_nof = not any(isinstance(op, JminusOp) for op in operators)
        if not all(isinstance(op, generator_types) for op in occupations):
            raise TypeError("Source keys must be supported annihilation generators")
        if not all(op.is_annihilation for op in occupations):
            raise ValueError("Source keys must be annihilation generators")
        self.operators = tuple(sorted(occupations, key=_operator_sort_key))
        self.coordinate_symbols = tuple(
            sympy.Dummy(f"target_{i}", integer=True, nonnegative=True)
            for i in range(len(operators))
        )
        substitutions = {
            (JzOp(op.name) if isinstance(op, JminusOp) else NumberOperator(op)): (
                symbol - sympy.Rational(size - 1, 2)
                if isinstance(op, JminusOp)
                else symbol
            )
            for op, symbol, size in zip(
                operators, self.coordinate_symbols, sizes, strict=True
            )
        }
        self.source_occupations = tuple(
            sympy.expand(sympy.sympify(occupations[op]).xreplace(substitutions))
            for op in self.operators
        )
        self._validate_occupations()
        self.phase = sympy.S.One
        if fermions:
            self.phase = self._fermion_phase()

    def _validate_occupations(self) -> None:
        """Prove the supported affine map is physical and injective."""
        rows = []
        origin = dict.fromkeys(self.coordinate_symbols, sympy.S.Zero)
        for operator, expression in zip(
            self.operators, self.source_occupations, strict=True
        ):
            constant = expression.xreplace(origin)
            coefficients = tuple(
                sympy.diff(expression, x) for x in self.coordinate_symbols
            )
            if not constant.is_Integer or any(not c.is_Integer for c in coefficients):
                raise ValueError(
                    "Occupations must be integer affine expressions in declared target numbers"
                )
            if (
                sympy.expand(
                    expression
                    - constant
                    - sum(
                        c * x
                        for c, x in zip(
                            coefficients, self.coordinate_symbols, strict=True
                        )
                    )
                )
                != 0
            ):
                raise ValueError("Only affine occupation rules are supported")
            minimum = constant + sum(
                min(0, c) * (size - 1)
                for c, size in zip(coefficients, self.target.dimensions, strict=True)
            )
            maximum = constant + sum(
                max(0, c) * (size - 1)
                for c, size in zip(coefficients, self.target.dimensions, strict=True)
            )
            if not isinstance(operator, LadderOp) and minimum < 0:
                raise ValueError("Source occupations must be nonnegative")
            if isinstance(operator, (FermionOp, SigmaMinus)) and maximum > 1:
                raise ValueError(
                    "Source spin and fermion occupations must be zero or one"
                )
            rows.append(coefficients)
        if self.coordinate_symbols and sympy.Matrix(rows).rank() != len(
            self.coordinate_symbols
        ):
            raise ValueError("Occupation rules must have full column rank (be injective)")

    def _fermion_phase(self) -> sympy.Expr:
        """Fix relative signs for direct retained fermionic generators."""
        if not all(isinstance(op, FermionOp) for op in self.operators):
            raise ValueError("Fermion targets require direct source fermion assignments")
        mapped = []
        occupied_fixed = 0
        phase = sympy.S.One
        for occupation in self.source_occupations:
            if occupation in self.coordinate_symbols:
                index = self.coordinate_symbols.index(occupation)
                phase *= (1 - 2 * occupation) ** (occupied_fixed % 2)
                for earlier in mapped:
                    if earlier > index:
                        phase *= 1 - 2 * self.coordinate_symbols[earlier] * occupation
                mapped.append(index)
            elif occupation == 1:
                occupied_fixed += 1
            elif occupation != 0:
                raise ValueError(
                    "Fermion targets require direct source fermion assignments"
                )
        if sorted(mapped) != list(range(len(self.coordinate_symbols))):
            raise ValueError("Each target fermion must map to exactly one source mode")
        return sympy.expand(phase)

    def encode(self, state: tuple[int, ...]) -> tuple[int, ...]:
        """Return source occupations for one target basis state (without its phase)."""
        if len(state) != len(self.target.dimensions) or any(
            sympy.sympify(value).is_Integer is not True or not 0 <= value < size
            for value, size in zip(state, self.target.dimensions)
        ):
            raise ValueError("State lies outside the target occupation basis")
        substitutions = dict(
            zip(self.coordinate_symbols, map(sympy.Integer, state), strict=True)
        )
        return tuple(
            int(value.xreplace(substitutions)) for value in self.source_occupations
        )
