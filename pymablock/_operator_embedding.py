"""Block diagonalization with an implicit discarded space.

W embeds the retained model into the source space; P = W W† and Q = 1 - P.
The retained block uses ordinary operators or matrices. Coupling blocks store
Q X W A, and complement blocks act within Q without enumerating its basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import sympy

from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main
from pymablock.number_ordered_form import NumberOrderedForm
from pymablock.operator_embedding import (
    Embedding,
    _NOFTransition,
    _ReferenceBasis,
)
from pymablock.series import BlockSeries, zero

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

Operator: TypeAlias = NumberOrderedForm | sympy.MatrixBase


def _is_zero(value: NumberOrderedForm | sympy.MatrixBase) -> bool:
    """Test exact structural zero without simplifying coefficients."""
    if isinstance(value, sympy.MatrixBase):
        return all(
            _is_zero(entry) if isinstance(entry, NumberOrderedForm) else entry == 0
            for entry in value
        )
    return not any(coefficient != 0 for coefficient in value.terms.values())


def _adjoint(value: Operator) -> Operator:
    """Return an algebra-native adjoint."""
    result = value.adjoint()
    if isinstance(value, NumberOrderedForm) and result.operators != value.operators:
        # SymPy may cache an equal expression with a different list of unused
        # generators. Restore the declared basis using the public conversion.
        return NumberOrderedForm.from_expr(result.as_expr(), operators=value.operators)
    return result


def _virtual_product(basis, left, right):
    """Return ``W† left Q right W``, retaining only discarded intermediate states."""
    return basis._pullback(left * right) - basis._pullback(left) * basis._pullback(right)


class _CouplingBlock:
    """Coupling from retained to discarded states: a sum of ``Q X W A``.

    Each stored pair contains a source operator X and a retained operator A.
    Products are evaluated in the source algebra before compression by W†.
    """

    __slots__ = ("basis", "terms")

    def __init__(
        self,
        basis,
        terms: Iterable[tuple[Operator, Operator]],
    ):
        """Combine equal source factors and discard exact structural zeros."""
        combined: dict[Operator, Operator] = {}
        for source, target in terms:
            if _is_zero(source) or _is_zero(target):
                continue
            combined[source] = combined.get(source, basis._target_zero) + target

        self.basis = basis
        self.terms = tuple(
            (source, target)
            for source, target in combined.items()
            if not _is_zero(target)
        )

    @classmethod
    def from_source(
        cls,
        basis,
        source: Operator,
    ):
        """Construct ``Q X W`` from one source-space operator ``X``."""
        return cls(
            basis,
            ((source, basis._target_identity),),
        ).or_zero()

    def or_zero(self):
        """Use Pymablock's structural zero sentinel for an empty map."""
        return self if self.terms else zero

    def __bool__(self) -> bool:
        return bool(self.terms)

    def _require_same_basis(self, other: _CouplingBlock) -> None:
        if self.basis is not other.basis:
            raise ValueError("Coupling blocks must use the same embedding")

    def __add__(self, other: object):
        if other is zero:
            return self
        if not isinstance(other, _CouplingBlock):
            return NotImplemented
        self._require_same_basis(other)
        return type(self)(
            self.basis,
            (*self.terms, *other.terms),
        ).or_zero()

    def __neg__(self):
        return self.right(-sympy.S.One)

    def __sub__(self, other: object):
        return self + (-other)

    def __mul__(self, factor: object):
        try:
            factor = sympy.sympify(factor)
        except sympy.SympifyError:
            return NotImplemented
        if not factor.is_commutative:
            return NotImplemented
        return self.right(factor)

    __rmul__ = __mul__

    def __truediv__(self, divisor: object):
        return self.right(sympy.S.One / divisor)

    def right(self, target: Operator):
        """Compose with a retained-space operator on the right."""
        return type(self)(
            self.basis,
            ((source, coefficient * target) for source, coefficient in self.terms),
        ).or_zero()

    def left(self, source: Operator):
        """Apply a source operator while preserving the outer projection.

        This uses ``Q S Q X W A = Q S X W A - Q S W (W† X W) A``.
        """
        terms = []
        for inner_source, target in self.terms:
            terms.append((source * inner_source, target))
            terms.append((source, -self.basis._pullback(inner_source) * target))
        return type(self)(self.basis, terms).or_zero()

    def inner(self, other: _CouplingBlock) -> Operator:
        """Return ``self† other`` in the retained operator algebra."""
        self._require_same_basis(other)

        result = self.basis._target_zero
        for left_source, left_target in self.terms:
            left_adjoint = left_source.adjoint()
            for right_source, right_target in other.terms:
                kernel = _virtual_product(self.basis, left_adjoint, right_source)
                result += _adjoint(left_target) * kernel * right_target
        return result

    def adjoint(self) -> _AdjointCouplingBlock:
        """Return the coupling from discarded to retained states."""
        return _AdjointCouplingBlock(self)


@dataclass(frozen=True, slots=True)
class _AdjointCouplingBlock:
    """Coupling from discarded to retained states, stored through its adjoint."""

    column: _CouplingBlock

    def __add__(self, other: object):
        if other is zero:
            return self
        if not isinstance(other, _AdjointCouplingBlock):
            return NotImplemented
        result = self.column + other.column
        return result.adjoint()

    def __neg__(self):
        result = -self.column
        return result.adjoint()

    def __sub__(self, other: object):
        return self + (-other)

    def __truediv__(self, divisor: object):
        result = self.column / sympy.conjugate(divisor)
        return result.adjoint()

    def adjoint(self) -> _CouplingBlock:
        """Return the underlying P→Q column."""
        return self.column


@dataclass(frozen=True)
class _ComplementBlock:
    """An operator within the discarded space, stored as forward and adjoint actions.

    The recurrence needs both actions, but never a matrix representation of Q.
    """

    action: Callable
    adjoint_action: Callable

    @classmethod
    def source(cls, source):
        """Represent ``Q source Q``."""
        return cls(lambda c: c.left(source), lambda c: c.left(source.adjoint()))

    @classmethod
    def outer(cls, left, right):
        """Represent ``left right†``."""
        left._require_same_basis(right)
        return cls(
            lambda c: left.right(right.inner(c)), lambda c: right.right(left.inner(c))
        )

    def apply(self, column):
        """Apply the action, propagating an empty column."""
        return zero if column is zero else self.action(column)

    def adjoint(self):
        """Exchange the forward and adjoint actions."""
        return type(self)(self.adjoint_action, self.action)

    def compose(self, other):
        """Compose actions; reverse the order for the adjoint."""
        return type(self)(
            lambda c: self.apply(other.apply(c)),
            lambda c: other.adjoint().apply(self.adjoint().apply(c)),
        )

    def __add__(self, other):
        if other is zero:
            return self
        if not isinstance(other, _ComplementBlock):
            return NotImplemented
        return type(self)(
            lambda c: self.apply(c) + other.apply(c),
            lambda c: self.adjoint().apply(c) + other.adjoint().apply(c),
        )

    def __neg__(self):
        return self / -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, divisor):
        return type(self)(
            lambda c: self.apply(c) * (sympy.S.One / divisor),
            lambda c: self.adjoint().apply(c) * (sympy.S.One / sympy.conjugate(divisor)),
        )


# The four block types have shapes PP, QP, PQ, QQ respectively.
# This is the complete multiplication table for conformable blocks.
_BLOCK_PRODUCTS = {
    (Operator, Operator): lambda a, b: a * b,
    (Operator, _AdjointCouplingBlock): lambda a, b: b.column.right(a.adjoint()).adjoint(),
    (_CouplingBlock, Operator): _CouplingBlock.right,
    (_CouplingBlock, _AdjointCouplingBlock): _ComplementBlock.outer,
    (_AdjointCouplingBlock, _CouplingBlock): lambda a, b: a.column.inner(b),
    (_AdjointCouplingBlock, _ComplementBlock): lambda a, b: b.adjoint()
    .apply(a.column)
    .adjoint(),
    (_ComplementBlock, _CouplingBlock): _ComplementBlock.apply,
    (_ComplementBlock, _ComplementBlock): _ComplementBlock.compose,
}


def multiply_projected(left, right):
    """Multiply conformable blocks of the projected algebra."""
    kinds = tuple(
        Operator if isinstance(x, (NumberOrderedForm, sympy.MatrixBase)) else type(x)
        for x in (left, right)
    )
    try:
        multiply = _BLOCK_PRODUCTS[kinds]
    except KeyError:
        raise TypeError(
            f"Cannot multiply projected blocks {type(left)} and {type(right)}"
        ) from None
    return multiply(left, right)


def _diagonal_energy(entry):
    zero_powers = (0,) * len(entry.operators)
    if any(
        powers != zero_powers and coefficient != 0
        for powers, coefficient in entry.terms.items()
    ):
        raise ValueError("Structured embeddings currently require diagonal H0")
    return sympy.expand(entry.terms.get(zero_powers, sympy.S.Zero))


def _generator_solver(h0, basis):
    """Compile energy division in symbolic target occupations."""
    source_energy = _diagonal_energy(h0)

    # Selecting occupation states of diagonal H0 is automatically invariant.
    target_energy = basis._at_occupations(source_energy, basis.source_occupations)

    def _channel_denominator(
        source: _NOFTransition,
        target: _NOFTransition,
    ) -> sympy.Expr:
        encoded_after_target = basis._shifted_occupations(target.powers)
        source_output = tuple(
            occupation - power
            for occupation, power in zip(encoded_after_target, source.powers, strict=True)
        )
        source_value = basis._at_occupations(source_energy, source_output)
        denominator = sympy.expand(target_energy - source_value)
        denominator = denominator.xreplace(
            basis._support_substitutions(source, target.powers)
        )
        return denominator.xreplace(basis._initial_to_middle(target.powers))

    def _has_no_virtual_action(
        source: _NOFTransition,
        target: NumberOrderedForm,
    ) -> bool:
        leakage = _virtual_product(basis, source.form.adjoint(), source.form)
        norm = target.adjoint() * leakage * target
        return all(sympy.cancel(coefficient) == 0 for coefficient in norm.terms.values())

    def _divide_channel(
        source: _NOFTransition, target: NumberOrderedForm, denominator: sympy.Expr
    ) -> NumberOrderedForm:
        """Divide middle coefficients, resolving finite occupation sectors."""
        variables = tuple(
            (symbol, size - abs(int(power)))
            for symbol, size, power in zip(
                basis._target_placeholders,
                basis._target_dimensions,
                next(iter(target.terms)),
                strict=True,
            )
            if size is not None and symbol in denominator.free_symbols
        )

        sectors = []
        has_zero = False

        def visit(expression, mask, remaining):
            nonlocal has_zero
            expression = sympy.expand(expression)
            remaining = tuple(x for x in remaining if x[0] in expression.free_symbols)
            if remaining:
                (symbol, size), *rest = remaining
                for value in range(size):
                    visit(
                        expression.xreplace({symbol: sympy.Integer(value)}),
                        mask
                        * (sympy.S.One if size == 1 else symbol if value else 1 - symbol),
                        rest,
                    )
                return
            expression = sympy.simplify(expression)
            if expression == 0:
                has_zero = True
                sector = target.applyfunc(lambda coefficient: coefficient * mask)
                if not _has_no_virtual_action(source, sector):
                    raise ZeroDivisionError(
                        "A virtual channel is degenerate with the retained space"
                    )
            else:
                sectors.append(mask / expression)

        visit(denominator, sympy.S.One, variables)
        inverse = sympy.Add(*sectors) if has_zero else 1 / denominator
        result = target.applyfunc(lambda coefficient: coefficient * inverse)
        return result * basis._target_identity

    def _solve_terms(terms):
        for source_form, target_form in terms:
            for source in _NOFTransition.from_form(source_form):
                for target in _NOFTransition.from_form(target_form):
                    denominator = _channel_denominator(source, target)
                    yield (
                        source.form,
                        _divide_channel(source, target.form, denominator),
                    )

    return _solve_terms


def _reference_solver(h0, basis):
    """Compile energy division for an ordered list of source states."""

    def _energy(component, state):
        return basis._at_occupations(source_energies[component], state)

    source_energies = [sympy.S.Zero] * h0.rows
    for (row, column), entry in h0.todok().items():
        if row != column and not _is_zero(entry):
            raise ValueError("Structured embeddings currently require diagonal H0")
        if row == column:
            source_energies[row] = _diagonal_energy(entry)
    reference_energies = [_energy(c, state) for c, state in basis._references]

    def _solve_terms(terms):
        for source, target in terms:
            for factor, j, state, _ in basis._actions(source):
                if state in basis._reference_indices:
                    continue  # Q removes the entire retained subspace.
                energy = _energy(*state)
                quotient = {}
                for (_, k), coefficient in target[j, :].todok().items():
                    denominator = sympy.simplify(reference_energies[k] - energy)
                    if denominator == 0:
                        raise ZeroDivisionError(
                            "A virtual channel is degenerate with the retained space"
                        )
                    quotient[j, k] = coefficient / denominator
                yield factor, sympy.ImmutableSparseMatrix(*target.shape, quotient)

    return _solve_terms


def block_diagonalize(
    hamiltonian: BlockSeries,
    embedding: Embedding,
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """Run the standard recurrence over a structured embedding."""
    basis = embedding._basis
    if hamiltonian.shape:
        raise ValueError("Structured embeddings require an unseparated Hamiltonian.")
    zero_order = (0,) * hamiltonian.n_infinite
    h0 = basis._source_form(hamiltonian[zero_order])
    source_shape = h0.shape if isinstance(h0, sympy.MatrixBase) else None
    divide = (
        _reference_solver if isinstance(basis, _ReferenceBasis) else _generator_solver
    )(h0, basis)

    def solve_sylvester(value, index):
        """Solve a P-Q equation using the compiled representation's channels."""
        if value is zero:
            return zero
        if index[:2] != (0, 1) or not isinstance(value, _AdjointCouplingBlock):
            raise TypeError("The embedding solver expects the P-Q block")
        return _CouplingBlock(basis, divide(value.column.terms)).or_zero().adjoint()

    def evaluate(*index):
        row, column, *order = index
        source = hamiltonian[tuple(order)]
        if source is zero:
            return zero
        source = basis._source_form(source)
        shape = source.shape if isinstance(source, sympy.MatrixBase) else None
        if shape != source_shape:
            raise ValueError(
                "All Hamiltonian coefficients must have the same source matrix shape"
            )
        if _is_zero(source):
            return zero
        if row == column == 0:
            result = basis._pullback(source)
            return zero if _is_zero(result) else result
        if tuple(order) == zero_order and row != column:
            return zero
        if (row, column) == (1, 0):
            return _CouplingBlock.from_source(basis, source)
        if (row, column) == (0, 1):
            column_map = _CouplingBlock.from_source(basis, source)
            return column_map.adjoint()
        return _ComplementBlock.source(source)

    outputs, _ = series_computation(
        {
            "H": BlockSeries(
                eval=evaluate,
                shape=(2, 2),
                n_infinite=hamiltonian.n_infinite,
                dimension_names=hamiltonian.dimension_names,
                name="H",
            )
        },
        algorithm=main,
        scope={
            "solve_sylvester": solve_sylvester,
            "use_linear_operator": np.zeros((2, 2), dtype=bool),
            "two_block_optimized": True,
            "commuting_blocks": [True, True],
        },
        operator=multiply_projected,
    )
    return outputs["H_tilde"], outputs["U"], outputs["U†"]
