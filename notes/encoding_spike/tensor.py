"""Sums of tensor products over native operator representations."""

# The arithmetic class is implementation scaffolding for the encoding spike.
# ruff: noqa: D101, D102, D105

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import sympy
from sympy.physics.quantum import Dagger

from pymablock.number_ordered_form import NumberOrderedForm

from .packed import PackedForm

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

NativeOperator = NumberOrderedForm | PackedForm


@cache
def _nof_identity(operators: tuple) -> NumberOrderedForm:
    return NumberOrderedForm(
        operators,
        {(0,) * len(operators): sympy.S.One},
    )


@cache
def _packed_identity(basis: tuple[str, ...]) -> PackedForm:
    return PackedForm.identity(basis)


def _native_identity(operator: NativeOperator) -> NativeOperator:
    if isinstance(operator, NumberOrderedForm):
        return _nof_identity(tuple(operator.operators))
    if isinstance(operator, PackedForm):
        return _packed_identity(operator.basis)
    raise TypeError(f"Unsupported tensor factor: {type(operator)}")


@cache
def _native_zero(identity: NativeOperator) -> NativeOperator:
    if isinstance(identity, NumberOrderedForm):
        return NumberOrderedForm(identity.operators, {}, validate=False)
    if isinstance(identity, PackedForm):
        return PackedForm.zero(identity.basis)
    raise TypeError(f"Unsupported tensor factor: {type(identity)}")


def _native_is_zero(operator: NativeOperator) -> bool:
    return not bool(operator)


def _native_adjoint(operator: NativeOperator) -> NativeOperator:
    if isinstance(operator, NumberOrderedForm):
        return Dagger(operator)
    if isinstance(operator, PackedForm):
        return operator.adjoint()
    raise TypeError(f"Unsupported tensor factor: {type(operator)}")


def _native_simplify(operator: NativeOperator) -> NativeOperator:
    if isinstance(operator, NumberOrderedForm):
        return NumberOrderedForm(
            operator.operators,
            (
                (powers, sympy.factor(sympy.cancel(coefficient)))
                for powers, coefficient in operator.terms.items()
            ),
            validate=False,
        )
    if isinstance(operator, PackedForm):
        return operator.simplify()
    raise TypeError(f"Unsupported tensor factor: {type(operator)}")


def _same_native_space(left: NativeOperator, right: NativeOperator) -> bool:
    if isinstance(left, NumberOrderedForm) and isinstance(right, NumberOrderedForm):
        return left.operators == right.operators
    if isinstance(left, PackedForm) and isinstance(right, PackedForm):
        return left.basis == right.basis
    return False


@dataclass(frozen=True, slots=True)
class TensorOperator:
    """A sparse sum of products over independent native operator factors.

    A term ``(factors, coefficient)`` represents
    ``coefficient * factors[0] ⊗ ... ⊗ factors[-1]``.  NOF and packed factors
    retain their own multiplication rules; this class only distributes sums and
    performs tensor-factor-wise arithmetic. Finite matrix structure is an outer
    layer, following Pymablock's existing matrix-of-operators convention.
    """

    identities: tuple[NativeOperator, ...]
    items: tuple[tuple[tuple[NativeOperator, ...], sympy.Expr], ...]

    @classmethod
    def build(
        cls,
        identities: Sequence[NativeOperator],
        terms: Iterable[tuple[Sequence[NativeOperator], object]],
    ) -> TensorOperator:
        identities = tuple(identities)
        if len(identities) == 1:
            total = _native_zero(identities[0])
            for raw_factors, raw_coefficient in terms:
                factors = tuple(raw_factors)
                coefficient = sympy.sympify(raw_coefficient)
                if len(factors) != 1:
                    raise ValueError("A tensor term has the wrong number of factors")
                factor = factors[0]
                if not _same_native_space(factor, identities[0]):
                    raise ValueError("A tensor term does not match its factor space")
                if coefficient != 0 and not _native_is_zero(factor):
                    total = total + factor * coefficient
            if _native_is_zero(total):
                return cls(identities, ())
            return cls(identities, (((total,), sympy.S.One),))

        combined: dict[tuple[NativeOperator, ...], sympy.Expr] = {}
        for raw_factors, raw_coefficient in terms:
            factors = tuple(raw_factors)
            coefficient = sympy.sympify(raw_coefficient)
            if len(factors) != len(identities):
                raise ValueError("A tensor term has the wrong number of factors")
            if coefficient == 0 or any(_native_is_zero(factor) for factor in factors):
                continue
            if any(
                not _same_native_space(factor, identity)
                for factor, identity in zip(factors, identities, strict=True)
            ):
                raise ValueError("A tensor term does not match its factor spaces")
            combined[factors] = combined.get(factors, sympy.S.Zero) + coefficient
            if combined[factors] == 0:
                del combined[factors]

        return cls(identities, tuple(combined.items()))

    @classmethod
    def from_factor(cls, operator: NativeOperator) -> TensorOperator:
        identity = _native_identity(operator)
        if _native_is_zero(operator):
            return cls((identity,), ())
        return cls((identity,), (((operator,), sympy.S.One),))

    @classmethod
    def from_factors(cls, factors: Sequence[NativeOperator]) -> TensorOperator:
        factors = tuple(factors)
        identities = tuple(_native_identity(factor) for factor in factors)
        return cls.build(identities, ((factors, sympy.S.One),))

    @classmethod
    def zero(cls, identities: Sequence[NativeOperator]) -> TensorOperator:
        return cls(tuple(identities), ())

    @classmethod
    def identity(cls, identities: Sequence[NativeOperator]) -> TensorOperator:
        identities = tuple(identities)
        return cls.build(identities, ((identities, sympy.S.One),))

    @property
    def num_factors(self) -> int:
        return len(self.identities)

    @property
    def native(self) -> NativeOperator:
        """Return the native value of a one-factor tensor."""
        if self.num_factors != 1:
            raise TypeError("A multi-factor tensor has no single native operator")
        if not self.items:
            return _native_zero(self.identities[0])
        factors, coefficient = self.items[0]
        return factors[0] if coefficient == 1 else factors[0] * coefficient

    def __bool__(self) -> bool:
        return bool(self.items)

    def _require_same_spaces(self, other: TensorOperator) -> None:
        if len(self.identities) != len(other.identities) or any(
            not _same_native_space(left, right)
            for left, right in zip(self.identities, other.identities, strict=True)
        ):
            raise ValueError("Tensor operators act on different factor spaces")

    def __add__(self, other: object) -> TensorOperator:
        if not isinstance(other, TensorOperator):
            return NotImplemented
        self._require_same_spaces(other)
        if self.num_factors == 1:
            return type(self).from_factor(self.native + other.native)
        return type(self).build(self.identities, (*self.items, *other.items))

    def __neg__(self) -> TensorOperator:
        if self.num_factors == 1:
            return type(self).from_factor(-self.native)
        return type(self).build(
            self.identities,
            ((factors, -coefficient) for factors, coefficient in self.items),
        )

    def __sub__(self, other: object) -> TensorOperator:
        if not isinstance(other, TensorOperator):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other: object) -> TensorOperator:
        if isinstance(other, TensorOperator):
            self._require_same_spaces(other)
            if self.num_factors == 1:
                return type(self).from_factor(self.native * other.native)
            return type(self).build(
                self.identities,
                (
                    (
                        tuple(
                            left * right
                            for left, right in zip(
                                left_factors,
                                right_factors,
                                strict=True,
                            )
                        ),
                        left_coefficient * right_coefficient,
                    )
                    for left_factors, left_coefficient in self.items
                    for right_factors, right_coefficient in other.items
                ),
            )
        try:
            if self.num_factors == 1:
                return type(self).from_factor(self.native * other)
            return type(self).build(
                self.identities,
                ((factors, coefficient * other) for factors, coefficient in self.items),
            )
        except Exception:
            return NotImplemented

    def __rmul__(self, other: object) -> TensorOperator:
        if isinstance(other, TensorOperator):
            return other * self
        return self * other

    def __truediv__(self, divisor: object) -> TensorOperator:
        return self * (sympy.S.One / divisor)

    def adjoint(self) -> TensorOperator:
        if self.num_factors == 1:
            return type(self).from_factor(_native_adjoint(self.native))
        return type(self).build(
            tuple(_native_adjoint(identity) for identity in self.identities),
            (
                (
                    tuple(_native_adjoint(factor) for factor in factors),
                    sympy.conjugate(coefficient),
                )
                for factors, coefficient in self.items
            ),
        )

    def simplify(self) -> TensorOperator:
        if self.num_factors == 1:
            return type(self).from_factor(_native_simplify(self.native))
        return type(self).build(
            self.identities,
            (
                (
                    tuple(_native_simplify(factor) for factor in factors),
                    sympy.factor(sympy.cancel(coefficient)),
                )
                for factors, coefficient in self.items
            ),
        )
