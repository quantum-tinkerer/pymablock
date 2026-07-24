"""Packed binary operator helpers for the algebraic embedding spike.

SymPy is kept at the coefficient boundary, while binary number operators are
represented by bits in packed monomial keys.
"""

# The protocol-like arithmetic classes below are implementation details of the spike.
# ruff: noqa: D101, D102, D103, D105

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import sympy

from pymablock._packed_binary import (
    masks_from_monomial,
    monomial_from_masks,
    multiply_monomials,
)
from pymablock.number_ordered_form import NumberOrderedForm

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from sympy.physics.quantum.fermion import FermionOp
    from sympy.physics.quantum.pauli import SigmaMinus


def _is_zero(value: object) -> bool:
    """Use exact zero recognition without asking SymPy for assumptions."""
    return value == 0


@dataclass(frozen=True, slots=True)
class PackedForm:
    """Hashable sparse operator backed by packed fermion monomial keys."""

    basis: tuple[str, ...]
    items: tuple[tuple[int, sympy.Expr], ...]

    @classmethod
    def build(
        cls,
        basis: Sequence[str],
        terms: dict[int, object] | Iterable[tuple[int, object]],
    ) -> PackedForm:
        combined: dict[int, sympy.Expr] = {}
        iterator = terms.items() if isinstance(terms, dict) else terms
        for monomial, coefficient in iterator:
            coefficient = sympy.sympify(coefficient)
            if _is_zero(coefficient):
                continue
            combined[monomial] = combined.get(monomial, sympy.S.Zero) + coefficient
            if _is_zero(combined[monomial]):
                del combined[monomial]
        return cls(tuple(basis), tuple(sorted(combined.items())))

    @classmethod
    def zero(cls, basis: Sequence[str]) -> PackedForm:
        return cls(tuple(basis), ())

    @classmethod
    def identity(
        cls, basis: Sequence[str], coefficient: object = sympy.S.One
    ) -> PackedForm:
        return cls.build(basis, {0: coefficient})

    @classmethod
    def monomial(
        cls, basis: Sequence[str], monomial: int, coefficient: object = sympy.S.One
    ) -> PackedForm:
        return cls.build(basis, {monomial: coefficient})

    @property
    def terms(self) -> dict[int, sympy.Expr]:
        return dict(self.items)

    @property
    def num_modes(self) -> int:
        return len(self.basis)

    def __bool__(self) -> bool:
        return bool(self.items)

    def _require_same_basis(self, other: PackedForm) -> None:
        if self.basis != other.basis:
            raise ValueError("Packed operators must use the same ordered basis")

    def __add__(self, other: object) -> PackedForm:
        if not isinstance(other, PackedForm):
            return NotImplemented
        self._require_same_basis(other)
        return type(self).build(self.basis, (*self.items, *other.items))

    def __neg__(self) -> PackedForm:
        return type(self).build(
            self.basis, ((monomial, -coefficient) for monomial, coefficient in self.items)
        )

    def __sub__(self, other: object) -> PackedForm:
        if not isinstance(other, PackedForm):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other: object) -> PackedForm:
        if isinstance(other, PackedForm):
            self._require_same_basis(other)
            return type(self).build(
                self.basis,
                (
                    (monomial, left_coefficient * right_coefficient * integer)
                    for left, left_coefficient in self.items
                    for right, right_coefficient in other.items
                    for monomial, integer in multiply_monomials(
                        left,
                        right,
                        num_modes=self.num_modes,
                    ).items()
                ),
            )
        try:
            return type(self).build(
                self.basis,
                ((monomial, coefficient * other) for monomial, coefficient in self.items),
            )
        except Exception:
            return NotImplemented

    def __rmul__(self, other: object) -> PackedForm:
        if isinstance(other, PackedForm):
            return other * self
        return self * other

    def __truediv__(self, divisor: object) -> PackedForm:
        return self * (sympy.S.One / divisor)

    def adjoint(self) -> PackedForm:
        """Return the adjoint, including canonical-order reversal signs."""
        terms = []
        for monomial, coefficient in self.items:
            creators, numbers, annihilators = masks_from_monomial(
                monomial, num_modes=self.num_modes
            )
            # A packed monomial is the product of its local factors in ascending
            # mode order. Adjoint reverses all odd local factors; number factors are
            # even and do not contribute to the reordering sign.
            odd_count = (creators | annihilators).bit_count()
            parity = odd_count * (odd_count - 1) // 2
            adjoint = monomial_from_masks(
                creators=annihilators,
                numbers=numbers,
                annihilators=creators,
                num_modes=self.num_modes,
            )
            terms.append(
                (
                    adjoint,
                    (-1 if parity % 2 else 1) * sympy.conjugate(coefficient),
                )
            )
        return type(self).build(self.basis, terms)

    def simplify(self) -> PackedForm:
        return type(self).build(
            self.basis,
            (
                (monomial, sympy.factor(sympy.cancel(coefficient)))
                for monomial, coefficient in self.items
            ),
        )


@dataclass(frozen=True, slots=True)
class BooleanPolynomial:
    """Sparse polynomial in idempotent binary variables.

    A monomial is the product of the variables selected by its integer mask.  Mask
    multiplication is bitwise OR, which implements ``n_i**2 = n_i`` directly.
    """

    variables: tuple[sympy.Symbol, ...]
    items: tuple[tuple[int, sympy.Expr], ...]

    @classmethod
    def build(
        cls,
        variables: Sequence[sympy.Symbol],
        terms: dict[int, object] | Iterable[tuple[int, object]],
    ) -> BooleanPolynomial:
        combined: dict[int, sympy.Expr] = {}
        iterator = terms.items() if isinstance(terms, dict) else terms
        for mask, coefficient in iterator:
            coefficient = sympy.sympify(coefficient)
            if _is_zero(coefficient):
                continue
            combined[mask] = combined.get(mask, sympy.S.Zero) + coefficient
            if _is_zero(combined[mask]):
                del combined[mask]
        return cls(tuple(variables), tuple(sorted(combined.items())))

    @classmethod
    def scalar(
        cls, variables: Sequence[sympy.Symbol], coefficient: object
    ) -> BooleanPolynomial:
        return cls.build(variables, {0: coefficient})

    @classmethod
    def variable(cls, variables: Sequence[sympy.Symbol], index: int) -> BooleanPolynomial:
        return cls.build(variables, {1 << index: sympy.S.One})

    @classmethod
    def from_expr(
        cls, expression: object, variables: Sequence[sympy.Symbol]
    ) -> BooleanPolynomial:
        return _boolean_from_expr(sympy.sympify(expression), tuple(variables))

    @property
    def terms(self) -> dict[int, sympy.Expr]:
        return dict(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def _require_same_variables(self, other: BooleanPolynomial) -> None:
        if self.variables != other.variables:
            raise ValueError("Boolean polynomials must use the same variables")

    def __add__(self, other: object) -> BooleanPolynomial:
        if not isinstance(other, BooleanPolynomial):
            other = type(self).scalar(self.variables, other)
        self._require_same_variables(other)
        return type(self).build(self.variables, (*self.items, *other.items))

    def __radd__(self, other: object) -> BooleanPolynomial:
        return self + other

    def __neg__(self) -> BooleanPolynomial:
        return type(self).build(
            self.variables,
            ((mask, -coefficient) for mask, coefficient in self.items),
        )

    def __sub__(self, other: object) -> BooleanPolynomial:
        return self + (-other)

    def __rsub__(self, other: object) -> BooleanPolynomial:
        return (-self) + other

    def __mul__(self, other: object) -> BooleanPolynomial:
        if not isinstance(other, BooleanPolynomial):
            other = type(self).scalar(self.variables, other)
        self._require_same_variables(other)
        return type(self).build(
            self.variables,
            (
                (left_mask | right_mask, left_coefficient * right_coefficient)
                for left_mask, left_coefficient in self.items
                for right_mask, right_coefficient in other.items
            ),
        )

    def __rmul__(self, other: object) -> BooleanPolynomial:
        return self * other

    def reciprocal(self) -> BooleanPolynomial:
        """Return the pointwise inverse on the Boolean cube."""
        return type(self).from_expr(1 / self.as_expr(), self.variables)

    def as_expr(self) -> sympy.Expr:
        return sympy.Add(
            *(
                coefficient
                * sympy.prod(
                    variable
                    for index, variable in enumerate(self.variables)
                    if mask & (1 << index)
                )
                for mask, coefficient in self.items
            )
        )

    def diagonal(self, basis: Sequence[str]) -> PackedForm:
        if len(basis) != len(self.variables):
            raise ValueError("Boolean variables and packed basis must have equal size")
        return PackedForm.build(
            basis,
            (
                (
                    monomial_from_masks(numbers=mask, num_modes=len(basis)),
                    coefficient,
                )
                for mask, coefficient in self.items
            ),
        )


@cache
def _boolean_from_expr(
    expression: sympy.Expr,
    variables: tuple[sympy.Symbol, ...],
) -> BooleanPolynomial:
    """Use Shannon reduction but store every intermediate in the mask domain."""
    used = expression.free_symbols.intersection(variables)
    if not used:
        return BooleanPolynomial.scalar(variables, expression)
    index = next(index for index, variable in enumerate(variables) if variable in used)
    variable = variables[index]
    low = _boolean_from_expr(expression.xreplace({variable: sympy.S.Zero}), variables)
    high = _boolean_from_expr(expression.xreplace({variable: sympy.S.One}), variables)
    return low + BooleanPolynomial.variable(variables, index) * (high - low)


def _single_fermion_factor(
    basis: Sequence[str], mode: int, *, dagger: bool
) -> PackedForm:
    monomial = monomial_from_masks(
        creators=1 << mode if dagger else 0,
        annihilators=0 if dagger else 1 << mode,
        num_modes=len(basis),
    )
    return PackedForm.monomial(basis, monomial)


def _spin_factor(basis: Sequence[str], mode: int, *, dagger: bool) -> PackedForm:
    """Represent a local spin transition through a Jordan-Wigner fermion."""
    variables = sympy.symbols(f"_jw_0:{len(basis)}")
    parity = BooleanPolynomial.scalar(variables, sympy.S.One)
    for earlier in range(mode):
        parity *= 1 - 2 * BooleanPolynomial.variable(variables, earlier)
    local = _single_fermion_factor(basis, mode, dagger=dagger)
    return parity.diagonal(basis) * local


def nof_to_packed_fermions(form: NumberOrderedForm) -> PackedForm:
    """Convert a purely fermionic NumberOrderedForm to packed monomials."""
    basis = tuple(str(operator.name) for operator in form.operators)
    variables = tuple(form._number_operator_placeholders)
    result = PackedForm.zero(basis)
    for powers, coefficient in form.terms.items():
        if any(abs(int(power)) > 1 for power in powers):
            raise ValueError("Fermionic powers must be -1, 0, or 1")
        term = PackedForm.identity(basis)
        for mode, power in enumerate(powers):
            if power < 0:
                term *= _single_fermion_factor(basis, mode, dagger=True)
        term *= BooleanPolynomial.from_expr(coefficient, variables).diagonal(basis)
        for mode in reversed(range(len(powers))):
            if powers[mode] > 0:
                term *= _single_fermion_factor(basis, mode, dagger=False)
        result += term
    return result


def spin_term_from_boolean(
    basis: Sequence[str],
    powers: Sequence[int],
    coefficient: BooleanPolynomial,
) -> PackedForm:
    """Build one number-ordered spin term in the packed Jordan-Wigner algebra."""
    term = PackedForm.identity(basis)
    for mode, power in enumerate(powers):
        if power < 0:
            term *= _spin_factor(basis, mode, dagger=True)
    term *= coefficient.diagonal(basis)
    for mode in reversed(range(len(powers))):
        if power := powers[mode]:
            if power > 0:
                term *= _spin_factor(basis, mode, dagger=False)
    return term


def packed_to_spin_nof(
    form: PackedForm, operators: Sequence[SigmaMinus]
) -> NumberOrderedForm:
    """Convert the packed Jordan-Wigner representation back at the readout boundary."""
    if len(operators) != form.num_modes:
        raise ValueError("Packed basis and spin operators must have equal size")
    zero = NumberOrderedForm(tuple(operators), {}, validate=False)
    one = NumberOrderedForm(
        tuple(operators), {(0,) * len(operators): sympy.S.One}, validate=False
    )
    placeholders = tuple(one._number_operator_placeholders)

    def parity(mode: int) -> NumberOrderedForm:
        result = one
        zero_powers = (0,) * len(operators)
        for earlier in range(mode):
            diagonal = NumberOrderedForm(
                tuple(operators),
                {zero_powers: 1 - 2 * placeholders[earlier]},
                validate=False,
            )
            result *= diagonal
        return result

    result = zero
    for monomial, coefficient in form.items:
        creators, numbers, annihilators = masks_from_monomial(
            monomial, num_modes=form.num_modes
        )
        term = one * coefficient
        for mode in range(form.num_modes):
            if creators & (1 << mode):
                powers = tuple(
                    -1 if index == mode else 0 for index in range(form.num_modes)
                )
                term *= parity(mode) * NumberOrderedForm(
                    tuple(operators), {powers: sympy.S.One}, validate=False
                )
            elif numbers & (1 << mode):
                term *= NumberOrderedForm(
                    tuple(operators),
                    {(0,) * form.num_modes: placeholders[mode]},
                    validate=False,
                )
            elif annihilators & (1 << mode):
                powers = tuple(
                    1 if index == mode else 0 for index in range(form.num_modes)
                )
                term *= parity(mode) * NumberOrderedForm(
                    tuple(operators), {powers: sympy.S.One}, validate=False
                )
        result += term
    return result


def packed_to_fermion_nof(
    form: PackedForm,
    operators: Sequence[FermionOp],
) -> NumberOrderedForm:
    """Convert packed monomials back to a fermionic NumberOrderedForm."""
    operators = tuple(operators)
    if len(operators) != form.num_modes:
        raise ValueError("Packed basis and fermion operators must have equal size")
    return NumberOrderedForm._from_packed_terms(
        operators,
        ((((), monomial), coefficient) for monomial, coefficient in form.items),
    )


def packed_powers(monomial: int, *, num_modes: int) -> tuple[int, ...]:
    """Return number-ordered transition powers for one packed monomial."""
    creators, _numbers, annihilators = masks_from_monomial(monomial, num_modes=num_modes)
    return tuple(
        1 if annihilators & (1 << mode) else -1 if creators & (1 << mode) else 0
        for mode in range(num_modes)
    )


def packed_monomial_term(
    form: PackedForm, monomial: int, coefficient: object
) -> PackedForm:
    return PackedForm.monomial(form.basis, monomial, coefficient)
