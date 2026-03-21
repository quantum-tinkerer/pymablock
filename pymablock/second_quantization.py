"""Second quantization tools.

See number_ordered_form_plan.md for the plan to implement NumberOrderedForm as an
Operator subclass for better representation of number-ordered expressions.
"""

from collections.abc import Callable
from functools import lru_cache

import numpy as np
import sympy
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOrderedForm,
)
from pymablock.series import zero

__all__ = [
    "apply_mask_to_operator",
    "expand_compact_denominators",
    "solve_sylvester_2nd_quant",
]

_COMPACT_DENOMINATOR_REGISTRY: dict[int, tuple[tuple[sympy.Symbol, ...], sympy.Expr]] = {}
_COMPACT_DENOMINATOR_REVERSE_REGISTRY: dict[
    tuple[tuple[sympy.Symbol, ...], sympy.Expr], int
] = {}
_NEXT_COMPACT_DENOMINATOR_ID = 0


class _CompactDenominator(sympy.Function):
    """Compact placeholder for a shifted Sylvester denominator."""

    is_commutative = True
    is_finite = True
    is_nonzero = True
    is_real = True
    is_zero = False

    def _eval_conjugate(self):
        return self


def _register_compact_denominator(
    placeholders: tuple[sympy.Symbol, ...],
    expr: sympy.Expr,
) -> sympy.Integer:
    global _NEXT_COMPACT_DENOMINATOR_ID
    key = (placeholders, expr)
    if key not in _COMPACT_DENOMINATOR_REVERSE_REGISTRY:
        token = _NEXT_COMPACT_DENOMINATOR_ID
        _NEXT_COMPACT_DENOMINATOR_ID += 1
        _COMPACT_DENOMINATOR_REVERSE_REGISTRY[key] = token
        _COMPACT_DENOMINATOR_REGISTRY[token] = key
    return sympy.Integer(_COMPACT_DENOMINATOR_REVERSE_REGISTRY[key])


def _make_compact_denominator(
    expr: sympy.Expr,
    placeholders: tuple[sympy.Symbol, ...],
) -> sympy.Expr:
    """Create a compact denominator atom that still shifts with placeholders."""
    used_placeholders = tuple(
        placeholder for placeholder in placeholders if expr.has(placeholder)
    )
    token = _register_compact_denominator(used_placeholders, expr)
    return _CompactDenominator(token, *used_placeholders)


def _expand_compact_denominator(expr: _CompactDenominator) -> sympy.Expr:
    return _expand_registered_compact_denominator(int(expr.args[0]), expr.args[1:])


@lru_cache(maxsize=8192)
def _expand_registered_compact_denominator(
    token: int,
    values: tuple[sympy.Expr, ...],
) -> sympy.Expr:
    placeholders, template = _COMPACT_DENOMINATOR_REGISTRY[token]
    if not placeholders:
        return sympy.expand(template)
    return sympy.expand(template.xreplace(dict(zip(placeholders, values, strict=True))))


def expand_compact_denominators(expr: sympy.Expr) -> sympy.Expr:
    """Expand compact Sylvester denominator atoms in an expression."""
    return expr.replace(
        lambda item: isinstance(item, _CompactDenominator),
        _expand_compact_denominator,
    )


def _is_infinite_order_only_operators(operators: tuple) -> bool:
    return bool(operators) and all(
        isinstance(op, (BosonOp, LadderOp)) for op in operators
    )


def _to_number_ordered_form(expr) -> NumberOrderedForm:
    if isinstance(expr, NumberOrderedForm):
        return expr
    return NumberOrderedForm.from_expr(expr)


def _extract_particle_conserving_coefficient(
    expr: NumberOrderedForm | sympy.Expr,
) -> sympy.Expr:
    """Extract the scalar coefficient from a particle-conserving expression."""
    if not isinstance(expr, NumberOrderedForm):
        return sympy.sympify(expr)

    if len(expr.args[1]) != 1:
        raise ValueError(
            "Diagonal second-quantized Hamiltonians must have a single scalar term."
        )

    powers, coeff = expr.args[1][0]
    if any(powers):
        raise ValueError(
            "Diagonal second-quantized Hamiltonians must contain only number operators."
        )
    return coeff


def _shift_number_operator_placeholders(
    coeff: sympy.Expr,
    operators: tuple,
    placeholders: tuple[sympy.Symbol, ...],
    operator_shifts: dict,
    *,
    positive: bool,
) -> sympy.Expr:
    """Shift bosonic/lattice number operators in a scalar coefficient."""
    replacements = {}
    for op, placeholder in zip(operators, placeholders):
        delta = operator_shifts.get(op, 0)
        if positive:
            if delta <= 0:
                continue
            shift_value = delta
        else:
            if delta >= 0:
                continue
            shift_value = -delta

        replacements[placeholder] = (
            placeholder + shift_value
            if isinstance(op, (BosonOp, LadderOp))
            else sympy.S.One
        )

    if not replacements:
        return coeff
    return coeff.xreplace(replacements)


def _make_sylvester_denominator_getter(
    H_ii: NumberOrderedForm | sympy.Expr,
    H_jj: NumberOrderedForm | sympy.Expr,
) -> Callable[[tuple, tuple[int, ...]], sympy.Expr]:
    """Build a cached compact denominator getter for one pair of diagonal energies."""
    coeff_ii = _extract_particle_conserving_coefficient(H_ii)
    coeff_jj = _extract_particle_conserving_coefficient(H_jj)
    ops_ii = tuple(H_ii.operators) if isinstance(H_ii, NumberOrderedForm) else ()
    ops_jj = tuple(H_jj.operators) if isinstance(H_jj, NumberOrderedForm) else ()
    placeholders_ii = (
        tuple(H_ii._number_operator_placeholders)
        if isinstance(H_ii, NumberOrderedForm)
        else ()
    )
    placeholders_jj = (
        tuple(H_jj._number_operator_placeholders)
        if isinstance(H_jj, NumberOrderedForm)
        else ()
    )
    cache: dict[tuple[tuple, tuple[int, ...]], sympy.Expr] = {}

    def get_denominator(operators: tuple, shift: tuple[int, ...]) -> sympy.Expr:
        key = (operators, shift)
        if key in cache:
            return cache[key]

        if not any(shift):
            denominator = coeff_ii - coeff_jj
        else:
            operator_shifts = dict(zip(operators, shift, strict=True))
            shifted_H_jj = _shift_number_operator_placeholders(
                coeff_jj,
                ops_jj,
                placeholders_jj,
                operator_shifts,
                positive=True,
            )
            shifted_H_ii = _shift_number_operator_placeholders(
                coeff_ii,
                ops_ii,
                placeholders_ii,
                operator_shifts,
                positive=False,
            )
            if shift < (0,) * len(shift):
                denominator = shifted_H_jj - shifted_H_ii
            else:
                denominator = shifted_H_ii - shifted_H_jj

        result = sympy.collect_const(denominator).doit()
        result = _make_compact_denominator(
            result,
            tuple(
                placeholder
                for placeholder in (*placeholders_ii, *placeholders_jj)
                if result.has(placeholder)
            ),
        )
        cache[key] = result
        return result

    return get_denominator


def _solve_scalar_with_denominator_impl(
    Y: sympy.Expr,
    denominator_getter: Callable[[tuple, tuple[int, ...]], sympy.Expr],
    diagonal: bool = False,
    *,
    expand_result: bool,
) -> NumberOrderedForm:
    """Solve a scalar Sylvester equation with a precomputed denominator getter."""
    if Y == 0:
        return sympy.S.Zero

    Y = NumberOrderedForm.from_expr(Y)
    operators = tuple(Y.operators)

    shifts = Y.terms
    new_shifts = {}
    for shift, coeff in shifts.items():
        # Ensure that the energy denominator always has the same sign to simplify the
        # expression. We do this by multiplying the denominator by -1 if the shift is
        # lexicographically negative.
        sign = -sympy.S.One if tuple(shift) < (0,) * len(shift) else sympy.S.One
        if diagonal and sign is sympy.S.One:
            continue
        denominator = denominator_getter(operators, tuple(shift))
        new_shifts[shift] = sign * denominator**-sympy.S.One * coeff

    result = (
        NumberOrderedForm(
            operators=Y.args[0],
            terms=new_shifts,
        )
        ._cancel_binary_operator_numbers()
        ._linearize_binary_operators()
    )

    if diagonal:
        result -= result.adjoint()

    if not expand_result:
        return result
    return result.applyfunc(expand_compact_denominators)


def _solve_scalar_with_denominator(
    Y: sympy.Expr,
    denominator_getter: Callable[[tuple, tuple[int, ...]], sympy.Expr],
    diagonal: bool = False,
) -> NumberOrderedForm:
    """Solve a scalar Sylvester equation and return the public expanded result."""
    return _solve_scalar_with_denominator_impl(
        Y,
        denominator_getter,
        diagonal,
        expand_result=True,
    )


def _solve_scalar_with_compact_denominator(
    Y: sympy.Expr,
    denominator_getter: Callable[[tuple, tuple[int, ...]], sympy.Expr],
    diagonal: bool = False,
) -> NumberOrderedForm:
    """Solve a scalar Sylvester equation and keep compact internal denominators."""
    return _solve_scalar_with_denominator_impl(
        Y,
        denominator_getter,
        diagonal,
        expand_result=False,
    )


def solve_scalar(
    Y: sympy.Expr,
    H_ii: sympy.Expr,
    H_jj: sympy.Expr,
    diagonal: bool = False,
) -> NumberOrderedForm:
    """Solve a scalar Sylvester equation with 2nd quantized operators.

    `H_ii` and `H_jj` are scalar expressions containing number operators of
    possibly several bosons, fermions, and spin operators.

    See more details of how this works in the :doc:`second quantization
    documentation <../second_quantization>`.

    Parameters
    ----------
    Y :
        Expression with raising and lowering bosonic, fermionic, and spin operators.
    H_ii :
        Sectors of the unperturbed Hamiltonian.
    H_jj :
        Sectors of the unperturbed Hamiltonian.
    diagonal : bool
        If True, we're evaluating the diagonal entry of the matrix operator,
        which means that `Y` is Hermitian and `H_ii` and `H_jj` are equal. This
        is used to speed up the computation.

    Returns
    -------
    NumberOrderedForm
        Result of the Sylvester equation for 2nd quantized operators.

    Notes
    -----
    See the second quantization documentation for the derivation.

    """
    return _solve_scalar_with_denominator(
        Y,
        _make_sylvester_denominator_getter(H_ii, H_jj),
        diagonal=diagonal,
    )


def _solve_sylvester_2nd_quant_impl(
    eigs: tuple[tuple[sympy.Expr, ...], ...],
    scalar_solver: Callable[
        [sympy.Expr, Callable[[tuple, tuple[int, ...]], sympy.Expr], bool],
        NumberOrderedForm,
    ],
) -> Callable:
    """Implement the second-quantized Sylvester solver."""
    eigs = tuple(
        [NumberOrderedForm.from_expr(eig) for eig in eig_block] for eig_block in eigs
    )
    if any(not eig.is_particle_conserving() for eig_block in eigs for eig in eig_block):
        raise ValueError(
            "The diagonal Hamiltonian blocks must contain only number-conserving expressions."
        )

    denominator_getters: dict[tuple[int, int, int, int], Callable] = {}

    def solve_sylvester(
        Y: sympy.MatrixBase,
        index: tuple[int, ...],
    ) -> sympy.MatrixBase:
        if Y is zero:
            return zero
        eigs_A, eigs_B = eigs[index[0]], eigs[index[1]]
        # Handle the case when a block is empty
        if not eigs_A:
            eigs_A = eigs[index[0]] = [sympy.S.Zero] * Y.shape[0]
        if not eigs_B:
            eigs_B = eigs[index[1]] = [sympy.S.Zero] * Y.shape[1]
        result = sympy.zeros(*Y.shape)
        for i in range(Y.rows):
            for j in range(Y.cols):
                # Only compute upper triangle of diagonal blocks
                if index[0] != index[1] or i >= j:
                    key = (index[0], index[1], i, j)
                    if key not in denominator_getters:
                        denominator_getters[key] = _make_sylvester_denominator_getter(
                            eigs_A[i], eigs_B[j]
                        )
                    result[i, j] = scalar_solver(
                        Y[i, j],
                        denominator_getters[key],
                        i == j and index[0] == index[1],
                    )
        for i in range(Y.rows):
            for j in range(Y.cols):
                # Fill the lower triangle with minus conjugate transpose
                if index[0] == index[1] and i < j:
                    result[i, j] = -result[j, i].adjoint()

        return result

    return solve_sylvester


def _solve_sylvester_2nd_quant_compact(
    eigs: tuple[tuple[sympy.Expr, ...], ...],
) -> Callable:
    """Compact internal Sylvester solver used by block diagonalization."""
    return _solve_sylvester_2nd_quant_impl(
        eigs,
        _solve_scalar_with_compact_denominator,
    )


def solve_sylvester_2nd_quant(
    eigs: tuple[tuple[sympy.Expr, ...], ...],
) -> Callable:
    """Solve a Sylvester equation for 2nd quantized diagonal Hamiltonians.

    Parameters
    ----------
    eigs :
        Tuple of lists of expressions representing the diagonal Hamiltonian blocks.

    Returns
    -------
    Callable
        A function that takes a matrix of operators and a tuple of indices, and
        computes the element-wise solution to the Sylvester equation for those
        diagonal Hamiltonian blocks.

    """
    return _solve_sylvester_2nd_quant_impl(
        eigs,
        _solve_scalar_with_denominator,
    )


def apply_mask_to_operator(
    operator: sympy.MatrixBase,
    mask: np.ndarray,
    keep: bool = True,
) -> sympy.Matrix:
    """Apply a mask to filter specific terms in a matrix operator.

    This function selectively keeps terms in a symbolic matrix operator based on
    their powers of creation and annihilation operators.

    See more details of how this works in the :doc:`second quantization
    documentation <../second_quantization>`.

    Parameters
    ----------
    operator :
        Matrix operator containing symbolic expressions with second quantized operators.
    mask :
        A matrix with `~pymablock.number_ordered_form.NumberOrderedForm` elements that
        define selection criteria. Specifically, the elements of the `operator[i, j]`
        with powers matching any `mask[i, j].terms` are selected.
    keep :
        If True (default), keep the terms that satisfy any of the conditions. If False
        discard the terms that satisfy any of the conditions. Used for inverting the
        mask.

    Returns
    -------
    filtered: `sympy.matrices.dense.MutableDenseMatrix`
        A new matrix with the same shape as the input, but containing only the
        selected terms.

    Examples
    --------
    Let's filter out terms in a Hamiltonian matrix based on their boson number operators:

    >>> import sympy
    >>> from sympy.physics.quantum import boson, Dagger
    >>> from pymablock.number_ordered_form import NumberOrderedForm
    >>> from pymablock.second_quantization import apply_mask_to_operator
    >>>
    >>> # Create bosonic operators
    >>> a = boson.BosonOp('a')
    >>> b = boson.BosonOp('b')
    >>>
    >>> # Create a matrix with different operator terms
    >>> H = sympy.Matrix([[a * Dagger(a) + b * Dagger(b), a * Dagger(b)],
            [b * Dagger(a), a * Dagger(a) - b * Dagger(b)]])
    >>> # Convert to NumberOrderedForm for easier handling
    >>> H_nof = H.applyfunc(NumberOrderedForm.from_expr)
    >>>
    >>> # Create a mask that selects only terms with a specific power pattern
    >>> # Select only terms with exactly one 'a' operator and one 'b' operator
    >>> mask = sympy.Matrix([[sympy.S.Zero, NumberOrderedForm([a, b], {(1, -1): sympy.S.One})],
                [NumberOrderedForm([a, b], {(1, 1): sympy.S.One}), sympy.S.Zero]])
    >>> H_filtered = apply_mask_to_operator(H_nof, mask, keep=True)
    >>> # H_filtered now contains only the terms that match the mask:
    >>> # [[0, Dagger(b)a], [0, 0]]

    """
    result = sympy.zeros(operator.rows, operator.cols)
    for i in range(operator.rows):
        for j in range(operator.cols):
            value = operator[i, j]
            if not value:
                continue
            if not mask[i, j]:
                if not keep:
                    result[i, j] = value
                continue
            value = _to_number_ordered_form(value)
            value, mask[i, j] = value._combine_operators(mask[i, j])
            assert isinstance(value, NumberOrderedForm)
            result[i, j] = value.filter_terms(tuple(mask[i, j].terms), keep)

    return result
