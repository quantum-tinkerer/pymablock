"""Second quantization tools.

See number_ordered_form_plan.md for the plan to implement NumberOrderedForm as an
Operator subclass for better representation of number-ordered expressions.
"""

from collections.abc import Callable

import numpy as np
import sympy
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOperator,
    NumberOrderedForm,
    _number_operator_to_placeholder,
)
from pymablock.series import zero

__all__ = [
    "apply_mask_to_operator",
    "solve_sylvester_2nd_quant",
]


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
        Must be Hermitian.
    H_ii :
        Sectors of the unperturbed Hamiltonian.
    H_jj :
        Sectors of the unperturbed Hamiltonian.
    diagonal : bool
        If True, we're evaluating the diagonal entry, which means that `Y` is Hermitian
        and `H_ii` and `H_jj` are equal. This is used to speed up the computation.

    Returns
    -------
    NumberOrderedForm
        Result of the Sylvester equation for 2nd quantized operators.

    Notes
    -----
    See the second quantization documentation for the derivation.

    """
    if Y == 0:
        return sympy.S.Zero

    Y = NumberOrderedForm.from_expr(Y)
    operators = Y.operators

    shifts = Y.terms
    new_shifts = {}
    for shift, coeff in shifts.items():
        # Ensure that the energy denominator always has the same sign to simplify the
        # expression. We do this by multiplying the denominator by -1 if the shift is
        # lexicographically negative.
        sign = -sympy.S.One if tuple(shift) < (0,) * len(shift) else sympy.S.One
        if diagonal and sign is sympy.S.One:
            continue
        # Commute H_ii and H_jj through creation and annihilation operators
        # respectively.
        #
        # Here we use a private function to get access to the placeholders
        shifted_H_jj = H_jj.xreplace(
            {
                _number_operator_to_placeholder(NumberOperator(op)): (
                    _number_operator_to_placeholder(NumberOperator(op)) + delta
                    if isinstance(op, (BosonOp, LadderOp))
                    else sympy.S.One
                )
                for delta, op in zip(shift, operators)
                if delta > 0
            }
        )
        shifted_H_ii = H_ii.xreplace(
            {
                _number_operator_to_placeholder(NumberOperator(op)): (
                    _number_operator_to_placeholder(NumberOperator(op)) - delta
                    if isinstance(op, (BosonOp, LadderOp))
                    else sympy.S.One
                )
                for delta, op in zip(shift, operators)
                if delta < 0
            }
        )
        if sign is sympy.S.One:
            denominator = shifted_H_ii - shifted_H_jj
        else:
            denominator = shifted_H_jj - shifted_H_ii
        # Denominators often simplify because linear powers of bosonic operators cancel.
        denominator = sympy.collect_const(
            next(iter((denominator.terms.values()))).simplify()
        ).doit()  # Not sure why doit is needed here, but it is.
        new_shifts[shift] = sign * (denominator) ** -sympy.S.One * coeff

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

    return result


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
    eigs = tuple(
        [NumberOrderedForm.from_expr(eig) for eig in eig_block] for eig_block in eigs
    )
    if any(not eig.is_particle_conserving() for eig_block in eigs for eig in eig_block):
        raise ValueError(
            "The diagonal Hamiltonian blocks must contain only number-conserving expressions."
        )

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
                    result[i, j] = solve_scalar(
                        Y[i, j],
                        eigs_A[i],
                        eigs_B[j],
                        diagonal=(i == j and index[0] == index[1]),
                    )
        for i in range(Y.rows):
            for j in range(Y.cols):
                # Fill the lower triangle with minus conjugate transpose
                if index[0] == index[1] and i < j:
                    result[i, j] = -result[j, i].adjoint()

        return result

    return solve_sylvester


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
            value = NumberOrderedForm.from_expr(value)
            value, mask[i, j] = value._combine_operators(mask[i, j])
            assert isinstance(value, NumberOrderedForm)
            result[i, j] = value.filter_terms(tuple(mask[i, j].terms), keep)

    return result
