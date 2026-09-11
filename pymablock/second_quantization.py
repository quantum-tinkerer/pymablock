"""Second quantization tools for number-ordered operators."""

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


def _diagonal_coefficient(expression: NumberOrderedForm | sympy.Expr) -> sympy.Expr:
    """Return the coefficient of an expression containing only number operators."""
    if not isinstance(expression, NumberOrderedForm):
        return sympy.sympify(expression)
    if not expression.is_particle_conserving():
        raise ValueError(
            "Diagonal second-quantized Hamiltonians must contain only number operators."
        )
    return next(iter(expression.terms.values()), sympy.S.Zero)


def _divide_binary_sectors(
    numerator: sympy.Expr,
    denominator: sympy.Expr,
    variables: tuple[sympy.Symbol, ...],
) -> sympy.Expr:
    """Solve denominator * x = numerator at each combination of occupations 0 and 1.

    Choose x = 0 when the numerator is zero, even if the denominator is zero.
    Raise ValueError when the denominator is zero but the numerator is not.
    Return an expression in the occupation variables that combines the results.
    """
    if numerator.is_zero:
        return sympy.S.Zero
    for index, variable in enumerate(variables):
        if variable not in numerator.free_symbols | denominator.free_symbols:
            continue
        samples = [
            _divide_binary_sectors(
                numerator.xreplace({variable: value}),
                denominator.xreplace({variable: value}),
                variables[index + 1 :],
            )
            for value in (sympy.S.Zero, sympy.S.One)
        ]
        return samples[0] + variable * (samples[1] - samples[0])
    if denominator.is_zero:
        raise ValueError(
            "Cannot solve the Sylvester equation: the right-hand side is nonzero "
            "but the energy difference is zero."
        )
    return numerator / denominator


def solve_scalar(
    Y: sympy.Expr,
    H_ii: sympy.Expr,
    H_jj: sympy.Expr,
    diagonal: bool = False,
) -> NumberOrderedForm:
    """Solve ``H_ii * X - X * H_jj = Y`` for the operator ``X``.

    `H_ii` and `H_jj` are scalar expressions containing number operators of
    possibly several bosons, fermions, and spin operators.

    See more details of how this works in the :doc:`second quantization
    documentation <../second_quantization>`.

    Parameters
    ----------
    Y :
        Right-hand side, expressed using boson, fermion, and spin operators.
    H_ii :
        Diagonal block of the unperturbed Hamiltonian.
    H_jj :
        Diagonal block of the unperturbed Hamiltonian.
    diagonal : bool
        If True, we're evaluating the diagonal entry of the matrix operator,
        which means that `Y` is Hermitian and `H_ii` and `H_jj` are equal. This
        is used to speed up the computation.

    Returns
    -------
    NumberOrderedForm
        The operator ``X`` satisfying the equation.

    Notes
    -----
    For fermion and spin occupation states, set each solution matrix element
    to zero when the corresponding right-hand side is zero, even if the energy
    difference is also zero. For example, solving ``N * X = N`` gives ``X = N``:
    at occupation 1, X must be 1; at occupation 0, we choose X = 0.
    Raise ``ValueError`` if the energy difference is identically zero but the
    right-hand side is nonzero.

    See the second quantization documentation for the derivation.

    """
    if Y == 0:
        return sympy.S.Zero

    Y = NumberOrderedForm.from_expr(Y)
    for diagonal_operator in (H_ii, H_jj):
        if isinstance(diagonal_operator, NumberOrderedForm):
            Y, _ = Y._combine_operators(diagonal_operator)
    operators = Y.operators
    H_ii = _diagonal_coefficient(H_ii)
    H_jj = _diagonal_coefficient(H_jj)
    binary_numbers = Y._number_operator_placeholders[Y._n_inf_order :]

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
        denominator = sympy.collect_const(denominator.simplify()).doit()
        # For fermions and spins, a† N = N a = 0. Set N = 0 in the
        # coefficient when the term contains a† or a for that mode.
        fixed = {
            number: sympy.S.Zero
            for number, power in zip(binary_numbers, shift[Y._n_inf_order :])
            if power
        }
        new_shifts[shift] = _divide_binary_sectors(
            sign * coeff.xreplace(fixed),
            denominator.xreplace(fixed),
            tuple(number for number in binary_numbers if number not in fixed),
        )

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
    *,
    hermitian: bool = True,
) -> Callable:
    """Solve a Sylvester equation for 2nd quantized diagonal Hamiltonians.

    Parameters
    ----------
    eigs :
        Tuple of lists of expressions representing the diagonal Hamiltonian blocks.
    hermitian :
        Whether to use anti-Hermitian symmetry of the solution within diagonal
        blocks. Set to False for general non-Hermitian sources.

    Returns
    -------
    Callable
        A function that takes a matrix of operators and a tuple of indices, and
        computes the element-wise solution to the Sylvester equation for those
        diagonal Hamiltonian blocks. Each entry is computed by ``solve_scalar``,
        including its choice of zero for undetermined fermion and spin matrix
        elements and its ``ValueError`` when the equation has no solution.

    """
    eigs = [
        [NumberOrderedForm.from_expr(eig) for eig in eig_block]
        if np.ndim(eig_block)
        else []
        for eig_block in eigs
    ]
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
                # Hermitian problems only need the lower triangle of diagonal blocks
                if not hermitian or index[0] != index[1] or i >= j:
                    result[i, j] = solve_scalar(
                        Y[i, j],
                        eigs_A[i],
                        eigs_B[j],
                        diagonal=(hermitian and i == j and index[0] == index[1]),
                    )
        for i in range(Y.rows):
            for j in range(Y.cols):
                # Fill the upper triangle with minus conjugate transpose
                if hermitian and index[0] == index[1] and i < j:
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
