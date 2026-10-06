"""Second quantization tools for number-ordered operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sympy
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOperator,
    NumberOrderedForm,
    _iter_fixed_number_indicators,
    _NOFTransition,
    _number_operator_to_placeholder,
)
from pymablock.operator_embedding import Embedding
from pymablock.series import zero

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any

    from pymablock.operator_embedding import _EmbeddingBlocks

__all__ = [
    "Embedding",
    "apply_mask_to_operator",
    "solve_sylvester_2nd_quant",
]


def _divide_by_energy_gap(
    coefficient: sympy.Expr,
    energy_gap: sympy.Expr,
    binary_numbers: Sequence[sympy.Symbol] = (),
    number_symbols: Sequence[sympy.Symbol] = (),
    amplitude: sympy.Expr | None = None,
    nonnegative_numbers: Sequence[sympy.Symbol] = (),
) -> sympy.Expr:
    """Divide a term's coefficient by its gap, choosing zero on inactive channels.

    ``number_symbols`` are scalar occupation coordinates; other free symbols are
    generic parameters. ``binary_numbers`` range over 0 and 1, while
    ``nonnegative_numbers`` exclude negative occupations (unlike bilateral ladders).

    ``amplitude`` determines where the transition is inactive. It defaults to the
    coefficient; embedding solves also include ladder amplitudes and fermionic
    signs. The returned coefficient is not multiplied by those factors again.

    Resolve immediately exposed zero gaps and fixed-number indicators locally.
    A nonzero amplitude at an exposed zero gap raises ValueError. Unresolved gaps
    remain symbolic poles; this does not certify nonresonance in every sector.
    The result is zero where the amplitude vanishes, including at a zero gap.
    """
    amplitude = coefficient if amplitude is None else amplitude
    if amplitude == 0:
        return sympy.S.Zero
    energy_gap = sympy.cancel(energy_gap)
    arguments = coefficient, energy_gap, amplitude

    def divide_after_substitution(
        substitution: Mapping[sympy.Expr, sympy.Expr],
    ) -> sympy.Expr:
        selected_coefficient, selected_gap, selected_amplitude = (
            expression.xreplace(substitution) for expression in arguments
        )
        return _divide_by_energy_gap(
            selected_coefficient,
            selected_gap,
            binary_numbers,
            number_symbols,
            selected_amplitude,
            nonnegative_numbers,
        )

    variables = set(number_symbols) | set(binary_numbers)
    gap = energy_gap.as_numer_denom()[0]
    for parameter in gap.free_symbols - variables:
        parameter_coefficient = gap.coeff(parameter)
        if (
            parameter_coefficient.is_Atom
            and parameter_coefficient.is_zero is False
            and not parameter_coefficient.free_symbols & variables
        ):
            return coefficient / energy_gap
    for n in binary_numbers:
        if n not in amplitude.free_symbols | energy_gap.free_symbols:
            continue
        if energy_gap == 0 or any(
            x.xreplace({n: v}) == 0
            for x in (amplitude, energy_gap)
            for v in (sympy.S.Zero, sympy.S.One)
        ):
            return (1 - n) * divide_after_substitution(
                {n: sympy.S.Zero}
            ) + n * divide_after_substitution({n: sympy.S.One})
    point = next(
        _iter_fixed_number_indicators(coefficient, variables & energy_gap.free_symbols),
        None,
    )
    if point is not None:
        delta, n, v = point
        return sympy.Piecewise(
            (divide_after_substitution({n: v}), sympy.Eq(n, v)),
            (divide_after_substitution({delta: sympy.S.Zero}), True),
        )
    if energy_gap == 0:
        raise ValueError(
            "Cannot solve the Sylvester equation: the right-hand side is nonzero "
            "but the energy difference is zero (degenerate channel)."
        )
    quotient = coefficient / energy_gap
    if not energy_gap.free_symbols & variables:
        return quotient
    # A constant plus occupations with the same sign cannot vanish. Inspect
    # numeric coefficients only, rather than asking the assumptions engine to
    # prove a general expression nonzero.
    constant, rest = energy_gap.as_coeff_Add()
    if constant and all(
        factor in set(nonnegative_numbers) | set(binary_numbers)
        and (coefficient * constant).is_positive is True
        for coefficient, factor in (
            term.as_coeff_Mul() for term in sympy.Add.make_args(rest)
        )
    ):
        return quotient
    factors = (factor.as_numer_denom()[0] for factor in sympy.Mul.make_args(amplitude))
    inactive = sympy.Or(
        *(
            sympy.Eq(factor, 0, evaluate=False)
            for factor in factors
            if factor.free_symbols & variables
        )
    )
    # A meromorphic amplitude has no defined zero at its poles. Its numerator
    # can vanish there without making the virtual channel inactive.
    amplitude_denominator = sympy.together(amplitude).as_numer_denom()[1]
    inactive = sympy.And(inactive, sympy.Ne(amplitude_denominator, 0))
    return sympy.Piecewise((0, inactive), (quotient, True), evaluate=False)


def _make_embedding_sylvester_solver(
    blocks: _EmbeddingBlocks, h0: sympy.MatrixBase
) -> Callable[[Any, tuple[int, ...]], Any]:
    """Validate H0 and return a solver for the embedding's Sylvester equations.

    H0 must be a source matrix of NOF entries, diagonal in both matrix indices and
    occupation numbers. Intra-block solves use the ordinary second-quantized solver.
    Rectangular solves evaluate outgoing and incoming energies on the embedding's
    reference states or symbolic target occupations, then divide each transition.
    The callback preserves the series zero sentinel before accessing matrix entries.
    """
    basis, finite = blocks.basis, blocks.finite
    modes, numbers = basis.operators, basis._source_placeholders
    if any(i != j or any(any(p) for p in x.terms) for (i, j), x in h0.todok().items()):
        raise ValueError("Structured embeddings currently require diagonal H0")
    vacuum = (0,) * len(modes)
    energies = [
        x.terms.get(vacuum, sympy.S.Zero) if x != 0 else sympy.S.Zero
        for x in h0.diagonal()
    ]
    if finite:
        occupations, coordinates, binary, coordinate_map, nonnegative = (
            vacuum,
            (),
            (),
            {},
            (),
        )
        incoming_energies = [
            basis._at_occupations(energies[row], state)
            for row, state in basis._references
        ]
    else:
        occupations, coordinates = basis.source_occupations, basis.coordinate_symbols
        binary = tuple(
            q for q, size in zip(coordinates, basis._target_dimensions) if size == 2
        )
        nonnegative = tuple(
            q
            for q, op in zip(coordinates, basis._target_operators)
            if not isinstance(op, LadderOp)
        )
        target_numbers = basis._occupation_left_inverse * (
            sympy.Matrix(numbers) - sympy.Matrix(basis.reference)
        )
        coordinate_map = dict(zip(coordinates, target_numbers))
        incoming_energies = [basis._at_occupations(energies[0], occupations)]

    # Within each diagonal block the operators already use target or source
    # coordinates. Only rectangular blocks require embedding-aware division.
    retained_energies = incoming_energies if finite else [basis._compress(h0[0, 0])]
    diagonal_solver = solve_sylvester_2nd_quant([retained_energies, h0.diagonal()])

    def divide_transition_entry(
        value: NumberOrderedForm | sympy.Expr, row: int, col: int
    ) -> NumberOrderedForm | sympy.Expr:
        """Solve one retained-to-source matrix entry using its actual transition gap."""
        if value == 0 or value.is_zero:
            return sympy.S.Zero
        value = basis._convert_operator(value.source)
        terms = {}
        for transition in _NOFTransition.from_form(value):
            powers = transition.powers
            action = transition.symbolic_action(occupations)
            if action.weight == 0:
                continue
            middle = [n - max(p, 0) for n, p in zip(occupations, powers)]
            coefficient = basis._at_occupations(transition.coefficient, middle)
            denominator = sympy.expand(
                basis._at_occupations(energies[row], action.output_state)
                - incoming_energies[col]
            )
            try:
                result = _divide_by_energy_gap(
                    coefficient,
                    denominator,
                    binary,
                    coordinates,
                    action.weight,
                    nonnegative,
                )
            except ValueError as error:
                raise ZeroDivisionError(str(error)) from error
            incoming = {n: n + max(p, 0) for n, p in zip(numbers, powers)}
            terms[powers] = result.xreplace(
                {
                    q: expression.xreplace(incoming)
                    for q, expression in coordinate_map.items()
                }
            )
        return NumberOrderedForm(
            basis.operators, terms, blocks.entry_embedding, 1, validate=False
        )

    def solve(value: Any, index: tuple[int, ...]) -> Any:
        """Dispatch by block; use anti-Hermitian symmetry for the reverse cross block."""
        if value is zero:
            return zero
        if index[0] == index[1]:
            return diagonal_solver(value, index)
        reverse = index[:2] == (0, 1)
        source = value.adjoint() if reverse else value
        result = sympy.ImmutableMatrix(
            source.rows,
            source.cols,
            lambda i, j: divide_transition_entry(source[i, j], i, j),
        )
        return -result.adjoint() if reverse else result

    return solve


def _diagonal_coefficient(expression: NumberOrderedForm | sympy.Expr) -> sympy.Expr:
    """Extract a diagonal NOF's scalar number coefficient, or pass a scalar through.

    Reject ladder shifts: the zero-shift coefficient is the only allowed term.
    """
    if not isinstance(expression, NumberOrderedForm):
        return sympy.sympify(expression)
    if not expression.is_particle_conserving():
        raise ValueError(
            "Diagonal second-quantized Hamiltonians must contain only number operators."
        )
    return next(iter(expression.terms.values()), sympy.S.Zero)


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
    H_ii, H_jj = (NumberOrderedForm.from_expr(h) for h in (H_ii, H_jj))
    for diagonal_operator in (H_ii, H_jj):
        Y, _ = Y._combine_operators(diagonal_operator)
    operators = Y.operators
    H_ii = _diagonal_coefficient(H_ii._expand_operators(operators))
    H_jj = _diagonal_coefficient(H_jj._expand_operators(operators))
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
        new_shifts[shift] = _divide_by_energy_gap(
            sign * coeff.xreplace(fixed),
            denominator.xreplace(fixed),
            tuple(number for number in binary_numbers if number not in fixed),
            tuple(Y._number_operator_placeholders),
            nonnegative_numbers=tuple(
                n
                for op, n in zip(Y.operators, Y._number_operator_placeholders)
                if not isinstance(op, LadderOp)
            ),
        )

    result = NumberOrderedForm(
        operators=Y.args[0],
        terms=new_shifts,
    )._cancel_binary_operator_numbers()

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
        elements. An exposed zero energy difference for a nonzero right-hand
        side raises ``ValueError``. Other occupation-dependent resonances may
        remain as symbolic poles; the result is valid only away from them.

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
    ...                   [b * Dagger(a), a * Dagger(a) - b * Dagger(b)]])
    >>> # Convert to NumberOrderedForm for easier handling
    >>> H_nof = H.applyfunc(NumberOrderedForm.from_expr)
    >>>
    >>> # Create a mask that selects only terms with a specific power pattern
    >>> # Select only terms with exactly one 'a' operator and one 'b' operator
    >>> mask = sympy.Matrix([[sympy.S.Zero, NumberOrderedForm([a, b], {(1, -1): sympy.S.One})],
    ...                      [NumberOrderedForm([a, b], {(1, 1): sympy.S.One}), sympy.S.Zero]])
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
