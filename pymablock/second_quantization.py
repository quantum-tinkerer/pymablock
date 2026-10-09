"""Second quantization tools for number-ordered operators."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import sympy
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOperator,
    NumberOrderedForm,
    _iter_fixed_number_indicators,
    _number_operator_to_placeholder,
)
from pymablock.operator_embedding import Embedding, _Lattice
from pymablock.series import zero

__all__ = [
    "Embedding",
    "apply_mask_to_operator",
    "solve_sylvester_2nd_quant",
    "solve_sylvester_embedding",
]


def _divide_by_energy_gap(
    coefficient: sympy.Expr,
    energy_gap: sympy.Expr,
    number_symbols: Sequence[sympy.Symbol] = (),
    amplitude: sympy.Expr | None = None,
    nonnegative_numbers: Sequence[sympy.Symbol] = (),
) -> sympy.Expr:
    """Return coefficient / energy_gap, choosing zero for inactive 0/0 channels.

    Ordinary cancellation can lose an inactive transition: n/n must be zero at
    n=0, not one everywhere. Occupations fixed by indicators are evaluated
    directly; otherwise the quotient is conditional on the transition being active.
    ``amplitude`` (default: ``coefficient``) also contains ladder factors and fermion
    signs and only decides inactivity. ``number_symbols`` are occupation coordinates
    and other symbols are generic parameters; ``nonnegative_numbers`` only avoids
    unnecessary conditionals. A literal zero gap with a nonzero amplitude raises
    ValueError; other resonances remain symbolic poles and are not searched for.
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
            number_symbols,
            selected_amplitude,
            nonnegative_numbers,
        )

    variables = set(number_symbols)
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
    # Generic scalar parameters are not tuned to occupation resonances. Keep
    # their quotients compact rather than expanding conditional expressions.
    gap = energy_gap.as_numer_denom()[0]
    for parameter in gap.free_symbols - variables:
        parameter_coefficient = gap.coeff(parameter)
        if (
            parameter_coefficient.is_Atom
            and parameter_coefficient.is_zero is False
            and not parameter_coefficient.free_symbols & variables
        ):
            return coefficient / energy_gap
    quotient = coefficient / energy_gap
    if not energy_gap.free_symbols & variables:
        return quotient
    # A constant plus occupations with the same sign cannot vanish. Inspect
    # numeric coefficients only, rather than asking the assumptions engine to
    # prove a general expression nonzero.
    constant, rest = energy_gap.as_coeff_Add()
    if constant and all(
        factor in set(nonnegative_numbers)
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


def _diagonal_coefficient(expression: NumberOrderedForm) -> sympy.Expr:
    """Extract a diagonal NOF's scalar number coefficient.

    Reject ladder shifts: the zero-shift coefficient is the only allowed term.
    """
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
        side raises ``ValueError``. Explicit occupation selections can expose
        such a gap, but binary sectors and integer roots are not searched.
        Other resonances remain symbolic poles; the result is valid only away
        from them.

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


def _divide_transitions(
    lattice: _Lattice,
    value: NumberOrderedForm,
    outgoing_energy: sympy.Expr,
    incoming_energy: sympy.Expr,
) -> NumberOrderedForm:
    """Divide a target NOF's transitions from this lattice by their energy gaps.

    Outgoing energy uses target number placeholders; incoming energy uses
    symbolic source occupations. Return a bare target NOF, preserving inactive
    zero-gap transitions as zero and unresolved gaps as symbolic denominators.
    """
    coordinates = lattice.source_occupations
    nonnegative = tuple(q for q in coordinates if q.is_nonnegative)

    def divide(coefficient, output, matrix_element):
        denominator = sympy.expand(
            lattice.evaluate_numbers(outgoing_energy, output) - incoming_energy
        )
        # Binary source identities can make a nominally resonant transition inactive.
        if denominator == 0 and coordinates and lattice.source_is_zero(matrix_element):
            return sympy.S.Zero
        try:
            return _divide_by_energy_gap(
                coefficient, denominator, coordinates, matrix_element, nonnegative
            )
        except ValueError as error:
            raise ZeroDivisionError(str(error)) from error

    return lattice.map_transition_coefficients(value, divide)


def solve_sylvester_embedding(h0: sympy.MatrixBase, embedding: Embedding) -> Callable:
    """Construct a Sylvester solver for retained and complement embedding blocks.

    Parameters
    ----------
    h0 :
        Square target Hamiltonian with NumberOrderedForm entries in the embedding's
        target mode order, diagonal in both matrix indices and occupation numbers.
    embedding :
        Isometry defining the retained states and their source representation.

    Returns
    -------
    Callable
        A function taking a matrix of operators and block indices and solving
        ``H_ii * X - X * H_jj = Y``. It preserves the series zero sentinel and uses
        anti-Hermitian symmetry for Hermitian perturbative problems.

    Notes
    -----
    Intra-block solves use ``solve_sylvester_2nd_quant``. Cross-block solves evaluate
    outgoing and incoming energies on the reference states or symbolic source
    occupations, then divide each transition. Non-diagonal H0 raises ValueError.
    Exposed resonant cross-block transitions raise ZeroDivisionError; inactive
    transitions contribute zero and unresolved resonances remain symbolic poles.

    """
    lattice = embedding._first_lattice
    modes = lattice.target_operators
    if any(i != j or any(any(p) for p in x.terms) for (i, j), x in h0.todok().items()):
        raise ValueError("Structured embeddings currently require diagonal H0")
    vacuum = (0,) * len(modes)
    energies = [
        x.terms.get(vacuum, sympy.S.Zero) if x != 0 else sympy.S.Zero
        for x in h0.diagonal()
    ]
    incoming_energies = [
        lattice.evaluate_numbers(energies[row], state)
        for row, state in embedding._column_states
    ]

    # Within each diagonal block the operators already use source or target
    # coordinates. Only rectangular blocks require embedding-aware division.
    w = embedding._retained_frame(h0.rows)
    retained_energies = (w.adjoint() * h0 * w).diagonal()
    diagonal_solver = solve_sylvester_2nd_quant([retained_energies, h0.diagonal()])

    def solve_sylvester(
        value: sympy.MatrixBase, index: tuple[int, ...]
    ) -> sympy.MatrixBase:
        """Dispatch by block; use anti-Hermitian symmetry for the reverse cross block."""
        if value is zero:
            return zero
        if index[0] == index[1]:
            return diagonal_solver(value, index)
        reverse = index[:2] == (0, 1)
        block = value.adjoint() if reverse else value
        entries = {}
        for (i, j), entry in block.todok().items():
            if entry.is_zero:
                continue
            divided = _divide_transitions(
                lattice, entry.target, energies[i], incoming_energies[j]
            )
            entries[i, j] = entry._rebuild(divided.args[1], operators=divided.operators)
        result = sympy.ImmutableSparseMatrix(*block.shape, entries)
        return -result.adjoint() if reverse else result

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
