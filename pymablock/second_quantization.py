"""Second quantization tools for bosonic and fermionic operators.

See number_ordered_form_plan.md for the plan to implement NumberOrderedForm as an
Operator subclass for better representation of number-ordered expressions.
"""

import itertools
from typing import Callable

import sympy
from packaging.version import parse
from sympy.physics.quantum import Operator, boson
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm
from pymablock.series import zero

# Monkey patch sympy to propagate adjoint to matrix elements.
if parse(sympy.__version__) < parse("1.14.0"):

    def _eval_adjoint(self):
        return self.transpose().applyfunc(lambda x: x.adjoint())

    def _eval_transpose(self):
        from sympy.functions.elementary.complexes import conjugate

        if self.is_commutative:
            return self
        if self.is_hermitian:
            return conjugate(self)
        if self.is_antihermitian:
            return -conjugate(self)
        return None

    sympy.MatrixBase.adjoint = _eval_adjoint
    sympy.Expr._eval_transpose = _eval_transpose

    # Monkey patch sympy to override the sum method to ExpressionRawDomain.
    def sum(self, items):  # noqa ARG001
        """Slower, but overridable version of sympy.Add."""
        if not items:
            return sympy.S.Zero
        result = items[0]
        for item in items[1:]:
            result += item
        return result

    sympy.polys.domains.expressionrawdomain.ExpressionRawDomain.sum = sum

    # Only implements skipping identity, and is deleted in 1.14.
    del BosonOp.__mul__
    del Operator.__mul__


def solve_monomial(
    Y: sympy.Expr,
    H_ii: sympy.Expr,
    H_jj: sympy.Expr,
    boson_operators: list[boson.BosonOp],
) -> NumberOrderedForm:
    """Solve a Sylvester equation for bosonic monomial.

    Given `Y`, `H_ii`, and `H_jj`, return `-(E_i_shifted - E_j_shifted) * Y_monomial`, per
    monomial of creation and annihilation operators in Y.

    `H_ii` and `H_jj` are scalar expressions containing number operators of
    possibly several bosons.

    See more details of how this works in the second quantization documentation.

    Parameters
    ----------
    Y :
        Expression with raising and lowering bosonic operators.
    H_ii :
        Sectors of the unperturbed Hamiltonian.
    H_jj :
        Sectors of the unperturbed Hamiltonian.
    boson_operators :
        List with all possible bosonic operators in the inputs.

    Returns
    -------
    Result of the Sylvester equation for bosonic operators.

    """
    # Plan:
    # 1. Find all the boson operators in Y, H_ii, and H_jj.
    # 2. Manipulate H_ii and H_jj so that are explicit functions of number
    #    operators.
    # 3. Separate the terms of Y into monomials.
    # 4. For each monomial, shift the numbers in H_jj
    # 5. Multiply by the corresponding monomial in Y.
    if Y == 0:
        return sympy.S.Zero

    Y = NumberOrderedForm.from_expr(Y)

    shifts = Y.terms
    new_shifts = {}
    for shift, coeff in shifts.items():
        # Commute H_ii and H_jj through creation and annihilation operators
        # respectively.
        shifted_H_jj = H_jj.subs(
            {
                NumberOperator(op): NumberOperator(op) + delta
                for delta, op in zip(shift, boson_operators)
                if delta > 0
            }
        )
        shifted_H_ii = H_ii.subs(
            {
                NumberOperator(op): NumberOperator(op) - delta
                for delta, op in zip(shift, boson_operators)
                if delta < 0
            }
        )
        new_shifts[shift] = (shifted_H_ii - shifted_H_jj) ** -1 * coeff

    return NumberOrderedForm(
        operators=Y.args[0],
        terms=new_shifts,
    )


def solve_sylvester_bosonic(
    eigs: tuple[tuple[sympy.Expr, ...], ...],
    boson_operators: list[boson.BosonOp],
) -> Callable:
    """Solve a Sylvester equation for bosonic diagonal Hamiltonians.

    Parameters
    ----------
    eigs :
        Tuple of lists of expressions representing the diagonal Hamiltonian blocks.
    boson_operators :
        List with all possible bosonic operators in the Hamiltonian.

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
    eigs = tuple(tuple(eig.as_expr() for eig in eig_block) for eig_block in eigs)

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
        return sympy.Matrix(
            [
                [
                    solve_monomial(Y[i, j], eigs_A[i], eigs_B[j], boson_operators)
                    for j in range(Y.cols)
                ]
                for i in range(Y.rows)
            ]
        )

    return solve_sylvester


def apply_mask_to_operator(
    operator: sympy.MatrixBase,
    mask: sympy.MatrixBase,
    keep: bool = True,
) -> sympy.MatrixBase:
    """Apply a mask to filter specific terms in a matrix operator.

    This function selectively keeps terms in a symbolic matrix operator based on
    their powers of creation and annihilation operators.

    See more details of how this works in the second quantization documentation.

    Parameters
    ----------
    operator :
        Matrix operator containing symbolic expressions with second quantized operators.
    mask :
        A matrix with `NumberOrderedForm` that define selection criteria. Specifically,
        the elements of the `operator[i, j]` with powers matching any `mask[i, j].terms`
        are selected.
    keep :
        If True (default), keep the terms that satisfy any of the conditions. If False
        discard the terms that satisfy any of the conditions. Used for inverting the
        mask.

    Returns
    -------
    sympy.MatrixBase
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
    >>> H_nof = sympy.Matrix([[NumberOrderedForm.from_expr(H[i, j]) for j in range(H.cols)]
                    for i in range(H.rows)])
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
            value, mask[i, j] = value._combine_operators(mask[i, j])
            assert isinstance(value, NumberOrderedForm)
            result[i, j] = value.filter_terms(list(mask[i, j].terms), keep)

    return result


def group_by_symbolic_denominators(expr: sympy.Expr) -> dict[sympy.Expr, sympy.Expr]:
    """Group terms in a sympy expression by their symbolic denominators.

    This function is designed to deal with nested fraction expressions, as generated by
    the Pymablock's perturbation theory calculations. It collects all terms with the
    same energy denominators.

    Parameters
    ----------
    expr : sympy.Expr
        The sympy expression to group.

    Returns
    -------
    dict[sympy.Expr, sympy.Expr]
        A dictionary mapping each symbolic denominator to the sum of its numerators.

    Examples
    --------
    >>> from sympy import symbols
    >>> a, b, c, d, e = symbols('a b c d e')
    >>> expr = (a/(2*b) + c/b + c/d) / e
    >>> group_by_symbolic_denominators(expr)
    {b*e: a/2 + c, d*e: c}

    """
    from sympy import Add, Mul

    # sympy .as_numer_denom also gathers multiple terms together; we want to
    # keep them separate.
    def split_numerator_denominator(expr):
        numerator = []
        denom = []
        for factor in expr.as_ordered_factors():
            if factor.is_number:
                numerator.append(factor)
            elif factor.is_Pow and factor.args[1] < 0:
                base, exp = factor.as_base_exp()
                denom.append(sympy.simplify(base) ** exp)
            else:
                numerator.append(factor)

        return Mul(*numerator), Mul(*denom)

    def one_level_expand(expr):
        """Expand only the top level of a product.

        Only expand the factors of a product, rather than their contents.
        """
        all_terms = [factor.as_ordered_terms() for factor in expr.as_ordered_factors()]
        # Distribute multiplication across all term combinations
        expanded = []
        for factors in itertools.product(*all_terms):
            expanded.append(Mul(*factors))
        return Add(*expanded)

    result: dict[sympy.Expr, sympy.Expr] = {}

    result = {}
    numerator, denom = split_numerator_denominator(expr)
    intermediate = {denom: one_level_expand(numerator)}

    while intermediate:
        denom, numerator = intermediate.popitem()
        if not isinstance(numerator, sympy.Add):
            # Single term, can't process further
            result[denom] = result.get(denom, 0) + numerator
            continue

        for term in numerator.as_ordered_terms():
            new_numerator, new_denom = split_numerator_denominator(term)
            new_denom = new_denom * denom
            new_numerator = one_level_expand(new_numerator)
            if not isinstance(new_numerator, sympy.Add):
                # Single term, can't process further
                result[new_denom] = result.get(new_denom, 0) + new_numerator
                continue

            intermediate[new_denom] = intermediate.get(new_denom, 0) + new_numerator

    return sympy.Add(*(numerator * denom for denom, numerator in result.items()))
