"""Second quantization tools for bosonic and fermionic operators.

See number_ordered_form_plan.md for the plan to implement NumberOrderedForm as an
Operator subclass for better representation of number-ordered expressions.
"""

from collections import defaultdict
from typing import Callable

import numpy as np
import sympy
import sympy.physics
from packaging.version import parse
from sympy.physics.quantum import Operator, boson, fermion
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm
from pymablock.series import zero

# Type aliases
Mask = tuple[
    list[sympy.physics.quantum.Operator],
    list[tuple[tuple[list[int], int | None], np.ndarray]],
]

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

    # Only implements skipping identity, and is deleted in 1.14.
    del BosonOp.__mul__
    del Operator.__mul__


def find_operators(expr: sympy.Expr):
    """Find all the boson and fermionic operators in an expression.

    Parameters
    ----------
    expr :
        Sympy expression with bosonic and fermionic operators.

    Returns
    -------
    List with all annihilation bosonic and fermionic operators in an expression.

    """
    # Replace all number operators with their evaluated form.
    expr = expr.subs({i: i.doit() for i in expr.atoms(NumberOperator)})
    return list(
        set(type(arg)(arg.name) for arg in expr.atoms(boson.BosonOp, fermion.FermionOp))
    )


def group_ordered(expr):
    """Group the terms in a number-ordered expression by powers of unmatched operators.

    Parameters
    ----------
    expr :
        Sympy expression with bosonic operators.

    Returns
    -------
    dict[tuple[sympy.core.expr.Expr, sympy.core.expr.Expr], sympy.core.expr.Expr]
        Dictionary with keys as tuples of monomials of creation and annihilation operators,
        and values as the terms with number operators.

    """
    result = defaultdict(lambda: sympy.S.Zero)
    for term in expr.as_ordered_terms():
        creation_powers = sympy.Mul(
            *(
                factor
                for factor in term.as_ordered_factors()
                if (
                    isinstance((factor.as_base_exp())[0], BosonOp)
                    and not (factor.as_base_exp())[0].is_annihilation
                )
            )
        )
        annihilation_powers = sympy.Mul(
            *(
                factor
                for factor in term.as_ordered_factors()
                if (
                    isinstance((factor.as_base_exp())[0], BosonOp)
                    and (factor.as_base_exp())[0].is_annihilation
                )
            )
        )
        result[(creation_powers, annihilation_powers)] += sympy.Mul(
            *(factor for factor in term.as_ordered_factors() if not factor.atoms(BosonOp))
        )
    return result


def simplify_number_expression(expr: sympy.Expr) -> sympy.Expr:
    """Simplify a second-quantized expression with only number operators.

    Parameters
    ----------
    expr :
        Sympy expression only containing number operators.

    Returns
    -------
    sympy.core.expr.Expr
        Simplified expression with number operators.

    Raises
    ------
    ValueError
        If the expression contains operators other than number operators.

    """
    if expr.atoms(boson.BosonOp):
        raise ValueError("Expression contains bosonic operators.")
    substitutions = {
        n: sympy.Symbol(f"dummy_{n.name}_{n.args[1]}", real=True)
        for n in expr.atoms(NumberOperator)
    }
    inverse = {value: key for key, value in substitutions.items()}
    return sympy.simplify(expr.subs(substitutions)).subs(inverse)


def solve_monomial(Y, H_ii, H_jj, boson_operators):
    """Solve a Sylvester equation for bosonic monomial.

    Given Y, H_ii, and H_jj, return -(E_i - E_j_shifted) * Y_monomial, per
    monomial of creation and annihilation operators in Y.

    H_ii and H_jj are scalar expressions containing number operators of
    possibly several bosons.

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

    shifts = NumberOrderedForm.from_expr(Y).terms

    result = sympy.S.Zero
    for shift, monomial in shifts.items():
        shifted_H_jj = H_jj.subs(
            {
                NumberOperator(op): NumberOperator(op) + delta
                for delta, op in zip(shift, boson_operators)
            }
        )
        result += simplify_number_expression(H_ii - shifted_H_jj) ** -1 * monomial

    return result


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
    mask: Mask,
) -> sympy.MatrixBase:
    """Apply a mask to filter specific terms in a matrix operator.

    This function selectively keeps terms in a symbolic matrix operator based on
    their powers of creation and annihilation operators.

    Parameters
    ----------
    operator :
        Matrix operator containing symbolic expressions with second quantized operators.
    mask :
        Specification of which terms to keep, see Notes for format details.

    Returns
    -------
    sympy.MatrixBase
        A new matrix with the same shape as the input, but containing only the
        selected terms.

    Notes
    -----
    The mask consists of two parts:
    1. A list of operators to check powers for
    2. A list of selection rules for keeping terms

    Each selection rule consists of:
    - One constraint per operator defining which powers to keep
    - A boolean matrix indicating which matrix elements to apply the rule to

    For each operator, the constraint is specified as ([n1, n2, ...], n_max) where:
    - [n1, n2, ...] is a list of specific powers of raising/lowering operators to keep
    - n_max is an optional threshold. If provided, all powers ≥ n_max will be kept
    - If n_max is None, only the explicitly listed powers are kept

    A term is kept if it satisfies ALL constraints in at least ONE selection rule.

    """
    operators, mask = mask
    result = sympy.zeros(operator.rows, operator.cols)
    for i in range(operator.rows):
        for j in range(operator.cols):
            value = operator[i, j]
            shifts = NumberOrderedForm.from_expr(value).terms
            for *mask_bosons, mask_matrix in mask:
                if not mask_matrix[i, j]:
                    continue
                found = []
                for shift, monomial in shifts.items():
                    if all(
                        abs(shift[i]) in ns
                        or (n_max is not None and abs(shift[i]) >= n_max)
                        for i, (ns, n_max) in enumerate(mask_bosons)
                    ):
                        result[i, j] += monomial
                        found.append(shift)
                for shift in found:
                    del shifts[shift]

    return result
