"""Second quantization tools for bosonic and fermionic operators.

See number_ordered_form_plan.md for the plan to implement NumberOrderedForm as an
Operator subclass for better representation of number-ordered expressions.
"""

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
        shifted_H_jj = H_jj.as_expr().subs(
            {
                NumberOperator(op): NumberOperator(op) + delta
                for delta, op in zip(shift, boson_operators)
            }
        )
        result += NumberOrderedForm.from_expr((H_ii - shifted_H_jj) ** -1 * monomial)
    return result.simplify()


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
    mask_operators, mask = mask
    result = sympy.zeros(operator.rows, operator.cols)
    for i in range(operator.rows):
        for j in range(operator.cols):
            value = operator[i, j]
            # We need to take special care because the mask might not contain all
            # operators appearing in the expression.
            operators, shifts = NumberOrderedForm.from_expr(value).args
            # Convert from sympy subclass because sympy dicts are immutable
            mask_indices = [operators.index(op) for op in mask_operators]
            for *mask_bosons, mask_matrix in mask:
                if not mask_matrix[i, j]:
                    continue
                keep = []
                for shift in shifts:
                    mask_shift = [shift[idx] for idx in mask_indices]
                    if all(
                        abs(op_shift) in ns
                        or (n_max is not None and abs(op_shift) >= n_max)
                        for op_shift, (ns, n_max) in zip(mask_shift, mask_bosons)
                    ):
                        keep.append(shift)
                result[i, j] = NumberOrderedForm(
                    operators=operators,
                    terms={
                        shift: value for shift, value in shifts.items() if shift in keep
                    },
                )

    return result
