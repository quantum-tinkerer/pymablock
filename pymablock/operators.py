"""Second quantization tools for bosonic and fermionic operators."""

from collections import defaultdict
from typing import Callable

import sympy
from packaging.version import parse
from sympy.physics.quantum import Dagger, boson
from sympy.physics.quantum.operatorordering import normal_ordered_form

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


def find_boson_operators(expr: sympy.Expr):
    """Find all the boson operators in an expression.

    Parameters
    ----------
    expr :
        Sympy expression with bosonic operators.

    Returns
    -------
    List with all annihilation bosonic operators in an expression.

    """
    annihilation_operators = [
        arg
        for arg in expr.free_symbols
        if isinstance(arg, boson.BosonOp) and arg.is_annihilation
    ]
    creation_operators = [
        Dagger(arg)
        for arg in expr.free_symbols
        if isinstance(arg, boson.BosonOp) and not arg.is_annihilation
    ]

    return list(set(annihilation_operators + creation_operators))


def convert_to_number_operators(expr: sympy.Expr, boson_operators: list):
    """Convert an expression to a function of number operators.

    Parameters
    ----------
    expr :
        Sympy expression with bosonic operators.
    boson_operators :
        List with all bosonic operators in the expression.

    Returns
    -------
    Equivalent expression in terms of number operators.

    """
    daggers = [Dagger(op) for op in boson_operators]
    terms = expr.expand().as_ordered_terms()
    total = sympy.S.Zero
    for term in terms:
        term = normal_ordered_form(term, independent=True)
        # 0. Expand again.
        # 1. Check that powers of creation and annihilation operators are the same.
        # 2. Compute the prefactor and all powers of the number operator.
        expression = term.expand()
        for term in expression.as_ordered_terms():
            result = sympy.S.One
            powers = defaultdict(int)
            for factor in term.as_ordered_factors():
                symbol, power = factor.as_base_exp()
                if symbol not in boson_operators and symbol not in daggers:
                    result *= factor
                powers[symbol] = power
            for op, op_dagger in zip(boson_operators, daggers):
                if powers[op] != powers[op_dagger]:
                    raise ValueError("Hamiltonian is not diagonal in number basis.")

                result *= sympy.Mul(*[(Dagger(op) * op - i) for i in range(powers[op])])
            total += result

    return total.expand().simplify()


def solve_monomial(Y, H_ii, H_jj):
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

    boson_operators_Y = find_boson_operators(Y)
    shifts = expr_to_shifts(Y)

    result = sympy.S.Zero
    for shift, monomial in shifts.items():
        shifted_H_jj = H_jj.subs(
            {
                Dagger(op) * op: Dagger(op) * op - delta
                for delta, op in zip(shift, boson_operators_Y)
            }
        )
        result -= ((H_ii - shifted_H_jj).expand().simplify()) ** -1 * monomial

    return result


def expr_to_shifts(expr: sympy.Expr) -> dict[tuple[int, ...], sympy.Expr]:
    """Decompose an expression to a dictionary of shifts.

    Parameters
    ----------
    expr :
        Sympy expression with bosonic operators.

    Returns
    -------
    Dictionary with shifts for each monomial.

    """
    boson_operators = find_boson_operators(expr)
    daggers = [Dagger(op) for op in boson_operators]

    shifts = defaultdict(lambda: sympy.S.Zero)
    for monomial in expr.expand().as_ordered_terms():
        shift = [0] * len(boson_operators)
        for term in monomial.as_ordered_factors():
            symbol, power = term.as_base_exp()
            if symbol in boson_operators:
                shift[boson_operators.index(symbol)] += power
            elif symbol in daggers:
                shift[daggers.index(symbol)] -= power
        shifts[tuple(shift)] += monomial

    return shifts


def solve_sylvester_bosonic(
    eigs: tuple[sympy.matrices.MatrixBase, ...],
) -> Callable:
    """Solve a Sylvester equation for bosonic diagonal Hamiltonians."""
    boson_operators = set.union(
        *(set(find_boson_operators(eig)) for eig_block in eigs for eig in eig_block)
    )

    eigs = tuple(
        [convert_to_number_operators(eig, boson_operators) for eig in eig_block]
        for eig_block in eigs
    )

    def solve_sylvester(
        Y: sympy.MatrixBase,
        index: tuple[int, ...],
    ) -> sympy.MatrixBase:
        if Y is zero:
            return zero
        eigs_A, eigs_B = eigs[index[0]], eigs[index[1]]
        return sympy.Matrix(
            [
                [solve_monomial(Y[i, j], eigs_A[i], eigs_B[j]) for j in range(Y.cols)]
                for i in range(Y.rows)
            ]
        )

    return solve_sylvester


def apply_mask_to_operator(
    operator: sympy.MatrixBase,
    mask: list[tuple[tuple[list[int], int | None], sympy.MatrixBase]],
) -> sympy.MatrixBase:
    """Apply a mask to an operator.

    Parameters
    ----------
    operator :
        Operator to apply the mask to.
    mask :
        Mask to apply to the operator.
    negate :
        Whether to negate the mask.

    Returns
    -------
    Operator with the mask applied.

    Notes
    -----
    For a single boson the mask has a format ([n_0, n_1, n_2, ...], n_max), where all
    n_i are nonnegative integers that label the powers of boson operators to eliminate.
    n_max may be None, and it indicates that powers above n_max are eliminated (or not
    if None). In a finite Hilbert space, the mask is an arbitrary symmetric binary
    matrix. A many-body mask is a list of tuples of masks for each boson/finite Hilbert
    space.

    For example [(([], 0), Matrix([[0, 1], [1, 0]])), (([], 1), Matrix([[1, 1], [1,
    1]]))] corresponds to full diagonalization of boson x spin.

    """
    result = sympy.zeros(operator.rows, operator.cols)
    for i in range(operator.rows):
        for j in range(operator.cols):
            value = operator[i, j]
            shifts = expr_to_shifts(value)
            for *mask_bosons, mask_matrix in mask:
                if not mask_matrix(i, j):
                    continue
                for shift, monomial in shifts.items():
                    if all(
                        abs(shift[i]) in ns
                        or (n_max is not None and abs(shift[i]) >= n_max)
                        for i, (ns, n_max) in enumerate(mask_bosons)
                    ):
                        result[i, j] += monomial

    return result
