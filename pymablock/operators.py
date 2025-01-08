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
    Y_daggers = [Dagger(op) for op in boson_operators_Y]

    result = sympy.S.Zero
    for monomial in Y.expand().as_ordered_terms():
        shift = [0] * len(boson_operators_Y)
        for term in monomial.as_ordered_factors():
            symbol, power = term.as_base_exp()
            if symbol in boson_operators_Y:
                shift[boson_operators_Y.index(symbol)] += power
            elif symbol in Y_daggers:
                shift[Y_daggers.index(symbol)] -= power

        shifted_H_jj = H_jj.subs(
            {
                Dagger(op) * op: Dagger(op) * op - delta_op
                for delta_op, op in zip(shift, boson_operators_Y)
            }
        )
        result -= ((H_ii - shifted_H_jj).expand().simplify()) ** (-1) * monomial

    return result


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
