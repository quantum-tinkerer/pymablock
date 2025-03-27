"""Second quantization tools for bosonic and fermionic operators."""

from collections import defaultdict
from typing import Callable

import numpy as np
import sympy
import sympy.physics
from packaging.version import parse
from sympy.physics.quantum import Dagger, HermitianOperator, boson, fermion
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.operatorordering import normal_ordered_form

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
    return list(
        set(
            type(arg)(arg.name)
            for arg in expr.free_symbols
            if isinstance(arg, (boson.BosonOp, fermion.FermionOp))
        )
    )


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

    shifts = expr_to_shifts(Y, boson_operators)

    result = sympy.S.Zero
    for shift, monomial in shifts.items():
        shifted_H_jj = H_jj.subs(
            {
                Dagger(op) * op: Dagger(op) * op - delta
                for delta, op in zip(shift, boson_operators)
            }
        )
        result -= ((H_ii - shifted_H_jj).expand().simplify()) ** -1 * monomial

    return result


def expr_to_shifts(
    expr: sympy.Expr, boson_operators: list[boson.BosonOp]
) -> dict[tuple[int, ...], sympy.Expr]:
    """Decompose an expression to a dictionary of shifts.

    Parameters
    ----------
    expr :
        Sympy expression with bosonic operators.
    boson_operators :
        List with all possible bosonic operators in the expression.

    Returns
    -------
    Dictionary with shifts for each monomial.

    """
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
            shifts = expr_to_shifts(value, operators)
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


class NumberOperator(HermitianOperator):
    """Number operator for bosonic and fermionic operators."""

    @property
    def name(self):
        """Return the name of the operator."""
        return self.args[0].name

    def __new__(cls, *args, **hints):
        """Construct a number operator for bosonic modes.

        Parameters
        ----------
        operator :
            Operator that the number operator counts.
        args :
            Length-1 list with the operator.
        hints :
            Unused; required for compatibility with sympy.

        """
        try:
            (operator,) = args
        except ValueError:
            raise ValueError("NumberOperator requires a single argument.")
        if not isinstance(operator, (boson.BosonOp, fermion.FermionOp)):
            raise TypeError("NumberOperator requires a bosonic or fermionic operator.")
        if not operator.is_annihilation:
            raise ValueError("Operator must be an annihilation operator.")

        return super().__new__(cls, operator, **hints)

    def doit(self, **hints):  # noqa: ARG002
        """Evaluate the operator.

        Returns
        -------
        sympy.QExpr
            The evaluated operator.

        """
        return Dagger(self.args[0]) * self.args[0]

    def _eval_commutator_NumberOperator(self, other):  # noqa: ARG002
        """Evaluate the commutator with another NumberOperator."""
        return sympy.S.Zero

    def _eval_commutator_BosonOp(self, other, **hints):
        """Evaluate the commutator with a Boson operator."""
        if isinstance(self.args[0], fermion.FermionOp):
            return sympy.S.Zero
        if other.name != self.name and hints.get("independent"):
            return sympy.S.Zero
        return normal_ordered_form(
            Commutator(self.doit(), other, **hints).doit(), **hints
        )

    def _eval_commutator_FermionOp(self, other, **hints):
        """Evaluate the commutator with a Fermion operator."""
        if isinstance(self.args[0], boson.BosonOp):
            return sympy.S.Zero
        if other.name != self.name and hints.get("independent"):
            return sympy.S.Zero
        return normal_ordered_form(
            Commutator(self.doit(), other, **hints).doit(), **hints
        )

    def _print_contents_latex(self, printer, *args):  # noqa: ARG002
        return r"{N_{%s}}" % str(self.name)

    def _print_contents(self, printer, *args):  # noqa: ARG002
        return r"N_%s" % str(self.name)

    def _print_contents_pretty(self, printer, *args):
        return printer._print("N_" + self.args[0], *args)
