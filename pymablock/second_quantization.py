"""Second quantization tools for bosonic and fermionic operators."""

from collections import defaultdict
from typing import Callable

import numpy as np
import sympy
import sympy.physics
from packaging.version import parse
from sympy.physics.quantum import Dagger, HermitianOperator, boson, fermion
from sympy.physics.quantum.boson import BosonOp
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
    # Replace all number operators with their evaluated form.
    expr = expr.subs({i: i.doit() for i in expr.atoms(NumberOperator)})
    return list(
        set(type(arg)(arg.name) for arg in expr.atoms(boson.BosonOp, fermion.FermionOp))
    )


def multiply_b(expr, operator):
    """Multiply a number-ordered expression by a boson annihilation operator from the right.

    Parameters
    ----------
    expr :
        Number-ordered sympy expression with bosonic operators.
    operator :
        Boson annihilation operator to multiply with.

    Returns
    -------
    sympy.core.expr.Expr
        Number-ordered product expr * operator.

    """
    n = NumberOperator(operator)

    number_ordered_terms = []
    for term in expr.as_ordered_terms():
        if not term.has(Dagger(operator)):  # only annihilations in term
            number_ordered_terms.append(term * operator)
        else:
            # Commute n and boson
            term = term.subs(n, n - 1) * n
            # Find Dagger(boson) in term
            daggered_operator = next(
                factor
                for factor in term.as_ordered_factors()
                if factor.has(Dagger(operator))
            )
            term = term.subs(
                daggered_operator,
                Dagger(operator) ** (daggered_operator.as_base_exp()[1] - 1),
            )
            number_ordered_terms.append(term)
    return sympy.Add(*number_ordered_terms)


def multiply_daggered_b(expr: sympy.Expr, daggered_operator):
    """Multiply a number-ordered expression by a boson creation operator from the right.

    Parameters
    ----------
    expr :
        Number-ordered sympy expression with bosonic operators.
    daggered_operator :
        Boson creation operator to multiply with.

    Returns
    -------
    sympy.core.expr.Expr
        Number-ordered product expr * daggered_operator.

    """
    operator = Dagger(daggered_operator)
    n = NumberOperator(operator)
    number_ordered_terms = []
    for term in expr.as_ordered_terms():
        try:
            boson_factor = next(
                (
                    factor
                    for factor in term.as_ordered_factors()
                    if factor.as_base_exp()[0] == operator
                )
            )
            term = term.subs(
                boson_factor, operator ** (boson_factor.as_base_exp()[1] - 1)
            )
            number_ordered_terms.append(multiply_fn(term, n + 1))
        except StopIteration:
            # Commute n and daggered operator
            number_ordered_terms.append(daggered_operator * term.subs(n, n + 1))
    return sympy.Add(*number_ordered_terms)


def multiply_fn(expr, nexpr):
    """Multiply a number-ordered expression by a function of number operators.

    Parameters
    ----------
    expr :
        Number-ordered sympy expression with bosonic operators.
    nexpr :
        Expression containing only number operators.

    Returns
    -------
    sympy.Expr
        Number-ordered product expr * nexpr.

    """
    number_ordered_terms = []
    for term in expr.as_ordered_terms():
        # Find common bosons
        boson_powers = [
            (base_exp[0], base_exp[1])
            for factor in term.as_ordered_factors()
            if isinstance((base_exp := factor.as_base_exp())[0], BosonOp)
            and base_exp[0].is_annihilation
        ]
        if not boson_powers:
            number_ordered_terms.append(term * nexpr)
            continue

        fn = nexpr
        for operator, power in boson_powers:
            fn = fn.subs(NumberOperator(operator), NumberOperator(operator) + power)
        number_ordered_terms.append(fn * term)
    return sympy.Add(*number_ordered_terms)


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


def number_ordered_form(expr, simplify=False):
    """Convert an expression to number-ordered form.

    Number ordered form is a form where:

    1. All creation operators are on the left.
    2. All annihilation operators are on the right.
    3. There are as many number operators as possible in the middle.

    This means that no term in a number-ordered expression can simultaneously contain
    both a creation and an annihilation operator for the same particle.

    Importantly, the part with number operators may contain arbitrary functions
    of number operators.

    This form makes it easy to manipulate complex expressions, because
    commuting a creation or annihilation operator through a function of a
    number operator amounts to replacing corresponding number operator `N` with
    `N ± 1`.

    Parameters
    ----------
    expr :
        Sympy expression with bosonic operators.
    simplify :
        Whether to simplify the number expressions in the result.

    Returns
    -------
    sympy.core.expr.Expr
        Equivalent expression in normal-ordered form with number operators.

    """
    result = sympy.S.Zero
    for term in expr.as_ordered_terms():
        composed_term = sympy.S.One
        for factor in term.as_ordered_factors():
            base, power = factor.as_base_exp()
            if isinstance(base, BosonOp) and base.is_annihilation:
                if power < 0:
                    raise ValueError(
                        f"Cannot have negative power of boson operator: {base}"
                    )
                for _ in range(power):
                    composed_term = multiply_b(composed_term, base)
                continue
            if isinstance(base, BosonOp):
                if power < 0:
                    raise ValueError(
                        f"Cannot have negative power of boson operator: {base}"
                    )
                for _ in range(power):
                    composed_term = multiply_daggered_b(composed_term, base)
                continue
            if not base.atoms(BosonOp):  # base is a function of number operators
                composed_term = multiply_fn(composed_term, base**power)
                continue

            # Composite number-ordered expression
            base = number_ordered_form(base)
            if power < 0 and base.atoms(BosonOp):
                raise ValueError(f"Cannot have negative power of boson operator: {base}")
            for _ in range(power):
                composed_term = sympy.Add(
                    *(
                        number_ordered_form(composed_term * base_term)
                        for base_term in base.as_ordered_terms()
                    )
                )
        result += composed_term

    # Group the terms in the result by unmatched powers of operators
    grouped_result = group_ordered(result)
    if simplify:
        grouped_result = {
            key: simplify_number_expression(value)
            for key, value in grouped_result.items()
        }
    return sympy.Add(*(i * value * j for (i, j), value in grouped_result.items()))


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
                NumberOperator(op): NumberOperator(op) + delta
                for delta, op in zip(shift, boson_operators)
            }
        )
        result -= (H_ii - shifted_H_jj) ** -1 * monomial

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
    eigs = tuple([number_ordered_form(eig) for eig in eig_block] for eig_block in eigs)

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
    """Number operator for bosonic and fermionic operators.

    Notes
    -----
    This class is used to simplify expressions with second-quantized operators. We do
    this ourselves, because sympy does not support this yet.

    """

    @property
    def name(self):
        """Return the name of the operator."""
        return self.args[0]

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
            if not isinstance(operator, (boson.BosonOp, fermion.FermionOp)):
                raise TypeError(
                    "NumberOperator requires a bosonic or fermionic operator."
                )
            if not operator.is_annihilation:
                raise ValueError("Operator must be an annihilation operator.")
            name = operator.name
            operator_type = "boson" if isinstance(operator, boson.BosonOp) else "fermion"
        except ValueError:
            name, operator_type = args

        return super().__new__(
            cls,
            name,
            operator_type,
            **hints,
        )

    def doit(self, **hints):  # noqa: ARG002
        """Evaluate the operator.

        For example,

            >>> from sympy.physics.quantum import boson
            >>> from pymablock.second_quantization import NumberOperator
            >>> b = boson.BosonOp('b')
            >>> n = NumberOperator(b)
            >>> n.doit()
            Dagger(b)*b

        Returns
        -------
        sympy.core.expr.Expr
            The evaluated operator.

        """
        op = (boson.BosonOp if self.args[1].name == "boson" else fermion.FermionOp)(
            self.args[0]
        )
        return Dagger(op) * op

    def _eval_commutator_NumberOperator(self, other):  # noqa: ARG002
        """Evaluate the commutator with another NumberOperator."""
        return sympy.S.Zero

    def _eval_commutator_BosonOp(self, other, **hints):
        """Evaluate the commutator with a Boson operator."""
        if self.args[1].name == "fermion":
            return sympy.S.Zero
        if other.name != self.name and hints.get("independent"):
            return sympy.S.Zero
        return normal_ordered_form(
            Commutator(self.doit(), other, **hints).doit(), **hints
        )

    def _eval_commutator_FermionOp(self, other, **hints):
        """Evaluate the commutator with a Fermion operator."""
        if self.args[1].name == "boson":
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
