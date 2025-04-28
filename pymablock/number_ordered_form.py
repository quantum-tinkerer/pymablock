"""Implementation of NumberOrderedForm as an Operator subclass.

This module provides a class-based implementation of number ordered form for quantum operators,
which represents operators with creation operators on the left, annihilation operators on the right,
and number operators in the middle.
"""

import uuid
from collections import defaultdict
from collections.abc import Callable

import sympy
from packaging.specifiers import SpecifierSet
from sympy.physics.quantum import Dagger, HermitianOperator, Operator
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

__all__ = [
    "NumberOperator",
    "NumberOrderedForm",
    "find_operators",
]


# Monkey patch sympy to propagate adjoint to matrix elements.
if sympy.__version__ in SpecifierSet("<1.14"):

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
    del _eval_adjoint
    del _eval_transpose

    # Only implements skipping identity, and is deleted in 1.14.
    try:
        del BosonOp.__mul__
        del Operator.__mul__
    except AttributeError:
        pass


# TODO: reimplement once https://github.com/sympy/sympy/issues/27385 is fixed.
# Monkey patch sympy to override the sum method to ExpressionRawDomain.
def _sum(self, items):  # noqa ARG001
    """Slower, but overridable version of sympy.Add."""
    if not items:
        return sympy.S.Zero
    result = items[0]
    for item in items[1:]:
        result += item
    return result


sympy.polys.domains.expressionrawdomain.ExpressionRawDomain.sum = _sum
del _sum


# Type aliases
operator_types = BosonOp, FermionOp
OperatorType = BosonOp | FermionOp
PowerKey = tuple[int, ...]
TermDict = dict[PowerKey, sympy.Expr]


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
            if not isinstance(operator, operator_types):
                raise TypeError(
                    "NumberOperator requires a bosonic or fermionic operator."
                )
            if not operator.is_annihilation:
                raise ValueError("Operator must be an annihilation operator.")
            name = operator.name
            operator_type = "boson" if isinstance(operator, BosonOp) else "fermion"
        except ValueError:
            name, operator_type = args

        return super().__new__(
            cls,
            name,
            operator_type,
            **hints,
        )

    def doit(self, **hints) -> sympy.Expr:  # noqa: ARG002
        """Evaluate the operator.

        For example,

        >>> from sympy.physics.quantum.boson import BosonOp
        >>> from pymablock.second_quantization import NumberOperator
        >>> b = BosonOp('b')
        >>> n = NumberOperator(b)
        >>> n.doit()
        Dagger(b)*b

        Returns
        -------
        result : `~sympy.core.expr.Expr`
            The evaluated operator.

        """
        op = (BosonOp if self.args[1].name == "boson" else FermionOp)(self.args[0])
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
        return printer._print("N_%s" % str(self.name), *args)


def find_operators(expr: sympy.Expr) -> list[OperatorType]:
    """Find all quantum operators in a SymPy expression.

    Parameters
    ----------
    expr :
        The expression to search for quantum operators.

    Returns
    -------
    operators : `list[OperatorType]`
        A list of unique quantum operators found in the expression. Boson operators are
        listed before fermion operators and both are sorted by their names.

    """
    # replace n -> a† * a.
    expanded = expr.subs({n: n.doit() for n in expr.atoms(NumberOperator)})
    return [
        op
        for particle in operator_types
        for op in sorted(
            {particle(atom.name) for atom in expanded.atoms(particle)},
            key=lambda op: str(op.name),
        )
    ]


class NumberOrderedForm(Operator):
    """Number ordered form of quantum operators.

    A number ordered form represents quantum operators where:
    1. All creation operators are on the left
    2. All annihilation operators are on the right
    3. Number operators (and other scalar expressions) are in the middle

    This representation makes it easy to manipulate complex quantum expressions, because
    commuting a creation or annihilation operator through a function of a number operator
    simply replaces the corresponding number operator `N` with `N ± 1`.

    See more details in the second quantization documentation.

    Parameters
    ----------
    operators :
        List of quantum operators (annihilation operators only).
    terms :
        Dictionary mapping operator power tuples to coefficient expressions.
        Negative powers represent creation operators, positive powers represent
        annihilation operators.
    **hints : dict
        Additional hints passed to the parent class.

    """

    # Same as dense matrices
    _op_priority = 10.01
    _class_priority = 4

    def __new__(
        cls,
        operators: list[OperatorType],
        terms: TermDict,
        *,
        validate: bool = True,
        **hints,
    ):
        """Create a new NumberOrderedForm instance.

        Parameters
        ----------
        operators :
            List of operators (annihilation operators only).
        terms :
            Dictionary mapping operator power tuples to coefficient expressions.
            Negative powers represent creation operators, positive powers represent
            annihilation operators.
        validate :
            Whether to validate the operators and terms, by default True.
        **hints : dict
            Additional hints passed to the parent class.

        Returns
        -------
        NumberOrderedForm
            A new NumberOrderedForm instance.

        """
        if validate:
            # Validate inputs before creating the object
            cls._validate_operators(operators)
            cls._validate_terms(terms, operators)

        if not isinstance(operators, sympy.core.containers.Tuple):
            operators = sympy.core.containers.Tuple(*operators)
        # Convert terms dict to a tuple of (power_tuple, coefficient) tuples
        if isinstance(terms, dict):
            terms = sympy.core.containers.Tuple(
                *(sympy.core.containers.Tuple(k, v) for k, v in terms.items())
            )
        elif not isinstance(terms, sympy.core.containers.Tuple):
            terms_items = list(terms)
            terms = sympy.core.containers.Tuple(
                *(sympy.core.containers.Tuple(k, v) for k, v in terms_items)
            )
        result = sympy.Expr.__new__(cls, operators, terms, **hints)
        result._n_fermions = sum(isinstance(op, FermionOp) for op in operators)
        return result

    @staticmethod
    def _validate_operators(operators: list[OperatorType]) -> None:
        """Validate the operators list.

        Parameters
        ----------
        operators :
            List of quantum operators to validate.

        Raises
        ------
        ValueError
            If the operators list is empty or if the order is incorrect.
        TypeError
            If an operator is not a valid quantum operator.
        ValueError
            If an operator is not an annihilation operator.

        """
        for op in operators:
            if not isinstance(op, OperatorType):
                raise TypeError(f"Expected BosonOp or FermionOp, got {type(op)}")
            if not op.is_annihilation:
                raise ValueError(f"Operator must be an annihilation operator: {op}")

        # Check that bosons come before fermions
        seen_fermion = False
        for op in operators:
            if isinstance(op, FermionOp):
                seen_fermion = True
            elif seen_fermion and isinstance(op, BosonOp):
                raise ValueError("Bosonic operators must come before fermionic operators")

    @staticmethod
    def _validate_terms(terms: TermDict, operators: list[OperatorType]) -> None:
        """Validate the terms dictionary.

        Parameters
        ----------
        terms : dict[tuple[int, ...], sympy.Expr]
            Dictionary mapping operator power tuples to coefficient expressions.
        operators : list[OperatorType]
            List of quantum operators to validate against.

        Raises
        ------
        ValueError
            If the terms dictionary is empty.
        ValueError
            If a powers tuple has incorrect length.
        TypeError
            If a coefficient is not a sympy expression.
        ValueError
            If a coefficient contains creation or annihilation operators.

        """
        if isinstance(terms, dict):
            terms = terms.items()

        for powers, coeff in terms:
            # Check that powers tuple has the right length
            if len(powers) != len(operators):
                raise ValueError(
                    f"Powers tuple length ({len(powers)}) doesn't match "
                    f"operators length ({len(operators)})"
                )

            for power in powers:
                if not isinstance(power, int) and not isinstance(power, sympy.Integer):
                    raise TypeError(f"Power must be an integer, got {type(power)}")

            # Check that the coefficient is a valid sympy expression
            if not isinstance(coeff, sympy.Expr):
                raise TypeError(
                    f"Coefficient must be a sympy expression, got {type(coeff)}"
                )

            # Check for unwanted creation or annihilation operators in the coefficient
            if coeff.has(*operator_types):
                raise ValueError(
                    f"Coefficient contains creation or annihilation operators: {coeff}"
                )

    @classmethod
    def from_expr(
        cls, expr: sympy.Expr, operators: list[OperatorType] | None = None
    ) -> "NumberOrderedForm":
        """Create a NumberOrderedForm instance from a sympy expression.

        Parameters
        ----------
        expr :
            Sympy expression with quantum operators.
        operators :
            List of quantum operators to use. If None, operators will be extracted from
            the expression.

        Returns
        -------
        NumberOrderedForm
            A NumberOrderedForm instance representing the expression.

        Examples
        --------
        Create a NumberOrderedForm from a sympy expression with bosonic operators:

        >>> from sympy.physics.quantum import boson
        >>> from pymablock.number_ordered_form import NumberOrderedForm
        >>> a = BosonOp('a')
        >>> expr = a.adjoint() * a + 1  # a^† * a + 1
        >>> nof = NumberOrderedForm.from_expr(expr)
        >>> nof
        1 + N_a

        Using a number operator:

        >>> from pymablock.number_ordered_form import NumberOperator
        >>> n_a = NumberOperator(a)
        >>> expr = n_a + 2
        >>> nof = NumberOrderedForm.from_expr(expr)
        >>> nof
        2 + N_a

        """
        if not isinstance(expr, sympy.Expr):
            try:
                expr = sympy.sympify(expr)
            except Exception:
                raise ValueError(f"Cannot convert {expr} to a sympy expression")

        if isinstance(expr, NumberOrderedForm):
            return expr

        # For scalar expressions (no operators)
        if not expr.has(*operator_types, NumberOperator):
            # Return a NumberOrderedForm with no operators and a single term
            return cls([], {(): expr}, validate=False)

        if not operators:
            operators = find_operators(expr)

        # Handle Add expressions by converting each term and summing
        if isinstance(expr, sympy.Add):
            terms = [
                NumberOrderedForm.from_expr(term, operators=operators)
                for term in expr.args
            ]
            return sum(terms, start=NumberOrderedForm([], {}, validate=False))

        # Handle Mul expressions by converting each factor and multiplying
        if isinstance(expr, sympy.Mul):
            factors = [
                NumberOrderedForm.from_expr(factor, operators=operators)
                for factor in expr.args
            ]
            result = factors[0]
            for factor in factors[1:]:
                result = result * factor
            return result

        # Handle Pow expressions
        if isinstance(expr, sympy.Pow):
            # Handle power expressions like a**2 or Dagger(a)**3
            base = expr.base
            exp = expr.exp

            # Handle exponentiation of single operators directly.
            if isinstance(base, OperatorType) and exp.is_integer and exp.is_positive:
                # Find the operator index in the operators list
                op = base if base.is_annihilation else type(base)(base.name)
                powers = tuple(
                    exp * (1 if base.is_annihilation else -1) if op == operator else 0
                    for operator in operators
                )
                return cls(operators, {powers: sympy.S.One}, validate=False)

            # Convert base to NumberOrderedForm
            base_nof = NumberOrderedForm.from_expr(base, operators=operators)

            # Use the __pow__ method to handle the exponentiation
            return base_nof**exp

        # Handle function calls (like exp, sin, etc.)
        if isinstance(expr, sympy.Function):
            # Convert each argument to NumberOrderedForm
            arg_nofs = [
                NumberOrderedForm.from_expr(arg, operators=operators) for arg in expr.args
            ]

            # Check that each argument has only number operators (no unmatched creation/annihilation operators)
            for arg_nof in arg_nofs:
                if not arg_nof.is_particle_conserving():
                    raise ValueError(
                        f"Cannot apply function {expr.func} to expression with unmatched "
                        f"creation or annihilation operators: {arg_nof}"
                    )

            # Now we can safely convert the arguments to expressions
            # Extract the coefficients from the zero keys for each argument
            zero_key = tuple(0 for _ in operators)
            arg_exprs = [
                next(iter(arg_nof.terms.values()), sympy.S.Zero) for arg_nof in arg_nofs
            ]

            # Return a new NumberOrderedForm with the function applied to the coefficients
            return cls(operators, {zero_key: expr.func(*arg_exprs)}, validate=False)

        # Handle BosonOp or FermionOp (both creation and annihilation operators)
        if isinstance(expr, OperatorType):
            # Find the corresponding annihilation operator in our operators list
            annihilation_op = expr if expr.is_annihilation else type(expr)(expr.name)

            if annihilation_op not in operators:
                raise ValueError(
                    f"Operator {annihilation_op} not found in operators list"
                )

            op_index = operators.index(annihilation_op)

            # Determine the power (1 for annihilation, -1 for creation)
            power = 1 if expr.is_annihilation else -1

            # Create a term with the appropriate power
            powers = tuple(0 if i != op_index else power for i in range(len(operators)))
            return cls(operators, {powers: sympy.S.One}, validate=False)

        # Handle NumberOperator
        if isinstance(expr, NumberOperator):
            # Number operator N_a should be represented as a scalar term
            # with no explicit creation or annihilation operators
            powers = tuple(0 for _ in range(len(operators)))
            coeff = expr  # Keep the NumberOperator as the coefficient

            return cls(operators, {powers: coeff}, validate=False)

        # If we've reached this point, we don't know how to handle this expression type
        raise ValueError(
            f"Cannot convert expression of type {type(expr)} to NumberOrderedForm: {expr}"
        )

    def as_expr(self) -> sympy.Expr:
        """Convert the NumberOrderedForm to a standard SymPy expression.

        Returns
        -------
        result : `~sympy.core.expr.Expr`
            A standard SymPy expression equivalent to this NumberOrderedForm.

        Examples
        --------
        Convert a NumberOrderedForm to a standard SymPy expression:

        >>> from sympy.physics.quantum.boson import BosonOp
        >>> from pymablock.number_ordered_form import (
        ...     NumberOrderedForm, NumberOperator
        ... )
        >>> a = BosonOp('a')
        >>> # Create NumberOrderedForm with creation and annihilation operators
        >>> nof = NumberOrderedForm.from_expr(a.adjoint() * a + 2)
        >>> nof
        2 + N_a
        >>> # Convert back to a standard SymPy expression
        >>> expr = nof.as_expr()
        >>> expr
        2 + N_a
        >>> # You can also use as_expr() with number operators
        >>> nof2 = NumberOrderedForm.from_expr(NumberOperator(a) + 3)
        >>> nof2.as_expr()
        3 + N_a

        """
        if not self.operators:
            # If there are no operators, just return the constant term
            return next(iter(self.terms.values())) if self.terms else sympy.S.Zero

        result = sympy.S.Zero
        reversed_operators = list(reversed(self.operators))

        for powers, coeff in self.args[1]:
            term = coeff
            for op, power in zip(reversed_operators, reversed(powers)):
                if not power > 0:
                    continue
                # Annihilation operator (positive power)
                term = term * op**power

            for op, power in zip(reversed_operators, reversed(powers)):
                if not power < 0:
                    continue
                # Creation operator (negative power)
                term = op.adjoint() ** (-power) * term

            result += term

        return result

    def doit(self, **hints) -> sympy.Expr:
        """Evaluate the NumberOrderedForm.

        Parameters
        ----------
        **hints :
            Additional hints passed to the parent class.

        Returns
        -------
        result : `~sympy.core.expr.Expr`
            The evaluated NumberOrderedForm.

        """
        return self.as_expr().doit(**hints)

    @property
    def operators(self) -> list[OperatorType]:
        """The list of included operators."""
        return self.args[0]

    @property
    def terms(self) -> TermDict:
        """The dictionary of terms.

        Notes
        -----
        Internally, terms are stored as a tuple of (key, value) tuples for better performance.
        This property converts the internal representation to a dictionary for compatibility.

        """
        # Convert tuple of tuples to dictionary
        return {k: v for k, v in self.args[1]}

    def _sympystr(self, printer):
        """Print the expression in a string format.

        Parameters
        ----------
        printer : object
            SymPy printer object.
        *args
            Additional arguments for the printer.

        Returns
        -------
        str
            String representation of the NumberOrderedForm.

        """
        return printer._print(self.as_expr())

    def _pretty(self, printer):
        """Return a pretty form of the expression.

        Parameters
        ----------
        printer : object
            SymPy pretty printer object.
        *args
            Additional arguments for the printer.

        Returns
        -------
        pretty print form
            Pretty representation of the NumberOrderedForm.

        """
        return printer._print(self.as_expr())

    def _latex(self, printer):
        """Return a LaTeX representation of the expression.

        Parameters
        ----------
        printer : object
            SymPy LaTeX printer object.
        *args
            Additional arguments for the printer.

        Returns
        -------
        str
            LaTeX representation of the NumberOrderedForm.

        """
        return printer._print(self.as_expr())

    def _multiply_op(self, op_index: int, op_power: int):
        """Multiply this NumberOrderedForm by an operator power.

        This implements multiplication by self.operators[op_index]^op_power,
        where positive op_power represents annihilation operators and
        negative op_power represents creation operators.

        Parameters
        ----------
        op_index : int
            The index of the operator in self.operators to multiply by.
        op_power : int
            The power of the operator. Negative for creation operators,
            positive for annihilation operators.

        Returns
        -------
        NumberOrderedForm
            The result of the multiplication.

        Raises
        ------
        ValueError
            If the op_index is out of range.

        """
        if op_index < 0 or op_index >= len(self.operators):
            raise ValueError(
                f"Operator index {op_index} out of range [0, {len(self.operators)})"
            )

        assert op_power != 0, "op_power must be non-zero"

        operator = self.operators[op_index]
        if isinstance(operator, FermionOp):
            raise NotImplementedError
        n_operator = NumberOperator(operator)

        # Create a new terms dictionary for the result
        new_terms = {}

        for powers, coeff in self.args[1]:
            orig_power = powers[op_index]  # Power of the operator at op_index
            new_power = orig_power + op_power
            new_powers = tuple(
                new_power if i == op_index else p for i, p in enumerate(powers)
            )
            if op_power > 0:  # Multiplying by an annihilation operator
                # Compute how many new number operators appear
                to_pair = min(op_power, max(-orig_power, 0))
                coeff = coeff.subs(n_operator, n_operator - to_pair)
                coeff = sympy.Mul(coeff, *(n_operator - i for i in range(to_pair)))
            else:
                to_pair = min(-op_power, max(orig_power, 0))
                # Create the new number operators from all pairs
                new_numbers = sympy.Mul(*[n_operator + i for i in range(1, to_pair + 1)])
                coeff = coeff * new_numbers
                if new_power > 0:
                    # Bring all unmatched annihilation operators to the right
                    coeff = coeff.subs(n_operator, n_operator + new_power)
                else:
                    # Bring all unmatched creation operators to the left
                    coeff = coeff.subs(n_operator, n_operator + (-op_power - to_pair))
            new_terms[new_powers] = coeff

        # Create the new NumberOrderedForm with the same operators but new terms
        return type(self)(self.operators, new_terms, validate=False)

    def _multiply_expr(self, expr: sympy.Expr):
        """Multiply by an expression without creation or annihilation operators.

        Parameters
        ----------
        expr :
            Expression to multiply by.
            This expression should not contain any creation or annihilation operators.

        Returns
        -------
        NumberOrderedForm
            The result of the multiplication.

        Raises
        ------
        ValueError
            If the expression contains creation or annihilation operators.

        """
        if expr.has(*operator_types):
            raise ValueError(
                "Expression contains creation or annihilation operators, "
                "which cannot be multiplied directly."
            )

        new_terms = {}
        for powers, coeff in self.args[1]:
            multiplier = expr
            for i, power in enumerate(powers):
                if power < 0:
                    continue
                n_i = NumberOperator(self.operators[i])
                multiplier = multiplier.subs(n_i, n_i + power)
            new_terms[powers] = coeff * multiplier

        # Return a new NumberOrderedForm instance with the updated terms
        return type(self)(self.operators, new_terms, validate=False)

    def _expand_operators(self, new_operators: list[OperatorType]) -> "NumberOrderedForm":
        """Expand the operators in this NumberOrderedForm.

        This method creates a new NumberOrderedForm with the same terms but expanded
        operators.

        Parameters
        ----------
        new_operators : list[OperatorType]
            List of new quantum operators to use. Has to contain at least all the
            original operators, and must be correctly ordered.

        Returns
        -------
        NumberOrderedForm
            A new NumberOrderedForm with the expanded operators.

        Notes
        -----
        Because this method is internal, it does not validate `new_operators`.

        """
        index_mapping = [
            self.operators.index(op) if op in self.operators else -1
            for op in new_operators
        ]
        new_terms = {
            tuple(
                powers[index_mapping[i]] if index_mapping[i] != -1 else 0
                for i in range(len(new_operators))
            ): coeff
            for powers, coeff in self.args[1]
        }
        return type(self)(new_operators, new_terms, validate=False)

    def __add__(self, other) -> "NumberOrderedForm":
        """Add this NumberOrderedForm with another object.

        Parameters
        ----------
        other : object
            Object to add to this NumberOrderedForm.

        Returns
        -------
        NumberOrderedForm
            The result of the addition.

        """
        if not isinstance(other, NumberOrderedForm):
            try:
                other = NumberOrderedForm.from_expr(sympy.sympify(other))
            except Exception:
                return NotImplemented

        self_expanded, other_expanded = self._combine_operators(other)

        new_terms = defaultdict(lambda: sympy.S.Zero)
        for powers, coeff in self_expanded.args[1]:
            new_terms[powers] += coeff
        for powers, coeff in other_expanded.args[1]:
            new_terms[powers] += coeff
        return type(self)(self_expanded.operators, new_terms, validate=False)

    def _combine_operators(self, other):
        """Convert this NumberOrderedForm and another to have the same operator list."""
        if other.operators != self.operators:
            new_operators = sorted(
                set(self.operators).union(other.operators),
                key=lambda op: (isinstance(op, BosonOp), str(op.name)),
            )
            self_expanded = self._expand_operators(new_operators)
            other_expanded = other._expand_operators(new_operators)
        else:
            self_expanded = self
            other_expanded = other
        return self_expanded, other_expanded

    def __radd__(self, other) -> "NumberOrderedForm":
        """Add another object with this NumberOrderedForm.

        This method is called when the left operand doesn't support addition with
        a NumberOrderedForm.

        Parameters
        ----------
        other : object
            Object to add with this NumberOrderedForm.

        Returns
        -------
        NumberOrderedForm
            The result of the addition.

        """
        return self.__add__(other)

    def __sub__(self, other) -> "NumberOrderedForm":
        """Subtract another object from this NumberOrderedForm.

        Parameters
        ----------
        other : object
            Object to subtract from this NumberOrderedForm.

        Returns
        -------
        NumberOrderedForm
            The result of the subtraction.

        """
        if not isinstance(other, NumberOrderedForm):
            try:
                other = NumberOrderedForm.from_expr(sympy.sympify(other))
            except Exception:
                return NotImplemented

        return self + (-other)

    def __neg__(self) -> "NumberOrderedForm":
        """Negate this NumberOrderedForm.

        Returns
        -------
        NumberOrderedForm
            The negated NumberOrderedForm.

        """
        new_terms = {powers: -coeff for powers, coeff in self.args[1]}
        return type(self)(self.operators, new_terms, validate=False)

    def __mul__(self, other) -> "NumberOrderedForm":
        """Multiply this NumberOrderedForm with another object.

        Parameters
        ----------
        other : object
            Object to multiply with this NumberOrderedForm.

        Returns
        -------
        NumberOrderedForm
            The result of the multiplication.

        """
        if not isinstance(other, NumberOrderedForm):
            if other.is_commutative:
                pass
            try:
                other = NumberOrderedForm.from_expr(sympy.sympify(other))
            except Exception:
                return NotImplemented

        self_expanded, other_expanded = self._combine_operators(other)

        result = type(self)(self_expanded.operators, {}, validate=False)
        for powers, coeff in other_expanded.args[1]:
            # First multiply by creation operators, those are with negative powers
            partial = NumberOrderedForm(
                self_expanded.operators, self_expanded.args[1], validate=False
            )
            for i, power in enumerate(powers):
                if not power < 0:
                    continue
                partial = partial._multiply_op(i, power)
            # Now multiply by the number part
            partial = partial._multiply_expr(coeff)
            # Finally, multiply by annihilation operators
            for i, power in enumerate(powers):
                if not power > 0:
                    continue
                partial = partial._multiply_op(i, power)
            # Add the result to the new terms
            result = result + partial

        return result

    def __rmul__(self, other) -> "NumberOrderedForm":
        """Right multiply this NumberOrderedForm with another object.

        This method is called when the left operand doesn't support multiplication with
        a NumberOrderedForm.

        Parameters
        ----------
        other : object
            Object to multiply with this NumberOrderedForm.

        Returns
        -------
        NumberOrderedForm
            The result of the multiplication.

        Notes
        -----
        Since NumberOrderedForm is non-commutative, this first converts the other object
        to a NumberOrderedForm, then applies regular multiplication: other * self.

        """
        try:
            other_nof = NumberOrderedForm.from_expr(sympy.sympify(other))
            return other_nof * self
        except Exception:
            return NotImplemented

    def _eval_adjoint(self):
        """Evaluate the adjoint of this NumberOrderedForm.

        This method is called by SymPy's adjoint operator.

        Returns
        -------
        NumberOrderedForm
            The adjoint of this NumberOrderedForm.

        """
        # Take the adjoint of each term and negate the powers
        new_terms = (
            (tuple(-power for power in powers), coeff.adjoint())
            for powers, coeff in self.args[1]
        )
        return type(self)(self.operators, new_terms, validate=False)

    def __eq__(self, other):
        """Evaluate equality between this NumberOrderedForm and another object.

        This method is called by SymPy's equality operator.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        sympy.Basic
            True if equal, False otherwise.

        """
        if not isinstance(other, NumberOrderedForm):
            try:
                other = NumberOrderedForm.from_expr(sympy.sympify(other))
            except Exception:
                return None  # Let SymPy handle the comparison
        return (
            all(op1 == op2 for op1, op2 in zip(self.operators, other.operators))
            and self.terms == other.terms
        )

    def __hash__(self):
        """Compute the hash of this NumberOrderedForm."""
        return super().__hash__()

    def _eval_is_zero(self):
        """Check if this NumberOrderedForm is zero.

        This method is used by SymPy to determine if an expression is zero.

        Returns
        -------
        bool or None
            True if zero, False if non-zero, None if undetermined.

        """
        if not self.args[1]:
            return True
        return None  # Let SymPy try other approaches

    def __bool__(self):
        """Check if the NumberOrderedForm is non-zero.

        This is kept for Python's boolean evaluation, but for SymPy operations,
        _eval_is_zero is preferred.

        Returns
        -------
        bool
            True if the form contains any terms, False otherwise.

        """
        return bool(self.args[1])

    def applyfunc(self, func: Callable, *args, **kwargs):
        """Apply a SymPy function to the terms of this NumberOrderedForm.

        This method temporarily replaces NumberOperators with unique symbols,
        applies the function, then substitutes back the NumberOperators.

        Parameters
        ----------
        func :
            SymPy function to apply (e.g., sympy.simplify, sympy.factor)
        *args
            Additional positional arguments for the function
        **kwargs
            Additional keyword arguments for the function

        Returns
        -------
        NumberOrderedForm
            A new NumberOrderedForm with the function applied to its terms

        """
        # Create number operators for all the operators in this NumberOrderedForm
        substitutions = {
            NumberOperator(op): sympy.Symbol(f"dummy_{uuid.uuid4().hex}", real=True)
            for op in self.operators
        }
        reverse = {v: k for k, v in substitutions.items()}

        # Create a new terms dictionary for the result
        new_terms = {}

        # Process each term in the terms dictionary
        for powers, coeff in self.args[1]:
            dummy_expr = coeff.subs(substitutions)
            result_expr = func(dummy_expr, *args, **kwargs)
            result_with_n_ops = result_expr.subs(reverse)
            new_terms[powers] = result_with_n_ops

        # Create a new NumberOrderedForm with the same operators but new terms
        return type(self)(self.operators, new_terms, validate=False)

    def _eval_simplify(self, **kwargs):
        """SymPy's hook for the simplify() function.

        This allows the SymPy simplify() function to work correctly with
        NumberOrderedForm instances.

        Parameters
        ----------
        **kwargs
            Keyword arguments to pass to sympy.simplify

        Returns
        -------
        NumberOrderedForm
            A simplified NumberOrderedForm

        """
        return self.applyfunc(sympy.simplify, **kwargs)

    def __pow__(self, exp: sympy.Expr) -> "NumberOrderedForm":
        """Raise this NumberOrderedForm to a power.

        Parameters
        ----------
        exp :
            The exponent to raise this NumberOrderedForm to.

        Returns
        -------
        NumberOrderedForm
            The result of raising this NumberOrderedForm to the given power.

        Raises
        ------
        ValueError
            If trying to raise an expression with unmatched creation/annihilation operators
            to a non-integer power.
        TypeError
            If the exponent is not a valid type.

        """
        if not isinstance(exp, (int, sympy.Integer, sympy.Expr)):
            return NotImplemented

        if exp == 0:
            return type(self)(self.operators, {(): sympy.S.One})

        # For integer exponents, convert to repeated multiplication
        if (isinstance(exp, int) or exp.is_Integer) and exp > 0:
            result = self
            for _ in range(exp - 1):
                result = result * self
            return result

        # For non-integer exponents, check that the expression only has
        # number operators (no unmatched creation/annihilation operators)
        if not self.is_particle_conserving():
            if len(self.terms) > 1:
                raise ValueError(
                    f"Cannot raise expression with unmatched creation or annihilation "
                    f"operators to non-integer power: {self}**{exp}"
                )

            # One term, may exponentiate if the coefficient is commutative.
            powers, coeff = next(iter(self.terms.items()))
            if not coeff.is_commutative or exp.is_negative:
                raise ValueError(
                    f"Cannot raise expression with unmatched creation or annihilation "
                    f"operators to non-integer power: {self}**{exp}"
                )

            # Note: does not take care of fermion nilpotency.
            return type(self)(
                self.operators,
                {tuple(i * exp for i in powers): coeff**exp},
                validate=False,
            )

        # Since the expression only contains number operators, it's safe to apply the power
        # We extract the coefficient (if exists) and raise it to the given exponent.
        return type(self)(
            self.operators,
            {key: value**exp for key, value in self.args[1]},
            validate=False,
        )

    def __truediv__(self, other) -> "NumberOrderedForm":
        """Divide this NumberOrderedForm by another object."""
        if not isinstance(other, NumberOrderedForm):
            try:
                other = NumberOrderedForm.from_expr(sympy.sympify(other))
            except Exception:
                return NotImplemented

        return self * (other**-1)

    def is_particle_conserving(self) -> bool:
        """Check if this expression conserves particle numbers.

        Returns
        -------
        bool
            True if the expression has no unpaired creation or annihilation operators,
            False otherwise.

        """
        return all(not any(powers) for powers, _ in self.args[1])

    def _eval_subs(self, old, new):
        if old in self.operators or new in self.operators:
            raise ValueError("Cannot substitute operators in NumberOrderedForm.")
        return type(self)(
            self.operators,
            {tuple(powers): coeff.subs(old, new) for powers, coeff in self.args[1]},
            validate=False,
        )

    def filter_terms(
        self, conditions: tuple[tuple[sympy.core.Expr, ...], ...], keep: bool = False
    ) -> "NumberOrderedForm":
        """Filter the terms of this NumberOrderedForm based on given conditions.

        Parameters
        ----------
        conditions :
            Tuples of conditions to filter the terms. Each condition is a tuple
            containing the powers of operators, possibly symbolic, e.g. `3 + n`.
        keep :
            If True, keep the terms that satisfy any of the conditions. If False
            (default), keep the terms that do not satisfy any of the conditions.

        Returns
        -------
        NumberOrderedForm
            A new NumberOrderedForm with only the terms that do not satisfy any of the
            conditions.

        """
        new_terms = {
            powers: coeff
            for powers, coeff in self.args[1]
            if not bool(keep)
            != any(
                all(
                    # is_zero is False when it is guaranteed that a solution does not
                    # exist. This takes care of e.g. 3 - n, where n is a positive
                    # integer.
                    (power - ref).is_zero is not False
                    for power, ref in zip(powers, condition)
                )
                for condition in conditions
            )
        }
        return type(self)(self.operators, new_terms, validate=False)
