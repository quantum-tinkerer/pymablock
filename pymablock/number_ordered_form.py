"""Implementation of NumberOrderedForm as an Operator subclass.

This module provides a class-based implementation of number ordered form for quantum operators,
which represents operators with creation operators on the left, annihilation operators on the right,
and number operators in the middle.
"""

import uuid
from collections import defaultdict
from typing import Dict, List, Tuple

import sympy
from sympy.physics.quantum import Dagger, HermitianOperator, Operator, boson, fermion
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

# Type aliases
operator_types = BosonOp, FermionOp
OperatorType = BosonOp | FermionOp
PowerKey = Tuple[int, ...]
TermDict = Dict[PowerKey, sympy.Expr]


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


def _find_operators(expr: sympy.Expr) -> List[OperatorType]:
    """Find all quantum operators in a SymPy expression.

    Parameters
    ----------
    expr :
        The expression to search for quantum operators.

    Returns
    -------
    List[OperatorType]
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

    Parameters
    ----------
    operators : List[OperatorType]
        List of quantum operators (annihilation operators only).
    terms : Dict[Tuple[int, ...], sympy.Expr]
        Dictionary mapping operator power tuples to coefficient expressions.
        Negative powers represent creation operators, positive powers represent
        annihilation operators.
    **hints : dict
        Additional hints passed to the parent class.

    Attributes
    ----------
    operators : List[OperatorType]
        Sorted list of quantum operators (bosons before fermions).
    terms : Dict[Tuple[int, ...], sympy.Expr]
        Dictionary mapping operator power tuples to coefficient expressions.

    """

    # Same as dense matrices
    _op_priority = 10.01
    _class_priority = 4

    def __new__(
        cls,
        operators: List[OperatorType],
        terms: TermDict,
        *,
        validate: bool = True,
        **hints,
    ):
        """Create a new NumberOrderedForm instance.

        Parameters
        ----------
        operators : List[OperatorType]
            List of quantum operators (annihilation operators only).
        terms : Dict[Tuple[int, ...], sympy.Expr]
            Dictionary mapping operator power tuples to coefficient expressions.
            Negative powers represent creation operators, positive powers represent
            annihilation operators.
        validate : bool, optional
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

        # Create the new object
        result = Operator.__new__(cls, operators, terms, **hints)
        result._n_fermions = sum(isinstance(op, FermionOp) for op in operators)
        return result

    @staticmethod
    def _validate_operators(operators: List[OperatorType]) -> None:
        """Validate the operators list.

        Parameters
        ----------
        operators : List[OperatorType]
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
    def _validate_terms(terms: TermDict, operators: List[OperatorType]) -> None:
        """Validate the terms dictionary.

        Parameters
        ----------
        terms : Dict[Tuple[int, ...], sympy.Expr]
            Dictionary mapping operator power tuples to coefficient expressions.
        operators : List[OperatorType]
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
        if not terms:
            raise ValueError("Empty terms dictionary")

        for powers, coeff in terms.items():
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
            for op in operators:
                if coeff.has(op) or coeff.has(Dagger(op)):
                    raise ValueError(
                        f"Coefficient contains creation or annihilation operators: {coeff}"
                    )

    @classmethod
    def from_expr(cls, expr: sympy.Expr, operators=None) -> "NumberOrderedForm":
        """Create a NumberOrderedForm instance from a sympy expression.

        Parameters
        ----------
        expr : sympy.Expr
            Sympy expression with quantum operators.
        operators : List[OperatorType], optional
            List of quantum operators to use. If None, operators will be extracted from
            the expression.

        Returns
        -------
        NumberOrderedForm
            A NumberOrderedForm instance representing the expression.

        """
        # For scalar expressions (no operators)
        if not expr.has(BosonOp, FermionOp, NumberOperator):
            # Return a NumberOrderedForm with no operators and a single term
            return cls([], {(): expr}, validate=False)

        if not operators:
            operators = _find_operators(expr)

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

            # Convert base to NumberOrderedForm
            base_nof = NumberOrderedForm.from_expr(base, operators=operators)

            # For integer exponents, convert to repeated multiplication
            if exp.is_Integer and exp.is_positive:
                result = base_nof
                for _ in range(int(exp) - 1):
                    result = result * base_nof
                return result

            # For non-integer exponents, check that the base has only
            # number operators (no unmatched creation/annihilation operators)
            zero_key = tuple(0 for _ in base_nof.operators)

            # Check if there are any keys other than the zero key
            if not all(powers == zero_key for powers in base_nof.terms):
                raise ValueError(
                    f"Cannot raise expression with unmatched creation or annihilation "
                    f"operators to non-integer power: {base}**{exp}"
                )

            # Since the base only contains number operators, it's safe to apply the power
            # We extract the coefficient and raise it to the given exponent
            return cls(operators, {zero_key: next(iter(base_nof.terms.values())) ** exp})

        # Handle function calls (like exp, sin, etc.)
        if isinstance(expr, sympy.Function):
            # Convert each argument to NumberOrderedForm
            arg_nofs = [
                NumberOrderedForm.from_expr(arg, operators=operators) for arg in expr.args
            ]

            # Check that each argument has only number operators (no unmatched creation/annihilation operators)
            for i, arg_nof in enumerate(arg_nofs):
                zero_key = tuple(0 for _ in arg_nof.operators)
                if not all(powers == zero_key for powers in arg_nof.terms):
                    raise ValueError(
                        f"Cannot apply function {expr.func} to expression with unmatched "
                        f"creation or annihilation operators: {expr.args[i]}"
                    )

            # Now we can safely convert the arguments to expressions
            # Extract the coefficients from the zero keys for each argument
            zero_key = tuple(0 for _ in operators)
            arg_exprs = [next(iter(arg_nof.terms.values())) for arg_nof in arg_nofs]

            # Return a new NumberOrderedForm with the function applied to the coefficients
            return cls(operators, {zero_key: expr.func(*arg_exprs)})

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
            return cls(operators, {powers: sympy.S.One})

        # Handle NumberOperator
        if isinstance(expr, NumberOperator):
            # Number operator N_a should be represented as a scalar term
            # with no explicit creation or annihilation operators
            powers = tuple(0 for _ in range(len(operators)))
            coeff = expr  # Keep the NumberOperator as the coefficient

            return cls(operators, {powers: coeff})

        # If we've reached this point, we don't know how to handle this expression type
        raise ValueError(
            f"Cannot convert expression of type {type(expr)} to NumberOrderedForm: {expr}"
        )

    def as_expr(self) -> sympy.Expr:
        """Convert the NumberOrderedForm to a standard SymPy expression.

        Returns
        -------
        sympy.Expr
            A standard SymPy expression equivalent to this NumberOrderedForm.

        """
        if not self.operators:
            # If there are no operators, just return the constant term
            return next(iter(self.terms.values())) if self.terms else sympy.S.Zero

        result = sympy.S.Zero
        reversed_operators = list(reversed(self.operators))

        for powers, coeff in self.terms.items():
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
                term = Dagger(op) ** (-power) * term

            result += term

        return result

    @property
    def operators(self) -> List[OperatorType]:
        """Get the list of operators.

        Returns
        -------
        List[OperatorType]
            The list of operators.

        """
        return list(self.args[0])

    @property
    def terms(self) -> TermDict:
        """Get the dictionary of terms.

        Returns
        -------
        Dict[Tuple[int, ...], sympy.Expr]
            The dictionary of terms.

        """
        return dict(self.args[1])

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

        if op_power == 0:
            return self  # Multiplying by op^0 = 1 doesn't change anything

        operator = self.operators[op_index]
        if isinstance(operator, FermionOp):
            raise NotImplementedError
        n_operator = NumberOperator(operator)

        # Create a new terms dictionary for the result
        new_terms = {}

        for powers, coeff in self.terms.items():
            orig_power = powers[op_index]  # Power of the operator at op_index
            new_power = orig_power + op_power
            new_powers = tuple(
                new_power if i == op_index else p for i, p in enumerate(powers)
            )
            if op_power > 0:  # Multiplying by an annihilation operator
                # Compute how many new number operators appear
                to_pair = min(op_power, max(-orig_power, 0))
                for _ in range(to_pair):
                    coeff = coeff.subs(n_operator, n_operator - 1) * n_operator
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
        return type(self)(self.operators, new_terms)

    def _multiply_expr(self, expr):
        """Multiply by an expression without creation or annihilation operators.

        Parameters
        ----------
        expr : sympy.Expr
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
        if expr.has(*self.operators) or expr.has(*(Dagger(op) for op in self.operators)):
            raise ValueError(
                "Expression contains creation or annihilation operators, "
                "which cannot be multiplied directly."
            )

        new_terms = {}
        for powers, coeff in self.terms.items():
            multiplier = expr
            for i, power in enumerate(powers):
                if power < 0:
                    continue
                n_i = NumberOperator(self.operators[i])
                multiplier = multiplier.subs(n_i, n_i + power)
            new_terms[powers] = coeff * multiplier

        # Return a new NumberOrderedForm instance with the updated terms
        return type(self)(self.operators, new_terms)

    def _expand_operators(self, new_operators: List[OperatorType]) -> "NumberOrderedForm":
        """Expand the operators in this NumberOrderedForm.

        This method creates a new NumberOrderedForm with the same terms but expanded
        operators.

        Parameters
        ----------
        new_operators : List[OperatorType]
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
            for powers, coeff in self.terms.items()
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

        new_terms = defaultdict(lambda: sympy.S.Zero)
        for powers, coeff in self_expanded.terms.items():
            new_terms[powers] += coeff
        for powers, coeff in other_expanded.terms.items():
            new_terms[powers] += coeff
        return type(self)(self_expanded.operators, dict(new_terms))

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

        result = type(self)(self_expanded.operators, {}, validate=False)
        for powers, coeff in other_expanded.terms.items():
            # First multiply by creation operators, those are with negative powers
            partial = NumberOrderedForm(self_expanded.operators, self_expanded.terms)
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
        print("Multiplying by", other)
        try:
            other_nof = NumberOrderedForm.from_expr(sympy.sympify(other))
            return other_nof * self
        except Exception:
            return NotImplemented

    def _eval_Eq(self, other):
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
        if isinstance(other, NumberOrderedForm):
            return (
                all(op1 == op2 for op1, op2 in zip(self.operators, other.operators))
                and self.terms == other.terms
            )
        return None  # Let SymPy handle other cases

    def _eval_is_zero(self):
        """Check if this NumberOrderedForm is zero.

        This method is used by SymPy to determine if an expression is zero.

        Returns
        -------
        bool or None
            True if zero, False if non-zero, None if undetermined.

        """
        if not self.terms:
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
        return bool(self.terms)

    def apply_sympy_func(self, func, *args, **kwargs):
        """Apply a SymPy function to the terms of this NumberOrderedForm.

        This method temporarily replaces NumberOperators with unique symbols,
        applies the function, then substitutes back the NumberOperators.

        Parameters
        ----------
        func : callable
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
        for powers, coeff in self.terms.items():
            dummy_expr = coeff.subs(substitutions)
            result_expr = func(dummy_expr, *args, **kwargs)
            result_with_n_ops = result_expr.subs(reverse)
            new_terms[powers] = result_with_n_ops

        # Create a new NumberOrderedForm with the same operators but new terms
        return type(self)(self.operators, new_terms)

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
        return self.apply_sympy_func(sympy.simplify, **kwargs)
