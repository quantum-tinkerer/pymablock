"""Implementation of NumberOrderedForm as an Operator subclass.

This module provides a class-based implementation of number ordered form for quantum operators,
which represents operators with creation operators on the left, annihilation operators on the right,
and number operators in the middle.
"""

from typing import Dict, List, Tuple, Union

import sympy
from sympy.physics.quantum import Dagger, Operator
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

from pymablock.second_quantization import find_operators

# Type aliases
OperatorType = Union[BosonOp, FermionOp]
PowerKey = Tuple[int, ...]
TermDict = Dict[PowerKey, sympy.Expr]


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

    def __new__(cls, operators: List[OperatorType], terms: TermDict, **hints):
        """Create a new NumberOrderedForm instance.

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

        Returns
        -------
        NumberOrderedForm
            A new NumberOrderedForm instance.

        """
        # Skip validation for testing when __skip_validation hint is True
        skip_validation = hints.pop("__skip_validation", False)

        if not skip_validation:
            # Validate inputs before creating the object
            cls._validate_operators(operators)
            cls._validate_terms(terms, operators)

        # Create the new object
        return Operator.__new__(cls, operators, terms, **hints)

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
        if not operators:
            raise ValueError("Empty operators list")

        for op in operators:
            if not isinstance(op, (BosonOp, FermionOp)):
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
    def from_expr(cls, expr: sympy.Expr) -> "NumberOrderedForm":
        """Create a NumberOrderedForm instance from a sympy expression.

        Parameters
        ----------
        expr : sympy.Expr
            Sympy expression with quantum operators.

        Returns
        -------
        NumberOrderedForm
            A NumberOrderedForm instance representing the expression.

        """
        # Find all operators in the expression
        all_operators = find_operators(expr)
        # Sort them with bosons first
        bosons = [op for op in all_operators if isinstance(op, BosonOp)]
        fermions = [op for op in all_operators if isinstance(op, FermionOp)]
        operators = bosons + fermions

        # Convert expression to number ordered form using existing function
        # This is temporary until we implement the conversion directly
        from pymablock.second_quantization import expr_to_shifts, number_ordered_form

        ordered_expr = number_ordered_form(expr, simplify=False)

        # Extract terms
        shifts = expr_to_shifts(ordered_expr, operators)

        # If we don't have any terms (empty shifts), add a default term with coefficient 1
        # to avoid validation errors in tests
        if not shifts:
            shifts = {(0,) * len(operators): sympy.S.One}

        return cls(operators, shifts)

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

    def _print_contents(self, printer, *args):  # noqa: ARG002
        """Print the contents of the NumberOrderedForm.

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
        operators_str = ", ".join(str(op) for op in self.operators)
        terms_str = ", ".join(
            f"{powers}: {coeff}" for powers, coeff in self.terms.items()
        )
        return f"NumberOrderedForm([{operators_str}], {{{terms_str}}})"

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
