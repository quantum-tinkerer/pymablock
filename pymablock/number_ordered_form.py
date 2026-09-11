"""Implementation of NumberOrderedForm as an Operator subclass.

This module provides a class-based implementation of number ordered form for quantum operators,
which represents operators with creation operators on the left, annihilation operators on the right,
and number operators in the middle.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence

import sympy
from packaging.specifiers import SpecifierSet
from sympy.core.logic import fuzzy_and
from sympy.physics.quantum import Dagger, HermitianOperator, Operator, pauli
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

from pymablock._packed_binary import (
    IDENTITY,
    adjoint_monomial,
    multiply_monomials,
    remap_monomial,
)
from pymablock._packed_terms import (
    Layout,
    PackedTerm,
    PowerKey,
    binary_powers,
    build_packed_terms,
    decode_monomial,
    pack_terms,
    unpack_terms,
)

__all__ = [
    "NumberOperator",
    "NumberOrderedForm",
    "find_operators",
]

# SymPy stores the expression arguments; packed binary arithmetic uses Python integers.
Zero = sympy.S.Zero
One = sympy.S.One
Tuple = sympy.Tuple


def _shift_number_placeholders(
    expr: sympy.Expr,
    placeholders: tuple[sympy.Symbol, ...],
    shifts: tuple[sympy.Expr, ...],
) -> sympy.Expr:
    replacements = {
        placeholder: placeholder + shift
        for placeholder, shift in zip(placeholders, shifts)
        if shift
    }
    if not replacements:
        return expr
    return expr.xreplace(replacements)


def _boson_ladder_product(
    left_powers: tuple[int, ...],
    right_powers: tuple[int, ...],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Return powers, coefficient shifts, and pair counts for bosons and ladders.

    Positive powers denote annihilation operators, negative powers creation
    operators. Pair counts distinguish a a† from a† a; they determine both
    the coefficient shifts and the additional bosonic number factors.
    """
    result_powers = []
    left_shifts = []
    right_shifts = []
    aa_dagger_counts = []
    a_dagger_a_counts = []
    for left, right in zip(left_powers, right_powers):
        # Pair left annihilation operators with right creation operators,
        # or left creation operators with right annihilation operators.
        aa_dagger_pairs = min(max(left, 0), max(-right, 0))
        a_dagger_a_pairs = min(max(-left, 0), max(right, 0))
        result_powers.append(left + right)
        left_shifts.append(max(-right, 0) - aa_dagger_pairs - a_dagger_a_pairs)
        right_shifts.append(max(left, 0) - aa_dagger_pairs - a_dagger_a_pairs)
        aa_dagger_counts.append(aa_dagger_pairs)
        a_dagger_a_counts.append(a_dagger_a_pairs)
    return (
        tuple(result_powers),
        tuple(left_shifts),
        tuple(right_shifts),
        tuple(aa_dagger_counts),
        tuple(a_dagger_a_counts),
    )


def _boson_number_factors(
    placeholders: tuple[sympy.Symbol, ...],
    n_bosons: int,
    result_powers: tuple[int, ...],
    aa_dagger_counts: tuple[int, ...],
    a_dagger_a_counts: tuple[int, ...],
) -> tuple[sympy.Expr, sympy.Expr]:
    """Build the bosonic number factors associated with a term product."""
    aa_dagger_factor = sympy.Mul(
        *(
            placeholders[i] + abs(result_powers[i]) + step
            for i, to_pair in enumerate(aa_dagger_counts[:n_bosons])
            for step in range(1, to_pair + 1)
        )
    )
    a_dagger_a_factor = sympy.Mul(
        *(
            placeholders[i] - step
            for i, to_pair in enumerate(a_dagger_a_counts[:n_bosons])
            for step in range(to_pair)
        )
    )
    return aa_dagger_factor, a_dagger_a_factor


# Monkey patch sympy to propagate adjoint to matrix elements.
if sympy.__version__ in SpecifierSet("<1.14"):  # pragma: no cover

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
    sympy.Expr._eval_transpose = _eval_transpose  # type: ignore
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
        return Zero
    result = items[0]
    for item in items[1:]:
        result += item
    return result


sympy.polys.domains.expressionrawdomain.ExpressionRawDomain.sum = _sum  # type: ignore
del _sum

if sympy.__version__ in SpecifierSet("<1.15"):
    # Define is_annihilation on spins for API uniformity
    pauli.SigmaPlus.is_annihilation = False  # type: ignore
    pauli.SigmaMinus.is_annihilation = True  # type: ignore


class LadderOp(Operator):
    """Ladder operator on a lattice.

    Notes
    -----
    Like a `~sympy.physics.quantum.boson.BosonOp`, but creation and annihilation
    operators commute. This is useful for simulating Floquet systems. The operator
    implementation is minimal, and is meant to be used in combination with the
    `~pymablock.number_ordered_form.NumberOperator` class.

    Implementation mostly copied from `~sympy.physics.quantum.boson.BosonOp`.

    """

    is_commutative = False
    is_hermitian = False
    is_antihermitian = False

    @property
    def name(self):
        return self.args[0]

    @property
    def is_annihilation(self):
        return bool(self.args[1])

    def __new__(cls, *args, **hints):  # noqa: ARG004
        if len(args) not in [1, 2]:
            raise ValueError("1 or 2 parameters expected, got %s" % args)

        if len(args) == 1:
            args = (args[0], One)

        if len(args) == 2:
            args = (args[0], sympy.Integer(args[1]))

        return Operator.__new__(cls, *args)

    def _eval_adjoint(self):
        return type(self)(self.name, not self.is_annihilation)

    def _print_contents_latex(self, printer, *args):  # noqa: ARG002
        if self.is_annihilation:
            return r"{%s}" % str(self.name)
        return r"{{%s}^\dagger}" % str(self.name)

    def _print_contents(self, printer, *args):  # noqa: ARG002
        if self.is_annihilation:
            return r"%s" % str(self.name)
        return r"Dagger(%s)" % str(self.name)

    def _print_contents_pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm

        pform = printer._print(self.args[0], *args)
        if self.is_annihilation:
            return pform
        return pform ** prettyForm("\N{DAGGER}")


# Type aliases
operator_types = BosonOp, LadderOp, pauli.SigmaOpBase, FermionOp
OperatorType = BosonOp | LadderOp | pauli.SigmaOpBase | FermionOp
generator_types = (BosonOp, LadderOp, pauli.SigmaMinus, FermionOp)
operator_type_by_name = {sympy.Symbol(op.__name__): op for op in operator_types}
TermDict = dict[PowerKey, sympy.Expr] | tuple[tuple[PowerKey, sympy.Expr], ...] | Tuple


class NumberOperator(HermitianOperator):
    """Number operator for bosonic, fermionic, and spin operators.

    Notes
    -----
    This class is used to simplify expressions with second-quantized operators. We do
    this ourselves, because sympy does not support this yet.

    """

    @property
    def name(self) -> sympy.Symbol:
        """Return the name of the operator."""
        return self.args[0]  # type: ignore

    def __new__(cls, *args, **hints):
        """Construct a number operator for bosonic, ladder, fermionic, or spin mode.

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
                    "NumberOperator requires a bosonic, ladder, fermionic, or spin operator."
                )
            name = operator.name
            operator_type = next(
                op.__name__ for op in operator_types if isinstance(operator, op)
            )
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
        if self.args[1].name == "SigmaOpBase":
            return (pauli.SigmaZ(self.args[0]) + sympy.S.One) / sympy.S(2)
        if self.args[1].name == "LadderOp":
            return self  # No alternative form of ladder number operators.
        op = operator_type_by_name[self.args[1]](self.args[0])
        return Dagger(op) * op

    def _eval_power(self, exp):
        """Evaluate the power of the operator.

        Parameters
        ----------
        exp :
            The exponent to raise the operator to.

        Returns
        -------
        result : `~sympy.core.expr.Expr`
            The evaluated operator raised to the given power.

        """
        if (
            exp.is_integer
            and exp.is_positive
            and self.args[1].name not in ("BosonOp", "LadderOp")
        ):
            return self  # Fermionic and spin number operators are idempotent.
        return super()._eval_power(exp)

    def _eval_commutator_NumberOperator(self, other):  # noqa: ARG002
        """Evaluate the commutator with another NumberOperator."""
        return Zero

    def _eval_commutator_BosonOp(self, other, **hints):
        """Evaluate the commutator with a boson operator."""
        if self.args[1].name != "BosonOp":
            return Zero
        if other.name != self.name and hints.get("independent"):
            return Zero
        return normal_ordered_form(
            Commutator(self.doit(), other, **hints).doit(), **hints
        )

    def _eval_commutator_FermionOp(self, other, **hints):
        """Evaluate the commutator with a fermion operator."""
        if self.args[1].name != "FermionOp":
            return Zero
        if other.name != self.name and hints.get("independent"):
            return Zero
        return normal_ordered_form(
            Commutator(self.doit(), other, **hints).doit(), **hints
        )

    def _eval_commutator_SigmaX(self, other, **hints):
        """Evaluate the commutator with a SigmaX operator."""
        if self.args[1].name != "SigmaOpBase":
            return Zero
        if other.name != self.name and hints.get("independent"):
            return Zero
        return normal_ordered_form(
            Commutator(self.doit(), other, **hints).doit(), **hints
        )

    def _eval_commutator_SigmaY(self, other, **hints):
        """Evaluate the commutator with a SigmaY operator."""
        if self.args[1].name != "SigmaOpBase":
            return Zero
        if other.name != self.name and hints.get("independent"):
            return Zero
        return normal_ordered_form(
            Commutator(self.doit(), other, **hints).doit(), **hints
        )

    def _eval_commutator_SigmaZ(self, other, **hints):  # noqa: ARG002
        """Evaluate the commutator with a SigmaZ operator."""
        return Zero

    def _print_contents_latex(self, printer, *args):  # noqa: ARG002
        return r"{N_{%s}}" % str(self.name)

    def _print_contents(self, printer, *args):  # noqa: ARG002
        return r"N_%s" % str(self.name)

    def _print_contents_pretty(self, printer, *args):
        return printer._print("N_%s" % str(self.name), *args)


def _operator_sort_key(operator: OperatorType) -> tuple[int, str]:
    return generator_types.index(type(operator)), str(operator.name)


def find_operators(expr: sympy.Expr) -> list[OperatorType]:
    """Find all quantum operators in a SymPy expression.

    Parameters
    ----------
    expr :
        The expression to search for quantum operators.

    Returns
    -------
    operators : `list[OperatorType]`
        Unique annihilation operators, sorted by type (boson, ladder, spin,
        fermion) and then by name. NumberOrderedForms contribute their operator
        lists, including unused modes, without expanding their terms.

    """
    operators = set()
    traversal = sympy.preorder_traversal(expr)
    for node in traversal:
        if isinstance(node, NumberOrderedForm):
            operators.update(node.operators)
        elif isinstance(node, NumberOperator):
            particle = operator_type_by_name[node.args[1]]
            generator = generator_types[operator_types.index(particle)]
            operators.add(generator(node.name))
        elif isinstance(node, operator_types):
            generator = next(
                generator
                for particle, generator in zip(operator_types, generator_types)
                if isinstance(node, particle)
            )
            operators.add(generator(node.name))
        else:
            continue
        traversal.skip()
    return sorted(operators, key=_operator_sort_key)


def _number_operator_to_placeholder(op: NumberOperator) -> sympy.Symbol:
    """Convert a NumberOperator to its placeholder symbol."""
    return sympy.Symbol(
        f"number_operator_placeholder_{op.args[0]}_{op.args[1]}",
        integer=True,
    )


class NumberOrderedForm(Operator):
    """Number ordered form of quantum operators.

    A number ordered form represents quantum operators where:
    1. All creation operators are on the left
    2. All annihilation operators are on the right
    3. Number operators (and other scalar expressions) are in the middle

    This representation makes it easy to manipulate complex quantum expressions, because
    commuting a creation or annihilation operator through a function of a number operator
    simply replaces the corresponding number operator `N` with `N ± 1`.

    Internally each term key is ``(boson_ladder_powers, binary_monomial)``. The first
    component stores boson and ladder powers, while the second is one spin/fermion
    monomial packed into an integer. The public :attr:`terms` property decodes this
    into full power tuples for public inspection and symbolic coefficient operations.

    See the :doc:`second quantization documentation <../second_quantization>` for a
    detailed description.

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

    # Number ordered forms may represent commutative expressions.
    is_commutative = None

    # Mode counts, fermion mask, and NumberOperator placeholder symbols
    _layout: Layout
    # Mapping from placeholder symbols to NumberOperator instances for reverse lookup
    _placeholder_to_number_operator: dict[sympy.Symbol, NumberOperator]
    # Inverse mapping
    _number_operator_to_placeholder: dict[NumberOperator, sympy.Symbol]
    args: tuple[sympy.Tuple, sympy.Tuple]

    def __new__(
        cls,
        operators: Sequence[OperatorType],
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
            annihilation operators. Fermion and spin powers must be -1, 0, or 1;
            coefficients may contain :class:`NumberOperator` instances. A
            ``sympy.Tuple`` is the packed storage that SymPy passes back when
            rebuilding an instance from its ``args``.
        validate :
            Whether to validate the operators and terms, by default True.
        **hints : dict
            Additional hints passed to the parent class.

        Returns
        -------
        NumberOrderedForm
            A new NumberOrderedForm instance.

        """
        if not isinstance(operators, Tuple):
            operators = Tuple(*operators)

        if validate:
            cls._validate_operators(operators)
        layout, to_placeholder = cls._layout_for(operators)

        packed = isinstance(terms, Tuple)
        if packed:
            # SymPy rebuilds ``func(*args)`` from the packed storage. Rebuilds such
            # as xreplace may have put NumberOperator instances into coefficients.
            terms = Tuple(
                *(Tuple(key, coeff.xreplace(to_placeholder)) for key, coeff in terms)
            )
        else:
            items = terms.items() if isinstance(terms, dict) else terms
            terms = [
                (
                    tuple(map(sympy.sympify, powers)),
                    sympy.sympify(coeff).xreplace(to_placeholder),
                )
                for powers, coeff in items
            ]
            if validate:
                cls._validate_terms(terms, operators)
            terms = pack_terms(layout, terms)

        result = cls._finalize(operators, terms, layout, to_placeholder, **hints)
        if packed:
            binary_numbers = set(layout.binary_placeholders)
            if any(coeff.free_symbols & binary_numbers for _, coeff in terms):
                # Replacements may introduce binary number factors into coefficients.
                # Move them into the keys using the public constructor's expansion.
                return cls(operators, result.terms, validate=False, **hints)
        return result

    @staticmethod
    def _layout_for(
        operators: Sequence[OperatorType],
    ) -> tuple[Layout, dict[NumberOperator, sympy.Symbol]]:
        """Return the mode layout and the NumberOperator placeholder mapping."""
        to_placeholder = {
            number: _number_operator_to_placeholder(number)
            for number in map(NumberOperator, operators)
        }
        layout = Layout.from_operators(operators, tuple(to_placeholder.values()))
        return layout, to_placeholder

    @classmethod
    def _finalize(
        cls,
        operators: Tuple,
        packed_terms: Tuple,
        layout: Layout,
        to_placeholder: dict[NumberOperator, sympy.Symbol],
        **hints,
    ) -> "NumberOrderedForm":
        """Store packed terms and the derived attributes."""
        result = sympy.Expr.__new__(cls, operators, packed_terms, **hints)
        result._layout = layout
        result._number_operator_to_placeholder = to_placeholder
        result._placeholder_to_number_operator = {
            placeholder: number for number, placeholder in to_placeholder.items()
        }
        return result

    @classmethod
    def _from_packed_terms(
        cls,
        operators: Sequence[OperatorType],
        terms: Iterable[PackedTerm],
    ) -> "NumberOrderedForm":
        """Combine equal keys, discard zero coefficients, and store the terms."""
        operators = Tuple(*operators)
        layout, to_placeholder = cls._layout_for(operators)
        return cls._finalize(
            operators, build_packed_terms(layout, terms), layout, to_placeholder
        )

    @staticmethod
    def _validate_operators(operators: Sequence[OperatorType]) -> None:
        """Validate the operators list.

        Parameters
        ----------
        operators :
            List of quantum operators to validate.

        Raises
        ------
        ValueError
            If the operator order is incorrect.
        TypeError
            If an operator is not a valid quantum operator.
        ValueError
            If an operator is not an annihilation operator.

        """
        if not all(isinstance(op, generator_types) for op in operators):
            raise TypeError(
                "Operators must be BosonOp, LadderOp, SigmaMinus, or FermionOp."
            )
        if not all(op.is_annihilation for op in operators):
            raise ValueError("Operators must be annihilation operators.")

        # Confirm operator sort order
        if list(operators) != sorted(operators, key=_operator_sort_key):
            raise ValueError("Operators must be sorted by type and name.")

    @staticmethod
    def _validate_terms(
        terms: Iterable[tuple[PowerKey, sympy.Expr]], operators: Sequence[OperatorType]
    ) -> None:
        """Validate operator powers and scalar coefficients.

        Parameters
        ----------
        terms :
            Pairs of operator power tuples and coefficients, with number
            operators already replaced by placeholder symbols.
        operators :
            List of quantum operators to validate against.

        Raises
        ------
        ValueError
            If a powers tuple has incorrect length.
        TypeError
            If a power is not an integer.
        ValueError
            If a coefficient is not commutative.

        """
        for powers, coeff in terms:
            # Check that powers tuple has the right length
            if len(powers) != len(operators):
                raise ValueError(
                    f"Powers tuple length ({len(powers)}) doesn't match "
                    f"operators length ({len(operators)})"
                )

            for power in powers:
                if not power.is_integer:
                    raise TypeError(f"Power must be an integer, got {power}")

            # Check for unwanted creation or annihilation operators in the coefficient
            if not coeff.is_commutative:
                raise ValueError(f"Coefficient {coeff} must be a commutative expression.")

    @classmethod
    def from_expr(
        cls, expr: sympy.Expr, operators: Sequence[OperatorType] | None = None
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

        scalar = not expr.has(*operator_types, NumberOperator)
        if not operators:
            operators = [] if scalar else find_operators(expr)
        operators = Tuple(*operators)
        layout, _ = cls._layout_for(operators)
        zero_powers = layout.zero_powers

        if scalar:
            return cls._from_packed_terms(operators, [((zero_powers, IDENTITY), expr)])

        if isinstance(expr, sympy.Add):
            forms = [cls.from_expr(term, operators=operators) for term in expr.args]
            # Some terms may already be NumberOrderedForms with different
            # operator lists. Use one shared list before combining their terms.
            forms = forms[0]._combine_operators(*forms[1:], operators=operators)
            return cls._from_packed_terms(
                forms[0].operators,
                (term for form in forms for term in form._packed_terms),
            )

        if isinstance(expr, sympy.Mul):
            factors = iter(expr.args)
            result = cls.from_expr(next(factors), operators=operators)
            for factor in factors:
                result *= cls.from_expr(factor, operators=operators)
            return result

        # Single boson and ladder powers can be stored without multiplication.
        # Binary powers still obey the nilpotence rules in __pow__.
        exponent = One
        if isinstance(expr, sympy.Pow):
            base, exponent = expr.args
            if not (
                isinstance(base, (*generator_types, pauli.SigmaPlus))
                and exponent.is_integer
                and exponent.is_positive
            ):
                return cls.from_expr(base, operators=operators) ** exponent
            expr = base

        if isinstance(expr, sympy.Function):
            arguments = [cls.from_expr(arg, operators=operators) for arg in expr.args]
            if any(not arg.is_particle_conserving() for arg in arguments):
                raise ValueError(
                    f"Cannot apply function {expr.func} to an expression with unmatched "
                    "creation or annihilation operators."
                )
            # A general function of occupations needs the Boolean expansion
            # performed by the constructor for number-dependent coefficients.
            coefficient = expr.func(*(arg._diagonal_coefficient() for arg in arguments))
            return cls(operators, {(Zero,) * len(operators): coefficient}, validate=False)

        if isinstance(expr, (*generator_types, pauli.SigmaPlus)):
            operator = expr if expr.is_annihilation else expr.adjoint()
            if operator not in operators:
                raise ValueError(f"Operator {operator} not found in operators list")
            index = operators.index(operator)
            if index < layout.n_boson_ladder:
                powers = list(zero_powers)
                powers[index] = exponent if expr.is_annihilation else -exponent
                return cls._from_packed_terms(
                    operators, [((tuple(powers), IDENTITY), One)]
                )
            bit = 1 << (index - layout.n_boson_ladder)
            monomial = (0, 0, bit) if expr.is_annihilation else (bit, 0, 0)
            result = cls._from_packed_terms(operators, [((zero_powers, monomial), One)])
            return result if exponent == One else result**exponent

        if isinstance(expr, (pauli.SigmaX, pauli.SigmaY, pauli.SigmaZ)):
            bit = 1 << (
                operators.index(pauli.SigmaMinus(expr.name)) - layout.n_boson_ladder
            )
            if isinstance(expr, pauli.SigmaZ):
                terms = [
                    ((zero_powers, (0, bit, 0)), 2 * One),
                    ((zero_powers, IDENTITY), -One),
                ]
            else:
                coefficients = (
                    (sympy.I, -sympy.I) if isinstance(expr, pauli.SigmaY) else (One, One)
                )
                terms = [
                    ((zero_powers, (0, 0, bit)), coefficients[0]),
                    ((zero_powers, (bit, 0, 0)), coefficients[1]),
                ]
            return cls._from_packed_terms(operators, terms)

        if isinstance(expr, NumberOperator):
            index = tuple(NumberOperator(operator) for operator in operators).index(expr)
            if index < layout.n_boson_ladder:
                return cls._from_packed_terms(
                    operators,
                    [((zero_powers, IDENTITY), _number_operator_to_placeholder(expr))],
                )
            bit = 1 << (index - layout.n_boson_ladder)
            return cls._from_packed_terms(operators, [((zero_powers, (0, bit, 0)), One)])

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
            return self.args[1][0][1] if self.args[1] else Zero

        terms = []
        reversed_operators = list(reversed(self.operators))

        for powers, coeff in self.terms.items():
            # Replace any placeholders with NumberOperator instances
            coeff = coeff.xreplace(self._placeholder_to_number_operator)
            term = coeff
            for op, power in zip(reversed_operators, reversed(powers)):
                if not power > Zero:
                    continue
                # Annihilation operator (positive power)
                term = term * op**power

            for op, power in zip(reversed_operators, reversed(powers)):
                if not power < Zero:
                    continue
                # Creation operator (negative power)
                term = op.adjoint() ** (-power) * term

            terms.append(term)

        # Build the sum once to avoid repeatedly collecting and sorting its terms
        # when constructing the expression for hashing.
        return sympy.Add(*terms)

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
    def terms(self) -> dict[PowerKey, sympy.Expr]:
        """The dictionary of terms.

        Notes
        -----
        Internally, a key is ``(boson_ladder_powers, binary_monomial)``. This
        property returns full power tuples with number factors in the coefficients.

        """
        return unpack_terms(self._layout, self._packed_terms)

    @property
    def _packed_terms(self) -> Iterator[PackedTerm]:
        """Iterate over ``((boson/ladder powers, packed monomial), coefficient)``."""
        num_binary = self._layout.num_binary
        return (
            ((tuple(key[0]), decode_monomial(int(key[1]), num_binary)), coefficient)
            for key, coefficient in self.args[1]
        )

    def _operator_powers(self) -> set[PowerKey]:
        """Return distinct creation/annihilation powers, ignoring number factors."""
        return {
            (*powers, *binary_powers(monomial, self._layout.num_binary))
            for (powers, monomial), _ in self._packed_terms
        }

    def _diagonal_coefficient(self) -> sympy.Expr:
        """Combine number factors into one coefficient of a diagonal operator.

        Callers must check that there are no unpaired creation or annihilation
        operators. Number operators use the internal placeholder symbols.
        """
        numbers = self._layout.binary_placeholders
        return sympy.Add(
            *(
                coefficient
                * sympy.prod(
                    number for i, number in enumerate(numbers) if number_mask & (1 << i)
                )
                for (_, (_, number_mask, _)), coefficient in self._packed_terms
            )
        )

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

    def _expand_operators(
        self, new_operators: Sequence[OperatorType]
    ) -> "NumberOrderedForm":
        """Expand the operators in this NumberOrderedForm.

        This method creates a new NumberOrderedForm with the same terms but expanded
        operators.

        Parameters
        ----------
        new_operators :
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
        if self.operators == new_operators:
            return self

        new_operators = Tuple(*new_operators)
        new_layout, _ = self._layout_for(new_operators)
        n_boson_ladder = self._layout.n_boson_ladder
        new_boson_ladder_ops = new_operators[: new_layout.n_boson_ladder]
        boson_ladder_mapping = [
            new_boson_ladder_ops.index(op) for op in self.operators[:n_boson_ladder]
        ]
        new_binary_ops = new_operators[new_layout.n_boson_ladder :]
        binary_mapping = [
            new_binary_ops.index(op) for op in self.operators[n_boson_ladder:]
        ]

        terms = []
        for (powers, monomial), coefficient in self._packed_terms:
            new_powers = [Zero] * new_layout.n_boson_ladder
            for power, new_index in zip(powers, boson_ladder_mapping):
                new_powers[new_index] = power
            new_monomial = remap_monomial(monomial, binary_mapping)
            terms.append(((tuple(new_powers), new_monomial), coefficient))
        return type(self)._from_packed_terms(new_operators, terms)

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
        return type(self)._from_packed_terms(
            self_expanded.operators,
            (*self_expanded._packed_terms, *other_expanded._packed_terms),
        )

    def _combine_operators(
        self,
        *others: "NumberOrderedForm",
        operators: Sequence[OperatorType] | None = None,
    ) -> tuple["NumberOrderedForm", ...]:
        """Give all forms a shared operator list, also including supplied operators."""
        forms = (self, *others)
        if operators is None:
            operators = self.operators
        if any(form.operators != operators for form in forms):
            operators = Tuple(
                *sorted(
                    set(operators).union(*(form.operators for form in forms)),
                    key=_operator_sort_key,
                )
            )
        return tuple(form._expand_operators(operators) for form in forms)

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
        return self.applyfunc(lambda coefficient: -coefficient)

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
            try:
                other = sympy.sympify(other)
                if other.is_commutative:
                    return self.applyfunc(lambda coefficient: coefficient * other)
                other = NumberOrderedForm.from_expr(other)
            except Exception:
                return NotImplemented

        self_expanded, other_expanded = self._combine_operators(other)
        layout = self_expanded._layout
        placeholders = layout.boson_ladder_placeholders
        result_terms: list[PackedTerm] = []
        right_terms = tuple(other_expanded._packed_terms)

        for (left_powers, left_binary), left_coefficient in self_expanded._packed_terms:
            for right_key, right_coefficient in right_terms:
                right_powers, right_binary = right_key
                binary_product = multiply_monomials(
                    left_binary, right_binary, layout.fermion_mask
                )
                if not binary_product:
                    continue
                (
                    result_powers,
                    left_coefficient_shift,
                    right_coefficient_shift,
                    aa_dagger_counts,
                    a_dagger_a_counts,
                ) = _boson_ladder_product(left_powers, right_powers)
                aa_dagger_factor, a_dagger_a_factor = _boson_number_factors(
                    placeholders,
                    layout.n_bosons,
                    result_powers,
                    aa_dagger_counts,
                    a_dagger_a_counts,
                )
                coefficient = _shift_number_placeholders(
                    left_coefficient,
                    placeholders,
                    left_coefficient_shift,
                )
                coefficient *= aa_dagger_factor * a_dagger_a_factor
                coefficient *= _shift_number_placeholders(
                    right_coefficient,
                    placeholders,
                    right_coefficient_shift,
                )
                for binary, sign in binary_product.items():
                    result_terms.append(((result_powers, binary), coefficient * sign))

        return type(self)._from_packed_terms(self_expanded.operators, result_terms)

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
            other = sympy.sympify(other)
            if other.is_commutative:
                return self.applyfunc(lambda coefficient: other * coefficient)
            other_nof = NumberOrderedForm.from_expr(other)
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
        terms = []
        for (powers, monomial), coefficient in self._packed_terms:
            adjoint, sign = adjoint_monomial(monomial, self._layout.fermion_mask)
            terms.append(
                (
                    (tuple(-power for power in powers), adjoint),
                    sympy.conjugate(coefficient) * sign,
                )
            )
        return type(self)._from_packed_terms(self.operators, terms)

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
        if not self or not other:
            return bool(self) == bool(other)
        if self.operators != other.operators:
            self, other = self._combine_operators(other)
        if self.args[1] == other.args[1]:
            return True
        return self.terms == other.terms

    def __hash__(self):
        """Compute the hash of this NumberOrderedForm."""
        # Equality ignores unused operators and accepts equivalent SymPy
        # expressions, so hash the represented expression rather than args.
        # _mhash is the hash cache inherited from sympy.Basic, initialized to
        # None. As in Basic.__hash__, construct and hash the expression only
        # on the first call; subsequent calls reuse the cached integer.
        cached_hash = self._mhash
        if cached_hash is None:
            cached_hash = hash(self.as_expr())
            self._mhash = cached_hash
        return cached_hash

    def _eval_is_zero(self):
        """Check if this NumberOrderedForm is zero.

        This method is used by SymPy to determine if an expression is zero.

        Returns
        -------
        bool or None
            True if zero, False if non-zero, None if undetermined.

        """
        return fuzzy_and(coeff.is_zero for _, coeff in self.args[1])

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
        """Apply a function to each stored coefficient, keeping the term keys.

        Notes
        -----
        Fermion and spin number factors belong to the packed keys. Boson and
        ladder number operators in coefficients use integer placeholder symbols.
        This method transforms coefficients independently; it does not evaluate
        a function of the complete operator.

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
            A new NumberOrderedForm with the function applied to each stored coefficient

        """
        return type(self)._from_packed_terms(
            self.operators,
            (
                (key, func(coefficient, *args, **kwargs))
                for key, coefficient in self._packed_terms
            ),
        )

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
        exp = sympy.sympify(exp)

        # Terms with the same creation and annihilation powers are nilpotent
        # if a binary mode is unpaired, even with different number factors.
        if self._layout.num_binary and exp.is_integer and (exp - 1).is_positive:
            operator_powers = self._operator_powers()
            if len(operator_powers) == 1 and any(
                next(iter(operator_powers))[self._layout.n_boson_ladder :]
            ):
                return type(self)._from_packed_terms(self.operators, ())

        # Positive symbolic bosonic powers are used as selective masks.
        if (
            exp.is_integer
            and exp.is_nonnegative
            and not exp.is_Integer
            and len(self.args[1]) == 1
        ):
            (powers, monomial), coeff = next(iter(self._packed_terms))
            if (
                any(powers)
                and monomial == IDENTITY
                and not coeff.has(*self._layout.placeholders)
            ):
                return type(self)._from_packed_terms(
                    self.operators,
                    [((tuple(power * exp for power in powers), IDENTITY), coeff**exp)],
                )

        if not self.is_particle_conserving() and not (
            exp.is_Integer and exp.is_nonnegative
        ):
            raise ValueError(
                "Expressions with unmatched creation or annihilation operators require a "
                "non-negative integer power."
            )

        if exp == 0:
            return type(self)._from_packed_terms(
                self.operators, [((self._layout.zero_powers, IDENTITY), One)]
            )

        # For integer exponents, convert to repeated multiplication
        if (isinstance(exp, int) or exp.is_Integer) and exp > 0:
            result = self
            for _ in range(exp - 1):
                result = result * self
            return result

        # Particle-conserving expressions contain only number operators.
        coefficient = self._diagonal_coefficient()
        return type(self)(
            self.operators,
            {(0,) * len(self.operators): coefficient**exp},
            validate=False,
        )

    def __truediv__(self, other) -> "NumberOrderedForm":
        """Divide this NumberOrderedForm by another object."""
        if not isinstance(other, NumberOrderedForm):
            try:
                other = NumberOrderedForm.from_expr(sympy.sympify(other))
            except Exception:
                return NotImplemented

        return self * (other**-One)

    def is_particle_conserving(self) -> bool:
        """Check if this expression conserves particle numbers.

        Returns
        -------
        bool
            True if the expression has no unpaired creation or annihilation operators,
            False otherwise.

        """
        return not any(
            any(powers) or creation or annihilation
            for (powers, (creation, _, annihilation)), _ in self._packed_terms
        )

    def _eval_subs(self, old, new):
        if old in self.operators or new in self.operators:
            raise ValueError("Cannot substitute operators in NumberOrderedForm.")

        old = old.xreplace(self._number_operator_to_placeholder)
        new = new.xreplace(self._number_operator_to_placeholder)

        # Scalar symbol substitutions leave the packed operator factors intact.
        if (
            old.is_Symbol
            and old.is_commutative
            and new.is_commutative
            and not old.has(*self._layout.placeholders)
            and not new.has(NumberOperator, *self._layout.placeholders)
        ):
            return self.applyfunc(lambda coefficient: coefficient.subs(old, new))

        return type(self)(
            self.operators,
            {powers: coeff.subs(old, new) for powers, coeff in self.terms.items()},
            validate=False,
        )

    def filter_terms(
        self, conditions: "tuple[PowerKey, ...] | NumberOrderedForm", keep: bool = False
    ) -> "NumberOrderedForm":
        """Filter the terms of this NumberOrderedForm based on given conditions.

        Parameters
        ----------
        conditions :
            Power tuples, or a NumberOrderedForm used as a mask. Powers may be
            symbolic, e.g. `3 + n`. For a NumberOrderedForm mask, the operator
            lists are aligned automatically; coefficients and number factors
            are ignored when matching creation and annihilation powers.
        keep :
            If True, keep the terms that satisfy any of the conditions. If False
            (default), keep the terms that do not satisfy any of the conditions.

        Returns
        -------
        NumberOrderedForm
            A new NumberOrderedForm with the matching terms kept or discarded
            according to ``keep``.

        """
        if isinstance(conditions, NumberOrderedForm):
            self, mask = self._combine_operators(conditions)
            conditions = tuple(mask._operator_powers())

        new_terms = []
        for key, coeff in self._packed_terms:
            boson_ladder_powers, monomial = key
            powers = (
                *boson_ladder_powers,
                *binary_powers(monomial, self._layout.num_binary),
            )
            matches = any(
                all(
                    # is_zero is False when it is guaranteed that a solution does not
                    # exist. This takes care of e.g. 3 - n, where n is a positive
                    # integer.
                    (power - ref).is_zero is not False
                    for power, ref in zip(powers, condition)
                )
                for condition in conditions
            )
            if matches == bool(keep):
                new_terms.append((key, coeff))
        return type(self)._from_packed_terms(self.operators, new_terms)

    def _poly_simplify(self) -> "NumberOrderedForm":
        """Simplify a NumberOrderedForm by converting it to polynomials and back.

        This uses as generators all possible expressions sympy finds, except for the
        number operators.
        """
        if not self.operators:
            # No operators, nothing to simplify
            return self

        new_terms = []
        for key, coeff in self._packed_terms:
            if not coeff.free_symbols:
                # If the coefficient is a constant, just keep it as is
                new_terms.append((key, coeff))
                continue

            # Convert the coefficient to a polynomial and extract the generators
            poly = sympy.poly(coeff)
            non_number_gens = tuple(
                gen for gen in poly.gens if gen not in self._layout.placeholders
            )
            if not non_number_gens:
                # If there are no non-number operator generators, keep the coefficient as is
                new_terms.append((key, coeff))
                continue

            # EXRAW keeps number factors such as (N + 1)**5 unexpanded.
            polynomial = sympy.poly(coeff, gens=non_number_gens, domain=sympy.EXRAW)
            simplified_terms = {
                term: sympy.collect_const(sympy.simplify(term_coeff)).doit()
                for term, term_coeff in polynomial.as_dict().items()
            }
            simplified_coefficient = sympy.Poly.from_dict(
                simplified_terms, gens=non_number_gens, domain=sympy.EXRAW
            ).as_expr()
            new_terms.append((key, simplified_coefficient))

        return type(self)._from_packed_terms(self.operators, new_terms)
