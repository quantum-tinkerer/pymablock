"""Implementation of NumberOrderedForm as an Operator subclass.

This module provides a class-based implementation of number ordered form for quantum operators,
which represents operators with creation operators on the left, annihilation operators on the right,
and number operators in the middle.
"""

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING

import sympy
from packaging.specifiers import SpecifierSet
from sympy.core.logic import fuzzy_and
from sympy.functions.elementary.piecewise import ExprCondPair, Piecewise
from sympy.physics.quantum import Dagger, HermitianOperator, Operator, pauli
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

if TYPE_CHECKING:
    import pymablock.operator_embedding

__all__ = [
    "NumberOperator",
    "NumberOrderedForm",
    "find_operators",
]

# To avoid back-and-forth conversion between sympy and pure Python types, this module
# tries to use sympy types as much as possible. The resulting code is unfortunately more
# verbose.
Zero = sympy.S.Zero
One = sympy.S.One
Tuple = sympy.Tuple


# SymPy infers Piecewise commutativity from branches alone. Include condition
# operands, respecting scalar wrappers and unknown assumptions. Keep branch-only
# scalar assumptions from overriding this inference, regardless of query order.
def _condition_commutativity(condition: sympy.logic.boolalg.Boolean) -> bool | None:
    """Combine operand commutativity across a Boolean condition.

    Inspect each expression as a whole rather than overriding its assumptions
    with those of its children. Return None when commutativity is unknown.
    """
    traversal = sympy.preorder_traversal(condition)
    values = []
    for node in traversal:
        if isinstance(node, sympy.Expr):
            values.append(node.is_commutative)
            traversal.skip()
    return fuzzy_and(values)


def _piecewise_commutative(self: ExprCondPair) -> bool | None:
    """Include the condition when determining a Piecewise branch's commutativity."""
    return fuzzy_and((self.expr.is_commutative, _condition_commutativity(self.cond)))


_piecewise_original_attribute = getattr(
    Piecewise, "_pymablock_original_attribute", Piecewise._eval_template_is_attr
)


def _piecewise_attribute(self: Piecewise, name: str) -> bool | None:
    """Avoid inferring scalar assumptions from branches with operator conditions."""
    # Scalar branch assumptions must not override condition dependence.
    if fuzzy_and(_condition_commutativity(c) for _, c in self.args) is not True:
        return None
    return _piecewise_original_attribute(self, name)


def _install_piecewise_patch() -> None:
    """Install once, only while upstream ignores operator-valued conditions."""
    if getattr(Piecewise, "_pymablock_condition_patch", False):
        return
    probe = Piecewise((1, sympy.Eq(Operator("_probe"), 0)), (0, True))
    if probe.is_commutative is False:
        return
    ExprCondPair.is_commutative = property(_piecewise_commutative)
    Piecewise._pymablock_original_attribute = _piecewise_original_attribute
    Piecewise._eval_template_is_attr = _piecewise_attribute
    Piecewise._pymablock_condition_patch = True


_install_piecewise_patch()


# TODO: reimplement once https://github.com/sympy/sympy/issues/27385 is fixed.
# Monkey patch sympy to override the sum method to ExpressionRawDomain.
def _sum_sequentially(
    _domain: sympy.polys.domains.expressionrawdomain.ExpressionRawDomain,
    items: Sequence[sympy.Expr],
) -> sympy.Expr:
    """Sum from left to right so operator subclasses can handle each addition."""
    if not items:
        return sympy.S.Zero
    result = items[0]
    for item in items[1:]:
        result += item
    return result


sympy.polys.domains.expressionrawdomain.ExpressionRawDomain.sum = _sum_sequentially  # type: ignore
del _sum_sequentially

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

    def __new__(cls, *args, **_hints):
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
PowerKey = tuple[int | sympy.Integer, ...]
TermDict = dict[PowerKey, sympy.Expr] | tuple[tuple[PowerKey, sympy.Expr], ...] | Tuple


def _operator_sort_key(operator: OperatorType) -> tuple[int, str]:
    """Order modes by algebra type, then name; fermion signs follow this order."""
    return generator_types.index(type(operator)), str(operator.name)


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


def find_operators(expr: sympy.Expr) -> list[OperatorType]:
    """Find all quantum operators in a SymPy expression.

    Parameters
    ----------
    expr :
        The expression to search for quantum operators.

    Returns
    -------
    operators : `list[OperatorType]`
        A list of unique quantum operators found in the expression. Boson and spin
        operators are listed before fermion operators and both are sorted by their
        names. For number-ordered forms, this includes unused declared modes.

    """
    # NOFs already include their declared operators in their SymPy arguments.
    return sorted(
        set().union(
            (
                generator(atom.name)
                for particle, generator in zip(operator_types, generator_types)
                for atom in expr.atoms(particle)
            ),
            (
                generator_types[
                    operator_types.index(operator_type_by_name[atom.args[1]])
                ](atom.name)
                for atom in expr.atoms(NumberOperator)
            ),
        ),
        key=_operator_sort_key,
    )


def _number_operator_to_placeholder(op: NumberOperator) -> sympy.Symbol:
    """Convert a NumberOperator to its placeholder symbol."""
    # Do not assume nonnegative: normal ordering replaces n by n-k. For example,
    # a guard at n=-1 in f(n) becomes a vacuum guard in a† f(N) a = N f(N-1).
    # A nonnegative assumption would discard the original guard before shifting,
    # allowing a pole in f(N-1) to cancel the zero ladder factor at the vacuum.
    return sympy.Symbol(
        f"number_operator_placeholder_{op.args[0]}_{op.args[1]}",
        integer=True,
    )


def _is_singular(expression: sympy.Expr) -> bool:
    """Return whether an evaluated expression contains an infinity or NaN."""
    return expression.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo)


def _equal_value_indicator(left: sympy.Expr, right: sympy.Expr | int) -> sympy.Expr:
    """Return the scalar indicator of ``left == right`` (1 if equal, else 0).

    Inputs are scalar occupation expressions or values, not quantum operators.
    A symbolic condition is represented by a two-branch SymPy Piecewise.
    """
    return sympy.Piecewise((1, sympy.Eq(left, right)), (0, True))


def _allowed_values_indicator(
    expression: sympy.Expr, allowed_values: Iterable[int | sympy.Expr] | sympy.Set
) -> sympy.Expr:
    """Return 1 when a scalar expression lies in the allowed set, else 0.

    ``allowed_values`` is either a finite collection of distinct values or
    ``sympy.S.Integers``. Integer membership is expressed by ``x == floor(x)``.
    This scalar coefficient can be used to construct a diagonal quantum projector.
    """
    if allowed_values is sympy.S.Integers:
        return _equal_value_indicator(expression, sympy.floor(expression))
    return sum(_equal_value_indicator(expression, value) for value in allowed_values)


def _iter_fixed_number_indicators(
    expression: sympy.Expr, number_symbols: Collection[sympy.Symbol]
) -> Iterator[tuple[sympy.Piecewise, sympy.Symbol, sympy.Expr]]:
    """Find 0/1 Piecewise indicators that fix one number symbol to a value.

    Yield ``(indicator, number_symbol, selected_value)`` for equalities linear
    in a single supplied symbol with a nonzero numeric slope. Other conditional
    expressions are ignored; no general equation solving is attempted.
    """
    for delta in expression.atoms(sympy.Piecewise) if number_symbols else ():
        if (
            len(delta.args) != 2
            or delta.args[0].expr != 1
            or delta.args[1] != (0, sympy.true)
            or not isinstance(delta.args[0].cond, sympy.Equality)
        ):
            continue
        variables = set(number_symbols) & delta.free_symbols
        if len(variables) != 1:
            continue
        (n,) = variables
        equation = sympy.expand(delta.args[0].cond.lhs - delta.args[0].cond.rhs)
        slope = equation.coeff(n)
        if slope.is_number and slope and not (equation - slope * n).has(n):
            yield delta, n, sympy.cancel(n - equation / slope)


@lru_cache(maxsize=1024)
def _simplify_on_fixed_numbers(
    coefficient: sympy.Expr,
    number_symbols: tuple[sympy.Symbol, ...],
    nonnegative: tuple[sympy.Symbol, ...],
) -> sympy.Expr:
    """Simplify a scalar coefficient where fixed-number indicators equal 1.

    Discard indicators selecting noninteger values or negative values for symbols
    listed in ``nonnegative``. For each remaining numeric selection ``n == v``,
    evaluate the coefficient at ``n = v`` while preserving its value elsewhere.
    Factor structurally equal correction weights when the background is free of
    that occupation, preserving per-mode selections without expanding coefficients.
    Leave it unsplit if a selected value is a pole, so undefined values cannot
    corrupt other sectors. The bounded cache stores scalar expressions only.
    """
    replacements, points = {}, {}
    for delta, n, value in _iter_fixed_number_indicators(coefficient, number_symbols):
        if not value.is_number:
            continue
        if value.is_integer is False or (n in nonnegative and value < 0):
            replacements[delta] = sympy.S.Zero
        else:
            replacements[delta] = _equal_value_indicator(n, value)
            points.setdefault(n, set()).add(value)
    coefficient = coefficient.xreplace(replacements)
    for n, values in sorted(
        points.items(), key=lambda item: sympy.default_sort_key(item[0])
    ):
        indicators = {_equal_value_indicator(n, v) for v in values}
        inactive = dict.fromkeys(indicators, sympy.S.Zero)
        background = coefficient.xreplace(inactive)
        # Keep a pole inside its original coefficient rather than spreading
        # undefined point values to the other occupation sectors.
        if any(
            _is_singular(expression.xreplace({n: v}))
            for expression in (coefficient, background)
            for v in values
        ):
            continue
        weights = {
            _equal_value_indicator(n, v): (
                coefficient.xreplace({n: v}) - background.xreplace({n: v})
            )
            for v in sorted(values, key=sympy.default_sort_key)
        }
        common = next(iter(weights.values()))
        # Extract weights even for affine sums so undistributed zeros cancel.
        # Equal weights can stay factored over all spectator modes. A background
        # depending on n can contain new indicators that need the old sum to cancel.
        coefficient = background + (
            common * sum(indicators)
            if not background.has(n)
            and all(weight == common for weight in weights.values())
            else sum(delta * weight for delta, weight in weights.items())
        )
    return coefficient


class NumberOrderedForm(Operator):
    """Number ordered form of quantum operators.

    A number ordered form represents quantum operators where:
    1. All creation operators are on the left
    2. All annihilation operators are on the right
    3. Number operators (and other scalar expressions) are in the middle

    This representation makes it easy to manipulate complex quantum expressions, because
    commuting a creation or annihilation operator through a function of a number operator
    simply replaces the corresponding number operator `N` with `N ± 1`.

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
    embedding, side :
        Optional fixed isometry and its side: +1 represents X W, -1 represents W† X.
        Omitting the embedding gives an ordinary operator X.
    **hints : dict
        Additional hints passed to the parent class.

    """

    # Same as dense matrices
    _op_priority = 10.01
    _class_priority = 4

    # Number ordered forms may represent commutative expressions.
    is_commutative = None

    # Attribute types
    _n_bosons: int
    _n_ladders: int
    # Number of infinite order operators (bosons + ladders)
    _n_inf_order: int
    _n_spins: int
    _n_fermions: int
    # List of placeholder symbols for NumberOperator instances, ordered like operators
    _number_operator_placeholders: list[sympy.Symbol]
    # Mapping from placeholder symbols to NumberOperator instances for reverse lookup
    _placeholder_to_number_operator: dict[sympy.Symbol, NumberOperator]
    # Inverse mapping
    _number_operator_to_placeholder: dict[NumberOperator, sympy.Symbol]
    args: tuple[tuple[tuple[sympy.Integer, ...], sympy.Expr], ...]

    def __new__(
        cls,
        operators: Sequence[OperatorType],
        terms: TermDict,
        embedding=None,
        side=0,
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
        embedding, side :
            Optional fixed isometry and attachment side: +1 for X W, -1 for W† X.
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

        if embedding is not None and tuple(operators) != embedding._target_operators:
            # Rebuilding after a mode rename can change the embedding's mode order;
            # _attach reorders the unvalidated form into the embedding's order.
            return embedding._attach(cls(operators, terms, validate=False), side)

        if validate:
            cls._validate_operators(operators)

        # Convert terms dict to a tuple of (power_tuple, coefficient) tuples
        if isinstance(terms, dict):
            terms = Tuple(*(Tuple(k, v) for k, v in terms.items()))

        elif not isinstance(terms, Tuple):
            terms_items = list(terms)
            terms = Tuple(*(Tuple(k, v) for k, v in terms_items))

        # Create placeholders for NumberOperators corresponding to each operator
        number_operators = [NumberOperator(op) for op in operators]
        number_operator_placeholders = [
            _number_operator_to_placeholder(op) for op in number_operators
        ]
        _placeholder_to_number_operator = {
            placeholder: n_op
            for placeholder, n_op in zip(number_operator_placeholders, number_operators)
        }
        replacements = {
            n_op: placeholder
            for placeholder, n_op in zip(number_operator_placeholders, number_operators)
        }

        if validate:
            # Replace NumberOperators with placeholders in terms
            new_terms = []
            for powers, coeff in terms:
                new_terms.append(Tuple(powers, coeff.xreplace(replacements)))
            terms = Tuple(*new_terms)

            # Validate only after conversion
            cls._validate_terms(terms, operators)

        numbers = tuple(number_operator_placeholders)
        nonnegative = tuple(
            n for op, n in zip(operators, numbers) if not isinstance(op, LadderOp)
        )
        simplified = (
            (powers, _simplify_on_fixed_numbers(coeff, numbers, nonnegative))
            if coeff.has(Piecewise)
            else (powers, coeff)
            for powers, coeff in terms
        )
        terms = Tuple(
            *(Tuple(powers, coeff) for powers, coeff in simplified if coeff != 0)
        )
        attachment = ()
        if embedding is not None and terms:
            if side not in (-1, 1):
                raise ValueError("An embedding attachment must be left (-1) or right (1)")
            attachment = (embedding, sympy.Integer(side))
        result = sympy.Expr.__new__(cls, operators, terms, *attachment, **hints)

        result._n_bosons = sum(isinstance(op, BosonOp) for op in operators)
        result._n_ladders = sum(isinstance(op, LadderOp) for op in operators)
        result._n_inf_order = result._n_bosons + result._n_ladders
        result._n_spins = sum(isinstance(op, pauli.SigmaMinus) for op in operators)
        result._n_fermions = sum(isinstance(op, FermionOp) for op in operators)
        result._placeholder_to_number_operator = _placeholder_to_number_operator
        result._number_operator_to_placeholder = replacements
        result._number_operator_placeholders = number_operator_placeholders

        return result

    @property
    def embedding(self) -> "pymablock.operator_embedding.Embedding | None":
        """Return the attached embedding, or None for an ordinary operator."""
        return self.args[2] if len(self.args) == 4 else None

    @property
    def side(self) -> int:
        """Return -1 for W† X, +1 for X W, or zero for no attachment."""
        return int(self.args[3]) if self.embedding is not None else 0

    @cached_property
    def target(self) -> "NumberOrderedForm":
        """The target operator without its embedding attachment."""
        return NumberOrderedForm(self.operators, self.args[1], validate=False)

    def _rebuild(
        self, terms: TermDict, operators: Sequence[OperatorType] | None = None
    ) -> "NumberOrderedForm":
        """Replace the terms while preserving the map's domain and codomain."""
        return type(self)(
            self.operators if operators is None else operators,
            terms,
            *self.args[2:],
            validate=False,
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
            If the operators list is empty or if the order is incorrect.
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
    def _validate_terms(terms: TermDict, operators: Sequence[OperatorType]) -> None:
        """Validate the terms dictionary.

        Parameters
        ----------
        terms :
            Dictionary mapping operator power tuples to coefficient expressions.
        operators :
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
                raise ValueError(
                    f"Coefficient {coeff} must be a commutative expression. Use "
                    "validate=True to convert NumberOperators to placeholders."
                )

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

        from pymablock.operator_embedding import Embedding

        if isinstance(expr, Embedding):
            return expr._attach(One, 1)
        if isinstance(expr, sympy.adjoint) and isinstance(expr.args[0], Embedding):
            return expr.args[0]._attach(One, -1)

        # For scalar expressions (no operators)
        if not expr.has(*operator_types, NumberOperator, Embedding):
            # Return a NumberOrderedForm with no operators and a single term
            operators = operators or []
            return cls(
                operators, Tuple(Tuple((Zero,) * len(operators), expr)), validate=False
            )

        if not operators:
            operators = find_operators(expr)

        # Handle Add expressions by converting each term and summing
        if isinstance(expr, sympy.Add):
            terms = [
                NumberOrderedForm.from_expr(term, operators=operators)
                for term in expr.args
            ]
            return sum(terms, start=NumberOrderedForm(operators, Tuple(), validate=False))

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
            if (
                isinstance(base, (*generator_types, pauli.SigmaPlus))
                and exp.is_integer
                and exp.is_positive
            ):
                # Find the operator index in the operators list
                op = base if base.is_annihilation else base.adjoint()
                powers = tuple(
                    exp * (One if base.is_annihilation else -One)
                    if op == operator
                    else Zero
                    for operator in operators
                )
                return cls(operators, Tuple(Tuple(powers, One)), validate=False)

            # Convert base to NumberOrderedForm
            base_nof = NumberOrderedForm.from_expr(base, operators=operators)

            # Use the __pow__ method to handle the exponentiation
            return base_nof**exp

        # Piecewise conditions and values share one diagonal occupation algebra.
        if isinstance(expr, sympy.Piecewise):
            coefficient = expr.xreplace(
                {
                    n: _number_operator_to_placeholder(n)
                    for n in expr.atoms(NumberOperator)
                }
            )
            if coefficient.has(*operator_types):
                raise ValueError(
                    "Piecewise requires diagonal number-operator expressions"
                )
            return cls(operators, {(0,) * len(operators): coefficient})

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
            zero_key = (Zero,) * len(operators)
            arg_exprs = [next(iter(arg_nof.terms.values()), Zero) for arg_nof in arg_nofs]

            # Return a new NumberOrderedForm with the function applied to the coefficients
            return cls(
                operators, Tuple(Tuple(zero_key, expr.func(*arg_exprs))), validate=False
            )

        # Handle creation/annihilation operators themselves
        if isinstance(expr, (*generator_types, pauli.SigmaPlus)):
            # Find the corresponding annihilation operator in our operators list
            annihilation_op = expr if expr.is_annihilation else expr.adjoint()

            if annihilation_op not in operators:
                raise ValueError(
                    f"Operator {annihilation_op} not found in operators list"
                )

            op_index = operators.index(annihilation_op)

            # Determine the power (1 for annihilation, -1 for creation)
            power = One if expr.is_annihilation else -One

            # Create a term with the appropriate power
            powers = tuple(
                Zero if i != op_index else power for i in range(len(operators))
            )
            return cls(operators, Tuple(Tuple(powers, One)), validate=False)

        # Manually handle Pauli x, y, z operators
        if isinstance(expr, pauli.SigmaZ):
            return cls(
                operators,
                Tuple(
                    Tuple(
                        (Zero,) * len(operators),
                        sympy.S(2) * _number_operator_to_placeholder(NumberOperator(expr))
                        - One,
                    )
                ),
                validate=False,
            )
        if isinstance(expr, pauli.SigmaX):
            op_index = operators.index(pauli.SigmaMinus(expr.name))
            return cls(
                operators,
                Tuple(
                    Tuple(
                        tuple(
                            Zero if i != op_index else One for i in range(len(operators))
                        ),
                        One,
                    ),
                    Tuple(
                        tuple(
                            Zero if i != op_index else -One for i in range(len(operators))
                        ),
                        One,
                    ),
                ),
            )
        if isinstance(expr, pauli.SigmaY):
            op_index = operators.index(pauli.SigmaMinus(expr.name))
            return cls(
                operators,
                Tuple(
                    Tuple(
                        tuple(
                            Zero if i != op_index else One for i in range(len(operators))
                        ),
                        sympy.I,
                    ),
                    Tuple(
                        tuple(
                            Zero if i != op_index else -One for i in range(len(operators))
                        ),
                        -sympy.I,
                    ),
                ),
            )

        if isinstance(expr, NumberOperator):
            return cls(
                operators,
                {(0,) * len(operators): _number_operator_to_placeholder(expr)},
                validate=False,
            )

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
        if self.embedding is not None:
            value = self.target.as_expr()
            return (
                value * self.embedding
                if self.side == 1
                else sympy.adjoint(self.embedding) * value
            )

        if not self.operators:
            # If there are no operators, just return the constant term
            return next(iter(self.terms.values())) if self.terms else Zero

        def export_coefficient(coeff):
            if coeff in self._placeholder_to_number_operator:
                return self._placeholder_to_number_operator[coeff]
            if not coeff.has(*self._number_operator_placeholders):
                return coeff
            args = tuple(export_coefficient(arg) for arg in coeff.args)
            if coeff.func == sympy.conjugate:
                # Scalar conjugation becomes operator adjunction. Evaluating a
                # radical's adjoint here can trigger invalid complex expansion.
                return sympy.adjoint(*args, evaluate=False)
            return coeff.func(*args)

        terms = []
        reversed_operators = list(reversed(self.operators))

        for powers, coeff in self.args[1]:
            term = export_coefficient(coeff)
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

        # Build the sum once: repeated addition repeatedly canonicalizes all
        # preceding terms, which is costly when this expression is hashed.
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
    def terms(self) -> TermDict:
        """The dictionary of terms.

        Notes
        -----
        Internally, terms are stored as a tuple of (key, value) tuples for better performance.
        This property converts the internal representation to a dictionary for compatibility.

        """
        # Convert tuple of tuples to dictionary
        return {k: v for k, v in self.args[1]}

    def act(
        self, occupations: Sequence[int | sympy.Expr]
    ) -> dict[tuple[int, ...], tuple[tuple[sympy.Expr, ...], sympy.Expr]]:
        """Apply each term to the Fock state with the given occupations.

        Occupations follow the order of ``operators`` and may be symbolic. Positive
        powers annihilate particles; negative powers create them. Each term applies
        its annihilation operators, evaluates its coefficient at the intermediate
        occupations, then applies its creation operators in reverse mode order.
        Fermion signs follow the order of ``operators``. All ladder factors are
        checked for zero before the coefficient is evaluated, so forbidden
        transitions never evaluate coefficient poles.

        Parameters
        ----------
        occupations :
            Occupation of each mode in the input state.

        Returns
        -------
        dict
            ``powers: (output_occupations, matrix_element)``, keyed as in
            ``terms``. The matrix element includes the evaluated coefficient,
            ladder amplitudes, and fermion signs. Terms with a literally zero
            matrix element are omitted; symbolic zeros are not inferred.

        Examples
        --------
        >>> a = BosonOp("a")
        >>> NumberOrderedForm.from_expr(Dagger(a) ** 2 * a).act((3,))
        {(-1,): ((4,), 6)}

        """
        is_fermion = [isinstance(op, FermionOp) for op in self.operators]
        numbers = self._number_operator_placeholders

        def apply_ladder(
            state: list[sympy.Expr], index: int, annihilate: bool
        ) -> sympy.Expr:
            """Apply one ladder operator to ``state`` in place; return its amplitude."""
            operator, n = self.operators[index], state[index]
            state[index] += -1 if annihilate else 1
            if isinstance(operator, BosonOp):
                return sympy.sqrt(n if annihilate else n + 1)
            if isinstance(operator, LadderOp):
                return sympy.S.One
            factor = n if annihilate else 1 - n
            if isinstance(operator, FermionOp):
                factor *= (-1) ** sum(
                    m for m, odd in zip(state[:index], is_fermion) if odd
                )
            return factor

        def apply_term(
            powers: tuple[int, ...], coefficient: sympy.Expr
        ) -> tuple[tuple[sympy.Expr, ...], sympy.Expr] | None:
            """Return the output occupations and matrix element, or None if zero."""
            state = list(map(sympy.sympify, occupations))
            factors = []
            for index, power in enumerate(powers):
                for _ in range(power):
                    factors.append(apply_ladder(state, index, annihilate=True))
            middle_occupations = tuple(state)
            for index, power in reversed(list(enumerate(powers))):
                for _ in range(-power):
                    factors.append(apply_ladder(state, index, annihilate=False))
            # Check the ladder factors first: the coefficient may be singular
            # where one of them vanishes.
            if any(factor == 0 for factor in factors):
                return None
            coefficient = coefficient.xreplace(
                dict(zip(numbers, middle_occupations, strict=True))
            )
            matrix_element = sympy.Mul(*factors) * coefficient
            return None if matrix_element == 0 else (tuple(state), matrix_element)

        result = {}
        for powers, coefficient in self.args[1]:
            powers = tuple(map(int, powers))
            if (action := apply_term(powers, coefficient)) is not None:
                result[powers] = action
        return result

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

    def _multiply_op(self, op_index: sympy.Integer, op_power: sympy.Integer):
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
        assert 0 <= op_index < len(self.operators), "op_index out of range"
        assert op_power != 0, "op_power must be non-zero"

        operator = self.operators[op_index]
        n_operator = self._number_operator_placeholders[op_index]

        # Create a new terms dictionary for the result
        new_terms = {}

        if op_index < self._n_inf_order:  # Bosons and ladders
            for powers, coeff in self.args[1]:
                orig_power = powers[op_index]  # Power of the operator at op_index
                new_power = orig_power + op_power
                new_powers = tuple(
                    new_power if i == op_index else p for i, p in enumerate(powers)
                )
                if op_power > 0:  # Multiplying by an annihilation operator
                    # Compute how many new number operators appear
                    to_pair = min(op_power, max(-orig_power, 0))
                    coeff = coeff.xreplace({n_operator: n_operator - to_pair})
                    if op_index < self._n_bosons:  # Bosons
                        # Test before multiplication can cancel a pole against a
                        # vanishing ladder factor. Those boundary states do not act.
                        inactive_poles = [
                            sympy.Eq(n_operator, i)
                            for i in range(to_pair)
                            if _is_singular(coeff.xreplace({n_operator: sympy.S(i)}))
                        ]
                        coeff = sympy.Mul(
                            coeff, *(n_operator - i for i in range(to_pair))
                        )
                        if inactive_poles:
                            coeff = sympy.Piecewise(
                                (Zero, sympy.Or(*inactive_poles)), (coeff, True)
                            )
                else:
                    to_pair = min(-op_power, max(orig_power, 0))
                    # Move unmatched creation operators to the left of the coefficient.
                    if new_power < 0:
                        coeff = coeff.xreplace(
                            {n_operator: n_operator - op_power - to_pair}
                        )
                    # Pairing operators produces number factors. Move these factors
                    # past the unmatched operators to restore number order.
                    if op_index < self._n_bosons:  # Bosons
                        new_numbers = sympy.Mul(
                            *[
                                n_operator + abs(new_power) + sympy.S(i)
                                for i in range(1, to_pair + 1)
                            ]
                        )
                        coeff = coeff * new_numbers
                new_terms[new_powers] = coeff
        else:  # Fermions and spins
            if abs(op_power) > One:
                # Fermionic and spin operators are nilpotent
                return self._rebuild(Tuple())
            for powers, coeff in self.args[1]:
                orig_power = powers[op_index]
                new_power = orig_power + op_power
                if abs(new_power) > One:
                    # Fermionic and spin operators are nilpotent
                    continue
                new_powers = tuple(
                    new_power if i == op_index else p for i, p in enumerate(powers)
                )
                if op_power is One:
                    # Annihilation operator, n_c * c = 0
                    coeff = coeff.xreplace({n_operator: Zero})
                    if orig_power:
                        # c† * c = n_c
                        coeff = n_operator * coeff
                else:
                    # For an existing annihilation operator, f(n)*c*c† = f(0)*(1-n).
                    # Otherwise, f(n)*c† = c†*f(1).
                    coeff = coeff.xreplace({n_operator: Zero if orig_power else One})
                    if orig_power:
                        # c * c† = 1 - n_c
                        coeff = (One - n_operator) * coeff

                # Handle fermionic anticommutation
                if isinstance(operator, FermionOp):
                    # Count the fermions with which we need to commute the new operator.
                    if orig_power == 1 or new_power == 1:
                        # Either multiplying annihilation by creation or nothing by
                        # annihilation => count all annihilation operators that are earlier
                        # than the current one.
                        preceding_fermions = sum(
                            int(pow == 1) for pow in powers[-self._n_fermions : op_index]
                        )
                    else:
                        # Multiplying creation by annihilation or nothing by creation =>
                        # count all annihilation operators and all creation operators that
                        # are later than the current one.
                        preceding_fermions = sum(
                            int(pow == One) for pow in powers[-self._n_fermions :]
                        ) + sum(int(pow == -One) for pow in powers[op_index + 1 :])

                    if preceding_fermions % 2:
                        # Fermionic sign change
                        coeff = -coeff

                new_terms[new_powers] = coeff

        # Create the new NumberOrderedForm with the same operators but new terms
        return self._rebuild(new_terms)

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
            replacements = {}
            for i, power in enumerate(powers):
                if power == 0:
                    continue
                n_i = self._number_operator_placeholders[i]
                if i < self._n_inf_order:  # Bosons or ladders
                    if power > 0:
                        # a * n_a = n_a + 1
                        replacements[n_i] = n_i + power
                else:  # Fermion or spin
                    if power < 0:
                        # c† * n_c = 0
                        replacements[n_i] = Zero
                    else:
                        # c * n_c = c.
                        replacements[n_i] = One

            new_terms[powers] = coeff * expr.xreplace(replacements)

        # Return a new NumberOrderedForm instance with the updated terms
        return self._rebuild(new_terms)

    def _cancel_binary_operator_numbers(self):
        """Cancel fermionic and spin number operators.

        If the coefficient has `n_f`, while the term has either `f` or `f†`,
        `n_f` may be safely replaced with `0` because of the fermionic nilpotence.

        Returns
        -------
        NumberOrderedForm
            A new NumberOrderedForm with the fermionic and spin number operators canceled.

        """
        if not (binary_ops := self.operators[self._n_inf_order :]):
            # No binary operators, nothing to do
            return self

        new_terms = {}
        for powers, coeff in self.args[1]:
            replacements = {}
            for p, op in zip(powers[self._n_inf_order :], binary_ops):
                if not p:
                    continue
                replacements[_number_operator_to_placeholder(NumberOperator(op))] = Zero
            coeff = coeff.xreplace(replacements)
            if coeff == 0:
                continue
            new_terms[powers] = coeff

        return self._rebuild(new_terms)

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
        return self._rebuild(new_terms, operators=new_operators)

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

        # Empty forms are neutral regardless of their embedding attachment.
        # Avoid symbolic zero inference on every intermediate coefficient.
        if not self:
            return other
        if not other:
            return self
        if self.args[2:] != other.args[2:]:
            raise ValueError("Addition requires matching embedding attachments")

        self_expanded, other_expanded = self._combine_operators(other)

        new_terms = defaultdict(lambda: Zero)
        for powers, coeff in (*self_expanded.args[1], *other_expanded.args[1]):
            new_terms[powers] += coeff
        return self._rebuild(new_terms, operators=self_expanded.operators)

    def _combine_operators(
        self, other
    ) -> tuple["NumberOrderedForm", "NumberOrderedForm"]:
        """Convert this NumberOrderedForm and another to have the same operator list."""
        if other.operators != self.operators:
            new_operators = sorted(
                set(self.operators).union(other.operators),
                key=_operator_sort_key,
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
        return self._rebuild(tuple((powers, -coeff) for powers, coeff in self.args[1]))

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
                other = NumberOrderedForm.from_expr(sympy.sympify(other))
            except Exception:
                return NotImplemented

        if self.embedding is not None or other.embedding is not None:
            return self._multiply_attached(other)

        self_expanded, other_expanded = self._combine_operators(other)

        # Binary ladder terms act only at middle occupation zero. Restrict their
        # coefficients before multiplication can cancel factors outside that domain.
        self_expanded = self_expanded._cancel_binary_operator_numbers()
        other_expanded = other_expanded._cancel_binary_operator_numbers()

        result = self._rebuild({}, operators=self_expanded.operators)
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
            # Apply annihilation operators in reverse mode order.
            for i, power in reversed(list(enumerate(powers))):
                if not power > 0:
                    continue
                partial = partial._multiply_op(i, power)
            # Add the result to the new terms
            result = result + partial._linearize_binary_operators()

        return result

    def _multiply_attached(
        self, other: "NumberOrderedForm"
    ) -> "NumberOrderedForm | sympy.Expr":
        """Compose operators when at least one factor carries an embedding.

        Matching opposite attachments give ``W† X Y W`` in source space or
        ``X W W† Y`` in target space. With one attachment, interpret the adjacent
        factor in the appropriate target or source basis and preserve the map.
        """

        def target_product(a, b):
            # Unit frame entries must not expand factored spectator coefficients.
            return b if a == 1 else a if b == 1 else a * b

        left, right = self.embedding, other.embedding
        if left is not None and right is not None:
            if left != right or self.side == other.side:
                raise ValueError("Composition requires opposite matching attachments")
            if self.side == -1:
                return left._compress(target_product(self.target, other.target))
            return self.target * left._projector * other.target
        if left is not None:
            value = left._lift(other) if self.side == 1 else left._convert_operator(other)
            result = target_product(self.target, value)
            return self._rebuild(result.args[1], operators=result.operators)
        value = right._lift(self) if other.side == -1 else right._convert_operator(self)
        result = target_product(value, other.target)
        return other._rebuild(result.args[1], operators=result.operators)

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
        new_terms = tuple(
            (tuple(-power for power in powers), coeff.adjoint())
            for powers, coeff in self.args[1]
        )
        return type(self)(
            self.operators, new_terms, self.embedding, -self.side, validate=False
        )

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
        if (self.embedding, self.side) != (other.embedding, other.side):
            return False
        if self.operators != other.operators:
            self, other = self._combine_operators(other)
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
        """Apply a SymPy function to the terms of this NumberOrderedForm.

        Notes
        -----
        `NumberOrderedForm` stores its coefficients with number operators replaced with
        integer placeholder symbols. This method does applies the function to those
        coefficients, but does not change the operators themselves.

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
        # Create a new terms dictionary for the result
        new_terms = {}

        # Process each term in the terms dictionary
        for powers, coeff in self.args[1]:
            result_expr = func(coeff, *args, **kwargs)
            new_terms[powers] = result_expr

        return self._rebuild(new_terms)

    def _linearize_binary_operators(self) -> "NumberOrderedForm":
        """Reduce binary-number dependence while preserving unresolved poles.

        Interpolate each fermion or spin number with ``f(n) = (1-n) f(0) + n f(1)``.
        If either endpoint is singular even after cancellation, leave that number's
        dependence unchanged so the pole cannot corrupt other occupation sectors.
        """
        if not (
            binary_numbers := [
                _number_operator_to_placeholder(NumberOperator(op))
                for op in self.operators[self._n_inf_order :]
            ]
        ):
            # No binary operators, nothing to do
            return self

        new_terms = {}
        for powers, coeff in self.args[1]:
            for power, number in zip(powers[self._n_inf_order :], binary_numbers):
                if power:
                    # Between binary creation and annihilation operators only
                    # occupation zero acts; do not sample a possible pole at one.
                    coeff = coeff.xreplace({number: Zero})
                    continue
                if number not in coeff.free_symbols:
                    continue
                values = tuple(coeff.xreplace({number: n}) for n in (Zero, One))
                # An unresolved pole must remain meromorphic. Evaluating at its
                # singular binary point would corrupt every other sector too.
                if any(map(_is_singular, values)):
                    reduced = sympy.cancel(coeff)
                    values = tuple(reduced.xreplace({number: n}) for n in (Zero, One))
                    if any(map(_is_singular, values)):
                        continue
                coeff = sympy.expand_mul(
                    (One - number) * values[0] + number * values[1], deep=False
                )
            new_terms[powers] = coeff
        return self._rebuild(new_terms)

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
        return self._linearize_binary_operators().applyfunc(sympy.simplify, **kwargs)

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

        if self.embedding is not None:
            if exp == 1:
                return self
            raise ValueError("A rectangular map cannot be raised to a power")

        # A single monomial with a binary mode is nilpotent, including for
        # symbolic integer exponents that are provably greater than one.
        if exp.is_integer and (exp - 1).is_positive and len(self.terms) == 1:
            powers = next(iter(self.terms))
            if any(powers[self._n_inf_order :]):
                return self._rebuild({})

        # Positive symbolic bosonic powers are used as selective masks.
        if not self.is_particle_conserving() and exp.is_integer and exp.is_nonnegative:
            if len(self.terms) == 1 and not exp.is_Integer:
                powers, coeff = next(iter(self.terms.items()))
                if not any(powers[self._n_inf_order :]) and not coeff.has(
                    *self._number_operator_placeholders
                ):
                    return self._rebuild(
                        {tuple(power * exp for power in powers): coeff**exp}
                    )

        if not self.is_particle_conserving() and not (
            exp.is_Integer and exp.is_nonnegative
        ):
            raise ValueError(
                "Expressions with unmatched creation or annihilation operators require a "
                "non-negative integer power."
            )

        if exp == 0:
            return self._rebuild(Tuple(Tuple((Zero,) * len(self.operators), One)))

        # For integer exponents, convert to repeated multiplication
        if (isinstance(exp, int) or exp.is_Integer) and exp > 0:
            result = self
            for _ in range(exp - 1):
                result = result * self
            return result

        # Since the expression only contains number operators, it's safe to apply the power
        # We extract the coefficient (if exists) and raise it to the given exponent.
        return self._rebuild({key: value**exp for key, value in self.args[1]})

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
        return all(not any(powers) for powers, _ in self.args[1])

    def _eval_subs(self, old: sympy.Basic, new: sympy.Basic) -> "NumberOrderedForm":
        """Substitute coefficients, reconstructing attachments with the new embedding.

        Bare operators retain their basis; direct mode replacement is rejected.
        """
        if old in self.operators or new in self.operators:
            raise ValueError("Cannot substitute operators in NumberOrderedForm.")

        if self.embedding is not None:
            attachment = self.embedding.subs(old, new)
            return attachment._attach(self.target.subs(old, new), self.side)
        old = old.xreplace(self._number_operator_to_placeholder)
        new = new.xreplace(self._number_operator_to_placeholder)
        return self._rebuild(
            Tuple(
                *(Tuple(powers, coeff.subs(old, new)) for powers, coeff in self.args[1])
            )
        )

    def _xreplace(
        self, rule: Mapping[sympy.Basic, sympy.Basic]
    ) -> tuple[sympy.Basic, bool]:
        """Replace exact subexpressions and report whether anything changed.

        Coefficients store number operators as placeholder symbols named after their
        modes, so renaming a mode also renames its placeholder. The constructor
        normalizes attached operators for the replaced embedding.
        """
        renames = {
            placeholder: _number_operator_to_placeholder(NumberOperator(new))
            for op, placeholder in zip(
                self.operators, self._number_operator_placeholders, strict=True
            )
            if isinstance(new := rule.get(op), generator_types) and new.is_annihilation
        }
        return super()._xreplace({**renames, **rule} if renames else rule)

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
        new_terms = tuple(
            Tuple(powers, coeff)
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
        )
        return self._rebuild(new_terms)

    def _poly_simplify(self) -> "NumberOrderedForm":
        """Simplify a NumberOrderedForm by converting it to polynomials and back.

        This uses as generators all possible expressions sympy finds, except for the
        number operators.
        """
        if not self.operators:
            # No operators, nothing to simplify
            return self

        new_terms = {}
        for powers, coeff in self.args[1]:
            if not coeff.free_symbols:
                # If the coefficient is a constant, just keep it as is
                new_terms[powers] = coeff
                continue

            # Convert the coefficient to a polynomial and extract the generators
            try:
                poly = sympy.poly(coeff)
            except sympy.polys.polyerrors.GeneratorsNeeded:
                # poly recurses into constant factors such as (1 + I).
                poly = sympy.Poly(coeff)
            number_gens = tuple(
                gen for gen in poly.gens if gen in self._number_operator_placeholders
            )
            non_number_gens = tuple(
                gen for gen in poly.gens if gen not in self._number_operator_placeholders
            )
            if len(number_gens) == len(poly.gens):
                # If there are no non-number operator generators, keep the coefficient as is
                new_terms[powers] = coeff
                continue

            new_terms[powers] = sympy.Poly.from_dict(
                {
                    # Here we simplify the polynomial of number operators (coeff).
                    # This may be improved in the future with better heuristics.
                    term: sympy.collect_const(sympy.simplify(coeff)).doit()
                    for term, coeff in sympy.poly(
                        # EXRAW domain does not expand its terms. This is important to
                        # not automatically expand expressions like (n + 1) ** 5
                        coeff,
                        gens=non_number_gens,
                        domain=sympy.EXRAW,
                    )
                    .as_dict()
                    .items()
                },
                gens=non_number_gens,
                domain=sympy.EXRAW,
            ).as_expr()

        return self._rebuild(new_terms)


def _occupation_dimension(operator: OperatorType) -> int | None:
    """Return 2 for a spin or fermion mode, and None for an infinite mode.

    Boson occupations range over nonnegative integers; bilateral ladder indices
    range over all integers. Both therefore have infinite-dimensional spaces.
    """
    return 2 if isinstance(operator, (FermionOp, pauli.SigmaMinus)) else None
