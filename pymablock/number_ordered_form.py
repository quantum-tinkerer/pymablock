"""Implementation of NumberOrderedForm as an Operator subclass.

This module provides a class-based implementation of number ordered form for quantum operators,
which represents operators with creation operators on the left, annihilation operators on the right,
and number operators in the middle.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from functools import lru_cache

import sympy
from packaging.specifiers import SpecifierSet
from sympy.core.logic import fuzzy_and
from sympy.physics.quantum import Dagger, HermitianOperator, Operator, pauli
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

from pymablock._packed_binary import masks_from_monomial, multiply_monomials
from pymablock._packed_nof import (
    anticommuting_mask,
    build_packed_terms,
    canonical_monomial,
    pack_terms,
    unpack_terms,
)

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
_FINALIZE_SMALL_COEFF_MAX_OPS = 200


@lru_cache(maxsize=8192)
def _cached_shift_number_placeholders(
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


@lru_cache(maxsize=8192)
def _cached_conjugate(expr: sympy.Expr) -> sympy.Expr:
    return sympy.conjugate(expr)


@lru_cache(maxsize=8192)
def _cached_power_key(powers: tuple) -> Tuple:
    """Convert a Python power key to its immutable SymPy storage form."""
    return Tuple(*powers)


@lru_cache(maxsize=8192)
def _infinite_operator_product(
    left_powers: tuple[int, ...],
    right_powers: tuple[int, ...],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Compute coefficient-independent data for an infinite-order term product."""
    current_powers = list(left_powers)
    total_shift = [0] * len(left_powers)
    pre_pair_counts = [0] * len(left_powers)
    post_pair_counts = [0] * len(left_powers)

    for i, power in enumerate(right_powers):
        if power >= 0:
            continue
        creation_power = -power
        orig_power = current_powers[i]
        to_pair = min(creation_power, max(orig_power, 0))
        pre_pair_counts[i] = to_pair
        new_power = orig_power + power
        total_shift[i] += new_power if new_power > 0 else creation_power - to_pair
        current_powers[i] = new_power

    right_coeff_shift = tuple(
        (power if power > 0 else 0) - shift
        for power, shift in zip(current_powers, total_shift)
    )

    for i, power in enumerate(right_powers):
        if power <= 0:
            continue
        orig_power = current_powers[i]
        to_pair = min(power, max(-orig_power, 0))
        post_pair_counts[i] = to_pair
        total_shift[i] -= to_pair
        current_powers[i] = orig_power + power

    return (
        tuple(current_powers),
        tuple(total_shift),
        right_coeff_shift,
        tuple(pre_pair_counts),
        tuple(post_pair_counts),
    )


@lru_cache(maxsize=8192)
def _boson_contraction_factors(
    placeholders: tuple[sympy.Symbol, ...],
    n_bosons: int,
    pre_pair_counts: tuple[int, ...],
    post_pair_counts: tuple[int, ...],
) -> tuple[sympy.Expr, sympy.Expr]:
    """Build the bosonic number factors associated with a term product."""
    pre_shift_multiplier = sympy.Mul(
        *(
            placeholders[i] + step
            for i, to_pair in enumerate(pre_pair_counts[:n_bosons])
            for step in range(1, to_pair + 1)
        )
    )
    post_shift_multiplier = sympy.Mul(
        *(
            placeholders[i] - step
            for i, to_pair in enumerate(post_pair_counts[:n_bosons])
            for step in range(to_pair)
        )
    )
    return pre_shift_multiplier, post_shift_multiplier


def _finalize_small_coefficient(expr: sympy.Expr) -> sympy.Expr:
    """Regroup repeated commutative factors in a small coefficient.

    This is intended for small post-substitution/readout expressions, not the hot
    symbolic path. It looks for multiplicative commutative factors that recur across
    several top-level additive terms and collects over them sequentially.
    """
    if not expr.is_Add or int(sympy.count_ops(expr)) > _FINALIZE_SMALL_COEFF_MAX_OPS:
        return expr

    factor_counts: Counter[sympy.Expr] = Counter()
    for term in sympy.Add.make_args(expr):
        term_factors = {
            factor
            for factor in sympy.Mul.make_args(term)
            if not factor.is_number and factor.is_commutative and not factor.is_Add
        }
        factor_counts.update(term_factors)

    candidates = [factor for factor, count in factor_counts.items() if count >= 2]
    if not candidates:
        return expr

    candidates.sort(
        key=lambda factor: (
            -factor_counts[factor],
            sympy.count_ops(factor),
            len(str(factor)),
        )
    )

    result = expr
    for factor in candidates:
        result = sympy.collect(result, factor)
    return result


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
PowerKey = tuple[int | sympy.Integer, ...]
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
            and exp != 0
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
        names.

    """
    # replace n -> a† * a and convert number ordered forms to expressions.
    # Number operator of ladder operators need to be included separately.
    expr = expr.doit()
    return sorted(
        set().union(
            (
                op
                for particle, generator in zip(operator_types, generator_types)
                for op in (generator(atom.name) for atom in expr.atoms(particle))
            ),
            (
                LadderOp(atom.name)
                for atom in expr.atoms(NumberOperator)
                if atom.args[1].name == "LadderOp"
            ),
        ),
        key=lambda op: (generator_types.index(type(op)), str(op.name)),
    )


def _number_operator_to_placeholder(op: NumberOperator) -> sympy.Symbol:
    """Convert a NumberOperator to its placeholder symbol."""
    return sympy.Symbol(
        f"number_operator_placeholder_{op.args[0]}_{op.args[1]}",
        integer=True,
    )


@lru_cache(maxsize=1024)
def _operator_metadata(operators: Tuple) -> tuple:
    """Build immutable metadata shared by forms with the same operators."""
    number_operators = tuple(NumberOperator(op) for op in operators)
    placeholders = tuple(map(_number_operator_to_placeholder, number_operators))
    placeholder_to_number_operator = dict(zip(placeholders, number_operators))
    number_operator_to_placeholder = dict(zip(number_operators, placeholders))
    n_bosons = sum(isinstance(op, BosonOp) for op in operators)
    n_ladders = sum(isinstance(op, LadderOp) for op in operators)
    n_spins = sum(isinstance(op, pauli.SigmaMinus) for op in operators)
    n_fermions = sum(isinstance(op, FermionOp) for op in operators)
    return (
        placeholders,
        placeholder_to_number_operator,
        number_operator_to_placeholder,
        n_bosons,
        n_ladders,
        n_spins,
        n_fermions,
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

    Internally each term key is ``(infinite_powers, binary_monomial)``. The first
    component stores boson and ladder shifts, while the second is one packed
    spin/fermion monomial. The public :attr:`terms` property decodes this into full
    power tuples for compatibility.

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

    # Attribute types
    _n_bosons: int
    _n_ladders: int
    # Number of infinite order operators (bosons + ladders)
    _n_inf_order: int
    _n_spins: int
    _n_fermions: int
    # List of placeholder symbols for NumberOperator instances, ordered like operators
    _number_operator_placeholders: tuple[sympy.Symbol, ...]
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
        _packed: bool = False,
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
        if not isinstance(operators, Tuple):
            operators = Tuple(*operators)

        if validate:
            cls._validate_operators(operators)

        (
            number_operator_placeholders,
            placeholder_to_number_operator,
            replacements,
            n_bosons,
            n_ladders,
            n_spins,
            n_fermions,
        ) = _operator_metadata(operators)

        # SymPy reconstructs expressions as ``func(*args)`` without private
        # constructor flags, so recognize our structural key unambiguously.
        if (
            not _packed
            and isinstance(terms, Tuple)
            and terms
            and len(terms[0][0]) == 2
            and isinstance(terms[0][0][0], Tuple)
        ):
            _packed = True

        if _packed:
            if not isinstance(terms, Tuple):
                terms = Tuple(*(Tuple(key, coeff) for key, coeff in terms))
        else:
            if isinstance(terms, dict):
                terms = Tuple(
                    *(Tuple(_cached_power_key(tuple(k)), v) for k, v in terms.items())
                )
            elif not isinstance(terms, Tuple):
                terms = Tuple(*(Tuple(_cached_power_key(tuple(k)), v) for k, v in terms))

            if validate:
                terms = Tuple(
                    *(
                        Tuple(powers, coeff.xreplace(replacements))
                        for powers, coeff in terms
                    )
                )
                cls._validate_terms(terms, operators)

            terms = pack_terms(
                operators,
                n_bosons + n_ladders,
                number_operator_placeholders,
                terms,
            )

        result = sympy.Expr.__new__(cls, operators, terms, **hints)

        result._n_bosons = n_bosons
        result._n_ladders = n_ladders
        result._n_inf_order = result._n_bosons + result._n_ladders
        result._n_spins = n_spins
        result._n_fermions = n_fermions
        result._placeholder_to_number_operator = placeholder_to_number_operator
        result._number_operator_to_placeholder = replacements
        result._number_operator_placeholders = number_operator_placeholders

        return result

    @classmethod
    def _from_packed_terms(
        cls,
        operators: Sequence[OperatorType],
        terms,
    ) -> "NumberOrderedForm":
        """Construct directly from canonical packed terms."""
        return cls(
            operators,
            build_packed_terms(terms),
            validate=False,
            _packed=True,
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
        if list(operators) != sorted(
            operators, key=lambda op: (generator_types.index(type(op)), str(op.name))
        ):
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

        # For scalar expressions (no operators)
        if not expr.has(*operator_types, NumberOperator):
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
        if not self.operators:
            # If there are no operators, just return the constant term
            return next(iter(self.terms.values())) if self.terms else Zero

        result = Zero
        reversed_operators = list(reversed(self.operators))
        placeholders = frozenset(self._number_operator_placeholders)

        for powers, coeff in self.terms.items():
            if coeff.free_symbols.isdisjoint(placeholders):
                coeff = _finalize_small_coefficient(coeff)
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
        Internally, a key is ``(infinite_powers, binary_monomial)``. This
        property decodes it to the historical full power tuple.

        """
        return unpack_terms(
            self.operators,
            self._n_inf_order,
            self._number_operator_placeholders,
            self.args[1],
        )

    @property
    def _packed_terms(self):
        """Iterate over ``((infinite powers, binary monomial), coefficient)``."""
        return (
            ((tuple(key[0]), int(key[1])), coefficient)
            for key, coefficient in self.args[1]
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
        powers = [0] * len(self.operators)
        powers[op_index] = int(op_power)
        factor = type(self)(self.operators, {tuple(powers): One}, validate=False)
        return self * factor

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

        scalar = type(self)(
            self.operators,
            {(0,) * len(self.operators): expr},
            validate=False,
        )
        return self * scalar

    def _cancel_binary_operator_numbers(self):
        """Cancel fermionic and spin number operators.

        If the coefficient has `n_f`, while the term has either `f` or `f†`,
        `n_f` may be safely replaced with `0` because of the fermionic nilpotence.

        Returns
        -------
        NumberOrderedForm
            A new NumberOrderedForm with the fermionic and spin number operators canceled.

        """
        return self

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

        self_expanded, other_expanded = self._combine_operators(other)
        return type(self)._from_packed_terms(
            self_expanded.operators,
            (*self_expanded._packed_terms, *other_expanded._packed_terms),
        )

    def _combine_operators(
        self, other
    ) -> tuple["NumberOrderedForm", "NumberOrderedForm"]:
        """Convert this NumberOrderedForm and another to have the same operator list."""
        if other.operators != self.operators:
            new_operators = sorted(
                set(self.operators).union(other.operators),
                key=lambda op: (generator_types.index(type(op)), str(op.name)),
            )
            self_expanded = self._expand_operators(new_operators)
            other_expanded = other._expand_operators(new_operators)
        else:
            self_expanded = self
            other_expanded = other
        return self_expanded, other_expanded

    @staticmethod
    def _shift_number_placeholders(
        expr: sympy.Expr,
        placeholders: Sequence[sympy.Symbol],
        shifts: Sequence[sympy.Expr],
    ) -> sympy.Expr:
        """Shift number-operator placeholders in a commutative expression."""
        return _cached_shift_number_placeholders(
            expr,
            tuple(placeholders),
            tuple(shifts),
        )

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
        return type(self)._from_packed_terms(
            self.operators,
            ((key, -coefficient) for key, coefficient in self._packed_terms),
        )

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
            other = sympy.sympify(other)
            if other.is_commutative:
                return type(self)._from_packed_terms(
                    self.operators,
                    (
                        (key, coefficient * other)
                        for key, coefficient in self._packed_terms
                    ),
                )
            try:
                other = NumberOrderedForm.from_expr(other)
            except Exception:
                return NotImplemented

        self_expanded, other_expanded = self._combine_operators(other)
        placeholders = self_expanded._number_operator_placeholders[
            : self_expanded._n_inf_order
        ]
        num_binary = len(self_expanded.operators) - self_expanded._n_inf_order
        fermion_mask = anticommuting_mask(
            self_expanded.operators, self_expanded._n_inf_order
        )
        result_terms = []

        for (left_powers, left_binary), left_coefficient in self_expanded._packed_terms:
            for (
                right_powers,
                right_binary,
            ), right_coefficient in other_expanded._packed_terms:
                (
                    result_powers,
                    total_shift,
                    right_coefficient_shift,
                    pre_pair_counts,
                    post_pair_counts,
                ) = _infinite_operator_product(left_powers, right_powers)
                pre_multiplier, post_multiplier = _boson_contraction_factors(
                    placeholders,
                    self_expanded._n_bosons,
                    pre_pair_counts,
                    post_pair_counts,
                )
                coefficient = self._shift_number_placeholders(
                    left_coefficient * pre_multiplier,
                    placeholders,
                    total_shift,
                )
                coefficient *= post_multiplier
                coefficient *= self._shift_number_placeholders(
                    right_coefficient,
                    placeholders,
                    tuple(
                        shift + base_shift
                        for shift, base_shift in zip(
                            right_coefficient_shift,
                            total_shift,
                            strict=True,
                        )
                    ),
                )
                result_terms.extend(
                    (
                        (result_powers, binary),
                        coefficient * integer,
                    )
                    for binary, integer in multiply_monomials(
                        left_binary,
                        right_binary,
                        num_modes=num_binary,
                        anticommuting_modes=fermion_mask,
                    ).items()
                )

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
                return type(self)._from_packed_terms(
                    self.operators,
                    (
                        (key, other * coefficient)
                        for key, coefficient in self._packed_terms
                    ),
                )
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
        num_binary = len(self.operators) - self._n_inf_order
        fermion_mask = anticommuting_mask(self.operators, self._n_inf_order)
        terms = []
        for (powers, monomial), coefficient in self._packed_terms:
            creators, numbers, annihilators = masks_from_monomial(
                monomial, num_modes=num_binary
            )
            original = canonical_monomial(
                num_binary,
                fermion_mask,
                creators,
                numbers,
                annihilators,
            )
            adjoint = canonical_monomial(
                num_binary,
                fermion_mask,
                annihilators,
                numbers,
                creators,
            )
            if original is None or adjoint is None:  # pragma: no cover
                raise AssertionError("A stored packed monomial decodes to zero")
            _, original_normalization = original
            adjoint_monomial, adjoint_normalization = adjoint
            terms.append(
                (
                    (
                        tuple(-power for power in powers),
                        adjoint_monomial,
                    ),
                    _cached_conjugate(coefficient)
                    * adjoint_normalization
                    / original_normalization,
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
        if self.operators != other.operators:
            self, other = self._combine_operators(other)
        return self.terms == other.terms

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

        if exp == 0:
            return type(self)(
                self.operators,
                Tuple(Tuple((Zero,) * len(self.operators), One)),
                validate=False,
            )

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

            powers, coeff = next(iter(self.terms.items()))

            if any(powers[self._n_inf_order :]) and exp > 1:
                return type(self)(self.operators, {}, validate=False)

            # One term, may exponentiate to a positive power if the coefficient is
            # commutative
            if coeff.has(*self._number_operator_placeholders):
                raise ValueError(
                    "Cannot raise expression with creation/annihilation and number "
                    f"operators to a non-integer power: {self}**{exp}"
                )
            if exp.is_negative:
                raise ValueError(
                    f"Cannot raise expression with unmatched creation or annihilation "
                    f"operators to non-positive power: {self}**{exp}"
                )

            return type(self)(
                self.operators,
                {tuple(i * exp for i in powers): coeff**exp},
                validate=False,
            )

        coefficient = next(iter(self.terms.values()), Zero)
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
        num_binary = len(self.operators) - self._n_inf_order
        for (powers, monomial), _ in self._packed_terms:
            creators, _, annihilators = masks_from_monomial(
                monomial, num_modes=num_binary
            )
            if any(powers) or creators or annihilators:
                return False
        return True

    def _eval_subs(self, old, new):
        if old in self.operators or new in self.operators:
            raise ValueError("Cannot substitute operators in NumberOrderedForm.")

        old = old.xreplace(self._number_operator_to_placeholder)
        new = new.xreplace(self._number_operator_to_placeholder)

        if old.is_Atom:
            replacements = {old: new}
            return type(self)(
                self.operators,
                {
                    powers: coeff.xreplace(replacements)
                    for powers, coeff in self.terms.items()
                },
                validate=False,
            )

        return type(self)(
            self.operators,
            {powers: coeff.subs(old, new) for powers, coeff in self.terms.items()},
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
        new_terms = tuple(
            Tuple(powers, coeff)
            for powers, coeff in self.terms.items()
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
        return type(self)(self.operators, new_terms, validate=False)

    def _poly_simplify(self) -> "NumberOrderedForm":
        """Simplify scalar coefficients with ``sympy.factor_terms``."""
        if not self.operators:
            # No operators, nothing to simplify
            return self

        number_placeholders = frozenset(self._number_operator_placeholders)
        new_terms = []
        for key, coeff in self._packed_terms:
            if not coeff.free_symbols:
                # If the coefficient is a constant, just keep it as is
                new_terms.append((key, coeff))
                continue
            if coeff.free_symbols <= number_placeholders:
                # There are no non-number symbols to factor out.
                new_terms.append((key, coeff))
                continue
            new_terms.append((key, sympy.factor_terms(coeff)))

        return type(self)._from_packed_terms(self.operators, new_terms)
