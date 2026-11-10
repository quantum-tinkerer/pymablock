"""Scalar occupation coefficients and their pole-handling contract.

Coefficients are amplitudes at middle occupations. Division first applies
occupations selected by indicators, including truncated boson levels and reference
lists. A defined zero transition has zero solution even at zero gap. An exposed
zero gap with nonzero coefficient raises ValueError; unresolved resonances remain
symbolic poles. Scalar parameters are generic. No binary sectors or integer roots
are enumerated, so a returned expression does not certify nonresonance everywhere.

Occupation conditions have four Piecewise shapes:

* Indicators: ((1, Eq(x, v)), (0, True)); sums and products build projectors.
* Boundary guards: ((0, Eq(n, i) | ...), (c, True)), suppressing poles where
  downward ladder factors vanish.
* Inactive transitions: ((0, inactive), (c / gap, True)).
* Point splits: ((value at v, Eq(n, v)), (rest, True)).

Only indicators are read back as fixed occupations. Other shapes remain branches:
multiplying a pole by a zero indicator would evaluate 0 * infinity. Affine
indicators split active sectors; indicator algebra recognizes zero coefficients
before a zero-gap error and zero maps on their retained domain. Some empty
combinations (mutually exclusive floor conditions, or delta(n+m, 1) * delta(n, 2)
with nonnegative occupations) are not recognized and can cause a spurious
ValueError. Nothing is evaluated outside the sector it belongs to.
"""

from collections.abc import Collection, Iterator, Mapping, Sequence
from functools import lru_cache

import sympy
from sympy.functions.elementary.piecewise import Piecewise


# The two scans below run on every coefficient built during perturbation theory.
# `Basic.has` walks the whole tree on each call, while coefficients of successive
# orders share most of their subtrees. Caching the recursion per node visits each
# shared subtree once; this removes most of the scanning overhead in spin and
# fermion models. The bounded caches hold only scalar SymPy expressions.
@lru_cache(maxsize=4096)
def _is_singular(expression: sympy.Basic) -> bool:
    """Return whether an evaluated expression contains an infinity or NaN."""
    return expression in (sympy.zoo, sympy.nan, sympy.oo, -sympy.oo) or any(
        _is_singular(arg) for arg in expression.args
    )


@lru_cache(maxsize=4096)
def _has_piecewise(expression: sympy.Basic) -> bool:
    """Return whether an expression contains a Piecewise node."""
    return isinstance(expression, Piecewise) or any(
        _has_piecewise(arg) for arg in expression.args
    )


def _equal_value_indicator(left: sympy.Expr, right: sympy.Expr | int) -> sympy.Expr:
    """Return the scalar indicator of ``left == right`` (1 if equal, else 0).

    Inputs are scalar occupation expressions or values, not quantum operators.
    A symbolic condition is represented by a two-branch SymPy Piecewise.
    """
    return sympy.Piecewise((1, sympy.Eq(left, right)), (0, True))


def _is_indicator(expression: Piecewise) -> bool:
    """Return whether a Piecewise has the two-branch 0/1 indicator shape."""
    return (
        len(expression.args) == 2
        and expression.args[0].expr == 1
        and expression.args[1] == (0, sympy.true)
    )


def _iter_fixed_number_indicators(
    expression: sympy.Expr, number_symbols: Collection[sympy.Symbol]
) -> Iterator[tuple[sympy.Piecewise, sympy.Symbol, sympy.Expr]]:
    """Find 0/1 Piecewise indicators that fix one number symbol to a value.

    Yield ``(indicator, number_symbol, selected_value)`` for equalities linear
    in one supplied symbol with a nonzero numeric slope; the selected value may
    depend affinely on other occupations. Other conditional
    expressions are ignored; no general equation solving is attempted.
    """
    for delta in expression.atoms(sympy.Piecewise) if number_symbols else ():
        if not _is_indicator(delta) or not isinstance(delta.args[0].cond, sympy.Equality):
            continue
        equation = sympy.expand(delta.args[0].cond.lhs - delta.args[0].cond.rhs)
        for n in sorted(
            set(number_symbols) & delta.free_symbols, key=sympy.default_sort_key
        ):
            slope = equation.coeff(n)
            if slope.is_number and slope and not (equation - slope * n).has(n):
                yield delta, n, sympy.cancel(n - equation / slope)
                break


def _vanishes_by_indicator_algebra(
    coefficient: sympy.Expr, number_symbols: Collection[sympy.Symbol] = ()
) -> bool:
    """Return whether a coefficient is zero by the algebra of its indicators.

    An indicator of an affine occupation equality splits the coefficient into
    the sector where it holds, evaluated by substitution, and its complement.
    The remaining indicators only use ``delta**2 == delta``, which SymPy applies
    when expanding. Some empty combinations, such as mutually exclusive floor
    conditions or ``delta(n + m, 1) * delta(n, 2)`` with nonnegative occupations,
    are not recognized. Nothing is evaluated outside the sector it belongs to.
    """
    if coefficient == 0:
        return True
    for delta, n, value in _iter_fixed_number_indicators(coefficient, number_symbols):
        return _vanishes_by_indicator_algebra(
            coefficient.xreplace({delta: sympy.S.Zero}), number_symbols
        ) and _vanishes_by_indicator_algebra(
            coefficient.xreplace({n: value}), number_symbols
        )
    indicators = [delta for delta in coefficient.atoms(Piecewise) if _is_indicator(delta)]
    if not indicators:
        return False
    try:
        return sympy.Poly(coefficient, *indicators, domain=sympy.EXRAW).is_zero
    except sympy.polys.polyerrors.PolynomialError:
        return False


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


def _divide_by_energy_gap(
    coefficient: sympy.Expr,
    energy_gap: sympy.Expr,
    number_symbols: Sequence[sympy.Symbol] = (),
    nonnegative_numbers: Sequence[sympy.Symbol] = (),
) -> sympy.Expr:
    """Return coefficient / energy_gap, choosing zero for inactive 0/0 channels.

    Ordinary cancellation can lose an inactive transition: n/n must be zero at
    n=0, not one everywhere. Occupations fixed by indicators are evaluated
    directly; otherwise the quotient is conditional on the transition being active.
    The NOF coefficient is the amplitude at the middle occupations.
    ``number_symbols`` are occupation coordinates
    and other symbols are generic parameters; ``nonnegative_numbers`` only avoids
    unnecessary conditionals. A literal zero gap with a nonzero coefficient raises
    ValueError; other resonances remain symbolic poles and are not searched for.
    """
    if coefficient == 0:
        return sympy.S.Zero
    energy_gap = sympy.cancel(energy_gap)
    arguments = coefficient, energy_gap

    def divide_after_substitution(
        substitution: Mapping[sympy.Expr, sympy.Expr],
    ) -> sympy.Expr:
        selected_coefficient, selected_gap = (
            expression.xreplace(substitution) for expression in arguments
        )
        return _divide_by_energy_gap(
            selected_coefficient,
            selected_gap,
            number_symbols,
            nonnegative_numbers,
        )

    variables = set(number_symbols)
    point = next(
        _iter_fixed_number_indicators(coefficient, variables & energy_gap.free_symbols),
        None,
    )
    if point is not None:
        delta, n, v = point
        return sympy.Piecewise(
            (divide_after_substitution({n: v}), sympy.Eq(n, v)),
            (divide_after_substitution({delta: sympy.S.Zero}), True),
        )
    if energy_gap == 0:
        if _vanishes_by_indicator_algebra(coefficient, variables):
            return sympy.S.Zero
        raise ValueError(
            "Cannot solve the Sylvester equation: the right-hand side is nonzero "
            "but the energy difference is zero (degenerate channel)."
        )
    # Generic scalar parameters are not tuned to occupation resonances. Keep
    # their quotients compact rather than expanding conditional expressions.
    gap = energy_gap.as_numer_denom()[0]
    for parameter in gap.free_symbols - variables:
        parameter_coefficient = gap.coeff(parameter)
        if (
            parameter_coefficient.is_Atom
            and parameter_coefficient.is_zero is False
            and not parameter_coefficient.free_symbols & variables
        ):
            return coefficient / energy_gap
    quotient = coefficient / energy_gap
    if not energy_gap.free_symbols & variables:
        return quotient
    # A constant plus occupations with the same sign cannot vanish. Inspect
    # numeric coefficients only, rather than asking the assumptions engine to
    # prove a general expression nonzero.
    constant, rest = energy_gap.as_coeff_Add()
    if constant and all(
        factor in set(nonnegative_numbers)
        and (coefficient * constant).is_positive is True
        for coefficient, factor in (
            term.as_coeff_Mul() for term in sympy.Add.make_args(rest)
        )
    ):
        return quotient
    factors = (factor.as_numer_denom()[0] for factor in sympy.Mul.make_args(coefficient))
    inactive = sympy.Or(
        *(
            sympy.Eq(factor, 0, evaluate=False)
            for factor in factors
            if factor.free_symbols & variables
        )
    )
    # A meromorphic coefficient has no defined zero at its poles. Its numerator
    # can vanish there without making the virtual channel inactive.
    coefficient_denominator = sympy.together(coefficient).as_numer_denom()[1]
    inactive = sympy.And(inactive, sympy.Ne(coefficient_denominator, 0))
    return sympy.Piecewise((0, inactive), (quotient, True), evaluate=False)
