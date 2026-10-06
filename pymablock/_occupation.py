"""Scalar occupation projectors and partial energy-gap division.

Coordinates and their domains come from the operator representation. This module
uses scalar SymPy expressions and never constructs quantum operators.
"""

from functools import lru_cache

import sympy


def _occupation_projector(left, right):
    return sympy.Piecewise((1, sympy.Eq(left, right)), (0, True))


def _spectral_projector(expression, spectrum):
    """Select a finite spectrum or the integers for a diagonal expression."""
    if spectrum is sympy.S.Integers:
        return _occupation_projector(expression, sympy.floor(expression))
    return sum(_occupation_projector(expression, value) for value in spectrum)


def _projectors(expression, numbers):
    """Yield point indicators and the occupation value they select."""
    for delta in expression.atoms(sympy.Piecewise) if numbers else ():
        if (
            len(delta.args) != 2
            or delta.args[0].expr != 1
            or delta.args[1] != (0, sympy.true)
            or not isinstance(delta.args[0].cond, sympy.Equality)
        ):
            continue
        variables = set(numbers) & delta.free_symbols
        if len(variables) != 1:
            continue
        (n,) = variables
        equation = sympy.expand(delta.args[0].cond.lhs - delta.args[0].cond.rhs)
        slope = equation.coeff(n)
        if slope.is_number and slope and not (equation - slope * n).has(n):
            yield delta, n, sympy.cancel(n - equation / slope)


@lru_cache(maxsize=1024)
def _reduce_projectors(coefficient, numbers, nonnegative):
    """Evaluate occupation functions on the support of point projectors."""
    replacements, points = {}, {}
    for delta, n, value in _projectors(coefficient, numbers):
        if not value.is_number:
            continue
        if value.is_integer is False or (n in nonnegative and value < 0):
            replacements[delta] = sympy.S.Zero
        else:
            replacements[delta] = _occupation_projector(n, value)
            points.setdefault(n, set()).add(value)
    coefficient = coefficient.xreplace(replacements)
    for n, values in sorted(
        points.items(), key=lambda item: sympy.default_sort_key(item[0])
    ):
        background = coefficient.xreplace(
            {_occupation_projector(n, v): sympy.S.Zero for v in values}
        )
        coefficient = background + sum(
            _occupation_projector(n, v)
            * (coefficient.xreplace({n: v}) - background.xreplace({n: v}))
            for v in sorted(values, key=sympy.default_sort_key)
        )
    return coefficient


def _divide_coefficients(
    numerator, denominator, binary=(), coordinates=(), weight=None, nonnegative=()
):
    """Divide using local support checks and a symbolic fallback.

    Choose zero where ``weight`` vanishes, including at zero gaps. The weight
    defaults to the numerator; embeddings include ladder amplitudes in it.
    Resolve exposed zero gaps and explicit point support, but leave unresolved
    resonances as poles. Parameters other than occupation coordinates are generic.
    This is a partial symbolic solver, not an exhaustive nonresonance check.
    """
    weight = numerator if weight is None else weight
    if weight == 0:
        return sympy.S.Zero
    denominator = sympy.cancel(denominator)
    arguments = numerator, denominator, weight

    def at(substitution):
        c, d, w = (x.xreplace(substitution) for x in arguments)
        return _divide_coefficients(c, d, binary, coordinates, w, nonnegative)

    variables = set(coordinates) | set(binary)
    gap = denominator.as_numer_denom()[0]
    for parameter in gap.free_symbols - variables:
        coefficient = gap.coeff(parameter)
        if (
            coefficient.is_Atom
            and coefficient.is_zero is False
            and not coefficient.free_symbols & variables
        ):
            return numerator / denominator
    for n in binary:
        if n not in weight.free_symbols | denominator.free_symbols:
            continue
        if denominator == 0 or any(
            x.xreplace({n: v}) == 0
            for x in (weight, denominator)
            for v in (sympy.S.Zero, sympy.S.One)
        ):
            return (1 - n) * at({n: sympy.S.Zero}) + n * at({n: sympy.S.One})
    point = next(_projectors(numerator, variables & denominator.free_symbols), None)
    if point is not None:
        delta, n, v = point
        return sympy.Piecewise(
            (at({n: v}), sympy.Eq(n, v)), (at({delta: sympy.S.Zero}), True)
        )
    if denominator == 0:
        raise ValueError(
            "Cannot solve the Sylvester equation: the right-hand side is nonzero "
            "but the energy difference is zero (degenerate channel)."
        )
    quotient = numerator / denominator
    if not denominator.free_symbols & variables:
        return quotient
    # A constant plus occupations with the same sign cannot vanish. Inspect
    # numeric coefficients only, rather than asking the assumptions engine to
    # prove a general expression nonzero.
    constant, rest = denominator.as_coeff_Add()
    if constant and all(
        factor in set(nonnegative) | set(binary)
        and (coefficient * constant).is_positive is True
        for coefficient, factor in (
            term.as_coeff_Mul() for term in sympy.Add.make_args(rest)
        )
    ):
        return quotient
    factors = (factor.as_numer_denom()[0] for factor in sympy.Mul.make_args(weight))
    inactive = sympy.Or(
        *(
            sympy.Eq(factor, 0, evaluate=False)
            for factor in factors
            if factor.free_symbols & variables
        )
    )
    # A meromorphic weight has no defined zero at its poles. Its numerator
    # can vanish there without making the virtual channel inactive.
    weight_denominator = sympy.together(weight).as_numer_denom()[1]
    inactive = sympy.And(inactive, sympy.Ne(weight_denominator, 0))
    return sympy.Piecewise((0, inactive), (quotient, True), evaluate=False)
