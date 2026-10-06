"""Scalar occupation projectors and support reduction.

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
