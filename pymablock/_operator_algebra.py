"""Private isometry protocol shared by rectangular operator arithmetic."""

from functools import wraps

import sympy


class _Isometry(sympy.Expr):
    """Structural boundary for attaching, converting, lifting and contracting.

    Subclasses provide `_attach`, `_convert_operator`, `_lift`, `_contract`,
    `_projector`, and `_source_expression` (undoing a compiled source basis).
    Operator arithmetic need not depend on a compiled basis or a solver.
    """

    is_commutative = False


def _cache_method(method):
    """Cache on the instance so dropped compiled bases can be collected."""

    @wraps(method)
    def cached(self, *args):
        memo = self.__dict__.setdefault("_memo_" + method.__name__, {})
        if args not in memo:
            memo[args] = method(self, *args)
        return memo[args]

    return cached
