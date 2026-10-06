"""Process-wide SymPy workarounds for operator arithmetic.

The Piecewise workaround is feature-detected and installed once. Remove it when
upstream handles operator-valued conditions and the Piecewise regression tests
pass without it. Other workarounds retain their upstream/version boundaries.
"""

import sympy
from packaging.specifiers import SpecifierSet
from sympy.core.logic import fuzzy_and
from sympy.functions.elementary.piecewise import ExprCondPair, Piecewise
from sympy.physics.quantum import Operator, pauli


# SymPy infers Piecewise commutativity from branches alone. Include condition
# operands, respecting scalar wrappers and unknown assumptions. Keep branch-only
# scalar assumptions from overriding this inference, regardless of query order.
def _condition_commutativity(condition):
    """Traverse Boolean structure, respecting each expression's assumptions."""
    traversal = sympy.preorder_traversal(condition)
    values = []
    for node in traversal:
        if isinstance(node, sympy.Expr):
            values.append(node.is_commutative)
            traversal.skip()
    return fuzzy_and(values)


def _piecewise_commutative(self):
    return fuzzy_and((self.expr.is_commutative, _condition_commutativity(self.cond)))


_piecewise_original_attribute = getattr(
    Piecewise, "_pymablock_original_attribute", Piecewise._eval_template_is_attr
)


def _piecewise_attribute(self, name):
    # Scalar branch assumptions must not override condition dependence.
    if fuzzy_and(_condition_commutativity(c) for _, c in self.args) is not True:
        return None
    return _piecewise_original_attribute(self, name)


def _install_piecewise_patch():
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
def _sum(self, items):  # noqa ARG001
    """Slower, but overridable version of sympy.Add."""
    if not items:
        return sympy.S.Zero
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
