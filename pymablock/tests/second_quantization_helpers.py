"""Occupation matrices and operator actions independent of number-ordered arithmetic."""

import numpy as np
import sympy
from scipy import sparse
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm


def occupation_matrices(operators, occupations):
    """Build ladder matrices, including Jordan-Wigner strings for fermion_mask."""
    identities = [sparse.eye(len(values), format="csr") for values in occupations]
    matrices = {}

    def tensor(factors):
        result = sparse.csr_matrix([[1.0]])
        for factor in factors:
            result = sparse.kron(result, factor, format="csr")
        return result

    for index, (operator, values) in enumerate(zip(operators, occupations)):
        weights = (
            np.sqrt(values[1:])
            if isinstance(operator, BosonOp)
            else np.ones(len(values) - 1)
        )
        factors = identities.copy()
        factors[index] = sparse.diags(weights, 1, shape=(len(values), len(values)))
        if isinstance(operator, FermionOp):
            for previous in range(index):
                if isinstance(operators[previous], FermionOp):
                    factors[previous] = sparse.diags(
                        (-1.0) ** np.asarray(occupations[previous])
                    )
        matrices[operator] = tensor(factors)
        matrices[operator.adjoint()] = matrices[operator].T
        factors = identities.copy()
        factors[index] = sparse.diags(values, dtype=float)
        matrices[NumberOperator(operator)] = tensor(factors)
    matrices[sympy.S.One] = tensor(identities)
    return matrices


def apply_local_factors(factors, state, fermion_mask):
    """Apply a sequence of local factors to one occupation state, right to left.

    Each (mode, factor) pair denotes annihilation (1), creation (2), or number
    (3). The fermion mask selects the modes that carry Jordan-Wigner signs.
    Return the final occupation bits and amplitude, or (None, 0).
    """
    sign = 1
    for mode, factor in reversed(factors):
        occupied = (state >> mode) & 1
        if (factor in (1, 3) and not occupied) or (factor == 2 and occupied):
            return None, 0
        if factor != 3:
            if fermion_mask & (1 << mode):
                preceding = state & ((1 << mode) - 1) & fermion_mask
                sign *= -1 if preceding.bit_count() % 2 else 1
            state ^= 1 << mode
    return state, sign


def operator_matrix(expression, matrices):
    """Evaluate a symbolic expression using ordinary matrix arithmetic."""
    if isinstance(expression, NumberOrderedForm):
        expression = expression.as_expr()
    if expression in matrices:
        return matrices[expression]
    if expression.is_number:
        return complex(expression) * matrices[sympy.S.One]
    if expression.is_Add:
        return sum(operator_matrix(term, matrices) for term in expression.args)
    if expression.is_Mul:
        result = matrices[sympy.S.One]
        for factor in expression.args:
            result = result @ operator_matrix(factor, matrices)
        return result
    if expression.is_Pow and expression.exp.is_Integer and expression.exp >= 0:
        return operator_matrix(expression.base, matrices) ** int(expression.exp)
    raise ValueError(f"Unsupported matrix reference expression: {expression}")
