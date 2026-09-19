"""Independent finite-matrix checks for the applications notebooks.

This file accompanies the downloadable notebooks. Source expressions are
interpreted directly as products of occupation matrices, without the embedding
compiler or number-ordered multiplication.
"""

from itertools import product

import numpy as np
import sympy
from scipy import sparse
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm, SpinOp


def occupation_matrices(operators, occupations):
    """Construct local ladder matrices and fermionic Jordan-Wigner strings."""
    identities = [sparse.eye(len(values), format="csr") for values in occupations]
    matrices = {}

    def tensor(factors):
        result = sparse.csr_matrix([[1.0]])
        for factor in factors:
            result = sparse.kron(result, factor, format="csr")
        return result

    for index, (operator, values) in enumerate(zip(operators, occupations, strict=True)):
        weights = np.ones(len(values) - 1)
        if isinstance(operator, BosonOp):
            weights = np.sqrt(values[1:])
        elif isinstance(operator, SpinOp):
            n = np.asarray(values[1:])
            weights = np.sqrt(n * (2 * float(operator.spin) + 1 - n))
        factors = identities.copy()
        factors[index] = sparse.diags(weights, 1, shape=(len(values), len(values)))
        if isinstance(operator, FermionOp):
            for previous in range(index):
                if isinstance(operators[previous], FermionOp):
                    factors[previous] = sparse.diags(
                        (-1.0) ** np.asarray(occupations[previous])
                    )
        matrices[operator] = tensor(factors)
        matrices[operator.adjoint()] = matrices[operator].conj().T
        factors = identities.copy()
        factors[index] = sparse.diags(values, dtype=float)
        matrices[NumberOperator(operator)] = tensor(factors)
    matrices[sympy.S.One] = tensor(identities)
    return matrices


def operator_matrix(expression, matrices):
    """Evaluate an expression using ordinary sparse-matrix products."""
    expression = sympy.sympify(expression)
    if isinstance(expression, NumberOrderedForm):
        expression = expression.as_expr()
    if expression in matrices:
        return matrices[expression]
    if expression.is_number and expression.is_commutative:
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


def second_order(energy, perturbation, retained):
    """Evaluate the Hermitian second-order resolvent sum for diagonal H0."""
    discarded = [i for i in range(len(energy)) if i not in retained]
    pq = perturbation[np.ix_(retained, discarded)]
    gaps = energy[retained, None] - energy[None, discarded]
    active = np.abs(pq) > 1e-13
    if np.any(np.abs(gaps[active]) < 1e-12):
        raise ValueError("A coupled virtual state is resonant")
    divided = np.zeros_like(pq, dtype=complex)
    divided[active] = pq[active] / gaps[active]
    result = divided @ pq.conj().T
    return (result + result.conj().T) / 2


def fourth_order_degenerate(energy, perturbation, retained):
    """Fourth-order resolvent formula when P is degenerate and P V P = 0."""
    np.testing.assert_allclose(energy[retained], energy[retained[0]], atol=1e-12)
    np.testing.assert_allclose(perturbation[np.ix_(retained, retained)], 0, atol=1e-12)
    discarded = [i for i in range(len(energy)) if i not in retained]
    gaps = energy[retained[0]] - energy[discarded]
    if np.min(np.abs(gaps)) < 1e-12:
        raise ValueError("Include the whole degenerate energy subspace in P")
    pq = perturbation[np.ix_(retained, discarded)]
    qq = perturbation[np.ix_(discarded, discarded)]
    r = np.diag(1 / gaps)
    h2 = pq @ r @ pq.conj().T
    norm = pq @ r @ r @ pq.conj().T
    return pq @ r @ qq @ r @ qq @ r @ pq.conj().T - (norm @ h2 + h2 @ norm) / 2


def occupation_indices(occupations, selected):
    """Locate explicitly specified product states in lexicographic order."""
    states = tuple(product(*occupations))
    return [states.index(tuple(state)) for state in selected]
