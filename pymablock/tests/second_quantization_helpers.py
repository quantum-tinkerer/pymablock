"""NOF matrix inspection and independent finite-occupation reference matrices."""

from itertools import product

import numpy as np
import sympy
from scipy import sparse
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOperator,
    NumberOrderedForm,
    _NOFTransition,
    _occupation_dimension,
)


def occupation_matrices(operators, occupations):
    """Build ladder matrices, including Jordan-Wigner strings for fermions."""
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


def operator_matrix(expression, matrices):
    """Evaluate a symbolic expression using ordinary matrix arithmetic."""
    if isinstance(expression, sympy.MatrixBase):
        return sparse.bmat(
            [
                [
                    operator_matrix(expression[i, j], matrices)
                    for j in range(expression.cols)
                ]
                for i in range(expression.rows)
            ],
            format="csr",
        )
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


def nof_matrix(form, occupations=None):
    """Evaluate on an explicitly chosen product occupation basis.

    Pauli spins and fermions use their full bases by default. For infinite
    modes, supply one occupation sequence per operator, for example
    ``[range(5), range(2)]`` for an oscillator and a fermion. The resulting
    matrix is a compression of this expression, with products evaluated
    before truncation. Rows and columns follow lexicographic occupation order.
    """
    if form.embedding is not None:
        raise ValueError(
            "Convert .source to a matrix and apply the embedding basis explicitly"
        )
    if occupations is None:
        dimensions = tuple(map(_occupation_dimension, form.operators))
        if None in dimensions:
            raise ValueError("Specify occupations for infinite modes")
        occupations = tuple(range(size) for size in dimensions)
    if len(occupations) != len(form.operators):
        raise ValueError("Supply one occupation sequence per operator")
    domains = tuple(tuple(map(sympy.sympify, domain)) for domain in occupations)
    for op, domain in zip(form.operators, domains, strict=True):
        size = _occupation_dimension(op)
        if len(set(domain)) != len(domain) or any(
            not n.is_Integer
            or (not isinstance(op, LadderOp) and n < 0)
            or (size is not None and n >= size)
            for n in domain
        ):
            raise ValueError(f"Invalid occupation basis for {op}")
    states = tuple(product(*domains))
    indices = {state: i for i, state in enumerate(states)}
    result = sympy.MutableSparseMatrix(len(states), len(states), {})
    for transition in _NOFTransition.from_form(form):
        for column, state in enumerate(states):
            action = transition.apply(state)
            if action is not None and action.output_state in indices:
                result[indices[action.output_state], column] += action.weight
    return sympy.ImmutableMatrix(result)
