# %%
from itertools import product

import numpy as np
import sympy
from sympy.physics.quantum.operatorordering import normal_ordered_form
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

from pymablock import block_diagonalize
# %%
def collect_constant(expr):
    """Collect constant terms in a fermionic expression.

    Parameters
    ==========

    expr : sympy expression

    Returns

    =======

    constant_terms : sympy expression
    """
    expr = normal_ordered_form(expr.expand(), independent=True)
    constant_terms = []
    for term in expr.as_ordered_terms():
        if not term.has(sympy.physics.quantum.Operator):
            constant_terms.append(term)
    return sum(constant_terms)


def matrix_elements(ham, basis):
    """Generate the matrix elements of a fermionic operator in a given basis.

    Parameters
    ==========

    ham : sympy FermionOp object
        The operator whose matrix elements are to be calculated

    basis : list of sympy FermionOp expressions
        The basis in which the matrix elements are to be calculated

    Returns

    =======

    matrix_elements : list of sympy expressions
        The matrix elements of the operator in the given basis
    """
    all_brakets = product(basis, basis)
    flat_matrix = [collect_constant(braket[0]*ham*Dagger(braket[1])) for braket in all_brakets]
    return flat_matrix

# %%
symbols = sympy.symbols(
    r"\omega_{1} \omega_{2} \alpha_{1} \alpha_{2} g",
    real=True,
    commutative=True,
    positive=True
)

omega1, omega2, alpha1, alpha2, g = symbols
# %%
a_1, a_2 = BosonOp('a_1'), BosonOp('a_2')

H_0 = []
for a, omega, alpha in zip([a_1, a_2], [omega1, omega2], [alpha1, alpha2]):
    H_0.append(omega * Dagger(a) * a + alpha * Dagger(a) * Dagger(a) * a * a)
H_0 = sum(H_0)

H_p = -g * a_1 * a_2 - g * Dagger(a_1) * Dagger(a_2) + g * a_1 * Dagger(a_2) + g * Dagger(a_1) * a_2

# %%
# Construct the matrix Hamiltonian
H_0 = normal_ordered_form(H_0.expand(), independent=True)
H_p = normal_ordered_form(H_p.expand(), independent=True)
# %%
occ_basis = [sympy.Rational(1), a_1, a_2]
unocc_basis = [a_1 * a_2, a_1 * a_1 / sympy.sqrt(2), a_2 * a_2 / sympy.sqrt(2), a_1 * a_1 * a_2 / sympy.sqrt(2), a_1 * a_2 * a_2 / sympy.sqrt(2)]
basis = occ_basis + unocc_basis

flat_matrix_0 = matrix_elements(H_0, basis)
flat_matrix_p = matrix_elements(H_p, basis)

N = len(basis)
H_0_matrix = sympy.Matrix(np.array(flat_matrix_0).reshape(N, N))
H_p_matrix = sympy.Matrix(np.array(flat_matrix_p).reshape(N, N))

H = H_0_matrix + H_p_matrix
# %%
# Perturbation theory
subspace_indices = [0, 0, 0, 1, 1, 1, 1, 1]
H_tilde, U, U_adjoint = block_diagonalize(H, subspace_indices=indices, symbols=[g])
# %%
H_tilde[0, 0, 2]
# %%
H
# %%
print(sympy.latex(H))
# %%
