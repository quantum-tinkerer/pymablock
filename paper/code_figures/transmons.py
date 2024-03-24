# %%
from itertools import product

import numpy as np
import sympy
from sympy.physics.quantum.operatorordering import normal_ordered_form
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
import matplotlib
import matplotlib.pyplot as plt

from pymablock import block_diagonalize

color_cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
figwidth = matplotlib.rcParams["figure.figsize"][0]
plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})


# %%
def collect_constant(expr):
    expr = normal_ordered_form(expr.expand(), independent=True)
    constant_terms = []
    for term in expr.as_ordered_terms():
        if not term.has(sympy.physics.quantum.Operator):
            constant_terms.append(term)
    return sum(constant_terms)


def matrix_elements(ham, basis):
    """Generate the matrix elements"""
    all_brakets = product(basis, basis)
    flat_matrix = [
        collect_constant(braket[0] * ham * Dagger(braket[1])) for braket in all_brakets
    ]
    return flat_matrix


# %%
symbols = sympy.symbols(
    r"\omega_{t} \omega_{r} \alpha g",
    real=True,
    commutative=True,
    positive=True,
)

omega_t, omega_r, alpha, g = symbols
# %%""
a_t, a_r = BosonOp("a_t"), BosonOp("a_r")

H_0 = (
    -omega_t * Dagger(a_t) * a_t
    + omega_t / 2
    + omega_r * (Dagger(a_r) * a_r + sympy.Rational(1) / 2)
)
H_0 += alpha * Dagger(a_t) * Dagger(a_t) * a_t * a_t / sympy.Rational(2)

H_p = (
    -g * a_t * a_r
    - g * Dagger(a_t) * Dagger(a_r)
    + g * a_t * Dagger(a_r)
    + g * Dagger(a_t) * a_r
)

# %%
# Construct the matrix Hamiltonian
H_0 = normal_ordered_form(H_0.expand(), independent=True)
H_p = normal_ordered_form(H_p.expand(), independent=True)
# %%
basis = [
    sympy.Rational(1),
    a_t,
    a_r,
    a_t * a_r,
    a_t * a_t / sympy.sqrt(2),
    a_r * a_r / sympy.sqrt(2),
    a_t * a_t * a_r / sympy.sqrt(2),
    a_t * a_r * a_r / sympy.sqrt(2),
    a_t * a_t * a_r * a_r / sympy.Rational(2),
]

flat_matrix_0 = matrix_elements(H_0, basis)
flat_matrix_p = matrix_elements(H_p, basis)

N = len(basis)
H_0_matrix = sympy.Matrix(np.array(flat_matrix_0).reshape(N, N))
H_p_matrix = sympy.Matrix(np.array(flat_matrix_p).reshape(N, N))

H = H_0_matrix + H_p_matrix
# %%
xi = 0
for id, sign in zip([3, 1, 2, 0], [1, -1, -1, 1]):
    subspace_indices = [1] * len(basis)
    subspace_indices[id] = 0
    H_tilde, U, U_adjoint = block_diagonalize(
        H, subspace_indices=subspace_indices, symbols=[g]
    )
    xi += sign * H_tilde[0, 0, 2][0, 0]
# %%
sympy.simplify(xi / 2)
# %%
