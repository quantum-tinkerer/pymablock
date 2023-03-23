# +
import os
from os import environ

import sympy
from sympy import MatrixSymbol, N
from sympy.physics.quantum.boson import BosonOp, BosonFockKet, BosonFockBra
from sympy.physics.quantum.operatorordering import normal_ordered_form
from sympy.physics.quantum import cartesian, qapply, InnerProduct, Operator, HermitianOperator, pauli, Dagger, Commutator
from operator import mul
import tinyarray as ta
import tinyarray as ta
import jupyterpost

from lowdin.block_diagonalization import general, to_BlockOperatorSeries, expand
from lowdin.series import _zero
# -

# # Problem setting

# +
sympy.init_printing(use_unicode=True)
# resonator frequency
wr = sympy.Symbol(
    r"\omega_r",
    real=True,
    commute=True,
    positive=True
)
# qubit frequency
wq = sympy.Symbol(
    r"\omega_q",
    real=True,
    commute=True,
    positive=True
)
# Rabi coupling
g = sympy.Symbol(
    "g",
    real=True,
    commute=True,
    positive=True
)

a = BosonOp("a")  # resonator photon annihilation operator

# Pauli matrices
sx = pauli.SigmaX()
sy = pauli.SigmaY()
sz = pauli.SigmaZ()

# qubit raising and lowering operators, notice the sympyin and qubit ladder operators are inversed
sminus = pauli.SigmaMinus()
sympylus = pauli.SigmaPlus()
# -

H_0_AA = wr * Dagger(a) * a + sympy.Rational(1 / 2) * wq
H_0_BB = wr * Dagger(a) * a - sympy.Rational(1 / 2) * wq
H_p_AB = {(1,): g *  a}
H_p_BA = {(1,): g * Dagger(a)}
H_p_AA = {}
H_p_BB = {}


# +
def exp_value(v, operator):
    return qapply(Dagger(v) * operator * v).doit().simplify()

def norm(v):
    return sympy.sqrt(exp_value(v, 1).factor()).doit().simplify()


# -

n = sympy.symbols("n", integer=True, positive=True)
basis = []
for i in [0, 1]:
    basis.append(BosonFockKet(n))


def divide_by_energies(rhs, basis=basis, H_0_AA=H_0_AA, H_0_BB=H_0_BB):
    V_AB = sympy.Rational(0)
    if rhs is not _zero:
        rhs = rhs.expand()
        terms = rhs.as_ordered_terms()
        for v in basis:
            E = exp_value(v, H_0_AA)
            for term in terms:
                v_term = qapply(term * v).doit()
                v_norm = norm(v_term)
                if not isinstance(v_norm, sympy.core.numbers.Zero):
                    v_term_norm = sympy.Rational(0)
                    for v_i in v_term.as_ordered_terms():
                        v_term_norm += v_i.as_ordered_factors()[-1]
                    v_term_norm = v_term_norm.as_ordered_factors()[-1]
                    E_term = exp_value(v_term_norm, H_0_BB)
                    denominator = (E_term - E).simplify()
                    V_AB += (term / denominator).simplify()
    return V_AB/2#.simplify()


H_tilde, U, U_adjoint = expand(to_BlockOperatorSeries(
    H_0_AA,
    H_0_BB,
    H_p_AA,
    H_p_BB,
    H_p_AB),
    solve_sylvester=divide_by_energies,
    op=mul
)

H_tilde.evaluated[0, 0, 8].simplify()

H_tilde.evaluated[1, 1, 4].simplify()


