# %% [markdown]
# # Jaynes-Cummings model

# %%
from operator import mul

import sympy
from sympy.physics.quantum.boson import BosonOp, BosonFockKet
from sympy.physics.quantum import qapply, Dagger

from lowdin.block_diagonalization import to_BlockSeries, expanded
from lowdin.series import zero

# %%
# resonator frequency
wr = sympy.Symbol(
    r"{\omega_r}",
    real=True,
    commute=True,
    positive=True
)

# qubit frequency
wq = sympy.Symbol(
    r"{\omega_q}",
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

# %%
H_0_AA = wr * Dagger(a) * a + sympy.Rational(1 / 2) * wq
H_0_BB = wr * Dagger(a) * a - sympy.Rational(1 / 2) * wq
H_p_AB = {(1,): g * Dagger(a)}
H_p_BA = {(1,): g * a}
H_p_AA = {}
H_p_BB = {}

H = to_BlockSeries(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB)

# %%
n = sympy.symbols("n", integer=True, positive=True)
basis_ket = BosonFockKet(n)


def expectation_value(v, operator):
    return qapply(Dagger(v) * operator * v).doit().simplify()


def norm(v):
    return sympy.sqrt(expectation_value(v, 1).factor()).doit().simplify()

def solve_sylvester(rhs):
    """
    Solves Sylvester's Equation
    rhs : zero or sympy expression

    Returns:
    V : zero or sympy expression for off-diagonal block of unitary transformation
    """
    if zero == rhs:
        return rhs

    E_i = expectation_value(basis_ket, H_0_AA)
    V = []
    for term in rhs.expand().as_ordered_terms():
        term_on_basis = qapply(term * basis_ket).doit()
        if not isinstance(norm(term_on_basis), sympy.core.numbers.Zero):
            normalized_ket = term_on_basis.as_ordered_factors()[-1]
            E_j = expectation_value(normalized_ket, H_0_BB)
            V.append(term / (E_j - E_i))
    return sum(V)


# %%
H_tilde, U, U_adjoint = expanded(H, solve_sylvester=solve_sylvester, op=mul)

# %%
H_tilde.evaluated[0, 0, 4].expand().simplify()
