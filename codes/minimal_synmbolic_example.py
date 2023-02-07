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

from polynomial_orders_U import compute_next_orders, H_tilde
# -

os.environ["JPY_API_TOKEN"] = "d0797e84f32243aebbb3711203615dab"
os.environ["JUPYTERPOST_URL"] = "https://services.io.quantumtinkerer.group/services/jupyterpost"

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

H_0_AA = wr * Dagger(a) * a + (1 / 2) * wq
H_0_BB = wr * Dagger(a) * a - (1 / 2) * wq
H_p_AB_term = g * Dagger(a)
H_p_BA_term = g * a
H_p_AB = {ta.array([1]): H_p_AB_term}
H_p_BA = {ta.array([1]): H_p_BA_term}
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
    V_AB = 0
    terms = rhs.as_ordered_terms()
    for v in basis:
        E = exp_value(v, H_0_AA)
        for term in terms:
            v_term = qapply(term * v).doit()
            v_norm = norm(v_term)
            if v_norm != 0:
                v_term_norm = 0
                for v_i in v_term.as_ordered_terms():
                    v_term_norm += v_i.as_ordered_factors()[-1]
                v_term_norm = v_term_norm.as_ordered_factors()[-1]
                E_term = exp_value(v_term_norm, H_0_BB)
                denominator = (E_term - E).simplify()
                V_AB += (term / denominator).simplify()
    return V_AB/2#.simplify()


# +
wanted_orders = [ta.array([4])]

exp_S = compute_next_orders(
    H_0_AA,
    H_0_BB,
    H_p_AA,
    H_p_BB,
    H_p_AB,
    wanted_orders=wanted_orders,
    divide_energies=divide_by_energies,
    op=mul
)

# +
# H_eff_AA, H_eff_BB = H_tilde(
#     H_0_AA,
#     H_0_BB,
#     H_p_AA,
#     H_p_BB,
#     H_p_AB,
#     wanted_orders=w_orders,
#     exp_S=exp_S,
#     compute_AB=False,
#     op=mul
# )
# -

# ### Relevant results
#
# We will use $\{V_n\}$ to compute $\tilde{H}_n^{AA}$ from an explicit expression that we will derive, so these terms are all we care about.

Vs = {}
for order, V in exp_S[0, 1].items():
    Vs[Operator(f"V_{order[0]}")] = V
    display(sympy.Equality(Operator(f"V_{order[0]}"), V))
    print(' ')

# Directly generating $\tilde{H}_{n}^{AA}$ produces a lot of terms, so we temporarily decide to recursively solve the general equations, to then replace the solutions $\{V_n\}$.

# ## Recursive simplifications, get rid of $H_0$ in general expression

# +
dim1 = 2
dim2 = 2

H_0_AA_s = HermitianOperator("{H_{0}^{AA}}")
H_0_BB_s = HermitianOperator("{H_{0}^{BB}}")

H_p_AA_s = {}
H_p_BB_s = {}
H_p_AB_s = {}

H_p_AA_s[ta.array([1])] = HermitianOperator("{H_{p}^{AA}}")
H_p_BB_s[ta.array([1])] = HermitianOperator("{H_{p}^{BB}}")   
H_p_AB_s[ta.array([1])] = Operator("{H_{p}^{AB}}")


# -

class EnergyDivider:
    def __init__(self):
        self.data = {}

    def __call__(self, rhs):
        new_entry = Operator(f"V_{len(self.data) + 1}")
        self.data[new_entry] = rhs
        return new_entry


# +
divider = EnergyDivider()
exp_S_s = compute_next_orders(
    H_0_AA_s, H_0_BB_s, H_p_AA_s, H_p_BB_s, H_p_AB_s,
    wanted_orders=wanted_orders,
    divide_energies=divider,
    op=mul,
)

H_tilde_AA_s, H_tilde_BB_s = H_tilde(
    H_0_AA_s, H_0_BB_s, H_p_AA_s, H_p_BB_s, H_p_AB_s, wanted_orders, exp_S_s, op=mul
)


# +
# sympy.simplify(list(divider.data.values())[-1])
# -

def simplify_recursively(H_0_AA, H_0_BB, dictionary, divider_data):
    i = 1
    for key in dictionary.keys():
        for _ in range(i):
            value = dictionary[key]
            value = sympy.expand(value.subs({H_0_AA * v: v * H_0_BB + rhs for v, rhs in divider_data.items()})) # H_0_AA to right
            value = sympy.expand(value.subs({H_0_BB * Dagger(v): Dagger(v) * H_0_AA - Dagger(rhs) for v, rhs in divider_data.items()})) # H_0_BB to right
            dictionary.update({key: value})
        # display(dictionary[key])
        i += 1
    return dictionary


divider.data = simplify_recursively(H_0_AA_s, H_0_BB_s, divider.data, divider.data)

H_tilde_AA_s = simplify_recursively(H_0_AA_s, H_0_BB_s, H_tilde_AA_s, divider.data)

# ### General expression for Hamiltonian
# This is  a general expression for $\tilde{H}_{4}^{AA}$, so we will replace the $\{V_n\}$ computed earlier for this problem.

expr = H_tilde_AA_s[(4,)]
expr

# ### Problem specific solutions

i = 1
for v, rhs in divider.data.items():
    rhs = sympy.expand(rhs.subs({key: value for key, value in Vs.items()}))
    rhs = sympy.expand(rhs.subs({H_p_AA_s[(1,)]: N(0), H_p_BB_s[(1,)]: N(0), H_p_AB_s[(1,)]: H_p_AB_term}))
    divider.data.update({v: rhs})
    display(sympy.Equality(Operator(f"Y_{{{i}}}"), rhs))
    i += 1

i = 1
for order, H in H_tilde_AA_s.items():
    H = sympy.expand(H.subs({key: divide_by_energies(value) for key, value in divider.data.items()}))
    H = sympy.expand(H.subs({H_p_AA_s[(1,)]: N(0), H_p_BB_s[(1,)]: N(0), H_p_AB_s[(1,)]: H_p_AB_term}))
    H = normal_ordered_form(H.simplify().factor()).simplify()
    H_tilde_AA_s.update({order: H})
    display(sympy.Equality(Operator(f"H_{{{i}}}"), H))
    i += 1
