# +
# import SymPy and define symbols
import sympy as sp
import tinyarray as ta
import polynomial_orders_U

sp.init_printing(use_unicode=True)
# resonator frequency
wr = sp.Symbol(
    r"\omega_r",
    real=True,
    commute=True,
    positive=True
)
# qubit frequency
wq = sp.Symbol(
    r"\omega_q",
    real=True,
    commute=True,
    positive=True
)
# Rabi coupling
g = sp.Symbol(
    "g",
    real=True,
    commute=True,
    positive=True
)

# +
# import operator relations and define them
from sympy.physics.quantum.boson import BosonOp, BosonFockKet, BosonFockBra

a = BosonOp("a")  # resonator photon annihilation operator
from sympy.physics.quantum import pauli, Dagger, Commutator
from sympy.physics.quantum.operatorordering import normal_ordered_form
from sympy.physics.quantum import cartesian, qapply, InnerProduct

# Pauli matrices
sx = pauli.SigmaX()
sy = pauli.SigmaY()
sz = pauli.SigmaZ()

# qubit raising and lowering operators, notice the spin and qubit ladder operators are inversed
sminus = pauli.SigmaMinus()
splus = pauli.SigmaPlus()
# -

H_0_AA = wr * Dagger(a) * a + (1 / 2) * wq
H_0_BB = wr * Dagger(a) * a - (1 / 2) * wq
H_p_AB = g * Dagger(a)
H_p_BA = g * a
H_p_AB = {ta.array([1]): H_p_AB}
H_p_BA = {ta.array([1]): H_p_BA}
H_p_AA = {}
H_p_BB = {}


# +
def exp_value(v, operator):
    return qapply(Dagger(v) * operator * v).doit().simplify()

def norm(v):
    return sp.sqrt(exp_value(v, 1).factor()).doit().simplify()


# -

n = sp.symbols("n", integer=True, positive=True)
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
    return V_AB.simplify()


from operator import mul

# +
w_orders = [ta.array([2])]

## generate trafo to effective

exp_S = polynomial_orders_U.compute_next_orders(
    H_0_AA,
    H_0_BB,
    H_p_AA,
    H_p_BB,
    H_p_AB,
    wanted_orders=w_orders,
    divide_energies=divide_by_energies,
    op=mul
)
# -

H_eff_AA, H_eff_BB = polynomial_orders_U.H_tilde(
    H_0_AA,
    H_0_BB,
    H_p_AA,
    H_p_BB,
    H_p_AB,
    wanted_orders=w_orders,
    exp_S=exp_S,
    compute_AB=False,
    op=mul
)

# +
for value in H_eff_AA.values():
    H_AA = value
    H_AA = normal_ordered_form(sp.nsimplify(H_AA).expand()).factor().simplify()

for value in H_eff_BB.values():
    H_BB = value
    H_BB = normal_ordered_form(sp.nsimplify(H_BB).expand()).factor().simplify()
# -

H_AA.factor()

H_BB.factor()
