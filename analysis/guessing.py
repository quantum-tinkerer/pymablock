#!/usr/bin/env python
# coding: utf-8
# %%
from fractions import Fraction
from itertools import count

import numpy as np
import tinyarray as ta
import sympy
from sympy import MatrixSymbol, N
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum import Operator

from polynomial_orders_U import compute_next_orders, H_tilde


# %%
# test
from sympy.physics.quantum import Operator, HermitianOperator
from operator import mul

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


# %%
class EnergyDivider:
    def __init__(self):
        self.data = {}

    def __call__(self, rhs):
        new_entry = Operator(f"V_{len(self.data) + 1}")
        self.data[new_entry] = rhs
        return new_entry


# %%
wanted_orders=[[5]]
divider = EnergyDivider()
exp_S = compute_next_orders(
    H_0_AA_s, H_0_BB_s, H_p_AA_s, H_p_BB_s, H_p_AB_s,
    wanted_orders=wanted_orders,
    divide_energies=divider,
    op=mul,
)


# %%
H_tilde_AA_s, H_tilde_BB_s = H_tilde(
    H_0_AA_s, H_0_BB_s, H_p_AA_s, H_p_BB_s, H_p_AB_s, wanted_orders, exp_S, op=mul
)


# %%
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


# %%
# divider.data = simplify_recursively(H_0_AA_s, H_0_BB_s, divider.data, divider.data)


# %%
# H_tilde_AA_s = simplify_recursively(H_0_AA_s, H_0_BB_s, H_tilde_AA_s, divider.data)


# %%
# with open("Y_orders.txt", "w") as f: 
#     for i, Y in enumerate(divider.data.values()):
#         f.write(sympy.srepr(Y))
#         f.write(", ,")


# %%
with open("Y_orders.txt") as f:
    Ys = f.read()

Ys = [sympy.sympify(data, locals=sympy.physics.quantum.__dict__) for data in Ys.split(', ,')[:-1]]

with open("H_tilde_AA_orders.txt") as f:
    Hs = f.read()

Hs = [sympy.sympify(data, locals=sympy.physics.quantum.__dict__) for data in Hs.split(', ,')[:-1]]


# %%
n = 3
sympy.Equality(Operator(f'H_{n+1}'), Hs[n].expand()) # 20/8


# %%
all_pairs = set()
for h in Hs:
    for term in h.as_ordered_terms():
        coeff, len_ = term.as_coeff_mul()[0], len(term.as_coeff_mul()[1])
        all_pairs.add((len_, coeff))

for pair in sorted(all_pairs):
    print(f"factors {pair[0]}: coeff {-pair[1]}")


# %%
orders = len(Ys)
table = np.zeros((orders, orders))

for n, Y in enumerate(Ys):
    Y = Y.expand()
    for term in Y.as_ordered_terms():
        prefactor = sympy.Rational(1)
        factor_left = sympy.Rational(1)
        ham = sympy.Rational(1)
        factor_right = sympy.Rational(1)
        len_left = 0
        len_right = 0
        past = False
        for factor in term.as_ordered_factors():
            if (not isinstance(factor, sympy.physics.quantum.operator.Operator) and
                not isinstance(factor, sympy.physics.quantum.operator.HermitianOperator) and
                not isinstance(factor, sympy.physics.quantum.dagger.Dagger)):
                prefactor *= factor
            elif factor.has(H_p_AB_s[(1,)]):
                past = True
                ham *= factor
            elif factor.has(H_p_AA_s[(1,)], H_p_BB_s[(1,)]):
                past = True
                ham *= factor
                continue
            elif not past:
                if not isinstance(factor, sympy.physics.quantum.dagger.Dagger):
                    len_left += int(repr(factor)[-1])
                else:
                    len_left += int(repr(factor)[-2])
                factor_left *= factor
            else:
                if not isinstance(factor, sympy.physics.quantum.dagger.Dagger):
                    len_right += int(repr(factor)[-1])
                else:
                    len_right += int(repr(factor)[-2])
                factor_right *= factor

        assert prefactor*factor_left*ham*factor_right == term, term
        if np.isclose(n - (len_left + len_right), 0) and not term.has(H_p_AA_s[(1,)], H_p_BB_s[(1,)]):
            table[len_left, len_right] += Fraction(prefactor)
            
table = table.tolist()


# %%
def fractional(x, y):
    if x != 0:
        return str(x)+'/'+str(y)
    else:
        return ''
teams_list = range(orders)
data = table
row_format ="{:>10}" * (len(teams_list) + 1)
print(row_format.format("", *teams_list))
print(' ')
for team, row in zip(teams_list, data):
    row = [fractional(Fraction(value).numerator, Fraction(value).denominator) for value in row]
    print(row_format.format(team, *row))


# %%
orders = len(Ys)
table = np.zeros((orders, orders))

for n, Y in enumerate(Ys):
    Y = Y.expand()
    for term in Y.as_ordered_terms():
        prefactor = sympy.Rational(1)
        factor_left = sympy.Rational(1)
        ham = sympy.Rational(1)
        factor_right = sympy.Rational(1)
        len_left = 0
        len_right = 0
        past = False
        for factor in term.as_ordered_factors():
            if (not isinstance(factor, sympy.physics.quantum.operator.Operator) and
                not isinstance(factor, sympy.physics.quantum.operator.HermitianOperator) and
                not isinstance(factor, sympy.physics.quantum.dagger.Dagger)):
                prefactor *= factor
            elif factor.has(H_p_AB_s[(1,)]):
                past = True
                ham *= factor
            elif factor.has(H_p_AA_s[(1,)], H_p_BB_s[(1,)]):
                past = True
                ham *= factor
                continue
            elif not past:
                if not isinstance(factor, sympy.physics.quantum.dagger.Dagger):
                    len_left += 1#int(repr(factor)[-1])
                else:
                    len_left += 1#int(repr(factor)[-2])
                factor_left *= factor
            else:
                if not isinstance(factor, sympy.physics.quantum.dagger.Dagger):
                    len_right += 1#int(repr(factor)[-1])
                else:
                    len_right += 1#int(repr(factor)[-2])
                factor_right *= factor

        assert prefactor*factor_left*ham*factor_right == term, term
        if not term.has(H_p_AA_s[(1,)], H_p_BB_s[(1,)]): # and np.isclose(n - (len_left + len_right), 0):
            table[n, len_left + len_right] = Fraction(prefactor)
            
table = table.tolist()


# %%
def fractional(x, y):
    if x != 0:
        return str(x)+'/'+str(y)
    else:
        return ''
teams_list = range(1, orders+1)
data = table
row_format ="{:>10}" * (len(teams_list) + 1)
print(row_format.format("", *teams_list))
print(' ')
for team, row in zip(teams_list, data):
    row = [fractional(Fraction(value).numerator, Fraction(value).denominator) for value in row]
    print(row_format.format(team, *row))

