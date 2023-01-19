# +
import kwant
import sympy
import scipy
import numpy as np
import matplotlib.pyplot as plt
import tinyarray as ta
from collections import defaultdict
from itertools import product

import poly_kpm
import polynomial_orders_U
import lowdin
import misc


# +
def sym_to_ta(hp):
    """
    hp must be dict {sympy.Symbol: numpy.array}
    so far...
    """
    all_keys = list(sum(hp.keys()).free_symbols)
    # generate keys
    hpn = {}
    for k, v in hp.items():
        # generate key array
        nkey = np.zeros_like(all_keys)
        
        if isinstance(k,sympy.core.symbol.Symbol):
            nkey[all_keys.index(k)] = 1
        
        if isinstance(k,sympy.core.power.Pow):
            nkey[all_keys.index(k.base)] = k.exp
            
        if isinstance(k,sympy.core.mul.Mul):
            for sub in k.args:
                if isinstance(sub, sympy.core.symbol.Symbol):
                    nkey[all_keys.index(k)] = 1
                if isinstance(sub, sympy.core.power.Pow):
                    nkey[all_keys.index(k.base)] = sub.exp
                    
        hpn[ta.array(nkey)] = v
        
    return hpn, all_keys

def reassemble_to_symb(hp,keys):
    hpn = {}
    for k,v in hp.items():
        n_key = sympy.prod([keys[i]**k[i] for i in range(len(k))])
        hpn[n_key] = v
    return hpn

def gen_all_ords(order,keys):
    base = np.arange(0,order+1)
    iterat = product(base, repeat=len(keys))
    return [ta.array(v) for v in iterat if sum(v)<=order]


# +
constants = {'hbar':scipy.constants.hbar/scipy.constants.eV,
             'e':scipy.constants.e/scipy.constants.eV, # this is already 1
             'me':scipy.constants.m_e/scipy.constants.eV*(1e-9)**2,
             'muB':scipy.constants.value('Bohr magneton')/scipy.constants.eV,
             'phi_0':2*scipy.constants.value('mag. flux quantum')*(1e-9)**(-2)
            }

ge_parameters = {'gamma_1' : 13.35,
                 'gamma_2'  : 4.25,
                 'gamma_3'  : 5.69,
                 'gamma_s'  :(2*4.25+3*5.69)/5,
                 'kappa'    : 3.41,
                 'b'        : -2.2,
                 'd'        : -5.0,
                 'a_c'      : 5.65791,
                 'mu_c'     : 55.6,
                 'lambda_c' : 39.8,
                 'a_s'      : 5.430,
                 'mu_s'     : 67.5,
                 'lambda_s' : 54.5,
                 'gamma'    : 0,
                 'muB'      :scipy.constants.value('Bohr magneton in eV/T')
                }

def square_shape(R):
    def square(site):
        (x,y) = site.pos
        return np.abs(x) <= R and np.abs(y) <= R
    return square

R_c = 7
n_steps = 10

# +
Jx = sympy.Matrix(
    [[0,sympy.sqrt(3)/2,0,0],
     [sympy.sqrt(3)/2,0,1,0],
     [0,1,0,sympy.sqrt(3)/2],
     [0,0,sympy.sqrt(3)/2,0]]
)

Jy = sympy.Matrix(
    [[0,-1j*sympy.sqrt(3)/2,0,0],
     [1j*sympy.sqrt(3)/2,0,-1j,0],
     [0,1j,0,-1j*sympy.sqrt(3)/2],
     [0,0,1j*sympy.sqrt(3)/2,0]]
)

Jz = sympy.Matrix(
    [[3/2,0,0,0],
     [0,1/2,0,0],
     [0,0,-1/2,0],
     [0,0,0,-3/2]]
)

local_j = {name: sympy.Symbol(name, commutative=False) 
           for name in ['J_x', 'J_y', 'J_z']}

j_matrices = {'I_4':np.eye(4),'J_x':Jx,'J_y':Jy,'J_z':Jz}

luttinger_kohn = kwant.continuum.sympify("""hbar**2/(2*me)*(
                                                    (gamma_1+5/2*gamma_2)*(k_x**2 + k_y**2 + k_z**2) * I_4
                                                   -2 * gamma_2*(k_x**2 * J_x**2 + k_y**2 * J_y**2 + k_z**2 * J_z**2)
                                                   -gamma_3*((k_x * k_y + k_y * k_x)*(J_x * J_y + J_y * J_x)
                                                                +(k_y * k_z + k_z * k_y)*(J_y * J_z + J_z * J_y)
                                                                +(k_z * k_x + k_x * k_z)*(J_z * J_x + J_x * J_z))
                                                    )""",
                                   locals=local_j)

electric_field = kwant.continuum.sympify("""-e * (E_x * x +
                                            E_y * y +
                                            E_z * z) * I_4 """,
                                   locals = local_j)

model = kwant.continuum.sympify(str(luttinger_kohn+electric_field),locals=j_matrices)
model = sympy.expand(model)
model
# -

h0syst, h1syst = lowdin.prepare_hamiltonian(model.subs(ge_parameters|constants|{'E_z':0,'z':0}),
                                             gens=['k_z','E_x','E_y'],
                                             coords='xy',
                                             grid=R_c/n_steps,
                                             shape=square_shape(R_c),
                                             start=(0,0))
kwant.plotter.plot(h0syst)

h0 = h0syst.hamiltonian_submatrix()
h1 = {key:val.hamiltonian_submatrix() for key,val in h1syst.items()}
h0.shape

# diagonalize
eigs,vecs = scipy.linalg.eigh(h0)
H0 = np.diag(eigs)
H_p = {key: vecs.T.conj() @ val @ vecs for key,val in h1.items()}
H_p, key_map = sym_to_ta(H_p)

# +
# Vanilla polyLowdin

## separate
N_A = 4
N_B = H0.shape[-1]-N_A

H_0_AA = H0[:N_A,:N_A]
H_0_BB = H0[N_A:,N_A:]

H_p_AA = {k:v[:N_A,:N_A] for k,v in H_p.items()}
H_p_BB = {k:v[N_A:,N_A:] for k,v in H_p.items()}
H_p_AB = {k:v[:N_A,N_A:] for k,v in H_p.items()}

w_orders = gen_all_ords(3,key_map)

## generate trafo to effective

exp_S = polynomial_orders_U.compute_next_orders(H_0_AA,
                                                H_0_BB, 
                                                H_p_AA,
                                                H_p_BB,
                                                H_p_AB,
                                                wanted_orders=w_orders)


H_eff_AA, H_eff_BB, H_eff_AB = polynomial_orders_U.H_tilde(H_0_AA,
                                                            H_0_BB, 
                                                            H_p_AA,
                                                            H_p_BB,
                                                            H_p_AB,
                                                            wanted_orders=w_orders,
                                                            exp_S=exp_S,
                                                            compute_AB=True)

# +
print(key_map)
print(H_eff_AA.keys())

reassemble_to_symb(H_eff_AA,key_map)

# +
# SumOfOperatorProducts

## separate
N_A = 4
N_B = H0.shape[-1]-N_A

H_0_AA = poly_kpm.SumOfOperatorProducts([[(H0[:N_A,:N_A],'AA')]])
H_0_BB = poly_kpm.SumOfOperatorProducts([[(H0[N_A:,N_A:], 'BB')]]) 

H_p_AA = {k:poly_kpm.SumOfOperatorProducts([[(v[:N_A,:N_A], 'AA')]]) for k,v in H_p.items()}
H_p_BB = {k:poly_kpm.SumOfOperatorProducts([[(v[N_A:,N_A:], 'BB')]]) for k,v in H_p.items()}
H_p_AB = {k:poly_kpm.SumOfOperatorProducts([[(v[:N_A,N_A:], 'AB')]]) for k,v in H_p.items()}

# +
w_orders = gen_all_ords(3,key_map)

## generate trafo to effective

exp_S = polynomial_orders_U.compute_next_orders(H_0_AA,
                                                H_0_BB, 
                                                H_p_AA,
                                                H_p_BB,
                                                H_p_AB,
                                                wanted_orders=w_orders,
                                                divide_energies=lambda x:poly_kpm.divide_energies(x,H_0_AA,H_0_BB))


H_eff_AA, H_eff_BB, H_eff_AB = polynomial_orders_U.H_tilde(H_0_AA,
                                                            H_0_BB, 
                                                            H_p_AA,
                                                            H_p_BB,
                                                            H_p_AB,
                                                            wanted_orders=w_orders,
                                                            exp_S=exp_S,
                                                            compute_AB=True)
# -

import poly_kpm
t_list = poly_kpm.t_list
test = poly_kpm.SumOfOperatorProducts(t_list)

test.terms

(-test).terms


