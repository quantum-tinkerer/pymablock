# +
import kwant
import sympy
import scipy
import qsymm
import numpy as np
import matplotlib.pyplot as plt
import tinyarray as ta
from collections import defaultdict
from itertools import product
from functools import reduce

import poly_kpm
import polynomial_orders_U

# + endofcell="--"
Jx = np.array(
    [[0,np.sqrt(3)/2,0,0],
     [np.sqrt(3)/2,0,1,0],
     [0,1,0,np.sqrt(3)/2],
     [0,0,np.sqrt(3)/2,0]]
)

Jy = np.array(
    [[0,-1j*np.sqrt(3)/2,0,0],
     [1j*np.sqrt(3)/2,0,-1j,0],
     [0,1j,0,-1j*np.sqrt(3)/2],
     [0,0,1j*np.sqrt(3)/2,0]]
)

Jz = np.array(
    [[3/2,0,0,0],
     [0,1/2,0,0],
     [0,0,-1/2,0],
     [0,0,0,-3/2]]
)
j_vec = np.array([Jx,Jy,Jz])

id2 = np.eye(2)
sx = np.array([[0,1],
               [1,0]])

sy = np.array([[0,-1j],
               [1j,0]])

sz = np.array([[1,0],
               [0,-1]])

sig_vec = np.array([id2,sx,sy,sz])


# -

def hilbert_schmidt_transform(matrix,decomp):
    """
    Hilbert-Schmidt transform from general operator to its representation of a 
    decomposition into \bigotimes SU(N) generators

    INPUTS:
    matrix:     np.ndarray, qsymm.Model, dict
                Object to be transformed

    decomp:     list of int
                integer dimensions that are to be replaced by SU(N) generators 

    """

    if isinstance(matrix,qsymm.Model) or isinstance(matrix,dict):
        for k, v in matrix.items():
            matrix[k] = hilbert_schmidt_transform(v,decomp)
        return matrix

    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    assert matrix.shape[0] == matrix.shape[1]

    decomp = list(map(int, str(decomp).split('+')))
    dim = np.prod(decomp)
    decomp = replace_with_gens(decomp)

    multi_kron = lambda M: reduce(np.kron, M)

    gen = (np.trace(matrix@multi_kron(M)*np.sqrt(2/dim)) for M in product(*decomp))

    tensor = np.fromiter(gen, dtype=complex).reshape(*[len(i) for i in decomp])

    norm = np.ones_like(tensor)/2
    zero_ind = tuple(np.zeros(shape=(len(tensor.shape)), dtype=int))
    norm[zero_ind] = 1/dim

    tensor = tensor*norm

    return tensor


def replace_with_gens(decomp):
    """
    Replace the integers in list by list of SU(N) generators of that dimension

    INPUT:
    decomp:     list of int
                integer dimensions that are to be replaced by SU(N) generators 

    RETURNS:
    decomp:     list of lists
                each integer is replaced by a list of generators of the SU(N), that is
                SU(2) : Pauli-matrices
                SU(3) : Gell-man matrices
                SU(4) : To be implemented

                The normalization of the generators under the Hilbert-Schmidt product
                is chosen such that Tr[a_i @ a_j] = 2 delta_{i,j}
    """
    su_2 = [np.eye(2),
            np.array([[0,1],
                      [1,0]]),
            np.array([[0,-1j],
                      [1j,0]]),
            np.array([[1,0],
                      [0,-1]])]

    su_3 = [np.eye(3),
           np.array([[0,1,0],
                     [1,0,0],
                     [0,0,0]]),
           np.array([[0,-1j,0],
                     [1j,0,0],
                     [0,0,0]]),
           np.array([[1,0,0],
                     [0,-1,0],
                     [0,0,0]]),
           np.array([[0,0,1],
                     [0,0,0],
                     [1,0,0]]),
           np.array([[0,0,-1j],
                     [0,0,0],
                     [1j,0,0]]),
           np.array([[0,0,0],
                     [0,0,1],
                     [0,1,0]]),
           np.array([[0,0,0],
                     [0,0,-1j],
                     [0,1j,0]]),
           np.array([[1,0,0],
                     [0,1,0],
                     [0,0,-2]])/np.sqrt(3)
           ]

    for i in range(len(decomp)):
        if decomp[i] == 2:
            decomp[i] = su_2
            continue
        if decomp[i] == 3:
            decomp[i] = su_3
            continue
        if decomp[i] >= 4:
            raise ValueError('Cannot decompose into SU(4) or above.')
            continue

    return decomp


def represent_hilbert_schmidt(matrix,decomp,symbols,prec=3):
    """
    Represent a Hamiltonian that either is or is not expanded in SU(N) generator as such
    and return them in a symbolic representation.

    INPUTS:
    matrix:     np.ndarray, qsymm.Model, dict
                Either Hamiltonian (the method determines this by checking hermiticity)
                or Hilbert-Schmidt representation of such
    decomp:     str; example '2+3+2'
                String that determines the decomposition. Must have form 'n_1+...+n_m'.
                The n_i determine the dimension of the SU(N) that shall be taken to
                represent the operator in. As such, a decomp='2+3+2' is the same as 
                expanding the Hamiltonian in the operator basis spanned by 
                SU(2) \otimes SU(3) \otimes SU(2).
    symbols:    list of strings
                List of your favorite symbols to use for the basis operators.
    prec:       int=3
                number of decimals that is kept upon rounding. 

    RETURNS:
    rep:        np.array, qsymm.Model, dict
                Object of same type as input in which the numeric matrices have been
                replaced with sympy expressions with the operators left as symbols.

    """
    # if matrix is Model or dict then apply method on each value
    if isinstance(matrix, qsymm.Model) or isinstance(matrix, dict):
        matrix = dict(matrix)
        for k, v in matrix.items():
            matrix[k] = represent_hilbert_schmidt(v, decomp, symbols, prec)
        return matrix

    assert matrix.shape[0] == matrix.shape[1]
    assert len(list(decomp.split('+'))) == len(symbols)

    # check if provided matrix is still a hamiltonian. The HS trafo should not be hermitian
    if np.allclose(matrix.T.conjugate(), matrix):
        matrix = hilbert_schmidt_transform(matrix,decomp)

    #round and bring into desired symbolic form
    hs_matrix = matrix
    rep = sum([np.round(np.sqrt(2/hs_matrix.shape[0])*ob[1], decimals=prec) *np.prod([sympy.sympify('{}_{}'.format(symbols[i],ob[0][i])) for i in range(len(ob[0]))]) for ob in np.ndenumerate(hs_matrix)])

    return rep
# --


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


# -

def assert_almost_zero(a, decimal, extra_msg=""):
    """Compare two dictionaries with array-like values."""
    for key, value in a.items():
        np.testing.assert_almost_equal(
            value, 0, decimal=decimal, err_msg=f"{key=} {extra_msg}"
        )


# # An example to demonstrate

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

def prepare_hamiltonian(ham, gens, coords, grid, shape, start=(0,0), locals=None):
    """Return systems corresponding to H0 and H1 part of full Hamiltonian.

    Parameters
    ----------
    ham : str or SymPy expression
        Symbolic representation of a continuous Hamiltonian.  It is
        converted to a SymPy expression using `kwant.continuum.sympify`.
    gens: sequence of sympy.Symbol objects or strings (optional)
        Generators of the perturbation. If this is a sequence of strings then
        corresponding symbols will be generated using `kwant.continuum.sympify`
        rules, especially regarding the commutative properties. If this is
        already a sequence of SymPy symbols then their commutative properties
        will be respected, i.e if symbol is defined as commutative in "gens" it
        will be casted to the commutative symbol in "ham". Commutative symbols
        will not however be casted to noncommutative.
    coords : sequence of strings, or ``None`` (default)
        The coordinates for which momentum operators will be treated as
        differential operators. May contain only "x", "y" and "z" and must be
        sorted.  If not provided, `coords` will be obtained from the input
        Hamiltonian by reading the present coordinates and momentum operators.
    grid : int or float, default: 1
        Spacing of the (quadratic or cubic) discretization grid.
    shape : callable
        A boolean function of site returning whether the site should be
        included in the system or not. The shape must be compatible
        with the system's symmetry.
    start : `Site` instance or iterable thereof or iterable of numbers
        The site(s) at which the the flood-fill starts.  If start is an
        iterable of numbers, the starting site will be
        ``template.closest(start)``.
    locals : dict or ``None`` (default)
        Additional namespace entries for `~kwant.continuum.sympify`.  May be
        used to simplify input of matrices or modify input before proceeding
        further. For example:
        ``locals={'k': 'k_x + I * k_y'}`` or
        ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.

    Returns
    -------
    H0: finalized "kwant.system"
    H1: dict: SymPy symbol -> finalized "kwant.system"

    "kwant" systems can be used to built corresponding Hamiltonian matrices
    """

    def _discretize_and_fill(operator, coords, grid, shape, start):
        """Discretize given operator and fill appropriate system.

        Use modified version of "kwant.continuum.discretize" to workaround
        flood-fill algorithm when discretizing operators.
        """
        tb = discretize_with_hoppings(
            operator, coords, grid_spacing=grid
        )
        syst = kwant.Builder()
        syst.fill(tb, shape, start);
        return syst.finalized()

    ham = kwant.continuum.sympify(ham, locals=locals)
    H0, H1 = separate_hamiltonian(ham, gens)

    H0 = _discretize_and_fill(H0, coords, grid, shape, start)
    H1 = {k: _discretize_and_fill(v, coords, grid, shape, start)
          for k, v in H1.items()}

    return H0, H1


h0syst, h1syst = prepare_hamiltonian(model.subs(ge_parameters|constants|{'E_z':0,'z':0}),
                                             gens=['k_z','E_x','E_y'],
                                             coords='xy',
                                             grid=R_c/n_steps,
                                             shape=square_shape(R_c),
                                             start=(0,0))
kwant.plotter.plot(h0syst)
print('prev second')

# +
h0 = h0syst.hamiltonian_submatrix()
h1 = {key:val.hamiltonian_submatrix() for key,val in h1syst.items()}
h0.shape
print(np.linalg.norm(h0))
print(np.linalg.cond(h0))


for val in h1.values():
    print(np.linalg.norm(val))
    print(np.linalg.cond(val))

# +
# diagonalize
eigs,vecs = scipy.linalg.eigh(h0)

print(np.linalg.norm(vecs))
print(np.linalg.cond(vecs))

H0 = np.diag(eigs)
H_p = {key: vecs.T.conj() @ val @ vecs for key,val in h1.items()}
for val in h1.values():
    print(np.linalg.norm(val))
    print(np.linalg.cond(val))
H_p, key_map = sym_to_ta(H_p)
# -

# ## PolyLowdin
#
#
# $k_z^2E_x$ $\rightarrow$ [ta.array([2,1,0])]

eigs[:6]

# +
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
# -

represent_hilbert_schmidt(reassemble_to_symb(H_eff_AA,key_map),decomp='2+2',symbols=['tau','sigma'],prec=3)

# +
# Is exp_S unitary?
transformed = polynomial_orders_U.H_tilde(np.eye(N_A), np.eye(N_B), {}, {}, {}, w_orders, exp_S, compute_AB=True)

for value, block in zip(transformed, "AA BB AB".split()):
    assert_almost_zero(value, 6, f"{block=}")
    print(f"{block=}"+'passed')
# -

# Does AB really vanish?
assert_almost_zero(H_eff_AB,6)

# ## PolyLowdin with B-subspace reduction

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

# matmul implementation is an issue. need op in SumOf.... to perform proper matrix mutiplication
# -

import poly_kpm
t_list = poly_kpm.t_list
t2_list = poly_kpm.t_list_2
test = poly_kpm.SumOfOperatorProducts(t_list)
test2 = poly_kpm.SumOfOperatorProducts(t2_list)

t = [(np.array([[1,2],[2,1]]),'AB')]
t2 = [(np.array([[5,4],[-4,2]]),'BA')]
