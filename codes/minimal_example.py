# +
import kwant
import sympy
import scipy
import scipy.stats
import qsymm
import numpy as np
import matplotlib.pyplot as plt
import tinyarray as ta
from collections import defaultdict
from itertools import product
from functools import reduce

import poly_kpm
import polynomial_orders_U
import lowdin
import misc
# -

N = 10
N_A = 4
N_B = N-N_A

# +
H_0 = np.sort(np.array([[i,i] for i in np.random.random(N//2)]).reshape(N))
H_0 = np.diag(H_0)
H_0

print('norm H0='+str(np.linalg.norm(H_0)))
print('cond H0='+str(np.linalg.cond(H_0)))

vecs = scipy.stats.unitary_group.rvs(N)
vecs /= 0.5*np.linalg.norm(vecs)
print('norm='+str(np.linalg.norm(vecs)))
print('cond='+str(np.linalg.cond(vecs)))

# +
N_p = 2

H_ps = []
for perturbation in range(N_p):
    H_p = np.random.random(size=(N, N)) + 1j * np.random.random(size=(N, N))
    H_p += H_p.conj().T
    H_p = (vecs.conj().T @ H_p @ vecs)/2
    print('norm H_{}='.format(perturbation)+str(np.linalg.norm(H_p)))
    print('cond H_{}='.format(perturbation)+str(np.linalg.cond(H_p)))
    
    H_ps.append(H_p)

orders = ta.array(np.eye(N_p))

# +
H_p_AA = {
    order: value[:N_A, :N_A]
    for order, value in zip(orders, H_ps)
}

H_p_BB = {
    order: value[N_A:, N_A:]
    for order, value in zip(orders, H_ps)
}

H_p_AB = {
    order: value[:N_A, N_A:]
    for order, value in zip(orders, H_ps)
}
# -

H_0_AA = H_0[:N_A,:N_A]
H_0_BB = H_0[N_A:,N_A:]


def gen_all_ords(order,N_p):
    base = np.arange(0,order+1)
    iterat = product(base, repeat=N_p)
    return [ta.array(v) for v in iterat if sum(v)<=order]


# +
w_orders = gen_all_ords(3,N_p)

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


# -

represent_hilbert_schmidt(H_eff_AA,decomp='2+2',symbols=['tau','sigma'],prec=3)


def assert_almost_zero(a, decimal, extra_msg=""):
    """Compare two dictionaries with array-like values."""
    for key, value in a.items():
        np.testing.assert_almost_equal(
            value, 0, decimal=decimal, err_msg=f"{key=} {extra_msg}"
        )


# +
# Is exp_S unitary?
transformed = polynomial_orders_U.H_tilde(np.eye(N_A), np.eye(N_B), {}, {}, {}, w_orders, exp_S, compute_AB=True)

for value, block in zip(transformed, "AA BB AB".split()):
    assert_almost_zero(value, 6, f"{block=}")
    print(f"{block=}"+'passed')
# -

# Does AB really vanish?
assert_almost_zero(H_eff_AB,6)






