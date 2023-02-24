from itertools import product, count

# %reload_ext autoreload
import pytest
import numpy as np
import tinyarray as ta
from scipy.linalg import eigh, block_diag
from scipy.stats import unitary_group

from poly_kpm import SumOfOperatorProducts, divide_energies, get_bb_action, create_div_energs
from polynomial_orders_U import compute_next_orders, H_tilde

def Ns():
    """
    Return a random number of states for each block (A, B).
    """
    return np.random.randint(1, high=5, size=2)


def hamiltonians(Ns, wanted_orders):
    """
    Produce random Hamiltonians to test.

    Ns: dimension of each block (A, B)
    wanted_orders: list of orders to compute

    Returns:
    hams: list of Hamiltonians
    """
    N_p = len(wanted_orders[0])
    orders = ta.array(np.eye(N_p))
    hams = []
    for i in range(2):
        hams.append(np.diag(np.sort(np.random.rand(Ns[i])) - i))

    def matrices_it(N_i, N_j, hermitian):
        """
        Generate random matrices of size N_i x N_j.

        N_i: number of rows
        N_j: number of columns
        hermitian: if True, the matrix is hermitian

        Returns:
        generator of random matrices
        """
        for i in count():
            H = np.random.rand(N_i, N_j) + 1j * np.random.rand(N_i, N_j)
            if hermitian:
                H += H.conj().T
            yield H

    for i, j, hermitian in zip([0, 1, 0], [0, 1, 1], [True, True, False]):
        matrices = matrices_it(Ns[i], Ns[j], hermitian)
        hams.append({order: matrix for order, matrix in zip(orders, matrices)})
    return hams


def random_term(n, m, length, start, end, rng=None):
    """Generate a random term.

    Parameters
    ----------
    n : int
        Size of "A" space
    m : int
        Size of "B" space
    length : int
        Number of operators in the term
    start, end : str
        Start and end spaces of the term (A or B)
    rng : np.random.Generator
        Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()
    spaces = "".join(np.random.choice(a=["A", "B"], size=length - 1))
    spaces = start + spaces + end
    op_spaces = ["".join(s) for s in zip(spaces[:-1], spaces[1:])]
    op_dims = [
        (n if dim[0] == "A" else m, m if dim[1] == "B" else n) for dim in op_spaces
    ]
    ops = [rng.random(size=dim) for dim in op_dims]
    return SumOfOperatorProducts([[(op, space) for op, space in zip(ops, op_spaces)]])

def assert_almost_zero(a, decimal=5, extra_msg=""):
    """
    Assert that all values in a are almost zero.

    a: array to check
    decimal: number of decimal places to check
    extra_msg: extra message to print if assertion fails
    """
    for key, value in a.items():
        np.testing.assert_almost_equal(
            value, 0, decimal=decimal, err_msg=f"{key=} {extra_msg}"
        )

# ############################################################################################

def test_array_vs_non_diag_proj(hamiltonians, wanted_orders):
    n_a, n_b = hamiltonians[0].shape[0], hamiltonians[1].shape[0]
    # initialize arrays
    H_0_AA_arr = hamiltonians[0]
    H_0_BB_arr = hamiltonians[1]
    
    h_arr = block_diag(H_0_AA_arr,H_0_BB_arr)

    # Perform unitary trafo on H_0 such that it is not diagonal anymore
    # Transform all others accordingly
    u = unitary_group.rvs(n_a+n_b)
    h_0 = u @ block_diag(hamiltonians[0], hamiltonians[1]) @ u.conj().T
    h_p = {A[0]: u @ np.block([[B[0],B[2]],[B[2].conj().T,B[1]]]) @ u.conj().T for A,B in zip(product(hamiltonians[2].keys(),
                                                                                      hamiltonians[3].keys(),
                                                                                      hamiltonians[4].keys()),
                                                                              product(hamiltonians[2].values(),
                                                                                      hamiltonians[3].values(),
                                                                                      hamiltonians[4].values()))
                                                                                       if (A[0]==A[1] and 
                                                                                           A[0]==A[2])}
    
    # now all hamiltonians are in u basis
    # get eigenvectors with eigh
    eigs_a = np.diag(H_0_AA_arr)
    eigs, vecs = eigh(h_0)
    order = [np.where(np.isclose(e,eigs))[0][0] for e in eigs_a]
    a_eigs, a_vecs =eigs[order], vecs[:,order]
    assert np.allclose(np.sort(eigs_a),np.sort(a_eigs))
    
    
    # initialize SOP with bookkeeping such that B is A+B
    # AB -> A, A+B
    h_ab = {A[0]:np.hstack((B[0],B[1])) for A,B in zip(product(hamiltonians[2].keys(),
                                                              hamiltonians[4].keys()),
                                                      product(hamiltonians[2].values(),
                                                              hamiltonians[4].values())
                                                      ) if A[0]==A[1] }
    
    # BB -> A+B, A+B
    h_bb = {A[0]:np.block([[B[0],B[2]],[B[2].conj().T,B[1]]]) for A,B in zip(product(hamiltonians[2].keys(),
                                                                                      hamiltonians[3].keys(),
                                                                                      hamiltonians[4].keys()),
                                                                              product(hamiltonians[2].values(),
                                                                                      hamiltonians[3].values(),
                                                                                      hamiltonians[4].values()))
                                                                                       if (A[0]==A[1] and 
                                                                                           A[0]==A[2])}


    H_0_AA_sop = SumOfOperatorProducts([[(hamiltonians[0],'AA')]])
    H_0_BB_sop = SumOfOperatorProducts([[(get_bb_action(h_0, a_vecs), 'BB')]])

    h_sop = block_diag(H_0_AA_sop.to_array(),(H_0_BB_sop.to_array() @ np.eye(H_0_BB_sop.to_array().shape[0])))

    assert np.all([np.any(np.isclose(e,eigh(h_sop)[0])) for e in eigh(h_arr)[0]])

    H_p_AA_sop = {key:SumOfOperatorProducts([[(val,'AA')]]) for key,val in hamiltonians[2].items()}
    H_p_AB_sop = {key:SumOfOperatorProducts([[(val,'AB')]]) for key,val in h_ab.items()}
    H_p_BB_sop = {key:SumOfOperatorProducts([[(get_bb_action(val, a_vecs),'BB')]]) for key,val in h_bb.items()}


    exp_S_sop = compute_next_orders(H_0_AA_sop,
                                    H_0_BB_sop,
                                    H_p_AA_sop,
                                    H_p_BB_sop,
                                    H_p_AB_sop,
                                    wanted_orders=wanted_orders,
                                    divide_energies=create_div_energs(H_0_AA_sop, H_0_BB_sop, mode='op')
                                    )

    # make all SOPs matrices
    ham = [H_0_AA_sop, H_0_BB_sop, H_p_AA_sop, H_p_BB_sop, H_p_AB_sop]
    H_AB = H_tilde(*ham, wanted_orders, exp_S_sop, compute_AB=True)[2]
    assert_almost_zero(H_AB, 6)



# +
hams = hamiltonians(Ns(), [ta.array([1,1,0])])

h_a = hams[0]
eigs_a = np.diag(h_a)
n_a = h_a.shape[0]
h_b = hams[1]
eigs_b = np.diag(h_b)
n_b = h_b.shape[0]

h_0 = block_diag(h_a,h_b)

u = unitary_group.rvs(h_0.shape[0])

h_u = u @ h_0 @ u.conj().T
print(eigs_a)
print(eigs_b)
print(np.diag(h_0))

eigs, vecs = eigh(h_u)
order = np.array([np.where(np.isclose(e,eigs))[0][0] for e in np.diag(h_0)])
eigs1, vecs1 = eigs[order], vecs[:,order]
vecs_a= vecs1[:,:n_a]
print(eigs1)
assert np.allclose(eigs1,np.diag(h_0))
# -

t_op = get_bb_action(h_u,u[:,:n_a])
tb = t_op @ np.eye(n_a+n_b)
ta = get_bb_action(h_u,u[:,n_a:]) @ np.eye(n_a+n_b)
assert np.allclose(h_u,ta+tb)

# +
import tinyarray as ta
n_a, n_b = 2,4

h_0 = np.random.random((n_a+n_b,n_a+n_b)) + 1j * np.random.random((n_a+n_b,n_a+n_b))
h_0 = h_0 + h_0.conj().T

eigs, vecs = eigh(h_0)

h_aa = np.diag(eigs[:n_a])
v_aa = vecs[:,:n_a]
#h_bb = get_bb_action(h_0,v_aa)
h_bb = np.diag(eigs[n_a:])

h_p = np.random.random((n_a+n_b,n_a+n_b)) + 1j * np.random.random((n_a+n_b,n_a+n_b))
h_p = (h_p + h_p.conj().T)

h_p = {ta.array([1]):h_p}

h_p_aa = {k:v[:n_a,:n_a] for k,v in h_p.items()}
h_p_bb = {k:v[n_a:,n_a:] for k,v in h_p.items()}
h_p_ab = {k:v[:n_a,n_a:] for k,v in h_p.items()}

exp_S = compute_next_orders(h_aa,
                            h_bb,
                            h_p_aa,
                            h_p_bb,
                            h_p_ab,
                            [ta.array([2])])
                            #divide_energies=lambda Y:divide_energies(Y,h_aa,h_bb,mode='arr'))
    
H_t = H_tilde(h_aa,
                h_bb,
                h_p_aa,
                h_p_bb,
                h_p_ab,
               wanted_orders=[ta.array([2])],
               exp_S=exp_S,
              compute_AB=True)
H_t[2]

# +
n_a, n_b = 2,4

h_0 = np.random.random((n_a+n_b,n_a+n_b)) + 1j * np.random.random((n_a+n_b,n_a+n_b))
h_0 = h_0 + h_0.conj().T

eigs, vecs = eigh(h_0)

h_aa = np.diag(eigs[:n_a])
v_aa = vecs[:,:n_a]
h_bb = get_bb_action(h_0,v_aa)


h_p = np.random.random((n_a+n_b,n_a+n_b)) + 1j * np.random.random((n_a+n_b,n_a+n_b))
h_p = (h_p + h_p.conj().T)

h_p = {ta.array([1,0]):h_p}

h_p2 = np.random.random((n_a+n_b,n_a+n_b)) + 1j * np.random.random((n_a+n_b,n_a+n_b))
h_p2 = (h_p2 + h_p2.conj().T)

h_p = h_p|{ta.array([0,1]):h_p2}

h_p_aa = {k:v[:n_a,:n_a] for k,v in h_p.items()}
h_p_bb = {k:get_bb_action(v,v_aa) for k,v in h_p.items()}
h_p_ab = {k:v[:n_a,:] for k,v in h_p.items()}


exp_S = compute_next_orders(h_aa,
                            h_bb,
                            h_p_aa,
                            h_p_bb,
                            h_p_ab,
                            [ta.array([1,1])],
                            divide_energies=create_div_energs(np.diag(h_aa), v_aa, h_bb))
    
H_t = H_tilde(h_aa,
                h_bb,
                h_p_aa,
                h_p_bb,
                h_p_ab,
               wanted_orders=[ta.array([1,1])],
               exp_S=exp_S,
              compute_AB=True)

{k:np.round(v,decimals=15) for k,v in H_t[2].items()}
# -




