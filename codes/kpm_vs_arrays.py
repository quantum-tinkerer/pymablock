# %reload_ext autoreload
import pytest
import numpy as np
import tinyarray as ta
from scipy.linalg import eigh, block_diag
from scipy.stats import unitary_group
from itertools import product, count

import sys
sys.path.append('../lowdin/')
from poly_kpm import SumOfOperatorProducts, divide_energies, create_div_energs_old, create_div_energs
from polynomial_orders_U import compute_next_orders, H_tilde
from linalg import complement_projected, ComplementProjector

# ### Diagonal $\mathcal{H}_0$
#
# #### Initialize common inputs

# +
n_a, n_b = 2,6

h_0 = np.diag(10*np.sort(np.random.random(n_a+n_b)))

h_p = np.random.random((n_a+n_b, n_a+n_b)) + 1j * np.random.random((n_a+n_b, n_a+n_b))
h_p = h_p + h_p.conj().T

h_p_dict = {ta.array([1]):h_p}

w_order = [ta.array([3])]
# -

# ##### Standard divide_by_energies

# +
h_aa = h_0[:n_a,:n_a]
h_bb = h_0[n_a:,n_a:]

h_p_aa = {k: v[:n_a, :n_a] for k, v in h_p_dict.items()}
h_p_bb = {k: v[n_a:, n_a:] for k, v in h_p_dict.items()}
h_p_ab = {k: v[:n_a, n_a:] for k, v in h_p_dict.items()}

exp_S = compute_next_orders(h_aa, h_bb, h_p_aa, h_p_bb, h_p_ab, wanted_orders=w_order)

H_t = H_tilde(
    h_aa,
    h_bb,
    h_p_aa,
    h_p_bb,
    h_p_ab,
    wanted_orders=w_order,
    exp_S=exp_S,
    compute_AB=True,
)
# -

# ##### Input characteristics

eig_h0_inp = eigh(block_diag(h_aa,h_bb))[0]
eig_hp_inp = {v1[0]:np.trace(block_diag(v1[1], v2[1])) for v1, v2 in product(h_p_aa.items(), h_p_bb.items()) if v1[0]==v2[0]}
print(eig_h0_inp)
print(eig_hp_inp)

trac_hp = [np.trace(block_diag(v[0], v[1])) for v in product(h_p_aa.values(), h_p_bb.values())]
trac_hp

# ##### Output characteristics

# +
# is H_tilde_AB really vanishing?
assert np.all([np.allclose(v, 0 ) for v in H_t[2].values()])

trac_ht = {i1[0]:np.trace(block_diag(i1[1], i2[1])) for i1,i2 in product(H_t[0].items(), H_t[1].items()) if i1[0]==i2[0]}
# -

eig_hta = {k:eigh(v)[0] for k, v in H_t[0].items()}
eig_htb = {k:eigh(v)[0] for k, v in H_t[1].items()}
print(eig_hta)
print(eig_htb)

H_t_trace = sum([i for i in trac_ht.values()])
H_t_trace



# ### old create divide by energies

# +
h_aa = h_0[:n_a,:n_a]
v_aa = np.eye(n_a+n_b)[:,:n_a]

h_bb = complement_projected(h_0, v_aa)

h_p_aa = {k: v[:n_a, :n_a] for k, v in h_p_dict.items()}
h_p_bb = {k: complement_projected(v,v_aa) for k, v in h_p_dict.items()}
h_p_ab = {k: v[:n_a, :] @ (np.eye(n_a+n_b) - v_aa @ v_aa.conj().T) for k, v in h_p_dict.items()}

# -

# ##### Inputs

print(eigh(h_aa)[0])
print(eigh(h_bb @ np.eye(n_a+n_b))[0])
print(np.diag(h_0))

tr_hp_op = sum(eigh(h_p_aa[(1,)])[0])+sum(eigh(h_p_bb[(1,)] @ np.eye(n_a+n_b) )[0])
np.isclose(tr_hp_op,trac_hp[0])

# +
exp_S = compute_next_orders(h_aa,
                            h_bb,
                            h_p_aa,
                            h_p_bb,
                            h_p_ab,
                            wanted_orders=w_order,
                            divide_energies = create_div_energs_old(np.diag(h_aa), v_aa, h_bb) )

H_t = H_tilde(
    h_aa,
    h_bb,
    h_p_aa,
    h_p_bb,
    h_p_ab,
    wanted_orders=w_order,
    exp_S=exp_S,
    compute_AB=True,
)
# -

# ##### Outputs

# These match approach 1!!
H_tilde_AA = H_t[0]
print({k: eigh(v)[0] for k, v in H_tilde_AA.items()})
H_tilde_BB = H_t[1]
print({k: eigh(v@np.eye(n_a+n_b))[0] for k, v in H_tilde_BB.items()})

H_tilde_AB = H_t[2]
sum(value for value in H_tilde_AB.values())



# ### KPM all vectors

# +
v_aa = np.eye(n_a+n_b)[:, :n_a]
v_bb = np.eye(n_a+n_b)[:, n_a:]

eigs_a = np.diag(h_0[:n_a, :n_a])
eigs_b = np.diag(h_0[n_a:, n_a:])

h_p_aa = {k: v[:n_a, :n_a] for k, v in h_p_dict.items()}
h_p_bb = {k: complement_projected(v,v_aa) for k, v in h_p_dict.items()}
h_p_ab = {k: v[:n_a, :] @ (np.eye(n_a+n_b) - v_aa @ v_aa.conj().T) for k, v in h_p_dict.items()}
# -

# ##### Inputs

print(np.diag(v_aa.conj().T @ h_0 @ v_aa))
print(np.diag(v_bb.conj().T @ h_0 @ v_bb))
print(np.diag(h_0))

tr_hp_op = sum(eigh(h_p_aa[(1,)])[0])+sum(eigh(h_p_bb[(1,)] @ np.eye(n_a+n_b) )[0])
np.isclose(tr_hp_op,trac_hp[0])

# +
exp_S = compute_next_orders(h_0[:n_a, :n_a],
                            complement_projected(h_0, v_aa),
                            h_p_aa,
                            h_p_bb,
                            h_p_ab,
                            wanted_orders=w_order,
                            divide_energies = create_div_energs(h_0, v_aa, eigs_a, v_bb, eigs_b) )

H_t = H_tilde(
    h_aa,
    h_bb,
    h_p_aa,
    h_p_bb,
    h_p_ab,
    wanted_orders=w_order,
    exp_S=exp_S,
    compute_AB=True,
)
# -

H_tilde_AA = H_t[0]
H_taa_exact = {k: eigh(v)[0] for k, v in H_tilde_AA.items()}
print(H_taa_exact)
H_tilde_BB = H_t[1]
H_tbb_exact = {k: eigh(v@np.eye(n_a+n_b))[0] for k, v in H_tilde_BB.items()}
print(H_tbb_exact)

H_tilde_AB = H_t[2]
sum(value for value in H_tilde_AB.values())



# ### KPM without all vectors

# +
v_aa = np.eye(n_a+n_b)[:, :n_a]
v_bb = np.eye(n_a+n_b)[:, n_a:]

eigs_a = np.diag(h_0[:n_a, :n_a])
eigs_b = np.diag(h_0[n_a:, n_a:])

h_p_aa = {k: v[:n_a, :n_a] for k, v in h_p_dict.items()}
h_p_bb = {k: complement_projected(v,v_aa) for k, v in h_p_dict.items()}
h_p_ab = {k: v[:n_a, :] @ (np.eye(n_a+n_b) - v_aa @ v_aa.conj().T) for k, v in h_p_dict.items()}
# -

# ##### Inputs

print(np.diag(v_aa.conj().T @ h_0 @ v_aa))
print(np.diag(v_bb.conj().T @ h_0 @ v_bb))
print(np.diag(h_0))

tr_hp_op = sum(eigh(h_p_aa[(1,)])[0])+sum(eigh(h_p_bb[(1,)] @ np.eye(n_a+n_b) )[0])
np.isclose(tr_hp_op,trac_hp[0])

# +
exp_S = compute_next_orders(h_0[:n_a, :n_a],
                            complement_projected(h_0, v_aa),
                            h_p_aa,
                            h_p_bb,
                            h_p_ab,
                            wanted_orders=w_order,
                            divide_energies = create_div_energs(h_0, v_aa, eigs_a) )

H_t = H_tilde(
    h_aa,
    h_bb,
    h_p_aa,
    h_p_bb,
    h_p_ab,
    wanted_orders=w_order,
    exp_S=exp_S,
    compute_AB=True,
)
# -

H_t[2]

{k:eigh(v)[0] for k,v in H_t[0].items()}

{k:eigh(v@np.eye(n_a+n_b))[0] for k,v in H_t[1].items()}

H_taa_exact

H_tbb_exact

# ## Non-diagonal systems

# #### Common inputs

# +
n_a, n_b = 2,6

h_0 = np.random.random((n_a+n_b, n_a+n_b)) + 1j * np.random.random((n_a+n_b, n_a+n_b))
h_0 = h_0 + h_0.conj().T

h_p = np.random.random((n_a+n_b, n_a+n_b)) + 1j * np.random.random((n_a+n_b, n_a+n_b))
h_p = h_p + h_p.conj().T

h_p_dict = {ta.array([1]):h_p}

w_order = [ta.array([3])]

# +
eigs, vecs = eigh(h_0)

eigs_a = eigs[:n_a]
vecs_a = vecs[:,:n_a]
print(np.round(eigs_a,3))

eigs_b = eigs[n_a:]
vecs_b = vecs[:,n_a:]
print(np.round(eigs_b,15))

sum(eigs_a)+sum(eigs_b)
# -

trace_h_p_0 = {k:np.trace(v) for k,v in h_p_dict.items()}
print(trace_h_p_0)

# +
h_aa = np.diag(eigs_a)
h_bb = complement_projected(h_0 ,vecs_a)

h_p_aa = {k: vecs_a.conj().T @ v @ vecs_a for k, v in h_p_dict.items()}
h_p_bb = {k: complement_projected(v,vecs_a) for k, v in h_p_dict.items()}
h_p_ab = {k: (vecs_a.conj().T @ v @ ComplementProjector(vecs_a)) for k, v in h_p_dict.items()} 
# -

print(eigh(h_aa)[0])
print(np.round(eigh(h_bb@np.eye(n_a+n_b))[0],15))
np.trace(h_aa)+sum(eigh(h_bb@np.eye(n_a+n_b))[0])

np.trace((h_p_aa[(1,)]))+np.trace(h_p_bb[(1,)]@np.eye(n_a+n_b))

# +
exp_S = compute_next_orders(h_aa,
                            h_bb,
                            h_p_aa,
                            h_p_bb,
                            h_p_ab,
                            wanted_orders=w_order,
                            divide_energies = create_div_energs(h_0, vecs_a, eigs_a) )

H_t = H_tilde(
    h_aa,
    h_bb,
    h_p_aa,
    h_p_bb,
    h_p_ab,
    wanted_orders=w_order,
    exp_S=exp_S,
    compute_AB=True,
)
# -

# #### Outputs KPM

{k:eigh(v)[0] for k,v in H_t[0].items()}

{k:eigh(v@np.eye(n_a+n_b))[0] for k,v in H_t[1].items()}

{k:v for k,v in H_t[2].items()}



# +
exp_S = compute_next_orders(h_aa,
                            h_bb,
                            h_p_aa,
                            h_p_bb,
                            h_p_ab,
                            wanted_orders=w_order,
                            divide_energies = create_div_energs(h_0, vecs_a, eigs_a, vecs_b, eigs_b) )

H_t = H_tilde(
    h_aa,
    h_bb,
    h_p_aa,
    h_p_bb,
    h_p_ab,
    wanted_orders=w_order,
    exp_S=exp_S,
    compute_AB=True,
)
# -

# #### Outputs exact

{k:eigh(v)[0] for k,v in H_t[0].items()}

{k:eigh(v@np.eye(n_a+n_b))[0] for k,v in H_t[1].items()}

{k:v for k,v in H_t[2].items()}



# +
exp_S = compute_next_orders(h_aa,
                            h_bb,
                            h_p_aa,
                            h_p_bb,
                            h_p_ab,
                            wanted_orders=w_order,
                            divide_energies = create_div_energs_old(eigs_a, vecs_a, h_bb@np.eye(n_a+n_b)) )

H_t = H_tilde(
    h_aa,
    h_bb,
    h_p_aa,
    h_p_bb,
    h_p_ab,
    wanted_orders=w_order,
    exp_S=exp_S,
    compute_AB=True,
)
# -

# #### Outputs old

{k:eigh(v)[0] for k,v in H_t[0].items()}

{k:eigh(v@np.eye(n_a+n_b))[0] for k,v in H_t[1].items()}

{k:v for k,v in H_t[2].items()}





# # Build Test of `create_div_energs`
#
# ### Utility

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



# #### Test

# +
def test_create_div_energs_kpm(hamiltonians):
    n_a = hamiltonians[0].shape[0]
    n_b = hamiltonians[1].shape[0]
    h_0 = block_diag(hamiltonians[0],hamiltonians[1])
    eigs, vecs = eigh(h_0)
    eigs_a, vecs_a = eigs[:n_a], vecs[:,:n_a]
    eigs_b, vecs_b = eigs[n_a:], vecs[:,n_a:]
    
    Y = []
    for _ in range(5):
        h_ab = np.random.random((n_a+n_b,n_a+n_b)) + 1j * np.random.random((n_a+n_b, n_a+n_b))
        h_ab += h_ab.conj().T
        Y.append(vecs_a.conj().T @ h_ab @ ComplementProjector(vecs_a))
    
    de_kpm_func =lambda Y: create_div_energs(h_0, vecs_a, eigs_a)(Y)
    de_exact_func =lambda Y: create_div_energs(h_0, vecs_a, eigs_a, vecs_b, eigs_b)(Y)
    
    #apply h_ab from left -> Y.conj() since G_0 is hermitian
    applied_exact = [de_exact_func(y.conj()) for y in Y]
    applied_kpm = [de_kpm_func(y.conj()) for y in Y]
    
    diff_approach = {i:np.abs(applied_exact[i] - applied_kpm[i]) for i in range(len(Y))}
    
    assert_almost_zero(diff_approach, decimal=2, extra_msg="")
    
    
    
    
# -

test_create_div_energs_kpm(hamiltonians(Ns(), [ta.array([1])]))


