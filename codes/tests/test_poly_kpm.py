from itertools import product, count

import pytest
import numpy as np
import tinyarray as ta
from scipy.linalg import eigh, block_diag

from codes.poly_kpm import SumOfOperatorProducts, divide_energies, get_bb_action
from codes.polynomial_orders_U import compute_next_orders

@pytest.fixture(
    scope="module",
    params=[
        [[3]],
        [[2, 2]],
        [[3, 1], [1, 3]],
        [[2, 2, 2], [3, 0, 0]],
    ],
)
def wanted_orders(request):
    """
    Return a list of orders to compute.
    """
    return request.param


@pytest.fixture(scope="module")
def Ns():
    """
    Return a random number of states for each block (A, B).
    """
    return np.random.randint(1, high=5, size=2)


@pytest.fixture(scope="module")
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

def test_shape_validation():
    """Test that only terms of compatible shapes are accepted.

    Instead of providing terms manually we rely on SumOfOperatorProducts
    creating new instances of itself on addition and multiplication.
    """
    n, m = 4, 10
    terms = {
        "AA": random_term(n, m, 1, "A", "A"),
        "AB": random_term(n, m, 1, "A", "B"),
        "BA": random_term(n, m, 1, "B", "A"),
        "BB": random_term(n, m, 1, "B", "B"),
    }
    for (space1, term1), (space2, term2) in product(terms.items(), repeat=2):
        # Sums should work if the spaces are the same
        if space1 == space2:
            # no error, moreover the result should simplify to a single term
            term1 + term2
            assert len(term1.terms) == 1
        else:
            with pytest.raises(ValueError):
                term1 + term2

        # Matmuls should work if start space of term2 matches end space of term1
        if space1[1] == space2[0]:
            term1 @ term2
        else:
            with pytest.raises(ValueError):
                term1 @ term2

def test_neg():
    """Test that negation works."""
    n, m = 4, 10
    term = random_term(n, m, 1, "A", "A")
    zero = term + -term
    # Should have one term with all zeros
    assert len(zero.terms) == 1
    np.testing.assert_allclose(zero.terms[0][0][0], 0)


# test if arrays and SumOfOperatorProducts generates the ame terms
def test_array_vs_sop(hamiltonians, wanted_orders):
    n_a, n_b = hamiltonians[0].shape[0], hamiltonians[1].shape[0]

    H_0 = np.diag(np.sort(np.random.random(n_a+n_b)))

    # initialize arrays
    H_0_AA_arr = hamiltonians[0]
    H_0_BB_arr = hamiltonians[1]

    H_p_AA_arr = hamiltonians[2]
    H_p_AB_arr = hamiltonians[4]
    H_p_BB_arr = hamiltonians[3]

    exp_S_arr = compute_next_orders(H_0_AA_arr,
                                    H_0_BB_arr,
                                    H_p_AA_arr,
                                    H_p_BB_arr,
                                    H_p_AB_arr,
                                    wanted_orders=wanted_orders
                                    )

    # initialize SumOfOps
    H_0_AA_sop = SumOfOperatorProducts([[(H_0_AA_arr,'AA')]])
    H_0_BB_sop = SumOfOperatorProducts([[(H_0_BB_arr,'BB')]])

    H_p_AA_sop = {key:SumOfOperatorProducts([[(val,'AA')]]) for key,val in H_p_AA_arr.items()}
    H_p_AB_sop = {key:SumOfOperatorProducts([[(val,'AB')]]) for key,val in H_p_AB_arr.items()}
    H_p_BB_sop = {key:SumOfOperatorProducts([[(val,'BB')]]) for key,val in H_p_BB_arr.items()}

    exp_S_sop = compute_next_orders(H_0_AA_sop,
                                    H_0_BB_sop,
                                    H_p_AA_sop,
                                    H_p_BB_sop,
                                    H_p_AB_sop,
                                    wanted_orders=wanted_orders,
                                    divide_energies=lambda Y:divide_energies(Y, H_0_AA_sop, H_0_BB_sop)
                                    )
    
    # make all SOPs matrices
    for i in (0,1):
        for j in (0,1):
            exp_S_sop[i,j] = {k:v.to_array() for k,v in exp_S_sop[i,j].items() if isinstance(v,SumOfOperatorProducts)}
            
    # subttract two results
    exp_S_diff = np.zeros_like(exp_S_arr)
    
    for i in (0,1):
        for j in (0,1):
            temp = {}
            for key,val in exp_S_arr[i,j].items():
                if isinstance(val,np.ndarray):
                    if not val.shape == exp_S_sop[i,j][key].shape:
                        print('hier')
                    temp[key]=val-exp_S_sop[i,j][key]
            exp_S_diff[i,j] = temp
    exp_S_diff = [exp_S_diff[0,0], exp_S_diff[1,1], exp_S_diff[0,1]]      
    for value, block in zip(exp_S_diff, "AA BB AB".split()):
        assert_almost_zero(value, 6, extra_msg=f"{block=}")



def test_does_bb_do_what_bb_do(hamiltonians):
    h0 = hamiltonians[0]
    h1 = hamiltonians[1]
    
    assert h0.shape[0] == h0.shape[1]
    assert h1.shape[0] == h1.shape[1]
    
    h_0_1 = np.concatenate((h0,np.zeros(shape=(h0.shape[0],h1.shape[1]))),axis=1)
    h_0_2 = np.concatenate((np.zeros(shape=(h1.shape[0],h0.shape[1])),h1),axis=1)
    h_0 = np.concatenate((h_0_1,h_0_2),axis=0)
    
    assert h_0.shape[0] == h_0.shape[1]
    
    eigs, vecs = eigh(h_0)
    inds = np.array([np.where(e==eigs) for e in np.diag(h0)]).flatten()
    vec_A = vecs[:,inds]
    
    proj = get_bb_action(h_0, vec_A)
    
    assert np.all(h1 == (proj @ np.eye(h_0.shape[0]))[h0.shape[0]:,h0.shape[1]:])
    
def test_array_vs_proj(hamiltonians, wanted_orders):
    n_a, n_b = hamiltonians[0].shape[0], hamiltonians[1].shape[0]

    # initialize arrays
    H_0_AA_arr = hamiltonians[0]
    H_0_BB_arr = hamiltonians[1]

    H_p_AA_arr = hamiltonians[2]
    H_p_AB_arr = hamiltonians[4]
    H_p_BB_arr = hamiltonians[3]

    exp_S_arr = compute_next_orders(H_0_AA_arr,
                                    H_0_BB_arr,
                                    H_p_AA_arr,
                                    H_p_BB_arr,
                                    H_p_AB_arr,
                                    wanted_orders=wanted_orders
                                    )

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
    H_0_BB_sop = SumOfOperatorProducts([[(get_bb_action(block_diag(hamiltonians[0],hamiltonians[1]), np.eye(n_a+n_b)[:,:n_a]), 'BB')]])

    H_p_AA_sop = {key:SumOfOperatorProducts([[(val,'AA')]]) for key,val in hamiltonians[2].items()}
    H_p_AB_sop = {key:SumOfOperatorProducts([[(val,'AB')]]) for key,val in h_ab.items()}
    H_p_BB_sop = {key:SumOfOperatorProducts([[(val,'BB')]]) for key,val in h_bb.items()}

    exp_S_sop = compute_next_orders(H_0_AA_sop,
                                    H_0_BB_sop,
                                    H_p_AA_sop,
                                    H_p_BB_sop,
                                    H_p_AB_sop,
                                    wanted_orders=wanted_orders,
                                    divide_energies=lambda Y:divide_energies(Y, H_0_AA_sop, H_0_BB_sop, mode='op')
                                    )
    
    # make all SOPs matrices
    for i in (0,1):
        for j in (0,1):
            if (i == 0 and j == 0):
                exp_S_sop[i,j] = {k:v.to_array() for k,v in exp_S_sop[i,j].items() if isinstance(v,SumOfOperatorProducts)}
            if (i == 0 and j == 1):
                exp_S_sop[i,j] = {k:v.to_array()[:,n_a:] for k,v in exp_S_sop[i,j].items() if isinstance(v,SumOfOperatorProducts)}
            if (i == 1 and j == 0):
                exp_S_sop[i,j] = {k:v.to_array()[n_a:,:] for k,v in exp_S_sop[i,j].items() if isinstance(v,SumOfOperatorProducts)}
            if (i == 1 and j == 1):
                exp_S_sop[i,j] = {k:v.to_array()[n_a:,n_a:] for k,v in exp_S_sop[i,j].items() if isinstance(v,SumOfOperatorProducts)}
            
    # subttract two results
    exp_S_diff = np.zeros_like(exp_S_arr)
    
    for i in (0,1):
        for j in (0,1):
            temp = {}
            for key,val in exp_S_arr[i,j].items():
                if isinstance(val,np.ndarray):
                    temp[key]=val-exp_S_sop[i,j][key]
            exp_S_diff[i,j] = temp
    exp_S_diff = [exp_S_diff[0,0], exp_S_diff[1,1], exp_S_diff[0,1]]      
    for value, block in zip(exp_S_diff, "AA BB AB".split()):
        assert_almost_zero(value, 6, extra_msg=f"{block=}")

