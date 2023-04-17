import numpy as np
import scipy
import sympy
import tinyarray as ta
from lowdin import linalg, block_diagonalization
from lowdin.tests import test_block_diagonalization
from lowdin.linalg import complement_projected, ComplementProjector
from lowdin.series import BlockSeries, zero

# +
n = 1
dim = 25

h_0 = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
h_0 += h_0.conj().transpose()

#h_0 = np.diag(np.sort(np.random.random(dim)))

eigs, vecs = scipy.linalg.eigh(h_0)

eigs_a = eigs[:2]
vecs_a = vecs[:,:2]

eigs_b = eigs[2:10]
vecs_b = vecs[:,2:10]

h = {ta.array([0 for _ in range(n)]): h_0}

for i in range(n):
    h_p = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
    h_p += h_p.conj().transpose()
    index = np.zeros(n,int)
    index[i] = 1
    h[ta.array(index)] = h_p
# -

H = BlockSeries(data={(0,0,0): h[ta.array([0])], (0,0,1): h[ta.array([1])]}, shape=(1,1), n_infinite=1)

h_t, u, u_adj = block_diagonalization.numerical(H, vecs_a, eigs_a, vecs_b, eigs_b, kpm_params={'num_moments':10000})

h_t.evaluated[0,1,6]


def test_solve_sylvester():
    n_trial = np.random.randint(low=3, high=10)
    n_dim = np.random.randint(low=5, high=100)
    a_dim = np.random.randint(low=1, high=int(n_dim/2))
    a_indices = np.random.randint(low=0, high=n_dim, size=a_dim)
    b_dim = n_dim-a_dim
    b_indices = np.delete(np.arange(0, n_dim), a_indices)
    
    h_0 = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
    h_0 += h_0.conjugate().transpose()
    
    eigs, vecs = scipy.linalg.eigh(h_0)
    eigs = 100 * np.random.random(1) * eigs
    h_0 = vecs @ np.diag(eigs) @ vecs.conjugate().transpose()
    eigs_a, vecs_a = eigs[a_indices], vecs[:, a_indices]
    eigs_b, vecs_b = eigs[b_indices], vecs[:, b_indices]
    print("min energy difference of A and B is {}".format(np.min(np.abs(eigs_a.reshape(-1,1) - eigs_b))))
    
    divide_energies_full_b = block_diagonalization.solve_sylvester_KPM(h_0, vecs_a, eigs_a, vecs_b, eigs_b)
    divide_energies_half_b = block_diagonalization.solve_sylvester_KPM(h_0, vecs_a, eigs_a, vecs_b[:,b_indices[:int(np.floor(len(b_indices)/2))]], eigs_b[b_indices[:int(np.floor(len(b_indices)/2))]], kpm_params = {'num_moments':10000})
    divide_energies_kpm = block_diagonalization.solve_sylvester_KPM(h_0, vecs_a, eigs_a, kpm_params={'num_moments': 20000})
    
    for _ in range(n_trial):
        y_trial = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
        y_trial += y_trial.conjugate().transpose()
        y_trial = y_trial[:len(eigs_a), :]
        
        y_full_b = np.abs(divide_energies_full_b(y_trial))
        y_half_b = np.abs(divide_energies_half_b(y_trial))
        y_kpm = np.abs(divide_energies_kpm(y_trial))
        
        diff_full_half = y_full_b - y_half_b 
        diff_full_kpm = y_full_b - y_kpm
        
        try: 
            np.testing.assert_allclose(diff_full_half, 0, atol=1e-2, err_msg="fail in full/half at max val {}".format(np.max(diff_full_half)))
        except Exception as e:
            print(e)
        try:
            np.testing.assert_allclose(diff_full_kpm, 0, atol=1e-2, err_msg="fail in full/kpm at max val {}".format(np.max(diff_full_kpm))) 
        except Exception as e:
            print(e)


for _ in range(10):
    print('############')
    try:
        test_solve_sylvester()
    except Exception as e:
        print(e)
        continue
print('finished')


def test_check_AB_KPM(wanted_orders: list[tuple[int, ...]]) -> None:
    """
    Test that H_AB is zero for a random Hamiltonian.

    Parameters
    ----------
    H: Hamiltonian
    wanted_orders: list of orders to compute
    """
    for order in wanted_orders:
        n_pert = len(order)
        n_dim = np.random.randint(low=5, high=100)
        a_dim = np.random.randint(low=1, high=int(n_dim/2))
        a_indices = np.random.randint(low=0, high=n_dim, size=a_dim)
        b_dim = n_dim-a_dim
        b_indices = np.delete(np.arange(0, n_dim), a_indices)
        
        assert not bool(set(a_indices) & set(b_indices))
        
        h_0 = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
        h_0 += h_0.conjugate().transpose()

        eigs, vecs = scipy.linalg.eigh(h_0)
        eigs = 1000 * eigs
        h_0 =  vecs @ np.diag(eigs) @ vecs.conjugate().transpose()
        
        eigs_a, vecs_a = eigs[a_indices], vecs[:, a_indices]
        eigs_b, vecs_b = eigs[b_indices], vecs[:, b_indices]
        print(np.min(np.abs(eigs_a.reshape(-1,1) - eigs_b)))
        
        
        H_input = BlockSeries(data={(0,0,0):h_0} )
        
        for i in range(n_pert):
            h_p = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
            h_p += h_p.conjugate().transpose()
            index = np.zeros(n_pert, int)
            index[i] = 1
            H_input.data[(0,0,*tuple(index))] = h_p
        
        H_input.shape = (1,1)
        H_input.n_infinite = n_pert
        
        H_tilde_full_b = block_diagonalization.numerical(H_input, vecs_a, eigs_a, vecs_b, eigs_b)[0]
        H_tilde_half_b = block_diagonalization.numerical(H_input, vecs_a, eigs_a, vecs_b[:,b_indices[:int(np.floor(b_dim/2))]], eigs_b[b_indices[:int(np.floor(int(b_dim/2)))]], kpm_params={'num_moments': 5000})[0]
        H_tilde_kpm = block_diagonalization.numerical(H_input, vecs_a, eigs_a, kpm_params={'num_moments': 10000})[0]
        
        # full b
        for order in wanted_orders:
            order = tuple(slice(None, dim_order + 1) for dim_order in order)
            for block in H_tilde_full_b.evaluated[(0, 1) + order].compressed():
                np.testing.assert_allclose(
                    block, 0, atol=1e-5, err_msg=f"{block=}, {order=}"
                )
        # half b
        for order in wanted_orders:
            order = tuple(slice(None, dim_order + 1) for dim_order in order)
            for block in H_tilde_half_b.evaluated[(0, 1) + order].compressed():
                np.testing.assert_allclose(
                    block, 0, atol=1e-1, err_msg=f"{block=}, {order=}"
                )   
        # KPM
        for order in wanted_orders:
            order = tuple(slice(None, dim_order + 1) for dim_order in order)
            for block in H_tilde_kpm.evaluated[(0, 1) + order].compressed():
                np.testing.assert_allclose(
                    block, 0, atol=1e-1, err_msg=f"{block=}, {order=}"
                )

test_check_AB_KPM([(3,)])


