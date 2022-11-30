from codes import polynomial_orders_U
import numpy as np
import tinyarray as ta
import sympy


def test_check_AB():
    # initialize randomized parameters
    precision_tol = 1e-5

    N_A = np.random.randint(0,high=10)
    N_B = np.random.randint(0,high=20)
    N = N_A + N_B

    H_0 = np.diag(np.sort(np.random.randn(N)))

    N_p = np.random.randint(1,high=5)

    wanted_orders = [ta.array(np.random.randint(0,high=5,size=N_p), int)]
    H_ps = []
    for perturbation in range(N_p):
        H_p = np.random.random(size=(N, N)) + 1j * np.random.random(size=(N, N))
        H_p += H_p.conj().T
        H_ps.append(H_p)

    H_0_AA = H_0[:N_A, :N_A]
    H_0_BB = H_0[N_A:, N_A:]

    orders = ta.array(np.eye(N_p))
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
    exp_S = polynomial_orders_U.compute_next_orders(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders=wanted_orders)
    
    H_AB = polynomial_orders_U.H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, exp_S, compute_AB=True)[2]

    assert np.all([np.allclose(v, 0, atol=precision_tol) for k,v in H_AB.items()]),\
            "H_AB does not vanish within requested tolerance of {}.".format(precision_tol)

def test_check_unitary():
    precision_tol = 1e-5

    N_A = np.random.randint(0, high=10)
    N_B = np.random.randint(0, high=20)
    N = N_A + N_B

    #Init randomized Hamiltonian to generate some exp_S
    
    H_0 = np.diag(np.sort(np.random.randn(N)))

    N_p = np.random.randint(1, high=5)

    wanted_orders = [ta.array(np.random.randint(1, high=5, size=N_p), int)]
    H_ps = []
    for perturbation in range(N_p):
        H_p = np.random.random(size=(N, N)) + 1j * np.random.random(size=(N, N))
        H_p += H_p.conj().T
        H_ps.append(H_p)

    H_0_AA = H_0[:N_A, :N_A]
    H_0_BB = H_0[N_A:, N_A:]

    orders = ta.array(np.eye(N_p))
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

    exp_S = polynomial_orders_U.compute_next_orders(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders=wanted_orders)
    
    #Reinitialize H as identities
    H_0 = np.eye(N)
    
    H_ps = []
    for perturbation in range(N_p):
        H_p = np.eye(N)
        H_ps.append(H_p)

    H_0_AA = H_0[:N_A, :N_A]
    H_0_BB = H_0[N_A:, N_A:]

    orders = ta.array(np.eye(N_p))
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

    H_AA, H_BB, H_AB = polynomial_orders_U.H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, exp_S, compute_AB=True)

    test_AA = np.all([np.allclose(value, 0, atol=precision_tol) for key ,value in H_AA.items()])
    test_BB = np.all([np.allclose(value, 0, atol=precision_tol) for key ,value in H_BB.items()])
    test_AB = np.all([np.allclose(value, 0, atol=precision_tol) for key ,value in H_AB.items()])

    assert test_AA, H_AA.items()
    assert test_BB, H_BB.items()
    assert test_AB, H_AB.items()
