from codes.polynomial_orders_U import compute_next_orders, H_tilde
import numpy as np
import tinyarray as ta
import sympy

rand_gen = np.random.default_rng(13012023)

def assert_almost_zero(a, decimal, extra_msg=""):
    """Compare two dictionaries with array-like values."""
    for key, value in a.items():
        np.testing.assert_almost_equal(
            value, 0, decimal=decimal, err_msg=f"{key=} {extra_msg}"
        )


def test_check_AB():
    # initialize randomized parameters
    decimal = 5

    N_A = rand_gen.integers(0,high=10)
    N_B = rand_gen.integers(0,high=20)
    N = N_A + N_B

    H_0 = np.diag(np.sort(rand_gen.normal(0, size=N)))

    N_p = rand_gen.integers(1,high=3)

    wanted_orders = [ta.array(rand_gen.integers(0,high=3,size=N_p), int)]
    H_ps = []
    for perturbation in range(N_p):
        H_p = rand_gen.normal(size=(N, N)) + 1j * rand_gen.normal(size=(N, N))
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
    exp_S = compute_next_orders(
        H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders=wanted_orders
    )

    H_AB = H_tilde(
        H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, exp_S, compute_AB=True
    )[2]

    assert_almost_zero(H_AB, decimal)


def test_check_unitary():
    decimal = 5

    N_A = rand_gen.integers(0, high=10)
    N_B = rand_gen.integers(0, high=20)
    N = N_A + N_B

    #Init randomized Hamiltonian to generate some exp_S

    H_0 = np.diag(np.sort(rand_gen.normal(0, size=N)))

    N_p = rand_gen.integers(1, high=3)

    wanted_orders = [ta.array(rand_gen.integers(1, high=3, size=N_p), int)]
    H_ps = []
    for perturbation in range(N_p):
        H_p = rand_gen.normal(size=(N, N)) + 1j * rand_gen.normal(size=(N, N))
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

    exp_S = compute_next_orders(
        H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders=wanted_orders
    )

    #Check unitarity

    transformed = H_tilde(np.eye(N_A), np.eye(N_B), {}, {}, {}, wanted_orders, exp_S, compute_AB=True)

    for value, block in zip(transformed, "AA BB AB".split()):
        assert_almost_zero(value, decimal, f"{block=}")


