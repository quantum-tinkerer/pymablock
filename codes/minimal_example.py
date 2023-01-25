# +
import scipy
import scipy.stats
import numpy as np
import tinyarray as ta
from itertools import product
from functools import reduce
import polynomial_orders_U


def assert_almost_zero(a, decimal, extra_msg=""):
    """
    Assert that all values in a are almost zero.

    a : dict of arrays
        The values to check.
    decimal : int
        The number of decimals to check.
    extra_msg : str

    Returns:
    None
    """
    for key, value in a.items():
        np.testing.assert_almost_equal(
            value, 0, decimal=decimal, err_msg=f"{key=} {extra_msg}"
        )


# -

N = 10
N_A = 4
N_B = N - N_A

# +
H_0 = np.diag(np.random.random(N))

print("norm H0=" + str(np.linalg.norm(H_0)))
print("cond H0=" + str(np.linalg.cond(H_0)))

vecs = scipy.stats.unitary_group.rvs(N)
vecs /= 0.5 * np.linalg.norm(vecs)
print("norm=" + str(np.linalg.norm(vecs)))
print("cond=" + str(np.linalg.cond(vecs)))

# +
N_p = 2

H_ps = []
for perturbation in range(N_p):
    H_p = np.random.random(size=(N, N)) + 1j * np.random.random(size=(N, N))
    H_p += H_p.conj().T
    H_p = (vecs.conj().T @ H_p @ vecs) / 2
    print("norm H_{}=".format(perturbation) + str(np.linalg.norm(H_p)))
    print("cond H_{}=".format(perturbation) + str(np.linalg.cond(H_p)))

    H_ps.append(H_p)

orders = ta.array(np.eye(N_p))

# +
H_p_AA = {order: value[:N_A, :N_A] for order, value in zip(orders, H_ps)}

H_p_BB = {order: value[N_A:, N_A:] for order, value in zip(orders, H_ps)}

H_p_AB = {order: value[:N_A, N_A:] for order, value in zip(orders, H_ps)}
# -

H_0_AA = H_0[:N_A, :N_A]
H_0_BB = H_0[N_A:, N_A:]

# +
w_orders = [
    ta.array(v) for v in product(np.arange(0, 3 + 1), repeat=N_p) if sum(v) <= 3
]

## generate trafo to effective

exp_S = polynomial_orders_U.compute_next_orders(
    H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders=w_orders
)


H_eff_AA, H_eff_BB, H_eff_AB = polynomial_orders_U.H_tilde(
    H_0_AA,
    H_0_BB,
    H_p_AA,
    H_p_BB,
    H_p_AB,
    wanted_orders=w_orders,
    exp_S=exp_S,
    compute_AB=True,
)

# +
# Is exp_S unitary?
transformed = polynomial_orders_U.H_tilde(
    np.eye(N_A), np.eye(N_B), {}, {}, {}, w_orders, exp_S, compute_AB=True
)

for value, block in zip(transformed, "AA BB AB".split()):
    assert_almost_zero(value, 6, f"{block=}")
    print(f"{block=}" + "passed")
# -

# Does AB really vanish?
assert_almost_zero(H_eff_AB, 6)
