# # The polynomial alternative to Lowdin perturbation theory
#
# See [this hackmd](https://hackmd.io/Rpt2C8oOQ2SGkGS9OYrlfQ?view) for the motivation and the expressions

# +
from itertools import count, product, chain

import numpy as np
import sympy
from sympy import (
    symbols, Symbol, MatrixSymbol, Matrix,
    diff, BlockMatrix, BlockDiagMatrix,
    ZeroMatrix, Identity, diag, eye, zeros
)
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum import TensorProduct
import matplotlib.pyplot as plt
import tinyarray as ta
# -

# ### Obtaining $\tilde{H}^{(n)AB}$

# +
N, M = symbols('N M')

U_AA = Symbol('U^{AA}')
U_AB = Symbol('U^{AB}')
U_BB = Symbol('U^{BB}')

V_AB = Symbol('V^{AB}')

# +
wanted_order = 4

H_AA = MatrixSymbol('H^{AA}_0', N, N)
H_BB = MatrixSymbol('H^{BB}_0', M, M)

H_1_AA = MatrixSymbol('H^{AA}_1', N, N)
H_1_BB = MatrixSymbol('H^{BB}_1', M, M)
H_2_AB = MatrixSymbol('H^{AB}_2', N, M)
H_2_BA = Dagger(H_2_AB)

U_AAn = [Identity(N), ZeroMatrix(N, N)]
U_BBn = [Identity(M), ZeroMatrix(M, M)]
U_AAn += [MatrixSymbol(f'{U_AA.name}_{{{n}}}', N, N)
          for n in range(2, wanted_order + 1)]
U_BBn += [MatrixSymbol(f'{U_BB.name}_{{{n}}}', M, M)
          for n in range(2, wanted_order + 1)]

V_ABn = [ZeroMatrix(N, M)]
V_ABn += [MatrixSymbol(V_AB.name + '_{}'.format(n), N, M)
          for n in range(1, wanted_order + 1)]

H_0 = BlockMatrix([[H_AA, ZeroMatrix(N, M)], [ZeroMatrix(M, N), H_BB]])
H_p = BlockMatrix([[H_1_AA, H_2_AB], [H_2_BA, H_1_BB]])

U_n = [BlockMatrix([[U_AA, ZeroMatrix(N, M)], [ZeroMatrix(M, N), U_BB]])
       for U_AA, U_BB in zip(U_AAn, U_BBn)
      ]
V_n = [BlockMatrix([[ZeroMatrix(N, N), V_AB], [-Dagger(V_AB), ZeroMatrix(M, M)]])
       for V_AB in V_ABn
      ]

zero = BlockMatrix([[ZeroMatrix(N, N), ZeroMatrix(N, M)],
                    [ZeroMatrix(M, N), ZeroMatrix(M, M)]
                    ])


# -

def H_tilde(H_0, H_p, wanted_order, U_n, V_n):
    """Returns H tilde to a certain order"""
    H_tilde_n = []

    for n in range(0, wanted_order+1):
        if isinstance(V_n[0], BlockMatrix):
            first_term = zero
            second_term = zero
        else:
            first_term = np.zeros_like(V_n[0])
            second_term = np.zeros_like(V_n[0])

        for i in range(0, n + 1):
            first_term += (U_n[n-i] - V_n[n-i]) @ H_0 @ (U_n[i] + V_n[i])
            if i < n:
                second_term += (U_n[n-i-1] - V_n[n-i-1]) @ H_p @ (U_n[i] + V_n[i])
        H_tilde_n.append(first_term + second_term)
    return H_tilde_n


H_tilde_n = H_tilde(H_0, H_p, wanted_order, U_n, V_n)

sympy.block_collapse(H_tilde_n[3]).blocks[0, 1]


# ### Computing $U_n$ and $V_n$

# +
def arrays_in_volume(length, total_sum):
    """ Generate all tinyarrays of a given length that add up to total_sum (int). """
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in arrays_in_volume(length - 1, total_sum - value):
                yield ta.array((value,) + permutation)


def generate_volume(wanted_order):
    """ Generate ordered array with all tinyarrays in the volume of wanted_orders. """
    N_p = len(wanted_order[0])
    max_order = np.max([sum(order) for order in wanted_order])
    possible_arrays = [list(arrays_in_volume(N_p, order)) for order in range(max_order+1)]
    possible_arrays = list(chain(*possible_arrays))
    id_keep = [np.any(np.all(array <= np.array(wanted_order), axis=1)) for array in possible_arrays]
    keep_arrays = np.array(possible_arrays)[id_keep]
    return keep_arrays


# +
# simple test
wanted_order = [ta.array([2, 1]), ta.array([1, 0]), ta.array([1, 4])]
for p in wanted_order:
    print(sum(p))
    
generate_volume(wanted_order)


# -

def product_by_order(wanted_order, *polynomials):
    """will write this later"""
    #     itertools.product(*(term.items() for term in terms))
    #     sum(keys) == wanted_order

    #     + U_AAn[n-i-1] @ H_p_AA @ V_ABn[i]
    #     + U_AAn[n-i-1] @ H_p_AB @ U_BBn[i]
    #     - V_ABn[n-i-1] @ Dagger(H_p_AB) @ V_ABn[i]
    #     - V_ABn[n-i-1] @ H_p_BB @ U_BBn[i]


def compute_next_orders(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_order, divide_energies=None):
    """
    H_0_AA : np Hamiltonian A block in eigenbasis and ordered by eigenenergy.
    H_0_BB : np Hamiltonian B block in eigenbasis and ordered by eigenenergy.
    H_p_AA : dictionary of perturbations A blocks in eigenbasis of H_0
    H_p_BB : dictionary of perturbations B blocks in eigenbasis of H_0
    H_p_AB : dictionary of perturbations AB blocks in eigenbasis of H_0
    wanted_order : int order of perturbation

    Returns:
    U_AAn : list of AA block matrices up to order wanted_order
    U_BBn : list of BB block matrices up to order wanted_order
    V_ABn : list of AB block matrices up to order wanted_order
    """
    N_p = len(H_p_AA)
    assert N_p == len(H_p_BB) == len(H_p_AB)

    if divide_energies is None:
        E_A = np.diag(H_0_AA)
        E_B = np.diag(H_0_BB)
        energy_denominators = 1/(E_A.reshape(-1, 1) - E_B)

        def divide_energies(Y):
            return Y * energy_denominators

    for n in range(2, wanted_order+1):
        U_AA_next = np.zeros((N_A, N_A), dtype=complex)
        U_BB_next = np.zeros((N_B, N_B), dtype=complex)
        Y_next = np.zeros_like(V_ABn[0])

        for i in range(n):
            Y_next = Y_next - (
                + product_by_order(wanted_order, [U_AAn, H_p_AA, V_ABn])
                + product_by_order(wanted_order, [U_AAn, H_p_AB, U_BBn])
                - product_by_order(wanted_order, [V_ABn, Dagger(H_p_AB), V_ABn])
                - product_by_order(wanted_order, [V_ABn, H_p_BB, U_BBn])
            )
        for i in range(1, n):
            Y_next = Y_next - U_AAn[n-i] @ H_0_AA @ V_ABn[i] - V_ABn[n-i] @ H_0_BB @ U_BBn[i]
            U_AA_next = U_AA_next - (U_AAn[n-i] @ U_AAn[i] + V_ABn[n-i] @ Dagger(V_ABn[i])) / 2
            U_BB_next = U_BB_next - (U_BBn[n-i] @ U_BBn[i] + Dagger(V_ABn[n-i]) @ V_ABn[i]) / 2

        # if isinstance(H_p_AA, np.ndarray):
        #     if any(not np.all(np.isfinite(mat)) for mat in (U_AA_next, U_BB_next, Y_next)):
        #         raise RuntimeError(f"Instability encountered in {n}th order.")
        U_AAn.append(U_AA_next)
        U_BBn.append(U_BB_next)
        V_ABn.append(divide_energies(Y_next))

    return U_AAn, U_BBn, V_ABn


# ### Testing

# +
wanted_order = 0
N_A = 2
N_B = 2
N = N_A + N_B
H_0 = np.diag(np.sort(np.random.randn(N)))

N_p = 4
H_ps = []
for perturbation in range(N_p):
    H_p = np.random.random(size=(N, N)) + 1j * np.random.random(size=(N, N))
    H_p += H_p.conj().T
    H_ps.append(H_p)

H_0_AA = H_0[:N_A, :N_A]
H_0_BB = H_0[N_A:, N_A:]


def l_i():
    for i in count():
        yield sympy.Symbol(f"lambda_{i}")


coeffs = l_i()
H_p_AA = {
    str(d_i): value[:N_A, :N_A]
    for d_i, value in zip(l_i(), H_ps)
}

coeffs = l_i()
H_p_BB = {
    str(d_i): value[N_A:, N_A:]
    for d_i, value in zip(l_i(), H_ps)
}

coeffs = l_i()
H_p_AB = {
    str(d_i): value[:N_A, N_A:]
    for d_i, value in zip(l_i(), H_ps)
}

wanted_order = [ta.array(np.random.randint(0, 4, size=(N_p))) for i in range(4)]

# +
U_AAn, U_BBn, V_ABn = compute_next_orders(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_order=wanted_order)

# %time U_AAn, U_BBn, V_ABn = compute_next_orders(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_order=wanted_order)

# +
U_n = [np.block([[U_AA, np.zeros((N_A, N_B))], [np.zeros((N_B, N_A)), U_BB]]) for U_AA, U_BB in zip(U_AAn, U_BBn)]
V_n = [np.block([[np.zeros((N_A, N_A)), V_AB], [-Dagger(V_AB), np.zeros((N_B, N_B))]]) for V_AB in V_ABn]

H_tilde_n = H_tilde(H_0, H_p, wanted_order, U_n, V_n)

# +
for H_tilde_ord in H_tilde_n:
    non_hermiticity = np.linalg.norm(H_tilde_ord - H_tilde_ord.T.conj())
    assert non_hermiticity < 1e-10, non_hermiticity
    assert np.linalg.norm(H_tilde_ord[:N_A, N_A:]) < 1e-10

    
def unitarity(strength, U_n, V_n):
    U_tot = sum(
        (strength**i * (U + V) for i, (U, V) in enumerate(zip(U_n, V_n))),
        np.zeros_like(U_n[0])
    )
    return np.linalg.norm(U_tot.T.conj() @ U_tot - np.identity(U_n[0].shape[0]))


def H_pert(strength, H_tilde_n):
    return sum(
        (strength**i * H for i, H in enumerate(H_tilde_n)),
        np.zeros_like(H_tilde_n[0])
    )


def E_pert(strength, H_tilde_n):
    return np.linalg.eigvalsh(H_pert(strength, H_tilde_n))


def E_exact(strength, H_0, H_p):
    return np.linalg.eigvalsh(H_0 + strength * H_p)


strengths = np.logspace(-3, -1)
pert_energies = np.array([E_pert(strength, H_tilde_n) for strength in strengths])
exact_energies = np.array([E_exact(strength, H_0, H_p) for strength in strengths])

plt.figure()
plt.plot(
    strengths,
    [unitarity(strength, U_n, V_n) for strength in strengths] / strengths**(wanted_order)
)
plt.loglog()
plt.title("Matrices are unitary to given order");