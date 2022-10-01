import numpy as np
import sympy
from sympy import (
    Symbol, MatrixSymbol, Matrix,
    diff, BlockMatrix, BlockDiagMatrix,
    ZeroMatrix, Identity
)
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum import TensorProduct
import matplotlib.pyplot as plt

# ### Obtaining $\tilde{H}^{(n)AB}$

# +
N = Symbol('N')

U_AA = Symbol('U^{AA}')
U_AB = Symbol('U^{AB}')
U_BB = Symbol('U^{BB}')

V_AB = Symbol('V^{AB}')

# +
wanted_order = 4

H_AA = MatrixSymbol('H^{AA}_0', N/2, N/2)
H_BB = MatrixSymbol('H^{BB}_0', N/2, N/2)

H_1_AA = MatrixSymbol('H^{AA}_1', N/2, N/2)
H_1_BB = MatrixSymbol('H^{BB}_1', N/2, N/2)
H_2_AB = MatrixSymbol('H^{AB}_2', N/2, N/2)
H_2_BA = Dagger(H_2_AB)

U_AAn = [Identity(N/2), ZeroMatrix(N/2, N/2)]
U_ABn = [ZeroMatrix(N/2, N/2), ZeroMatrix(N/2, N/2)]
U_BAn = [ZeroMatrix(N/2, N/2), ZeroMatrix(N/2, N/2)]
U_BBn = [Identity(N/2), ZeroMatrix(N/2, N/2)]
U_AAn += [MatrixSymbol(U_AA.name + '_{}'.format(n), N/2, N/2) for n in range(2, wanted_order + 1)]
U_ABn += [MatrixSymbol(U_AB.name + '_{}'.format(n), N/2, N/2) for n in range(2, wanted_order + 1)]
U_BAn += [Dagger(U_ABn[n]) for n in range(2, wanted_order + 1)]
U_BBn += [MatrixSymbol(U_BB.name + '_{}'.format(n), N/2, N/2) for n in range(2, wanted_order + 1)]

V_ABn = [ZeroMatrix(N/2, N/2), Identity(N/2)]
V_BAn = [ZeroMatrix(N/2, N/2), -Identity(N/2)]
V_ABn += [MatrixSymbol(V_AB.name + '_{}'.format(n), N/2, N/2) for n in range(2, wanted_order + 1)]
V_BAn += [-Dagger(V_ABn[n]) for n in range(2, wanted_order + 1)]

H_0 = BlockMatrix([[H_AA, ZeroMatrix(N/2, N/2)], [ZeroMatrix(N/2, N/2), H_BB]])
H_p = BlockMatrix([[H_1_AA, H_2_AB], [H_2_BA, H_1_BB]])

U_n = [BlockMatrix([[U_AAn[n], U_ABn[n]], [U_BAn[n], U_BBn[n]]]) for n in range(0, wanted_order + 1)]
V_n = [BlockMatrix([[ZeroMatrix(N/2, N/2), V_ABn[n]], [V_BAn[n], ZeroMatrix(N/2, N/2)]]) for n in range(0, wanted_order + 1)]

zero = BlockMatrix([[ZeroMatrix(N/2, N/2), ZeroMatrix(N/2, N/2)], [ZeroMatrix(N/2, N/2), ZeroMatrix(N/2, N/2)]])


# -

# $$
# \tilde{H}^{(n)} = \sum_{i=0}^n (U_{n-i} - V_{n-i}) H_0 (U_i + V_i) + \sum_{i=0}^{n-1} (U_{n-i-1} - V_{n-i-1}) H_p (U_i + V_i).
# $$

def H_tilde(H_0, H_p, wanted_order, U_n, V_n):
    """Returns H tilde to a certain order"""
    H_tilde_n = []
    
    if isinstance(V_n[0], sympy.matrices.expressions.blockmatrix.BlockMatrix):
        first_term = zero
        second_term = zero
    else:
        first_term = V_n[0]
        second_term = V_n[0]
    for n in range(0, wanted_order+1):
        for i in range(0, n + 1):
            first_term += (U_n[n-i] - V_n[n-i]) @ H_0 @ (U_n[i] + V_n[i])
            if i<n:
                second_term += (U_n[n-i-1] - V_n[n-i-1]) @ H_p @ (U_n[i] + V_n[i])
        H_tilde_n.append(first_term + second_term)
    return H_tilde_n


H_tilde_n = H_tilde(H_0, H_p, wanted_order, U_n, V_n)

sympy.block_collapse(H_tilde_n[1]).blocks[0, 1] # should be 0 and give condition for V^AB

sympy.block_collapse(H_tilde_n[1]).blocks[1, 0] # should be 0 and give condition for V^AB

# ### Computing $U_n$ and $V_n$

# These are the blocks computed recursively
# $$
# \begin{align}
# -2U_n^{AA} &= \sum_{i=1}^{n-i} \left(U_{n-i}^{AA}U_i^{AA} + U_{n-i}^{AB}(U_i^{AB})^\dagger + U_{n-i}^{AB}(V_i^{AB})^\dagger + V_{n-i}^{AB}(U_i^{AB})^\dagger + V_{n-i}^{AB}(V_i^{AB})^\dagger\right)\\
# -2U_n^{BB} &= \sum_{i=1}^{n-i} \left(U_{n-i}^{BB}U_i^{BB} + (U_{n-i}^{AB})^\dagger U_i^{AB} - (U_{n-i}^{AB})^\dagger V_i^{AB} - (V_{n-i}^{AB})^\dagger U_i^{AB} + (V_{n-i}^{AB})^\dagger V_i^{AB}\right)\\
# -2U_n^{AB} &= \sum_{i=1}^{n-i} \left(U_{n-i}^{AA}U_i^{AB} + U_{n-i}^{AB}U_i^{BB} - V_{n-i}^{AB} U_i^{BB} + U_{n-i}^{AA} V_i^{AB}\right)\\
# \end{align}.
# $$

np.arange(4).reshape(-1, 1) + np.arange(5)


def compute_next_orders(H_0, H_p, wanted_order, N_A=None):
    """
    H_0 : np Hamiltonian in eigenbasis and ordered by eigenenergy.
    H_p : np Hamiltonian in eigenbasis of H_0
    wanted_order : int order of perturbation
    
    Returns:
    U_AAn : list of AA block matrices up to order wanted_order
    U_ABn : list of AB block matrices up to order wanted_order
    U_BBn : list of BB block matrices up to order wanted_order
    V_ABn : list of AB block matrices up to order wanted_order
    """
    N = H_0.shape[0]
    if N_A is None:
        N_A = N // 2
    N_B = N - N_A
    H_p_AA = H_p[:N_A, :N_A]
    H_p_AB = H_p[:N_A, N_A:]
    H_p_BB = H_p[N_A:, N_A:]

    # Blocks of U and V
    # 0th order
    U_AAn = [np.eye(N_A, dtype=complex)]
    U_ABn = [np.zeros((N_A, N_B), dtype=complex)]
    U_BBn = [np.eye(N_B, dtype=complex)]
    V_ABn = [np.zeros((N_A, N_B), dtype=complex)]
    if wanted_order = 0:
        return U_AAn, U_ABn, U_BBn, V_ABn
    
    #1st order
    E_A = np.diag(H_0)[:N_A]
    E_B = np.diag(H_0)[N_A:]
    energy_denominators = 1/(E_A.reshape(-1, 1) - E_B)
    
    U_AAn.append(np.zeros((N_A, N_A), dtype=complex))
    U_ABn.append(np.zeros((N_A, N_B), dtype=complex))
    U_BBn.append(np.zeros((N_B, N_B), dtype=complex))
    V_ABn.append((       
                 U_AAn[0] @ H_p_AA @ (U_ABn[0] + V_ABn[0]) +
                 U_AAn[0] @ H_p_AB @ U_BBn[0] +
                 (U_ABn[0] - V_ABn[0]) @ H_p_AB.conj().T @ (U_ABn[0] + V_ABn[0]) +
                 (U_ABn[0] - V_ABn[0]) @ H_p_BB @ U_BBn[0]
                 ) * energy_denominators
                ) # V_1
    if wanted_order = 1:
        return U_AAn, U_ABn, U_BBn, V_ABn

    for n in range(2, wanted_order+1):
        U_AA_next = np.zeros((N_A, N_A), dtype=complex)
        U_AB_next = np.zeros((N_A, N_B), dtype=complex)
        U_BB_next = np.zeros((N_B, N_B), dtype=complex)
        Y_next = -(
            U_AAn[n-1] @ H_p_AA @ (U_ABn[0] + V_ABn[0]) +
            U_AAn[n-1] @ H_p[:N_A, N_A:] @ U_BBn[0] +
            (U_ABn[n-1] - V_ABn[n-1]) @ H_p[:N_A, N_A:].conj().T @ (U_ABn[0] + V_ABn[0]) +
            (U_ABn[n-1] - V_ABn[n-1]) @ H_p[N_A:, N_A:] @ U_BBn[0]
        )
        for i in range(1, n):
            U_AA_next -= (
                U_AAn[n-i] @ U_AAn[i] + U_ABn[n-i] @ U_ABn[i].conj().T +
                U_ABn[n-i] @ V_ABn[i].conj().T + V_ABn[n-i] @ U_ABn[i].conj().T +
                V_ABn[n-i] @ V_ABn[i].conj().T
            ) / 2
            U_AB_next -= (
                U_AAn[n-i] @ U_ABn[i] + U_ABn[n-i] @ U_BBn[i] -
                V_ABn[n-i] @ U_BBn[i] + U_AAn[n-i] @ V_ABn[i]
            ) / 2
            U_BB_next -= (
                U_BBn[n-i] @ U_BBn[i] + U_ABn[n-i].conj().T @ U_ABn[i] -
                U_ABn[n-i].conj().T @ V_ABn[i] - V_ABn[n-i].conj().T @ U_ABn[i] +
                V_ABn[n-i].conj().T @ V_ABn[i]
            ) / 2
            Y_next -= (
                U_ABn[n-i] @ H_p_AB.T.conj() @ (U_ABn[i] + V_ABn[i]) +
                (U_ABn[n-i] - V_ABn[n-i]) @ H_p_BB @ U_BBn[i] +
                U_AAn[n-i-1] @ H_p_AA @ (U_ABn[i] + V_ABn[i]) +
                U_AAn[n-i-1] @ H_p_AB @ U_BBn[i] +
                (U_ABn[n-i-1] - V_ABn[n-i-1]) @ H_p_AB.conj().T @ (U_ABn[i] + V_ABn[i]) +
                (U_ABn[n-i-1] - V_ABn[n-i-1]) @ H_p_BB @ U_BBn[i]
            )

        U_AAn.append(U_AA_next)
        U_ABn.append(U_AB_next)
        U_BBn.append(U_BB_next)

        Y_next -= H_0[:N_A, :N_A] @ U_AB_next + U_AB_next @ H_0[N_A:, N_A:]
        V_ABn.append(Y_next * energy_denominators)
        
    return U_AAn, U_ABn, U_BBn, V_ABn


# ### Testing

# +
wanted_order = 4
N_A = 4
N_B = 5
N = N_A + N_B
H_0 = np.diag(np.sort(np.random.randn(N)))

strength = 0.0001
H_p = np.random.random(size=(N, N)) + 1j * np.random.random(size=(N, N))
H_p += H_p.conj().T
H_p = strength * H_p
# -

U_AAn, U_ABn, U_BBn, V_ABn = compute_next_orders(H_0, H_p, wanted_order, N_A=N_A)

evals, evecs = np.linalg.eigh(H_0 + H_p)

# +
U_n = [np.block([[U_AA, U_AB], [U_AB.conj().T, U_BB]]) for U_AA, U_AB, U_BB in zip(U_AAn, U_ABn, U_BBn)]
V_n = [np.block([[np.zeros((N_A, N_A)), V_AB], [-V_AB.conj().T, np.zeros((N_B, N_B))]]) for V_AB in V_ABn]

H_tilde_n = H_tilde(H_0, H_p, wanted_order, U_n, V_n)
# -

# Exact diagonalization
plt.imshow(np.diag(evals.real), cmap='seismic')
plt.colorbar()

if wanted_order==0:
    H_tilde_sum = H_tilde_n[0]
else:
    H_tilde_sum = np.sum(H_tilde_n, axis=0)

# Lowdin result
plt.imshow(H_tilde_sum.real, cmap='seismic')
plt.colorbar()

# Difference between Lowdin and exact diagonalization
plt.imshow(np.diag(evals.real)-H_tilde_sum.real/(2*wanted_order+2),
           vmin=-np.max(np.abs(np.diag(evals.real)-H_tilde_sum.real/(2*wanted_order+2))),
           vmax=np.max(np.abs(np.diag(evals.real)-H_tilde_sum.real/(2*wanted_order+2))),
           cmap='seismic'
          ) # factor of 2 mismatch
plt.colorbar()

# :) :tada:

# **There is a factor of 2 mismatch per order of H**, but that's for later 

# #### Other plots

H_tilde_order = H_tilde_n[1].real
plt.imshow(H_tilde_order,
           vmin=-np.max(np.abs(H_tilde_order)),
           vmax=np.max(np.abs(H_tilde_order)),
           cmap='seismic')
plt.colorbar()


