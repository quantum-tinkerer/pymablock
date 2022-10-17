# # The polynomial alternative to Lowdin perturbation theory
#
# See [this hackmd](https://hackmd.io/Rpt2C8oOQ2SGkGS9OYrlfQ?view) for the motivation and the expressions

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
U_AAn += [MatrixSymbol(f'{U_AA.name}_{{{n}}}', N, N) for n in range(2, wanted_order + 1)]
U_BBn += [MatrixSymbol(f'{U_BB.name}_{{{n}}}', M, M) for n in range(2, wanted_order + 1)]

V_ABn = [ZeroMatrix(N, M)]
V_ABn += [MatrixSymbol(V_AB.name + '_{}'.format(n), N, M) for n in range(1, wanted_order + 1)]

H_0 = BlockMatrix([[H_AA, ZeroMatrix(N, M)], [ZeroMatrix(M, N), H_BB]])
H_p = BlockMatrix([[H_1_AA, H_2_AB], [H_2_BA, H_1_BB]])

U_n = [BlockMatrix([[U_AA, ZeroMatrix(N, M)], [ZeroMatrix(M, N), U_BB]]) for U_AA, U_BB in zip(U_AAn, U_BBn)]
V_n = [BlockMatrix([[ZeroMatrix(N, N), V_AB], [-Dagger(V_AB), ZeroMatrix(M, M)]]) for V_AB in V_ABn]

zero = BlockMatrix([[ZeroMatrix(N, N), ZeroMatrix(N, M)], [ZeroMatrix(M, N), ZeroMatrix(M, M)]])


# -

def H_tilde(H_0, H_p, wanted_order, U_n, V_n):
    """Returns H tilde to a certain order"""
    H_tilde_n = []

    for n in range(0, wanted_order+1):
        if isinstance(V_n[0], sympy.matrices.expressions.blockmatrix.BlockMatrix):
            first_term = zero
            second_term = zero
        else:
            first_term = np.zeros_like(V_n[0])
            second_term = np.zeros_like(V_n[0])

        for i in range(0, n + 1):
            first_term += (U_n[n-i] - V_n[n-i]) @ H_0 @ (U_n[i] + V_n[i])
            if i<n:
                second_term += (U_n[n-i-1] - V_n[n-i-1]) @ H_p @ (U_n[i] + V_n[i])
        H_tilde_n.append(first_term + second_term)
    return H_tilde_n


H_tilde_n = H_tilde(H_0, H_p, wanted_order, U_n, V_n)

sympy.block_collapse(H_tilde_n[3]).blocks[0, 1] # should be 0 and give condition for V^AB

sympy.block_collapse(H_tilde_n[1]).blocks[1, 0] # should be 0 and give condition for V^AB


# ### Computing $U_n$ and $V_n$

def compute_next_orders(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_order):
    """
    H_0 : np Hamiltonian in eigenbasis and ordered by eigenenergy.
    H_p : np Hamiltonian in eigenbasis of H_0
    wanted_order : int order of perturbation
    
    Returns:
    U_AAn : list of AA block matrices up to order wanted_order
    U_BBn : list of BB block matrices up to order wanted_order
    V_ABn : list of AB block matrices up to order wanted_order
    """
    H_0_AA = np.array(H_0_AA)
    H_0_BB = np.array(H_0_BB)
    H_p_AA = np.array(H_p_AA)
    H_p_BB = np.array(H_p_BB)
    H_p_AB = np.array(H_p_AB)
    
    N_A = H_0_AA.shape[0]
    N_B = H_0_BB.shape[0]
    
    assert H_p_AA.shape[0]==N_A
    assert H_p_BB.shape[0]==N_B
    assert H_p_AB.shape[0]==N_A
    assert H_p_AB.shape[1]==N_B
        
    # Blocks of U and V
    # 0th order
    U_AAn = [np.eye(N_A, dtype=complex)]
    U_BBn = [np.eye(N_B, dtype=complex)]
    V_ABn = [np.zeros((N_A, N_B), dtype=complex)]
    
    if wanted_order == 0:
        return U_AAn, U_BBn, V_ABn
        
    #1st order
    E_A = np.diag(np.array(H_0_AA))
    E_B = np.diag(np.array(H_0_BB))
    energy_denominators = 1/(E_A.reshape(-1, 1) - E_B)
    
    U_AAn.append(np.zeros((N_A, N_A), dtype=complex))
    U_BBn.append(np.zeros((N_B, N_B), dtype=complex))
    V_ABn.append(-H_p_AB * energy_denominators)
    if wanted_order == 1:
        return U_AAn, U_BBn, V_ABn

    for n in range(2, wanted_order+1):
        U_AA_next = np.zeros((N_A, N_A), dtype=complex)
        U_BB_next = np.zeros((N_B, N_B), dtype=complex)
        Y_next = np.zeros_like(V_ABn[0])

        for i in range(n):
            Y_next = Y_next - (
                + U_AAn[n-i-1] @ H_p_AA @ V_ABn[i]
                + U_AAn[n-i-1] @ H_p_AB @ U_BBn[i]
                - V_ABn[n-i-1] @ Dagger(H_p_AB) @ V_ABn[i]
                - V_ABn[n-i-1] @ H_p_BB @ U_BBn[i]
            )
        for i in range(1, n):
            Y_next = Y_next - U_AAn[n-i] @ H_0_AA @ V_ABn[i] - V_ABn[n-i] @ H_0_BB @ U_BBn[i]
            U_AA_next = U_AA_next - (U_AAn[n-i] @ U_AAn[i] + V_ABn[n-i] @ Dagger(V_ABn[i])) / 2
            U_BB_next = U_BB_next - (U_BBn[n-i] @ U_BBn[i] + Dagger(V_ABn[n-i]) @ V_ABn[i]) / 2
        
        if isinstance(H_p, np.ndarray):
            if any(not np.all(np.isfinite(mat)) for mat in (U_AA_next, U_BB_next, Y_next)):
                raise RuntimeError(f"Instability encountered in {n}th order.")
        U_AAn.append(U_AA_next)
        U_BBn.append(U_BB_next)
        V_ABn.append(Y_next * energy_denominators)
    if type(H_p) == type(Matrix()):
        U_AAn = [Matrix(u) for u in U_AAn]
        U_BBn = [Matrix(u) for u in U_BBn]
        V_ABn = [Matrix(u) for u in V_ABn]
        
    return U_AAn, U_BBn, V_ABn


# ### Testing

# +
wanted_order = 4
N_A = 2
N_B = 2
N = N_A + N_B
H_0 = np.diag(np.sort(np.random.randn(N)))

kx, ky, kz = symbols('k_x k_y k_z')
a, b, c = symbols('a b c')
H_p = Matrix([[a*(kx**2+ky**2)/2,0,c*kz,0],[0,b*(kx**2+ky**2)/2,0,c*kz],[0,0,a*(kx**2+ky**2)/2,0],[0,0,0,b*(kx**2+kz**2)/2]])
H_p += Dagger(H_p).T
H_p = H_p

H_0_AA = H_0[:N_A, :N_A]
H_0_BB = H_0[N_B:, N_B:]
H_p_AA = H_p[:N_A, :N_A]
H_p_BB = H_p[N_B:, N_B:]
H_p_AB = H_p[:N_A, N_B:]

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
# -

H_tilde_n[3]

H_p
