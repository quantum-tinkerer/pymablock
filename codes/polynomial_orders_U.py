# # The polynomial alternative to Lowdin perturbation theory
#
# See [this hackmd](https://hackmd.io/Rpt2C8oOQ2SGkGS9OYrlfQ?view) for the motivation and the expressions

# +
from itertools import count, product
from functools import reduce
from operator import matmul

import numpy as np
import sympy
from sympy import (
    symbols, Symbol, MatrixSymbol, Matrix,
    diff, BlockMatrix, BlockDiagMatrix,
    ZeroMatrix, Identity, diag, eye, zeros
)
from IPython.display import display_latex
from sympy.physics.quantum import TensorProduct, Dagger, Operator, HermitianOperator
import matplotlib.pyplot as plt
import tinyarray as ta
# -

sympy.init_printing()


# +
class Zero(np.ndarray):
    """A class that skips itself in additions

    It is derived from a numpy array for its implementation of right addition
    and subtraction to take priority.
    """

    def __add__(self, other):
        return other

    __radd__ = __rsub__ = __add__

    def __sub__(self, other):
        return -other

    def __neg__(self):
        return self

    def __truediv__(self, other):
        return self

_zero = Zero(0)


class One(np.ndarray):
    """A class that skips itself in matrix multiplications

    It is derived from a numpy array for its implementation of right
    multiplication to take priority.
    """

    def __add__(self, other):
        raise NotImplementedError

    __radd__ = __rsub__ = __sub__ = __neg__ = __add__

    def __matmul__(self, other):
        return other

    __rmatmul__ = __matmul__

_one = One(1)


# -

# ### Computing $U_n$ and $V_n$

def generate_volume(wanted_orders):
    """
    Generate ordered array with tinyarrays in volume of wanted_orders.
    
    wanted_orders : list of tinyarrays containing the wanted order of each perturbation.
    
    Returns:
    List of sorted tinyarrays contained in the volume of required orders to compute the wanted orders.
    """
    wanted_orders = np.array(wanted_orders)
    N_o, N_p = wanted_orders.shape
    max_order = np.max(wanted_orders, axis=0)
    possible_orders = np.array(
        np.meshgrid(*(np.arange(order+1) for order in max_order))
    ).reshape(len(max_order), -1)
    indices = np.any(np.all(possible_orders.reshape(N_p, -1, 1)
                            <= wanted_orders.T.reshape(N_p, 1, -1), axis=0), axis=1)
    keep_arrays = possible_orders.T[indices]
    return (ta.array(i) for i in sorted(keep_arrays, key=sum) if any(i))


def product_by_order(order, *terms):
    """
    Compute sum of all product of terms of wanted order.

    wanted_orders : list of tinyarrays containing the wanted order of each perturbation.

    Returns:
    Sum of all contributing products.
    """
    contributing_products = []
    for combination in product(*(term.items() for term in terms)):
        if sum(key for key, _ in combination) == order:
            contributing_products.append(reduce(matmul, (value for _, value in combination)))
    return sum(contributing_products, start=_zero)


def compute_next_orders(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, divide_energies=None):
    """
    Computes transformation to diagonalized Hamiltonian with multivariate perturbation.

    H_0_AA : unperturbed Hamiltonian A block in eigenbasis and ordered by eigenenergy.
    H_0_BB : unperturbed Hamiltonian B block in eigenbasis and ordered by eigenenergy.
    H_p_AA : dictionary of perturbations A blocks in eigenbasis of H_0
    H_p_BB : dictionary of perturbations B blocks in eigenbasis of H_0
    H_p_AB : dictionary of perturbations AB blocks in eigenbasis of H_0
    wanted_orders : list of tinyarrays containing the wanted order of each perturbation.

    Returns:
    U_AAn : list of AA block matrices up to order wanted_order
    U_BBn : list of BB block matrices up to order wanted_order
    V_ABn : list of AB block matrices up to order wanted_order
    """
    H_p_BA = {key: Dagger(value) for key, value in H_p_AB.items()}
    H_p = {
        "AA": H_p_AA,
        "AB": H_p_AB,
        "BA": H_p_BA,
        "BB": H_p_BB,
    }

    if divide_energies is None:
        E_A = np.diag(H_0_AA)
        E_B = np.diag(H_0_BB)
        energy_denominators = 1/(E_A.reshape(-1, 1) - E_B)

        def divide_energies(Y):
            return Y * energy_denominators

    H_0 = {
        "AA": {ta.zeros([len(wanted_orders[0])]): H_0_AA},
        "AB": {},
        "BA": {},
        "BB": {ta.zeros([len(wanted_orders[0])]): H_0_BB},
    }

    U_AA = {ta.zeros([len(wanted_orders[0])]): _one}
    U_BB = {ta.zeros([len(wanted_orders[0])]): _one}
    V_AB = {}
    V_BA = {}
    exp_S = {
        "AA": U_AA,
        "AB": V_AB,
        "BA": V_BA,
        "BB": U_BB,
    }
    inner_indices = ["AA", "AB", "BA", "BB"]
    indices = [
        ["A" + first, first + second, second + "B"]
        for first, second in inner_indices
    ]
    needed_orders = generate_volume(wanted_orders)

    for order in needed_orders:
        Y = sum(
            (
                (-1 if a != "AB" else +1) * product_by_order(order, exp_S[a], H[b], exp_S[c])
                for a, b, c in indices
                for H in [H_0, H_p]
            ),
            start=_zero
        )
        if Y is not _zero:
            V_AB[order] = divide_energies(Y)
            V_BA[order] = -Dagger(V_AB[order])

        new_U_AA = (
            - product_by_order(order, U_AA, U_AA)
            + product_by_order(order, V_AB, V_BA)
        )/2
        if new_U_AA is not _zero:
            U_AA[order] = new_U_AA

        new_U_BB = (
            - product_by_order(order, U_BB, U_BB)
            + product_by_order(order, V_BA, V_AB)
        )/2
        if new_U_BB is not _zero:
            U_BB[order] = new_U_BB

    return U_AA, U_BB, V_AB


def compute_next_orders_old(H_0, H_p, wanted_order, N_A=None):
    """    
    H_0 : np Hamiltonian in eigenbasis and ordered by eigenenergy.
    H_p : np Hamiltonian in eigenbasis of H_0
    wanted_order : int order of perturbation
    
    Returns:
    U_AAn : list of AA block matrices up to order wanted_order
    U_BBn : list of BB block matrices up to order wanted_order
    V_ABn : list of AB block matrices up to order wanted_order
    """
    N = H_0.shape[0]
    if N_A is None:
        N_A = N // 2
    N_B = N - N_A

    H_0_AA = H_0[:N_A, :N_A]
    H_0_BB = H_0[N_A:, N_A:]
    H_p_AA = H_p[:N_A, :N_A]
    H_p_AB = H_p[:N_A, N_A:]
    H_p_BB = H_p[N_A:, N_A:]

    # Blocks of U and V
    # 0th order
    U_AAn = [np.eye(N_A, dtype=complex)]
    U_BBn = [np.eye(N_B, dtype=complex)]
    V_ABn = [np.zeros((N_A, N_B), dtype=complex)]
    if wanted_order == 0:
        return U_AAn, U_BBn, V_ABn
    
    #1st order
    E_A = np.diag(H_0)[:N_A]
    E_B = np.diag(H_0)[N_A:]
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
            Y_next -= (
                + U_AAn[n-i-1] @ H_p_AA @ V_ABn[i]
                + U_AAn[n-i-1] @ H_p_AB @ U_BBn[i]
                - V_ABn[n-i-1] @ H_p_AB.conj().T @ V_ABn[i]
                - V_ABn[n-i-1] @ H_p_BB @ U_BBn[i]
            )
        for i in range(1, n):
            Y_next -= U_AAn[n-i] @ H_0_AA @ V_ABn[i] - V_ABn[n-i] @ H_0_BB @ U_BBn[i]
            U_AA_next -= (U_AAn[n-i] @ U_AAn[i] + V_ABn[n-i] @ V_ABn[i].conj().T) / 2
            U_BB_next -= (U_BBn[n-i] @ U_BBn[i] + V_ABn[n-i].conj().T @ V_ABn[i]) / 2

        if any(not np.all(np.isfinite(mat)) for mat in (U_AA_next, U_BB_next, Y_next)):
            raise RuntimeError(f"Instability encountered in {n}th order.")
        U_AAn.append(U_AA_next)
        U_BBn.append(U_BB_next)
        V_ABn.append(Y_next * energy_denominators)

    return U_AAn, U_BBn, V_ABn


# + product_by_order(order, V_AB, V_BA)
def H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, U_AA, U_BB, V_AB):
    """
    Computes block-diagonal form of Hamiltonian with multivariate perturbation.
    
    H_0_AA : unperturbed Hamiltonian A block in eigenbasis and ordered by eigenenergy.
    H_0_BB : unperturbed Hamiltonian B block in eigenbasis and ordered by eigenenergy.
    H_p_AA : dictionary of perturbations A blocks in eigenbasis of H_0
    H_p_BB : dictionary of perturbations B blocks in eigenbasis of H_0
    H_p_AB : dictionary of perturbations AB blocks in eigenbasis of H_0
    wanted_orders : list of tinyarrays containing the wanted order of each perturbation.
    U_AAn : list of AA block matrices up to order wanted_order
    U_BBn : list of BB block matrices up to order wanted_order
    V_ABn : list of AB block matrices up to order wanted_order
    
    Returns:
    H_AA : dictionary of orders of transformed perturbed Hamiltonian A block
    H_BB : dictionary of orders of transformed perturbed Hamiltonian A block
    """
    H_0 = {
    "AA": {ta.zeros([len(wanted_orders[0])]): H_0_AA},
    "AB": {},
    "BA": {},
    "BB": {ta.zeros([len(wanted_orders[0])]): H_0_BB},
    }
    H_p_BA = {key: Dagger(value) for key, value in H_p_AB.items()}
    H_p = {
        "AA": H_p_AA,
        "AB": H_p_AB,
        "BA": H_p_BA,
        "BB": H_p_BB,
    }

    H_AA = {}
    H_BB = {}
    H_AB = {}
    V_BA = {}

    exp_S = {
        "AA": U_AA,
        "AB": V_AB,
        "BA": V_BA,
        "BB": U_BB,
    }
    inner_indices = ["AA", "AB", "BA", "BB"]
    indices_AA = [
        ["A" + first, first + second, second + "A"]
        for first, second in inner_indices
    ]
    indices_BB = [
        ["B" + first, first + second, second + "B"]
        for first, second in inner_indices
    ]
    needed_orders = generate_volume(wanted_orders)

    for order in needed_orders:
        V_BA[order] = - Dagger(V_AB[order])
        new_H_AA = sum(
            (
                (-1 if a == "AB" else +1) * product_by_order(order, exp_S[a], H[b], exp_S[c])
                for a, b, c in indices_AA
                for H in [H_0, H_p]
            ),
            start=_zero
        )
        new_H_BB = sum(
            (
                (-1 if a == "BA" else +1) * product_by_order(order, exp_S[a], H[b], exp_S[c])
                for a, b, c in indices_BB
                for H in [H_0, H_p]
            ),
            start=_zero
        )
            
        if new_H_AA is not _zero:
            H_AA[order] = new_H_AA
        if new_H_BB is not _zero:
            H_BB[order] = new_H_BB
    return H_AA, H_BB


# + product_by_order(order, V_AB, V_BA)
def H_tilde_old(H_0, H_p, wanted_order, U_n, V_n):
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


# -

# ### Testing

# #### An initial attempt to get a symbolic expression

# +
N_A = 2
N_B = 2
N = N_A + N_B
H_0 = np.diag(np.sort(np.random.randn(N)))

N_p = 1
wanted_orders = [ta.array(np.random.randint(0, 3, size=N_p)) for i in range(2)]
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
# -

wanted_orders = [ta.array([5])]
wanted_order = 5

U_AA, U_BB, V_AB = compute_next_orders(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders=wanted_orders)
U_AA_old, U_BB_old, V_AB_old = compute_next_orders_old(H_0, H_p, wanted_order=wanted_order)

U_n = [np.block([[U_AA, np.zeros((N_A, N_B))], [np.zeros((N_B, N_A)), U_BB]]) for U_AA, U_BB in zip(U_AA_old, U_BB_old)]
V_n = [np.block([[np.zeros((N_A, N_A)), V_AB], [-Dagger(V_AB), np.zeros((N_B, N_B))]]) for V_AB in V_AB_old]


H_tilde_n_old = H_tilde_old(H_0, H_p, wanted_order, U_n, V_n)
H_tilde_n_old

H_tilde_n = H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, wanted_orders, U_AA, U_BB, V_AB)
H_tilde_n[0]

# ### Symbolic terms

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
H_2_BA = H_2_AB.T.conjugate()

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
V_n = [BlockMatrix([[ZeroMatrix(N, N), V_AB], [-V_AB.T.conjugate(), ZeroMatrix(M, M)]])
       for V_AB in V_ABn
]

zero = BlockMatrix([
    [ZeroMatrix(N, N), ZeroMatrix(N, M)],
    [ZeroMatrix(M, N), ZeroMatrix(M, M)]
])
