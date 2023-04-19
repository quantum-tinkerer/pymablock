import numpy as np
import scipy
import sympy
import tinyarray as ta
import pdb
from lowdin import linalg, block_diagonalization
from lowdin.tests import test_block_diagonalization
from lowdin.linalg import complement_projected, ComplementProjector
from lowdin.series import BlockSeries, zero


def generate_kpm_hamiltonian(n_dim, n_infinite, a_dim):
    a_indices = slice(None, a_dim)
    b_indices = slice(a_dim, None)

    h_0 = np.random.randn(n_dim, n_dim) + 1j * np.random.randn(n_dim, n_dim)
    h_0 += h_0.conjugate().transpose()

    eigs, vecs = scipy.linalg.eigh(h_0)
    eigs[a_indices] -= 10.0  # introduce an energy gap
    h_0 = vecs @ np.diag(eigs) @ vecs.conjugate().transpose()
    eigs_a, vecs_a = eigs[a_indices], vecs[:, a_indices]
    eigs_b, vecs_b = eigs[b_indices], vecs[:, b_indices]

    H_input = BlockSeries(data={(0, 0, *((0,) * n_infinite)): h_0})

    for i in range(n_infinite):
        h_p = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
        h_p += h_p.conjugate().transpose()
        index = np.zeros(n_infinite, int)
        index[i] = 1
        H_input.data[(0, 0, *tuple(index))] = h_p

    H_input.shape = (1, 1)
    H_input.n_infinite = n_infinite

    return H_input, eigs_a, vecs_a, eigs_b, vecs_b


# +
wanted_orders = [(3,)]
n_infinite = len(wanted_orders[0])
n_dim = 36
a_dim = 5
b_dim = n_dim - a_dim

H_input, eigs_a, vecs_a, eigs_b, vecs_b = generate_kpm_hamiltonian(
    n_dim, n_infinite, a_dim
)

H_tilde_expl = block_diagonalization.numerical(H_input, vecs_a, eigs_a, vecs_b, eigs_b)[0]
H_tilde_kpm = block_diagonalization.numerical(H_input, vecs_a, eigs_a)[0]
# -

for i in range(3):
    print(H_tilde_expl.evaluated[1,1,i])

for i in range(3):
    print(H_tilde_kpm.evaluated[1,1,i])

# +
# This I wanted to keep for a proper test later


def test_BB_does_what_BB_do(wanted_orders: list[tuple[int, ...]]) -> None:
    """
    Test that the BB block of H_tilde is a) a LinearOperator type and
    b) the same as the AA block on exchanging veca_a and vecs_b
    
    Parameters:
    ----------
    wanted_orders:
        list of orders to compute
    """
    n_infinite = len(wanted_orders[0])
    n_dim = 36
    a_dim = 5
    b_dim = n_dim - a_dim

    H_input, eigs_a, vecs_a, eigs_b, vecs_b = generate_kpm_hamiltonian(
        n_dim, n_infinite, a_dim
    )
    
    H_tilde_AA = block_diagonalization.numerical(H_input, vecs_a, eigs_a, vecs_b, eigs_b)[0]
    H_tilde_BB = block_diagonalization.numerical(H_input, vecs_b, eigs_b, vecs_a, eigs_a)[0]
    
    for order in wanted_orders:
        order = tuple(slice(None, dim_order + 1) for dim_order in order)
        h_aa = H_tilde_AA.evaluated[(0, 0) + order]
        h_bb = H_tilde_BB.evaluated[(1, 1) + order]
        
        assert isinstance(h_bb, scipy.sparse.linalg.LinearOperator)
        
        eigs_aa = scipy.linalg.eigh(h_aa)[0]
        eigs_bb = scipy.linalg.eigh(h_bb @ np.eye(h_bb.shape[0]))[0]
        
        is_contained = [np.any(np.isclose(e, eigs_bb)) for e in eigs_aa]
        assert np.all(is_contained)
        
# -


