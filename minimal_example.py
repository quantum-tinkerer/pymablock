import numpy as np
import scipy
import sympy
import tinyarray as ta
from lowdin import linalg, block_diagonalization
from lowdin.tests import test_block_diagonalization
from lowdin.linalg import complement_projected, ComplementProjector

# +
n = 1
dim = 25

h_0 = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
h_0 += h_0.conj().transpose()

#h_0 = np.diag(np.sort(np.random.random(dim)))

eigs, vecs = scipy.linalg.eigh(h_0)

eigs_a = eigs[:2]
vecs_a = vecs[:,:2]

eigs_b = eigs[2:]
vecs_b = vecs[:,2:]

h = {ta.array([0 for _ in range(n)]): h_0}

for i in range(n):
    h_p = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
    h_p += h_p.conj().transpose()
    index = np.zeros(n,int)
    index[i] = 1
    h[ta.array(index)] = h_p
# -

h_tilde, u, u_adj = block_diagonalization.numerical(h, vecs_a, eigs_a, kpm_params={'num_moments':1000})

h_tilde.evaluated[0,0,4]

h_tilde.evaluated[1,1,3]


