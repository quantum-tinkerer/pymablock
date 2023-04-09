import numpy as np
import scipy
import sympy
from lowdin import poly_kpm, linalg
from lowdin.tests import test_poly_kpm
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

h = {1: h_0}

for i in range(n):
    h_p = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
    h_p += h_p.conj().transpose()
    h[sympy.symbols('p_{}'.format(i))] = h_p
    
# -


# # Check inputs and bookkeeping

# ### Inputs

# +
h_p = {k:v for k,v in h.items() if k!=1}

det_inp = scipy.linalg.det(sum(h.values()))
trace_inp = np.trace(sum(h.values()))
trace2_inp = sum(scipy.linalg.eigh(sum(h.values()))[0])
assert np.isclose(trace_inp, trace2_inp)

det_h0_inp = scipy.linalg.det(h[1])
det_hp_inp = scipy.linalg.det(sum(h_p.values()))

trace_h0_inp = np.trace(h[1])
trace_hp_inp = np.trace(sum(h_p.values()))
trace2_hp_inp = sum(scipy.linalg.eigh(sum(h_p.values()))[0])

assert np.isclose(trace_hp_inp, trace2_hp_inp)
# -

# ### Impose $B \rightarrow A+B$ bookkeeping and check consistency
#
# $B\rightarrow A+B$ bookkeeping uses that $\mathcal{H_0}=\mathcal{D}[E]+\mathcal{H}_0^\prime$ where $\mathcal{D}[E]$ is a small diagonal matrix in the $AA$ space while $\mathcal{H}_0^\prime=\mathcal{P}_B \mathcal{H}_0 \mathcal{P}_B$ is a dense matrix in $A+B$. We therefore split up $\mathcal{H}_0$ into blocks reading
# \begin{align}
# \mathcal{H}_0=\begin{pmatrix}\mathcal{D}[E]&v_A^\dagger \mathcal{H}_0 \mathcal{P}_B \\ (v_A^\dagger \mathcal{H}_0 \mathcal{P}_B)^\dagger &\mathcal{P}_B\mathcal{H}_0\mathcal{P}_B\end{pmatrix}
# \end{align}
# The same goes for $\mathcal{H}_p$ where
# \begin{align}
# H_p = \begin{pmatrix}v_A^\dagger \mathcal{H}_p v_A & v_A^\dagger \mathcal{H}_p \mathcal{P}_B \\ (v_A^\dagger \mathcal{H}_p \mathcal{P}_B)^\dagger & \mathcal{P}_B \mathcal{H}_P \mathcal{P}_B \end{pmatrix}
# \end{align}
# In both cases, we have used $v_A\in \mathbb{C}^{A+B \times A}$ begin the eigenvectors of the $A$ subspace.

# +
h_0_aa = np.diag(eigs_a)
h_0_bb = complement_projected(h[1], vecs_a) @ np.eye(vecs_a.shape[0])
h_0_ab = vecs_a.conj().T @ h[1] @ ComplementProjector(vecs_a)

h_p_aa = {k: vecs_a.conj().T @ v @ vecs_a for k,v in h_p.items()}
h_p_ab = {k: vecs_a.conj().T @ v @ ComplementProjector(vecs_a) for k,v in h_p.items()}
h_p_bb = {k: complement_projected(v, vecs_a)@np.eye(vecs_a.shape[0]) for k,v in h_p.items()}
# -

# $\mathcal{H}_0^{AB}=v_A^\dagger \mathcal{H}_0 \mathcal{P}_B\equiv 0$ since
# \begin{align}
# \mathcal{H}_0^{AB}\cdot \vec{v} &= \mathcal{H}_0^{AB} \left(\sum_{A,B}c_A v_A + c_B v_B\right) \\
# &=v_A^\dagger [\mathcal{D}[E]+\mathcal{P}_B\mathcal{H}_0\mathcal{P}_B]\left(Id-v_A v_A^\dagger\right)  \left(\sum_{A,B}c_A v_A + c_B v_B\right) \\
# &=\mathcal{D}[E]\sum_{B}c_B v_B \\
# &\equiv 0
# \end{align}
# where we used that $v_A^\dagger \mathcal{P}_B = 0$ and $v_A^\dagger \cdot v_B =0$.

assert np.allclose(h_0_ab, 0)

# Depite the bookkeeping we should retain
# \begin{align}
# Tr[\mathcal{H}] &= Tr[\mathcal{H}_{book}]
# \end{align}
# while 
# \begin{align}
# \det[\mathcal{H}_{book}]\equiv 0
# \end{align}
# which is because the bookkeping sends the eigenvalues of the $A$ space to $0$ upon $\mathcal{P}_B$

# +
h_book = scipy.linalg.block_diag(h_0_aa,h_0_bb)+np.block([[sum(h_p_aa.values()),sum(h_p_ab.values())],
                                                          [sum(h_p_ab.values()).conjugate().transpose(),sum(h_p_bb.values())]])

det_book = scipy.linalg.det(h_book)
trace_book = np.trace(h_book)

det_h0_book = scipy.linalg.det(scipy.linalg.block_diag(h_0_aa,h_0_bb))
trace_h0_book = np.trace(scipy.linalg.block_diag(h_0_aa,h_0_bb))
trace2_h0_book = sum(scipy.linalg.eigh(scipy.linalg.block_diag(h_0_aa,h_0_bb))[0])

assert np.isclose(trace_h0_book, trace2_h0_book)

h_p_book = np.block([[sum(h_p_aa.values()),sum(h_p_ab.values())],
                     [sum(h_p_ab.values()).conjugate().transpose(),sum(h_p_bb.values())]])

det_hp_book = scipy.linalg.det(h_p_book)
trace_hp_book = np.trace(h_p_book)
trace2_hp_book = sum(scipy.linalg.eigh(h_p_book)[0])

assert np.isclose(trace_hp_book, trace2_hp_book)

# +
assert np.isclose(trace_inp, trace_book)
assert np.isclose(trace_h0_inp, trace_h0_book)
assert np.isclose(trace_hp_inp, trace_hp_book)

assert np.isclose(det_book, 0)
assert np.isclose(det_h0_book, 0)
assert np.isclose(det_hp_book, 0)
# -

# But the eigenvalues should be retained

np.round(scipy.linalg.eigh(h[1])[0],3)

np.round(scipy.linalg.eigh(scipy.linalg.block_diag(h_0_aa,h_0_bb))[0],3)

np.round(scipy.linalg.eigh(sum(h_p.values()))[0],3)

np.round(scipy.linalg.eigh(h_p_book)[0],3)

# Hence we see that the inputs and transformed inputs are compatible with each other!

# # KPM and Green's function application
#
# we want to perform a division by energies from the right on objects $Y\in \mathbb{C}^{A\times A+B}$. In case all eigenvalues are known this constitutes an elementwise operation
# \begin{align}
# Y_{m,l} \rightarrow Y_{m,l} \frac{1}{E_{A,m}-E_{B,l}}
# \end{align}
# In the KPM case this changes not only because of the application of the interpolated Green's function, but also because of the possible presence of explicit parts of the spectrum. Both need to be treated separaterly
#
# ### KPM pre-processing:
# We first need to separate the inputs of the energy division into the explicit $\vec{v}_B$-part and the implicit GF part of the spectrum.
# \begin{align}
# Y_{KPM} &= Y \cdot \mathcal{P}_B\cdot \left(Id - \mathcal{P}_{B,exp}\right) \\
# Y_{Dir} &= Y \cdot \mathcal{P}_B \cdot \vec{v}_{B,exp}
# \end{align}
#
# ### Explicit part:
# proceeding with $Y_{Dir}$, since all eigenvalues that are relevant for the object are known, we can directly multiply by the respective energy differences as
# \begin{align}
# Y_{Dir; m,l} \rightarrow Y_{Dir; m,l} \frac{1}{E_{A; m} - E_{B,exp; l}} \vec{v}_{B,exp}^\dagger
# \end{align}
# such that all-together
# \begin{align}
# Y_{exp} = Y \mathcal{P}_b \vec{v}_{B,exp} \frac{1}{E_A - E_{B,exp}} \vec{v}_{B,exp}^\dagger
# \end{align}
#
# Note for the code: When using `SumOfOperatorProducts`, the resulting elements need sub-space flags. Overall, $Y$ is an object in $A\times B$, while $\vec{B,exp}$ is an object in $B\times B_exp$. Hence, to avoid indexing conflicts (and not introduce an additional dummy flag into `SumOfOperatorProducts`) we will utilize the convention that $Y\cdot\mathcal{P}_B \cdot v_{B,exp}$ is an object of $A\times A$. This not only distinguishes this object from the original $B \times B$, but also enforces its evaluation since $A\times A$ objects will be evaluated by `SumOfOperatorProducts`.
#
# The note is actually not true

h_tilde, u, u_adj = poly_kpm.numerical(h, vecs_a, eigs_a, vecs_b, eigs_b)

h_tilde.evaluated[0,1,3].to_array()

a = np.array([1,2,3,4,5,56,6,7])
b = np.array([3,5,6])

np.array([np.where(a == c)[0] for c in b]).flatten()



# ### GF part:
# Applying the Green's function $G_0(E_A)$ to $Y_{KPM}$ is an operation where the GF is applied from the right $\rightarrow Y_{KPM}\cdot G_0(E_A)$. To perform this operation we make use of 
# \begin{align}
# Y_{KPM}G_0(E_A)&=\bigg[\left(Y_{KPM} G_0(E_A)\right)^\dagger \bigg]^\dagger \\
# &=\bigg[G_0(E_A)Y_{KPM}^\dagger \bigg]^\dagger
# \end{align}
# where we have used that $G_0(E_A)$ is hermitian.









test_poly_kpm.test_create_div_energs_kpm([h_0[:2,:2],h_0[2:,2:]])

H_t, U, U_adj = poly_kpm.numerical(h, vecs_a, eigs_a, vecs_b, eigs_b)

H_t.evaluated[0,1,4].to_array()

test_poly_kpm.test_ab_is_zero()

t = np.random.random((4,4))
test = poly_kpm.SumOfOperatorProducts([[(t,'AB')]])


