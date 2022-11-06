import numpy as np
import scipy
import qsymm
import sympy
import matplotlib.pyplot as plt
import polynomial_orders_U as polyLow
import higher_order_lowdin as oldLow

# +
N = 6
order = 9
A = np.random.random((N,N)) + 1j*np.random.random((N,N))

H_0 = np.diag(np.sort(np.random.random((N))))
H_p = A+A.conj().T

N_A = 2
N_B = H_0.shape[0]-N_A

eigs, vecs = scipy.linalg.eigh(H_0)
"""
Here, I determine all eigenvectors (although they are trivial), just to be able to supply them all
to the higher_order_lowdin.effective_model() method as to not allow it to use KPM.
"""

display(np.linalg.norm(H_p))
display(np.linalg.cond(H_p))
display(sympy.Matrix(H_p))
# -

H_0_AA, H_p_AA, A_vecs = H_0[:N_A,:N_A], H_p[:N_A,:N_A], vecs[:,:N_A]
H_0_BB, H_p_BB, B_vecs = H_0[N_A:,N_A:], H_p[:N_A,:N_A], vecs[:,N_A:]
H_p_AB = H_p[:N_A,N_A:]

U_AA, U_BB, V_AB = polyLow.compute_next_orders_old(H_0, H_p, wanted_order=order, N_A=N_A)
U_n = [np.block([[U_AA, np.zeros((N_A, N_B))], [np.zeros((N_B, N_A)), U_BB]]) for U_AA, U_BB in zip(U_AA, U_BB)]
V_n = [np.block([[np.zeros((N_A, N_A)), V_AB], [-V_AB.conj().T, np.zeros((N_B, N_B))]]) for V_AB in V_AB]

# Get H_tilde using the new poly Lowdin method
H_tilde_new = []
for i in range(order):
    H_tilde_new.append(polyLow.H_tilde(H_0,H_p,i,U_n,V_n)[-1][:N_A,:N_A])
display(H_tilde_new[1])
display(scipy.linalg.eigh(H_tilde_new[1])[0])

# Get H_tilde using the old higher_order_lowdin code
H_tilde_old = []
for i in range(order):
    H_tilde_old.append( oldLow.effective_model(H_0, {1:H_p}, A_vecs, B_vecs,order= i)['1'] )
display(H_tilde_old[1])
display(scipy.linalg.eigh(H_tilde_old[1])[0])

Diffs = [H_tilde_new[i]-H_tilde_old[i] for i in range(order)]
display(Diffs)
[np.linalg.norm(d) for d in Diffs]

display([np.linalg.norm(H) for H in H_tilde_new])
display([np.linalg.norm(H) for H in H_tilde_old])

# Is there a trfo in between?
Sylv = []
for i in range(order):
    Sylv.append(scipy.linalg.solve_sylvester(-H_tilde_new[i],H_tilde_old[i],np.zeros((N_A,N_A))))
Sylv

# So are they the same? $\rightarrow$ Sure does not look like it.

