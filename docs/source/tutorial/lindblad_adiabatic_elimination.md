---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Adiabatic elimination for a Lindblad equation

This tutorial shows a complete non-Hermitian workflow with {autolink}`~pymablock.block_diagonalize`.
We start from a Lindblad master equation.
We separate fast and slow sectors in Liouville space.
We build a second-order effective Liouvillian for the slow sector.

The example is a three-level $\Lambda$ system with states $|0\rangle$, $|1\rangle$, and $|e\rangle$.

## 1. Physics in one page

We use
\begin{equation}
\dot\rho = -i[H,\rho] + \sum_k \mathcal{D}[J_k]\rho,
\qquad
\mathcal{D}[J]\rho = J\rho J^\dagger - \frac{1}{2}\{J^\dagger J,\rho\}.
\end{equation}

The Hamiltonian is split as
\begin{equation}
H = H_0 + g H_1.
\end{equation}

We choose
\begin{equation}
H_0
= \Delta |e\rangle\langle e|
+ \frac{\delta}{2}\left(|1\rangle\langle1|-|0\rangle\langle0|\right),
\end{equation}
\begin{equation}
H_1
= \Omega_0\left(|e\rangle\langle0|+|0\rangle\langle e|\right)
+ \Omega_1\left(|e\rangle\langle1|+|1\rangle\langle e|\right).
\end{equation}

The jump operator is pure dephasing of $|e\rangle$:
\begin{equation}
J_\phi = \sqrt{\Gamma_\phi}\,|e\rangle\langle e|.
\end{equation}

Physical regime:

1. $\Delta$ and $\Gamma_\phi$ are large (fast optical coherences).
2. $g$ is small (weak coupling).
3. We eliminate fast coherences perturbatively and keep slow dynamics.

## 2. Build the model in code

Before writing code, we fix the vectorization convention.
We stack matrix rows into a vector:
\begin{equation}
\mathrm{vec}(\rho) = (\rho_{00}, \rho_{01}, \dots, \rho_{0N}, \rho_{10}, \dots)^{\mathsf T}.
\end{equation}
With this choice,
\begin{equation}
A \rho B \;\longmapsto\; \left(A \otimes B^{\mathsf T}\right)\mathrm{vec}(\rho).
\end{equation}
So:

1. $H\rho \mapsto H \otimes I$
2. $\rho H \mapsto I \otimes H^{\mathsf T}$
3. $J\rho J^\dagger \mapsto J \otimes (J^\dagger)^{\mathsf T} = J \otimes J^*$

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from pymablock import block_diagonalize
from pymablock.series import zero
```

```{code-cell} ipython3
def ket(n, i):
    out = np.zeros((n, 1), dtype=complex)
    out[i, 0] = 1
    return out


def projector(n, i):
    v = ket(n, i)
    return v @ v.conj().T


def coherent_superoperator(H):
    """-i[H, rho] in vec(rho), with A rho B -> kron(A, B^T)."""
    n = H.shape[0]
    I = np.eye(n, dtype=complex)
    return -1j * (np.kron(H, I) - np.kron(I, H.T))


def dissipator_superoperator(J):
    """D[J] in vec(rho) representation."""
    n = J.shape[0]
    I = np.eye(n, dtype=complex)
    JdagJ = J.conj().T @ J
    return np.kron(J, J.conj()) - 0.5 * (
        np.kron(JdagJ, I) + np.kron(I, JdagJ.T)
    )


def liouvillian(H, jumps):
    L = coherent_superoperator(H)
    for J in jumps:
        L = L + dissipator_superoperator(J)
    return L
```

```{code-cell} ipython3
# Basis: |0⟩, |1⟩, |e⟩.
n = 3
P0 = projector(n, 0)
P1 = projector(n, 1)
Pe = projector(n, 2)

# Parameters.
Delta = 8.0
delta = 0.5
Gamma_phi = 6.0
Omega_0 = 1.0
Omega_1 = 0.8

# Hamiltonian split H = H0 + g H1.
H0 = Delta * Pe + 0.5 * delta * (P1 - P0)
H1 = (
    Omega_0 * (ket(n, 2) @ ket(n, 0).conj().T + ket(n, 0) @ ket(n, 2).conj().T)
    + Omega_1 * (ket(n, 2) @ ket(n, 1).conj().T + ket(n, 1) @ ket(n, 2).conj().T)
)

# Jump operators split the same way: J = J0 + g J1.
jumps_0 = [np.sqrt(Gamma_phi) * Pe]
jumps_1 = []

L0 = liouvillian(H0, jumps_0)
L1 = liouvillian(H1, jumps_1)
```

```{code-cell} ipython3
evals_L0 = np.linalg.eigvals(L0)

plt.figure(figsize=(4.8, 3.6))
plt.scatter(np.real(evals_L0), np.imag(evals_L0), s=35)
plt.axvline(0.0, color="0.75", lw=1)
plt.xlabel(r"$\mathrm{Re}\,\lambda$")
plt.ylabel(r"$\mathrm{Im}\,\lambda$")
plt.title(r"Spectrum of $\mathcal{L}_0$")
plt.tight_layout()
```

The spectrum is complex, as expected for a Lindbladian generator.

## 3. Choose slow and fast sectors

In Liouville space, basis vectors correspond to operators $|i\rangle\langle j|$.
We mark as fast all optical coherences that involve exactly one $|e\rangle$:
\[
|e\rangle\langle0|,\ |e\rangle\langle1|,\ |0\rangle\langle e|,\ |1\rangle\langle e|.
\]
Everything else is slow.

```{code-cell} ipython3
def vec_index(i, j, n):
    # Row-major index of |i⟩⟨j| in vec(rho).
    return j + n * i


subspace_indices = np.zeros(n * n, dtype=int)  # 0: slow, 1: fast
slow_vec_indices = []

for j in range(n):
    for i in range(n):
        k = vec_index(i, j, n)
        is_fast = (i == 2) ^ (j == 2)
        subspace_indices[k] = int(is_fast)
        if not is_fast:
            slow_vec_indices.append(k)

slow_dim = int(np.sum(subspace_indices == 0))
fast_dim = int(np.sum(subspace_indices == 1))
slow_dim, fast_dim
```

```{code-cell} ipython3
# Requirement for perturbation theory: L0 has no slow-fast coupling.
L0_sf = L0[np.ix_(subspace_indices == 0, subspace_indices == 1)]
L0_fs = L0[np.ix_(subspace_indices == 1, subspace_indices == 0)]
float(np.max(np.abs(L0_sf))), float(np.max(np.abs(L0_fs)))
```

```{code-cell} ipython3
slow_order = np.where(subspace_indices == 0)[0]
fast_order = np.where(subspace_indices == 1)[0]
perm = np.r_[slow_order, fast_order]

L0_perm = L0[np.ix_(perm, perm)]
L0_slow = L0_perm[:slow_dim, :slow_dim]
L0_fast = L0_perm[slow_dim:, slow_dim:]

evals_slow = np.linalg.eigvals(L0_slow)
evals_fast = np.linalg.eigvals(L0_fast)

plt.figure(figsize=(5.2, 3.6))
plt.scatter(np.real(evals_slow), np.imag(evals_slow), s=45, label="slow block (kept)")
plt.scatter(np.real(evals_fast), np.imag(evals_fast), s=45, label="fast block (eliminated)")
plt.axvline(0.0, color="0.75", lw=1)
plt.xlabel(r"$\mathrm{Re}\,\lambda$")
plt.ylabel(r"$\mathrm{Im}\,\lambda$")
plt.title(r"Block-resolved spectrum of $\mathcal{L}_0$")
plt.legend()
plt.tight_layout()
```

Interpretation:

1. We keep the eigenvalues of the slow block (`evals_slow`): they govern the reduced dynamics.
2. We eliminate the eigenvalues of the fast block (`evals_fast`): they correspond to optical coherences involving exactly one $|e\rangle$.
3. In this model, these fast modes have large detuning/dephasing scales, which is why perturbative elimination is accurate.

## 4. Apply Pymablock

This is the core call.
Because the Liouvillian is non-Hermitian, we set `hermitian=False`.

```{code-cell} ipython3
L_tilde, U, U_inv = block_diagonalize(
    [L0, L1],
    subspace_indices=subspace_indices,
    hermitian=False,
)
```

```{code-cell} ipython3
# First-order slow-fast block is eliminated.
L_tilde[0, 1, 1] is zero
```

## 5. Build the second-order effective Liouvillian

```{code-cell} ipython3
def slow_block_term(series, order):
    term = series[0, 0, order]
    if term is zero:
        return np.zeros_like(series[0, 0, 0])
    return term


def effective_liouvillian(g, max_order=2):
    Leff = np.zeros_like(L_tilde[0, 0, 0])
    for order in range(max_order + 1):
        Leff = Leff + (g**order) * slow_block_term(L_tilde, order)
    return Leff
```

## 6. Compare full vs effective dynamics

We compare dynamics projected to the slow sector.

```{code-cell} ipython3
# Projection from full vec(rho) to slow vec(rho).
P_slow = np.zeros((slow_dim, n * n), dtype=complex)
for row, k in enumerate(slow_vec_indices):
    P_slow[row, k] = 1.0

# Map (i, j) -> index inside the slow vector.
slow_pos = {}
for row, k in enumerate(slow_vec_indices):
    i = k // n
    j = k % n
    slow_pos[(i, j)] = row

rho0 = projector(n, 0)
rho0_vec = rho0.reshape(-1, order="C")
rho0_slow = P_slow @ rho0_vec

pop1_idx = slow_pos[(1, 1)]
pope_idx = slow_pos[(2, 2)]
```

```{code-cell} ipython3
g = 0.12
times = np.linspace(0, 10, 160)

L_full = L0 + g * L1
L_eff2 = effective_liouvillian(g, max_order=2)

full_slow = np.array(
    [P_slow @ (scipy.linalg.expm(L_full * t) @ rho0_vec) for t in times]
)
eff_slow = np.array([scipy.linalg.expm(L_eff2 * t) @ rho0_slow for t in times])

rho11_full = np.real(full_slow[:, pop1_idx])
rho11_eff = np.real(eff_slow[:, pop1_idx])
rhoee_full = np.real(full_slow[:, pope_idx])
rhoee_eff = np.real(eff_slow[:, pope_idx])
```

```{code-cell} ipython3
fig, axes = plt.subplots(ncols=2, figsize=(8.2, 3.2), sharex=True)

axes[0].plot(times, rho11_full, label="full")
axes[0].plot(times, rho11_eff, "--", label="effective")
axes[0].set_title(r"$\rho_{11}(t)$")
axes[0].set_xlabel("t")
axes[0].legend()

axes[1].plot(times, rhoee_full, label="full")
axes[1].plot(times, rhoee_eff, "--", label="effective")
axes[1].set_title(r"$\rho_{ee}(t)$")
axes[1].set_xlabel("t")

plt.tight_layout()
```

```{code-cell} ipython3
float(np.max(np.abs(full_slow - eff_slow)))
```

The mismatch is small in the perturbative regime.
This is exactly what we want from adiabatic elimination.

## 7. What to reuse in your own project

If your model is non-Hermitian (Liouvillian, non-normal effective generator, or gain/loss model), the same pattern applies:

1. Write it as $L_0 + \lambda L_1 + \lambda^2 L_2 + \dots$.
2. Choose subspaces where $L_0$ is block-diagonal.
3. Call `block_diagonalize(..., hermitian=False)`.
4. Read `L_tilde[0, 0, n]` for the slow-sector effective generator at order `n`.
