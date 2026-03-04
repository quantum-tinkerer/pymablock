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

# Adiabatic elimination in a Lindblad problem

In this tutorial we use the non-Hermitian block-diagonalization algorithm on a Lindblad master equation.
The goal is to eliminate fast optical coherences of a three-level $\Lambda$-system and obtain an effective reduced generator.

We write the Liouvillian as
\begin{equation}
\mathcal{L} = \mathcal{L}_0 + g\,\mathcal{L}_1,
\end{equation}
where $g$ is the perturbative coupling strength.

The physical regime is a standard adiabatic-elimination limit:

- The excited level $|e\rangle$ is far detuned by $\Delta$.
- Excited-state phase coherence decays quickly with rate $\Gamma_\phi$.
- Coherent couplings (scaled by $g$) are weak compared to these fast scales.

So the fast optical coherences $|e\rangle\langle 0|$, $|e\rangle\langle 1|$ and their
Hermitian conjugates relax quickly, while the remaining sector evolves slowly.
At second order, eliminating the fast sector produces effective slow-sector
physics: Raman-like coupling between $|0\rangle$ and $|1\rangle$, plus dissipative
corrections.

## 1. Build the Liouvillian

We use the Lindblad equation
\begin{equation}
\dot{\rho}
= -i[H,\rho]
+ \sum_k \mathcal{D}[J_k]\rho,
\qquad
\mathcal{D}[J]\rho
= J\rho J^\dagger - \frac{1}{2}\{J^\dagger J,\rho\}.
\end{equation}

The Hamiltonian is split as
\begin{equation}
H = H_0 + g H_1,
\end{equation}
with
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

For dissipation we use one jump operator in $\mathcal{L}_0$:
\begin{equation}
J_\phi = \sqrt{\Gamma_\phi}\,|e\rangle\langle e|.
\end{equation}

```{code-cell} ipython3
import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from pymablock import block_diagonalize
from pymablock.series import zero
```

```{code-cell} ipython3
def liouvillian_coherent(H):
    n = H.shape[0]
    eye = np.eye(n, dtype=complex)
    return -1j * (np.kron(eye, H) - np.kron(H.T, eye))


def liouvillian_dissipator(c):
    n = c.shape[0]
    eye = np.eye(n, dtype=complex)
    cdgc = c.conj().T @ c
    return np.kron(c.conj(), c) - 0.5 * (
        np.kron(eye, cdgc) + np.kron(cdgc.T, eye)
    )


def liouvillian_from_hamiltonian_and_jumps(H, jumps):
    return liouvillian_coherent(H) + sum(liouvillian_dissipator(j) for j in jumps)
```

```{code-cell} ipython3
# Basis states |0⟩, |1⟩, |e⟩.
n = 3
ket0 = np.array([[1], [0], [0]], dtype=complex)
ket1 = np.array([[0], [1], [0]], dtype=complex)
kete = np.array([[0], [0], [1]], dtype=complex)

P0 = ket0 @ ket0.conj().T
P1 = ket1 @ ket1.conj().T
Pe = kete @ kete.conj().T
```

```{code-cell} ipython3
# Unperturbed part: diagonal Hamiltonian + strong dephasing on |e⟩.
Delta = 8.0
delta = 0.5
Gamma_phi = 6.0
Omega_0 = 1.0
Omega_1 = 0.8

H0 = Delta * Pe + 0.5 * delta * (P1 - P0)
H1 = (
    Omega_0 * (kete @ ket0.conj().T + ket0 @ kete.conj().T)
    + Omega_1 * (kete @ ket1.conj().T + ket1 @ kete.conj().T)
)

jump_operators_0 = [np.sqrt(Gamma_phi) * Pe]
jump_operators_1 = []

L0 = liouvillian_from_hamiltonian_and_jumps(H0, jump_operators_0)
L1 = liouvillian_from_hamiltonian_and_jumps(H1, jump_operators_1)

np.linalg.norm((L0 + 0.1 * L1) - (L0 + 0.1 * L1).conj().T)
```

The nonzero norm confirms that the generator is not Hermitian.

## 2. Define slow and fast sectors

We keep the slow sector
\[
\{|0\rangle\langle0|, |1\rangle\langle0|, |0\rangle\langle1|,
|1\rangle\langle1|, |e\rangle\langle e|\},
\]
and eliminate the fast sector
\[
\{|e\rangle\langle0|, |e\rangle\langle1|, |0\rangle\langle e|,
|1\rangle\langle e|\}.
\]

We intentionally keep $|e\rangle\langle e|$ in the slow block.
This makes $\mathcal{L}_0$ exactly block-diagonal under the chosen split and avoids
accidental Sylvester singularities from shared unperturbed eigenvalues.

```{code-cell} ipython3
subspace_indices = np.zeros(n * n, dtype=int)
slow_indices = []

for j in range(n):
    for i in range(n):
        idx = i + n * j  # column-major vec index for |i⟩⟨j|
        is_fast = (i == 2) ^ (j == 2)
        subspace_indices[idx] = 1 if is_fast else 0
        if not is_fast:
            slow_indices.append(idx)

slow_dim = len(slow_indices)
fast_dim = np.sum(subspace_indices == 1)
slow_dim, fast_dim
```

```{code-cell} ipython3
# L0 must already be block diagonal for perturbation theory.
L0_sf = L0[np.ix_(subspace_indices == 0, subspace_indices == 1)]
L0_fs = L0[np.ix_(subspace_indices == 1, subspace_indices == 0)]
float(np.max(np.abs(L0_sf))), float(np.max(np.abs(L0_fs)))
```

```{code-cell} ipython3
L_tilde, U, U_inv = block_diagonalize(
    [L0, L1],
    subspace_indices=subspace_indices,
    hermitian=False,
)
```

```{code-cell} ipython3
# Off-diagonal blocks are eliminated order by order.
L_tilde[0, 1, 1] is zero
```

## 3. Compare spectrum with the full Liouvillian

```{code-cell} ipython3
def series_block(series, order):
    block = series[0, 0, order]
    if block is zero:
        return np.zeros_like(series[0, 0, 0])
    return block


def effective_generator(g, max_order=2):
    out = np.zeros_like(L_tilde[0, 0, 0])
    for order in range(max_order + 1):
        out = out + (g ** order) * series_block(L_tilde, order)
    return out


def low_lying_eigs(M, n_keep):
    eigs = np.linalg.eigvals(M)
    order = np.argsort(np.real(eigs))[::-1]
    return eigs[order[:n_keep]]


def match_distance(a, b):
    # Small helper for set-to-set spectral distance.
    best = np.inf
    for perm in itertools.permutations(range(len(a))):
        best = min(best, np.max(np.abs(a - b[list(perm)])))
    return best
```

```{code-cell} ipython3
g_values = np.logspace(-2, -0.7, 8)
errors = []

for g in g_values:
    full = low_lying_eigs(L0 + g * L1, n_keep=slow_dim)
    eff = np.linalg.eigvals(effective_generator(g, max_order=2))
    errors.append(match_distance(full, eff))

errors = np.array(errors)
errors
```

```{code-cell} ipython3
guide = errors[0] * (g_values / g_values[0]) ** 3

plt.figure(figsize=(5.2, 3.6))
plt.loglog(g_values, errors, "o-", label="spectral mismatch")
plt.loglog(g_values, guide, "--", label=r"$\propto g^3$ guide")
plt.xlabel(r"coupling $g$")
plt.ylabel("max eigenvalue mismatch")
plt.legend()
plt.tight_layout()
```

The cubic scaling is what we expect from truncating the effective generator at second order.
In other words, the missing terms start at order $g^3$.

## 4. Compare reduced dynamics

```{code-cell} ipython3
projection = np.zeros((slow_dim, n * n), dtype=complex)
for row, idx in enumerate(slow_indices):
    projection[row, idx] = 1

rho0 = (ket0 @ ket0.conj().T).reshape(-1, order="F")
rho0_slow = projection @ rho0

g = 0.12
L_full = L0 + g * L1
L_eff = effective_generator(g, max_order=2)
times = np.linspace(0, 10, 120)

full_slow = np.array([projection @ (scipy.linalg.expm(L_full * t) @ rho0) for t in times])
eff_slow = np.array([scipy.linalg.expm(L_eff * t) @ rho0_slow for t in times])

# In the chosen slow basis: index 3 is ρ₁₁ and index 4 is ρₑₑ.
rho11_full = np.real(full_slow[:, 3])
rho11_eff = np.real(eff_slow[:, 3])
rhoee_full = np.real(full_slow[:, 4])
rhoee_eff = np.real(eff_slow[:, 4])
```

```{code-cell} ipython3
fig, axes = plt.subplots(ncols=2, figsize=(8, 3.2), sharex=True)

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

This gives a compact Lindblad example where `hermitian=False` is essential.
The non-Hermitian algorithm provides an effective reduced generator that reproduces
the slow-sector spectrum and dynamics up to the expected perturbative error.
