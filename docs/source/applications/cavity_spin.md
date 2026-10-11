---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 480
---

# Artificial higher spins in a driven cavity

Can the lowest Fock states of a microwave cavity behave as a spin larger than
one half? Roy et al. engineer such a spin with a driven ancilla qubit.[^roy]
The effective model keeps $2s+1$ cavity states, $n=0,\ldots,2s$, which play the
role of spin projections $m=n-s$. To first order in the cavity drive amplitude
$\epsilon$, and with the comb phases of Roy et al., the effective Hamiltonian
is a spin rotation,

$$
H^{(1)} = \frac{\epsilon}{2\sqrt{2s}}\left(e^{i\varphi}S_- + e^{-i\varphi}S_+\right).
$$

The drive phase $\varphi$ sets the rotation axis. We confirm this for $s=1$ and
$s=3/2$, and compute the second-order correction for $s=1$.

## Model

A cavity mode $c$ has dispersive shift $\chi$ to an ancilla qubit with states
$|g\rangle,|e\rangle$. A comb of drive tones $k=0,\ldots,2s$ acts on the
ancilla. Each tone has amplitude $\Omega$ and phase $\phi_k$, and tone $k$ is
resonant when the cavity holds $k$ photons. We work in the rotating-wave
approximation for the carrier frequencies and keep all comb tones. An integer
ladder $\ell$ (a Floquet index) records the detuning $(k-n)\chi$ of tone $k$ in
cavity sector $n$; it contributes $\chi N_\ell$ to the energy.

In sector $n\le 2s$, the resonant tone dresses the ancilla into states
$|\pm_n\rangle$ with energies $\pm\Omega/2$. We write the ancilla as a
two-by-two matrix in this dressed basis. Its columns in the $(g,e)$ basis form

$$
R_n = \frac{1}{\sqrt2}\begin{pmatrix}
e^{i\phi_n/2} & -e^{i\phi_n/2}\\ e^{-i\phi_n/2} & e^{-i\phi_n/2}
\end{pmatrix}.
$$

In sector $n=2s+1$ no tone is resonant, and $R_n$ is the identity.

```{code-cell} ipython3
import sympy as sp
from IPython.display import display
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import LadderOp
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.second_quantization import Embedding
```

```{code-cell} ipython3
chi, Omega, epsilon = sp.symbols("chi Omega epsilon", real=True, nonzero=True)
varphi = sp.Symbol("varphi", real=True)
c, ell = BosonOp("c"), LadderOp("ell")
sigma_minus = sp.Matrix([[0, 1], [0, 0]])
g, e = sp.diag(1, 0), sp.diag(0, 1)

def phases(s):
    return sp.symbols(f"phi_0:{2 * s + 1}", real=True)

def R(n, s):
    if n > 2 * s:
        return sp.eye(2)
    phi = phases(s)[n]
    return sp.Matrix([[sp.exp(sp.I * phi / 2), -sp.exp(sp.I * phi / 2)],
                      [sp.exp(-sp.I * phi / 2), sp.exp(-sp.I * phi / 2)]]) / sp.sqrt(2)
```

The projector $P_n$ onto $n$ photons enters every term. We write it as the
polynomial in $N_c$ that equals one at $N_c=n$ and zero at the other values
$0,\ldots,2s+1$. To second order, processes starting from the kept states never
leave this range of photon numbers.

```{code-cell} ipython3
def P(n, s):
    return sp.prod((N(c) - j) / (n - j) for j in range(2 * s + 2) if j != n)
```

The unperturbed Hamiltonian contains the Floquet energy and the dressed
splitting, with $\tau_z={\rm diag}(1,-1)$ in the dressed basis:

$$
H_0 = \chi N_\ell + \frac{\Omega}{2}\sum_{n=0}^{2s} P_n\,\tau_z.
$$

```{code-cell} ipython3
def H_0(s):
    resonant = sum(P(n, s) for n in range(2 * s + 1))
    return chi * N(ell) * sp.eye(2) + Omega / 2 * resonant * sp.diag(1, -1)
```

The comb tones that are not resonant in sector $n$ form the first part of the
perturbation. With $\sigma_n^-=R_n^\dagger\sigma^-R_n$ the ancilla lowering
operator in the dressed basis,

$$
V_{\rm comb} = \sum_{n=0}^{2s+1}\sum_{k\ne n} \frac{\Omega}{2}e^{i\phi_k}
P_n\,\sigma_n^-\,(\ell^\dagger)^{k-n} + \mathrm{h.c.}
$$

Here $(\ell^\dagger)^{k-n}$ means $\ell^{n-k}$ when $k<n$.

```{code-cell} ipython3
def V_comb(s):
    V = sp.zeros(2)
    for n in range(2 * s + 2):
        sigma_n = R(n, s).adjoint() * sigma_minus * R(n, s)
        for k, phi_k in enumerate(phases(s)):
            if k == n:
                continue  # The resonant tone is part of H_0.
            shift = Dagger(ell)**(k - n) if k > n else ell**(n - k)
            term = Omega / 2 * sp.exp(sp.I * phi_k) * P(n, s) * sigma_n * shift
            V += term + Dagger(term)
    return V
```

The cavity drive has two tones with amplitude $\epsilon$ and phase $\varphi$.
For each ancilla state one tone is resonant, and the other is detuned by
$\chi$. Removing a photon from sector $n$ changes the dressed basis, which gives
the overlaps $R_{n-1}^\dagger|g\rangle\langle g|R_n$ and
$R_{n-1}^\dagger|e\rangle\langle e|R_n$:

$$
V_{\rm drive} = \frac{\epsilon}{2}e^{i\varphi}\sum_{n=1}^{2s+1} cP_n\,
R_{n-1}^\dagger\Big[|g\rangle\langle g|(1+\ell^\dagger)
+ |e\rangle\langle e|(1+\ell)\Big]R_n + \mathrm{h.c.}
$$

```{code-cell} ipython3
def V_drive(s):
    V = sp.zeros(2)
    for n in range(1, 2 * s + 2):
        g_path = R(n - 1, s).adjoint() * g * R(n, s) * (1 + Dagger(ell))
        e_path = R(n - 1, s).adjoint() * e * R(n, s) * (1 + ell)
        term = epsilon / 2 * sp.exp(sp.I * varphi) * c * P(n, s) * (g_path + e_path)
        V += term + Dagger(term)
    return V
```

## What is kept

We keep the upper dressed state $|+_n\rangle$ for photon numbers
$n=0,\ldots,2s$, with Floquet index zero. The upper dressed state is the first
row of the ancilla matrix, which is the default for each reference state. The
order of the list fixes the order of rows in the effective Hamiltonian. All
other states, including the lower dressed branch and nonzero Floquet indices,
appear only in virtual processes.

```{code-cell} ipython3
H_eff = {}
for s in (sp.Integer(1), sp.Rational(3, 2)):
    kept = Embedding({}, reference=[{c: n, ell: 0} for n in range(2 * s + 1)])
    H_eff[s], *_ = block_diagonalize(
        [H_0(s), V_comb(s) + V_drive(s)], subspace_eigenvectors=kept
    )
```

## Perturbation theory

Both $\Omega$-sidebands and the cavity drive count as first order. At first
order, the cavity drive couples neighboring photon numbers. Roy et al. find the
amplitude[^roy]

$$
H^{(1)}_{n-1,n} = \frac{\epsilon\sqrt{n}}{2}\,e^{i\varphi}
\cos\frac{\phi_n-\phi_{n-1}}{2},
$$

where the cosine is the overlap of neighboring dressed states. We check this
for both spins.

```{code-cell} ipython3
def tidy(x):
    return sp.factor(sp.simplify(x.rewrite(sp.cos)).rewrite(sp.exp))

for s in H_eff:
    phi = phases(s)
    for n in range(1, 2 * s + 1):
        expected = epsilon * sp.sqrt(n) / 2 * sp.exp(sp.I * varphi) * sp.cos((phi[n] - phi[n - 1]) / 2)
        assert tidy(H_eff[s][0, 0, 1][n - 1, n] - expected) == 0
```

## Result and check

Roy et al. choose the comb phases so that
$\cos[(\phi_n-\phi_{n-1})/2]=\sqrt{(2s+1-n)/2s}$, Eq. (27) of their paper.
Then $\sqrt n\cos[(\phi_n-\phi_{n-1})/2]=\sqrt{n(2s+1-n)}/\sqrt{2s}$, and
$\sqrt{n(2s+1-n)}$ is the matrix element of $S_-$ between $m=n-s$ and $m-1$.
We check that this gives the spin rotation stated at the top of the page.

```{code-cell} ipython3
def roy_phases(s):
    values = [sp.S.Zero]
    for n in range(1, 2 * s + 1):
        values.append(values[-1] + 2 * sp.acos(sp.sqrt(sp.Rational(2 * s + 1 - n, 2 * s))))
    return dict(zip(phases(s), values))

for s in H_eff:
    H_1 = H_eff[s][0, 0, 1].subs(roy_phases(s)).applyfunc(tidy)
    S_minus = sp.Matrix(2 * s + 1, 2 * s + 1,
                        lambda i, j: sp.sqrt(j * (2 * s + 1 - j)) if i == j - 1 else 0)
    spin_rotation = epsilon / (2 * sp.sqrt(2 * s)) * (
        sp.exp(sp.I * varphi) * S_minus + sp.exp(-sp.I * varphi) * S_minus.T)
    assert (H_1 - spin_rotation).applyfunc(tidy).is_zero_matrix
    display(H_1)
```

The second-order correction for $s=1$, in the same basis, has these entries:

```{code-cell} ipython3
H_2 = H_eff[sp.Integer(1)][0, 0, 2].subs(roy_phases(sp.Integer(1))).applyfunc(tidy)
for i in range(3):
    for j in range(i, 3):
        display(sp.Eq(sp.Symbol(f"H^{{(2)}}_{{{i}{j}}}"), H_2[i, j]))
```

Without the cavity drive, the off-resonant comb tones only shift the levels.
The shifts of $m=\pm1$ are equal, so they act as a uniaxial anisotropy
$S_z^2$ rather than as a field along $z$.

```{code-cell} ipython3
assert sp.simplify(H_2[0, 0] - H_2[2, 2]).subs(epsilon, 0) == 0
assert all(H_2[i, j].subs(epsilon, 0) == 0 for i, j in ((0, 1), (0, 2), (1, 2)))
```

## Discussion

At first order, the effective model is exactly the engineered spin rotation.
At second order, the off-resonant comb tones add an $S_z^2$ anisotropy, terms linear in
$\epsilon$ correct the ladder amplitudes, and the $\epsilon^2$ terms include a
direct coupling between $m=-1$ and $m=1$. The poles at $\Omega=\pm\chi$ and
$\Omega=\pm2\chi$ mark resonances between the upper dressed branch and the
lower one shifted by one or two Floquet steps. This coherent model omits dissipation and the
counter-rotating corrections at the carrier frequencies.

[^roy]: S. Roy et al.,
    [Synthetic high angular momentum spin dynamics in a microwave oscillator](https://doi.org/10.1103/PhysRevX.15.021009),
    Physical Review X **15**, 021009 (2025).
    [Open preprint, Appendix B](https://arxiv.org/html/2405.15695v4#A2).
