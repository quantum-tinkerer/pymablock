---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
import sympy
import scipy
import kwant

from lowdin import block_diagonalization
```

# Bilayer Grahepene in tight-binding approximation

+++

## Real-space Hamltonians
We begin in real space where we consider the different allowed hoppings seperately. The unit cell consists of 4 atoms, 2 from the bottom layer and 2 from the top layer. We denote each by their own creation and annihilation operators reading
$$\{c_{B1}, c_{B2}, c_{T1}, c_{T2}\}.$$
We set $c_{B1}$ to be in the origin and $c_{T1}$ to be the AA on top at $(0,0,b)^T$, where $b$ is the distance between the two layers.
The hoppings between nearest neighbors are the same (covalent carbon bonds) in plane, but differ out of plane.

```{code-cell} ipython3

```

```{code-cell} ipython3

```

Finally we arrive at the Bloch-matrix representation of the bilayer tight-binding lattice
\begin{align}
\mathcal{H}=\sum_k \left(\begin{array}{cc}c^\dagger_{k,B1}\\ c^{\dagger}_{k,B2}\\ c^{\dagger}_{k,T1}\\ c^{\dagger}_{k,T2}\end{array}\right)\left(\begin{array}{cc}0&f_{B1,B2}(k)&f_{B1,T1}(k)&f_{B1,T2}(k)\\ f_{B1,B2}(k)^\dagger&0&f_{B2,T1}(k)&f_{B2,T2}(k)\\f_{B1,T1}^\dagger(k)&f_{B2,T1}^\dagger(k)&0&f_{T1,T2}(k)\\f_{B1,T2}^\dagger(k)&f_{B2,T2}^\dagger(k)&f_{T1,T2}^\dagger(k)&0\end{array}\right) \left(\begin{array}{cc}c_{k,B1}\\ c_{k,B2}\\ c_{k,T1}\\ c_{k,T2}\end{array}\right)
\end{align}
where we have found
\begin{align}
f_{B1,B2}(k)&=-t_i(e^{ik_xa}+e^{-ik_xa/2}2\cos(k_ya\sqrt{3}/2)) \\
f_{B1,T1}(k)&=-t_{B1,T1}e^{ik_zb}(e^{-ik_xa}+e^{ikxa/2}2\cos(k_ya\sqrt{3}/2)) \\
f_{B1,T2}(k)&= -t_{B1,T2}e^{ik_zb}(e^{-ik_xa}+e^{ik_xa/2}2\cos(k_ya\sqrt{3}/2))\\
f_{B2,T1}(k)&= -t_{B1,T2}e^{ik_zb}(e^{ik_xa/2}2\cos(k_ya\sqrt{3}/2)+e^{-ik_xa})\\
f_{B2,T2}(k)&= -t_{B2,T2}e^{ik_zb}(e^{ik_xa}+e^{-ik_xa/2}2\cos(k_ya\sqrt{3}/2))\\
f_{T1,T2}(k)&=-t_i(e^{ik_xa}+e^{-ik_xa/2}2\cos(k_ya\sqrt{3}/2)) \\
\end{align}
where the $t$ correspond to the hopping ampltiudes across the indexed bond, $k_i$ are the momenta, and $a$ is the lattice constant.
We see that the electrons have no amplitude to stay on the same site but are strictly intienerant. If the second layer of graphene was not present, we would recover the usual graphene Hamiltonian with its bands
\begin{align}
\varepsilon_\pm=\pm t\sqrt{3+2\cos(k_ya\sqrt{3})+4\cos(k_xa3/2)\cos(k_ya\sqrt{3}/2)}
\end{align}
and its emergent Dirac cones at the K-points
\begin{align}
\vec{K}&=\frac{2\pi}{3a}(1,1/\sqrt{3})^T \\
\vec{K^\prime}&=\frac{2\pi}{3a}(1,-1/\sqrt{3})^T.
\end{align}
What though when we slowly turn on the interlayer coupling? To analyse this, we want to treat the presence of the second layer in perturbation theory. Looking at the Hamiltonian, this means we have to remove the block off-diagonal coupling blocks between the layers in perturbation theory. Wouldn't it be convenient if there was a package for that?

Let us begin by creating the Hamiltonian as a `sympy` Matrix. For brevity we will assume that all interlayer hoppings are equal and denote them as $t_l$

```{code-cell} ipython3
t_i, t_l, a, k_x, k_y = sympy.symbols('t_i t_l a k_x k_y')
H = sympy.Matrix
```

```{code-cell} ipython3

```
