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
We set $c_{B1}$ to be in the origin and $c_{T1}$ to be on top of $B1$ at $(0,0,b)^T$, where $b$ is the distance between the two layers.
The hoppings between nearest neighbors are the same (covalent carbon bonds) in plane, but differ out of plane. We also allow for a back gate potential which we set to zero in the bottom layer and to a finite value in the top layer.

+++

Finally we arrive at the Bloch-matrix representation of the bilayer tight-binding lattice
\begin{align}
\mathcal{H}=\sum_k \left(\begin{array}{cc}c^\dagger_{k,B1} \\c^{\dagger}_{k,B2}\\ c^{\dagger}_{k,T1}\\ c^{\dagger}_{k,T2}\end{array}\right)\left(\begin{array}{cc} 0 &f_{B1,B2}(k) & f_{B1,T1}(k) & 0\\ f_{B1,B2}(k)^\dagger & 0 & 0 & 0 \\ f_{B1,T1}^\dagger(k) & 0 & A & f_{T1,T2}(k) \\ 0 & 0 & f_{T1,T2}^\dagger(k) & A \end{array}\right) \left(\begin{array}{cc}c_{k,B1}\\ c_{k,B2}\\ c_{k,T1}\\ c_{k,T2}\end{array}\right)
\end{align}
where we have found
\begin{align}
f_{B1,B2}(k)&=-t_i(e^{ik_xa}+e^{-ik_xa/2}2\cos(k_ya\sqrt{3}/2)) \\
f_{B1,T1}(k)&=-t_l \\
f_{T1,T2}(k)&=-t_i(e^{ik_xa}+e^{-ik_xa/2}2\cos(k_ya\sqrt{3}/2)) \\
\end{align}
where the $t_j$ correspond to the hopping ampltiudes across the indexed bond, $k_i$ are the momenta, and $a$ is the lattice constant.
We see that the electrons have no amplitude to stay on the same site but are strictly intienerant. If the second layer of graphene was not present, we would recover the usual graphene Hamiltonian with its bands
\begin{align}
\varepsilon_\pm=\pm t_i\sqrt{3+2\cos(k_ya\sqrt{3})+4\cos(k_xa3/2)\cos(k_ya\sqrt{3}/2)}
\end{align}
and its emergent Dirac cones at the K-points
\begin{align}
\vec{K}&=\frac{2\pi}{3a}(1,1/\sqrt{3})^T \\
\vec{K^\prime}&=\frac{2\pi}{3a}(1,-1/\sqrt{3})^T.
\end{align}

What, however, happens to the emergent Dirac cones in presence of the other layer? Will the linear dispersion around the $K$ (and $K^\prime$) points persist?

Let us begin by writing the Hamiltonian as a `sympy.Matrix`.

```{code-cell} ipython3
t_i, t_l, a, b = sympy.symbols('t_i t_l a b', real=True, positive=True, commutative=True)
k_x, k_y, k_z = sympy.symbols('k_x k_y k_z', commutative=True, real=True)
A = sympy.symbols('A', real=True, positive=True, commutative=True)

f_12 = -t_i*(sympy.exp(sympy.I*k_x*a)+sympy.exp(-sympy.I*k_x*a/2)*2*sympy.cos(k_y*a*sympy.sqrt(3)/2))
f_13 = -t_l
f_34 = -t_i*(sympy.exp(sympy.I*k_x*a)+sympy.exp(-sympy.I*k_x*a/2)*2*sympy.cos(k_y*a*sympy.sqrt(3)/2))

H = sympy.Matrix([[0,f_12,f_13,0],
                  [0,0,0,0],
                  [0,0,A/2,f_34],
                  [0,0,0,A/2]])
H += H.conjugate().T
display(H)
```

Let us see if we indeed find the Dirac cones in the $K$ point:

Define the $K$-point vector in `sympy` as well as the deviation, $\vec{q}$ in our nomenclature, from it

```{code-cell} ipython3
:tags: []

K = sympy.sympify(2*sympy.pi/(3*a)*np.array([1,1/sympy.sqrt(3)]))

q_x, q_y = sympy.symbols('q_x q_y', real=True, positive=True, commutative=True)
delta_K = sympy.sympify(np.array([q_x,q_y]))

display(sympy.simplify(K+delta_K))
```

The Hamiltonian in the vicinity of $K$ then takes the form

```{code-cell} ipython3
H_K_point = sympy.simplify(H.subs({k_x:K[0]+delta_K[0],k_y:K[1]+delta_K[1]}))
display(H_K_point)
```

+++ {"tags": []}

If we turn off the interlayer coupling for a second and diagonalize the Hamiltonian in the vicinity of $K$ we should recover the emerget Dirac cone

```{code-cell} ipython3
:tags: []

Layer_vecs, Layer_eigs = H[:2,:2].diagonalize()
Layer_eigs = sympy.Matrix([[sympy.simplify(Layer_eigs[i,j].rewrite(sympy.exp).simplify().trigsimp()) for j in range(2)] for i in range(2)])
Layer_eigs
```

Indeed, we recover the proper energy dispersions in the individual layer as expected, and, expanding the dispersions around for $\vec{q}\approx 0$ around $\vec{k}=K+\vec{q}$

```{code-cell} ipython3
:tags: []

sympy.sqrt((sympy.expand(((((Layer_eigs[0,0].subs({k_x:K[0]+q_x,k_y:K[1]+q_y}).trigsimp())**2).series(q_x,0,3).removeO()).series(q_y,0,3).removeO()))+sympy.O(q_x*q_y)).removeO()).simplify()
```

We find the expected Dirace cones.

Now let's move on and turn on the interlayer coupling, $t_l \neq 0$. With the two layers being coupled, we can again

```{code-cell} ipython3
P, D = H_K_point.subs({q_x:0,q_y:0}).diagonalize()
D
```

```{code-cell} ipython3
P
```

Now we perform the perturbative expansion in the momentum differences from the $K$-point and collect the result up to third order. If the Dirac cone survived, we could recover it again form the eigenvalues of the result through Taylor expansion.

```{code-cell} ipython3
:tags: []

H_tilde, U, U_adj = block_diagonalization.block_diagonalize(H_K_point, symbols=[q_x, q_y], subspace_eigenvectors=(P[:,:2],P[:,2:]))
```

```{code-cell} ipython3
#collect A-space Hamiltonian up to second order
indices = np.mgrid[:3, :3]
(0, 0) + tuple(indices[(slice(None),) + np.where(np.sum(indices, axis=0) == 2)])

H_effective = H_tilde[0,0,0,0]+np.ma.sum(H_tilde[(0, 0) + tuple(indices[(slice(None),) + np.where(np.sum(indices, axis=0) == 2)])])
sympy.simplify(H_effective)
```

Diagonalizing yields the dispersions

```{code-cell} ipython3
:tags: []

P_effective, D_effective = H_effective.diagonalize()
D_effective
```

+++ {"tags": []}

Let us take the `[0,0]` entry and see if we recover the Dirac cone

```{code-cell} ipython3
:tags: []

sympy.expand(sympy.expand(D_effective[0,0].series(q_x,0,3).removeO()).series(q_y,0,3).removeO()).simplify()
```

We find that the linear dispersing part of the band has renormalised to a quadratic dispersion modified by the interlayer hopping and the back-gate strength of the second layer.
