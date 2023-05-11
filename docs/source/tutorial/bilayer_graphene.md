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
We set $c_{B1}$ to be in the origin and $c_{T1}$ to be the AA on top at $(0,0,b)^T$, where $b$ is the distance between the two layers.
The hoppings between nearest neighbors are the same (covalent carbon bonds) in plane, but differ out of plane.

+++

Finally we arrive at the Bloch-matrix representation of the bilayer tight-binding lattice
\begin{align}
\mathcal{H}=\sum_k \left(\begin{array}{cc}c^\dagger_{k,B1}\\ c^{\dagger}_{k,B2}\\ c^{\dagger}_{k,T1}\\ c^{\dagger}_{k,T2}\end{array}\right)\left(\begin{array}{cc}0&f_{B1,B2}(k)&f_{B1,T1}(k)&0\\ f_{B1,B2}(k)^\dagger&0&0&0\\f_{B1,T1}^\dagger(k)&0&0&f_{T1,T2}(k)\\0&0&f_{T1,T2}^\dagger(k)&0\end{array}\right) \left(\begin{array}{cc}c_{k,B1}\\ c_{k,B2}\\ c_{k,T1}\\ c_{k,T2}\end{array}\right)
\end{align}
where we have found
\begin{align}
f_{B1,B2}(k)&=-t_i(e^{ik_xa}+e^{-ik_xa/2}2\cos(k_ya\sqrt{3}/2)) \\
f_{B1,T1}(k)&=-t_{B1,T1}\\
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
t_i, t_l, a, b = sympy.symbols('t_i t_l a b', real=True, positive=True, commutative=True)
k_x, k_y, k_z = sympy.symbols('k_x k_y k_z', commutative=True, real=True)

f_12 = -t_i*(sympy.exp(sympy.I*k_x*a)+sympy.exp(-sympy.I*k_x*a/2)*2*sympy.cos(k_y*a*sympy.sqrt(3)/2))
f_13 = -t_l
f_14 = 0
f_23 = 0
f_24 = 0
f_34 = -t_i*(sympy.exp(sympy.I*k_x*a)+sympy.exp(-sympy.I*k_x*a/2)*2*sympy.cos(k_y*a*sympy.sqrt(3)/2))

H = sympy.Matrix([[0,f_12,f_13,f_14],
                  [0,0,f_23,f_24],
                  [0,0,0,f_34],
                  [0,0,0,0]])
H += H.conjugate().T
display(H)
```

```{code-cell} ipython3
:tags: []

K = sympy.sympify(2*sympy.pi/(3*a)*np.array([1,1/sympy.sqrt(3)]))
K
```

```{code-cell} ipython3
:tags: []

q_x, q_y = sympy.symbols('q_x q_y')
delta_K = sympy.sympify(2*sympy.pi/(3*a)*np.array([q_x,q_y]))
delta_K
```

```{code-cell} ipython3
:tags: []

K+delta_K
```

```{code-cell} ipython3
H_K_point = sympy.simplify(H.subs({k_x:K[0]+delta_K[0],k_y:K[1]+delta_K[1]}))
display(H_K_point)
```

```{code-cell} ipython3
P, D = H_K_point.subs({q_x:0,q_y:0}).diagonalize()
```

```{code-cell} ipython3
D
```

```{code-cell} ipython3
P
```

```{code-cell} ipython3
:tags: []

H_tilde, U, U_adj = block_diagonalization.block_diagonalize(H_K_point, symbols=[q_x, q_y], subspace_eigenvectors=(P[:,:2],P[:,2:]))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# To-do
# treat the intrAlayer hopping as perturbation
# spectrum will have two zero modes 
# delta_k_x and delta_k_y are perturbations too


# for start up to second order in delta_k
```
