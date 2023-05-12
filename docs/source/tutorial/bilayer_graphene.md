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

# Bilayer-graphene in tight-binding approximation

```{code-cell} ipython3
import numpy as np
import sympy
import scipy
import kwant

from lowdin import block_diagonalization
```

## The set-up

Over the past few years, graphene was at the forefront in materials research. From its [interesting properties as conductor](http://dx.doi.org/10.1103/PhysRevB.78.085415) to its [surprising features when stacked on top of each other in particular ways](https://doi.org/10.1038/s41563-020-00840-0), graphene has many interesting aspect to study. The one feature that we want to take a closer look on are the emergent Dirac cones at the, so called, $K$-point, one of the points on the boundary of the Brillouin zone.
In this point, the two energy bands of single layer graphene touch (at $E=0$) and in its vicinity, say a small vector $\vec{q}$ away, the dispersion is linear. This constitutes a Dirac cone and is one of the peculiar features [linked to topology](https://doi.org/10.1103/PhysRevD.13.3398). 
Since we hinted earlier that stacking multiple layers of graphene can change the electronic properties significantly, one can ask the question what happens with the Dirac cone in said $K$-point when we introduce a second layer of graphene into the mix.

+++

## The Hamiltonian
We begin in real space where we consider the different allowed hoppings seperately. The unit cell consists of 4 atoms, 2 from the bottom layer and 2 from the top layer. We denote each by their own creation and annihilation operators reading
$$\{c_{B1}, c_{B2}, c_{T1}, c_{T2}\}.$$
We consider $c_{B1}$ atom to be in the 3-vertex of the bottom layer and fix our coordinate system such that the origin coincides with its position. We place the second layer of graphene on top such that the $c_{T1}$ carbon atom, which itself sits in the 3-vertex of the top layer which is rotated by an angle of $60\degree$ with respect to the bottom layer, at $(0, 0, b)^T$.

Within the two layers, the nearest neighbour hopping amplitudes are all equal and we denote them by $t_i$. Between the layers, we only consider the hopping between the $c_{B1}$ and $c_{T1}$ atoms and denote its hopping amplitude by $t_l$. All other hoppings are kept at zero which means that we will not recover socalled [_trigonal warping_](https://doi.org/10.1016/j.ssc.2007.03.054). Lastly, we also apply a back gate volatage to our bilayer sample which leads to different on-site potentials of the two layers. We keep that at zero for the bottom layer and retain a finite on-site potential $A$ in the top layer. 

Preforming a Fourier transform in the $x-$ and $y-$directions, we recover the Bloch-Hamiltonian of the system, reading
\begin{align}
\mathcal{H}=\sum_k \left(\begin{array}{cc}c^\dagger_{k,B1} \\c^{\dagger}_{k,B2}\\ c^{\dagger}_{k,T1}\\ c^{\dagger}_{k,T2}\end{array}\right)\left(\begin{array}{cc} 0 &f_{B1,B2}(k) & f_{B1,T1}(k) & 0\\ f_{B1,B2}(k)^\dagger & 0 & 0 & 0 \\ f_{B1,T1}^\dagger(k) & 0 & A & f_{T1,T2}(k) \\ 0 & 0 & f_{T1,T2}^\dagger(k) & A \end{array}\right) \left(\begin{array}{cc}c_{k,B1}\\ c_{k,B2}\\ c_{k,T1}\\ c_{k,T2}\end{array}\right)
\end{align}
where we have found
\begin{align}
f_{B1,B2}(k)&=-t_i(e^{ik_xa}+e^{-ik_xa/2}2\cos(k_ya\sqrt{3}/2)) \\
f_{B1,T1}(k)&=-t_l \\
f_{T1,T2}(k)&=-t_i(e^{ik_xa}+e^{-ik_xa/2}2\cos(k_ya\sqrt{3}/2)) \\
\end{align}
The $t_j$ are the aforementioned hopping amplitudes, $a$ is the lattice constant, and $k_x, k_y$ are the momenta in $x-$ and $y-$direction. 

Let us now make this Hamiltonian a `sympy.Matrix`, which allows us to calculate the physical peroperties we want to study and is furthermore one of the data formats that is natively understood by _Lowdin_

```{code-cell} ipython3
t_i, t_l, a, b, k_x, k_y, k_z, A = sympy.symbols('t_i t_l a b k_x k_y k_z A', real=True, positive=True, commutative=True)

f_12 = -t_i*(sympy.exp(sympy.I*k_x*a)+sympy.exp(-sympy.I*k_x*a/2)*2*sympy.cos(k_y*a*sympy.sqrt(3)/2))
f_13 = -t_l
f_34 = -t_i*(sympy.exp(sympy.I*k_x*a)+sympy.exp(-sympy.I*k_x*a/2)*2*sympy.cos(k_y*a*sympy.sqrt(3)/2))

H = sympy.Matrix([[0, f_12, f_13, 0],
                  [0, 0, 0, 0],
                  [0, 0, A/2, f_34],
                  [0, 0, 0, A/2]])
H += H.conjugate().T
H
```

## The Dirac cones

As stated in the introduction, we expect to find a Dirac cone at the so called $K$-point. Note, that a Dirac cone is characterised by a degeneracy in a single point and a linear dispersion around. 

Let us define the $K$-point vector $K=\frac{2\pi}{3a}(1,1/\sqrt{3})^T$, and a deviation $\vec{q}$ from it, in `sympy` and see if the single layer Hamiltonian (`H[:2,:2]` for the bottom layer for instance) indeed yields a Dirac cone. Note that this amounts to fixing $t_l=0$ for the time being. The Hamiltonian then separates.

```{code-cell} ipython3
:tags: []

K = sympy.sympify(2*sympy.pi/(3*a)*np.array([1,1/sympy.sqrt(3)]))

q_x, q_y = sympy.symbols('q_x q_y', real=True, positive=True, commutative=True)
delta_K = sympy.sympify(np.array([q_x,q_y]))

display(sympy.simplify(K+delta_K))
```

In vicinity of the $K$-point the single layer Hamiltonian therefore takes the form

```{code-cell} ipython3
H_K_single_layer = sympy.simplify(H[:2, :2].subs({k_x:K[0]+delta_K[0],k_y:K[1]+delta_K[1]}))
display(H_K_single_layer)
```

Diagonalizing gives us the dispersion around $K$, which we want to expand for small deviations $\vec{q}$

```{code-cell} ipython3
:tags: []

Layer_eigs = H_K_single_layer.diagonalize()[1]
Layer_eigs = sympy.Matrix([[sympy.simplify(Layer_eigs[i,j].rewrite(sympy.exp).simplify().trigsimp()) for j in range(2)] for i in range(2)])
display(Layer_eigs)
```

Since both only differ by a minus sign it sufices to take one of the two (we will take the `[0,0]` entry) and expand it $\vec{q}\approx 0$. With a little bit of `sympy` tinkering we finally get

```{code-cell} ipython3
:tags: []

sympy.sqrt((sympy.expand(((((Layer_eigs[0,0].trigsimp())**2).series(q_x,0,3).removeO()).series(q_y,0,3).removeO()))+sympy.O(q_x*q_y)).removeO()).simplify()
```

Indeed we find that the spectrum takes the form of a Dirac cone and that the energies at $\vec{q}=0$ will be twice degenerate (because of the second band) at $E=0$.

+++

## The second layer

What happens to this very characteristic property if we turn on the interlayer hopping $t_l\neq 0$? Will the linearity of the dispersion be retained?

We can answer these questions using _quasi-degenerate perturbation theory_ using the very package of which you are currently reading the tutorial (how convenient). The perturbation in this case, which might seem counter intuitive at first, will be the in-plane motion of the electrons, i.e. $\vec{q}$ itself. To perform the procedure itself, we have to find the eigenvectors of our Hamiltonian in the point of interest, namely the $K$-point, determine which states we are interested in and which we want to separate, and call `block_diagonalize` to deliver its magic.

This little amount of information suffices, because we know that in the $K$ point, the Hamiltonian will be diagonal in the eigenvectors and the eigenenergies can be read-off from the diagonal. 

Let us set up the spectrum in the $K$-point for further use and see which of the states we want to keep

```{code-cell} ipython3
P, D = sympy.simplify(H.subs({k_x:K[0], k_y:K[1]})).diagonalize()
```

We find the energies of the four-band, two-layer system to be

```{code-cell} ipython3
:tags: []

display(D)
display(P)
```

From the energies, we see that one of the bands of each layer retains the energy as if the second layer was not present while the other two bands move because of the layer coupling. The former are the bands we want to focus on (our $A$ subspace) while the latter bands we want to remove perturbatively (the $B$ subspace). 

So lets input our Hamiltonian at $\vec{k}=K+\vec{q}$ and the eigenvectors of the $A$ and $B$ subspace into `block_diagonalize` and see what we find:

```{code-cell} ipython3
:tags: []

H_K_point = sympy.simplify(H.subs({k_x:K[0]+delta_K[0],k_y:K[1]+delta_K[1]}))
```

```{code-cell} ipython3
:tags: []

H_tilde, U, U_adj = block_diagonalization.block_diagonalize(sympy.simplify(H.subs({k_x:K[0]+q_x, k_y:K[1]+q_y})), 
                                                            symbols = [q_x, q_y],
                                                            subspace_eigenvectors = (P[:,:2], P[:,2:])
                                                           )
```

`block_diagonalize` now gave us back three objects of type `lowdin.series.BlockSeries`. These encode everything we need to know of our problem. `H_tilde` represents the effective (block-diagonalized) Hamiltonian, `U` and `U_adj` represent the unitary transformation that achieves this block-diagonalization. 

We now simply call orders of the perturbation theory of the blocks (in the sense of the $A$ and $B$ subspaces) we are interested in as if we were dealing with a `np.ndarray` object. To call e.g. the second order contribution in $q_x$ within the $A$ block (to remind, those are our bands of interest) we query

```{code-cell} ipython3
:tags: []

H_tilde[0,0,2,0]
```

Et voilá! The queries proceed with the first two indices specifying the block, i.e. whether $A$ or $B$ block, while the remaining indices directly query orders of the perturbation.

But back to our graphene problem: To see what happened to our Dirac cones, let us collect all perturbation terms up to second order in $q_x, q_y$ and sum them

```{code-cell} ipython3
#collect A-space Hamiltonian up to second order
indices = np.mgrid[:3, :3]
query = (0, 0) + tuple(indices[(slice(None),) + np.where(np.sum(indices, axis=0) == 2)])

H_effective = H_tilde[0,0,0,0]+np.ma.sum(H_tilde[query])
sympy.simplify(H_effective)
```

We can already see that, instead of the two bands being oblivious of the other, now the two bands are in fact coupled through $t_l$. But let us see the problem to the end and diagonlize our second-order effective Hamiltonian:

```{code-cell} ipython3
:tags: []

P_effective, D_effective = H_effective.diagonalize()
D_effective
```

+++ {"tags": []}

We see that our spectrum looks quite a bit different now. What happens for $vec{q}\approx 0$ now? Let us expand the `[0,0]` entry for small $\vec{q}$ again and see what became of our once linear dispersion:

```{code-cell} ipython3
:tags: []

sympy.expand(sympy.expand(D_effective[0,0].series(q_x,0,3).removeO()).series(q_y,0,3).removeO()).simplify()
```

And there you have it: what was once fascinating and topological now almost looks like what you find in old semiconductor textbooks. All because that second layer of graphene was too pushy.
