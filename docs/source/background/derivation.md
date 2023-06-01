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

+++ {"user_expressions": []}

# Polynomial form of the block diagonalizing transformation

(sec:derivation)=
## Derivation

The solution of a stationary Schödinger equation for the Hamiltonian $\mathcal{H}_0\in \mathbb{C}^{n \times n}$ in quantum mechanics generally yields infinitely many states $|i\rangle$, satisying $\mathcal{H}_0|i \rangle=E_i |i \rangle$, that can be occupied by particles.
Typically, however, one is only interested in a small number of states $|i\rangle \in A=\{i\in\mathbb{N}^+: i\leq N_A\}$ and discards the remaining states $|i\rangle\in B=\{i\in\mathbb{N}^+:i\notin A\}$ since, in an eigenbasis, these share no overlap with the subspace of interest as $\langle i, j\rangle=\delta_{i,j}$.
A general perturbation $\mathcal{H}_p\in \mathbb{C}^{N\times N}$ that polynomially depends on a perturbation parameter $\lambda$ changes this and leads to mixing of the previously orthogonal states.
Writing the Hamiltonian $\mathcal{H}=\mathcal{H}_0+\mathcal{H}_p$ in terms of the block indices $A$ and $B$ yields

```{math}
:label: hamiltonian
\begin{align}
\mathcal{H}=
\begin{pmatrix}
\mathcal{H}^{AA} & \mathcal{H}^{AB} \\
\mathcal{H}^{BA} & \mathcal{H}^{BB}
\end{pmatrix}=
\begin{pmatrix}
\mathcal{H}_0^{AA}+\mathcal{H}_p^{AA} & \mathcal{H}_p^{AB} \\
\mathcal{H}_p^{BA} & \mathcal{H}^{BB}+\mathcal{H}_p^{BB}
\end{pmatrix}
\end{align}
```

where we assume that $\mathcal{H}_0$ has already been doagonalized and is hence block diagonal (lifting this asumption is discussed in section {ref}`sec:kpm`). The off diagonal blocks in eq. {eq}`hamiltonian` facilitate the need for _quasi degenerate perturbation theory_, i.e. a unitary transformation $U\in \mathbb{C}^{N\times N}$ that block diagonalizes $\mathcal{H}$ perturbatively in $\lambda$ to yield

```{math}
:label: effective_hamiltonian
\begin{align}
\tilde{\mathcal{H}}=U^\dagger \mathcal{H} U
\end{align}
```

where $\tilde{\mathcal{H}}$, in the following referred to as _effective Hamiltonian_, is block diagonal in $A$ and $B$.
We address this problem by letting 
\begin{align}
\tilde{\mathcal{H}} = U^\dagger \mathcal{H} U = (W + V)^\dagger \mathcal{H} (W + V),\quad W = \sum_{i=0}^\infty W_n, \quad V = \sum_{i=0}^\infty V_n,
\end{align}
where $W^\dagger=W$ is hermitian, $V^\dagger=-V$ is anti hermitian, and $W_n,V_n\propto \lambda^n$. Due to unitarity, we require $(W+V)^\dagger (W+V)=1$ yielding $W_0=1$, $V=0$, and

```{math}
:label: unitarity
\begin{align}
\forall (n \geq 1):\quad \sum_{i=0}^n (W_{n-i} - V_{n-i})(W_i + V_i) &= 0\\
\Rightarrow \sum_{i=0}^n \left[(W_{n-i} - V_{n-i})(W_i + V_i) + (W_{n-i} + V_{n-i})(W_i - V_i)\right] &= 0\\
\Rightarrow 2 W_n &= -\sum_{i=1}^{n-1}(W_{n-i}W_i - V_{n-i}V_i).
\end{align}
```

where we added the hermitian conjugate of itself to the equation to arrive at the second identity. Note how the unitary $W_n$ now only depends on lower orders of the expansion. To determine the $V_n$ that block diagonalize the perturbed Hamiltonian we go back to {eq}`hamiltonian` and plug in the expanded forms of $W$ and $V$, and truncate the series at some $n$ to give

\begin{align}
\tilde{\mathcal{H}}^{(n)} = \sum_{i=0}^n (W_{n-i} - V_{n-i}) \mathcal{H}_0 (W_i + V_i) + \sum_{i=0}^{n-1} (W_{n-i-1} - V_{n-i-1}) \mathcal{H}_p (W_i + V_i).
\end{align}

We choose the gauge of the operators $W$ and $V$ in block matrix space such that we can write 

\begin{align}
V_n &= \begin{pmatrix}
0 & V_n^{AB}\\
-(V_n^{AB})^\dagger & 0
\end{pmatrix} 
\quad 
W_n = \begin{pmatrix}
W_n^{AA} & 0 \\
0 & W_n^{BB}
\end{pmatrix}
\end{align}

The second identity is a consequence of eq. {eq}`unitarity` and the structure of the $V_n$. We can now write down the conditions for each of the three unknowns $W_n^{AA}$, $W_n^{BB}$, and $V_n^{AB}$ reading

```{math}
:label: w_conditions
\begin{align}
-2W_n^{AA} &= \sum_{i=1}^{n-1} \left(W_{n-i}^{AA}W_i^{AA} + V_{n-i}^{AB}(V_i^{AB})^\dagger\right)\\
-2W_n^{BB} &= \sum_{i=1}^{n-1} \left(W_{n-i}^{BB}W_i^{BB} + ( V_{n-i}^{AB})^\dagger V_{i}^{AB}\right)
\end{align}
```

and

```{math}
:label: v_condition
\begin{align}
0 = \tilde{\mathcal{H}}^{(n)AB}&=[\mathcal{H}_0,V_n]^{AB} +\sum_{i=1}^{n-1}\left[W_{n-i}^{AA}\mathcal{H}_0^{AA}V_i^{AB}-V_{n-i}^{AB} \mathcal{H}_0^{BB}W_i^{BB}\right] \\
&+\sum_{i=0}^{n-1}\bigg[W_{n-i-1}^{AA}\mathcal{H}_p^{AA}V_i^{AB}+W_{n-i-1}^{AA}\mathcal{H}_p^{AB}W_i^{BB}
-V_{n-i-1}^{AB}(\mathcal{H}_p^{AB})^\dagger V_i^{AB} -V_{n-i-1}^{AB} \mathcal{H}_p^{BB}W_i^{BB}\bigg]\\
&\equiv \mathcal{H}_0^{AA} V_n^{AB} - V_n^{AB} \mathcal{H}_0^{BB} - Y_n.
\end{align}
```

where $Y_n$ only depends on $W_i, V_i$ to lower orders. Note how the structure of the previous equations is that of a Cauchy product. Solving these three conditions for a fixed $n$ leads to perturbative decoupling of the two blocks.

+++ {"tags": [], "user_expressions": []}

## The polynomial and Schrieffer-Wolff representation

Contrary to the present polynomial parametrization of the unitary $U$, the most commonly seen representation of $U$ makes use of the fact that it realizes  a rotation in block matrix space {ref}`[1]<bravyi_divincenzo_loss>`. Therefore, often times, $U$ is chosen as $U=e^\mathcal{S}$, with $\mathcal{S}^\dagger=-\mathcal{S}$ anti hermitian, where $\mathcal{S}$ is a generator of said rotation {ref}`[1]<bravyi_divincenzo_loss>`. As we will now show, both parametrizations, albeit different in their formulation generate the same rotation up to an overall phase in block matrix space.

### Proof:

Assume $\mathcal{S}\in\mathbb{C}^{N\times N}$ solves {eq}`w_conditions` and {eq}`v_condition` for some $\mathcal{H}$. We rewrite 

```{math}
:label: exp_s_expansion
\begin{align}
U=\exp{\left(\mathcal{S}\right)}=\exp{\left(\sum_{i=0}^\infty \mathcal{S}_n\right)}=1+\sum_{n=1}^\infty \left[\frac{1}{n!}\left(\sum_{j=1}^\infty \mathcal{S}_n\right)^n\right]
\end{align}
```

where $\mathcal{S}_n\in\mathbb{C}^{N\times N}$ inherits the property $\mathcal{S}_n^\dagger=-\mathcal{S}_n$. We truncate the series at some finite $n$.
To establish coincidence of the two formulations it suffices to show that a truncated series of eq. {eq}`exp_s_expansion` only containes terms that are block diagonal and hermitian or block offdiagonal and anti hermitian as outlined in the parametrization presented above. 
Observe that, through the multinomial theorem, each generated term of the truncated expansion of {eq}`exp_s_expansion` at a given order consists out of itself and its order reversed partner. Furthermore observe

```{math}
:label: s_relation
\begin{align}
\prod_{\sum_i k_i=N}\mathcal{S}_{k_i} = (-1)^N\left(\prod_{\sum_ik_i=N}S_{k_{N-i}}\right)^\dagger
\end{align}
```

for all $n\in\mathbb{N}$. The indices in the previous equation refer to all vectors $\vec{k}\in\{\mathbb{N}^N:\sum_ik_i=N\}$ that are permissible by the multinomial theorem. To realize that eq. {eq}`s_relation` is true note that the adjoint operation, on one hand, maps $k_i\rightarrow k_{N-i}$ reversing the order of the terms, and, on the other hand, because of the anti hermiticity property we collect a minus for each factor in the product. In conjunction with the earlier observation that each term comes with its reversed partner and the fact that even number products of purely block offdiagonal matrices lead to a purely block diagonal matrix we can conclude that a truncation of the series {eq}`exp_s_expansion` only contains purely block diagonal unitaries or purely block offdiagonal anti hermitian matrices. Since at $\lambda=0$ both parametrizations must be proportional to the identity we can conclude coincidence of both forumulations up to a trivial global phase of the unitary $U$.

We want to point out that this proof establishes coincidence of the two parametrizations given the same basis ordering of the original Hamiltonian $\mathcal{H}$. Basis reordering pertains to gauges in block matrix space of the form

\begin{align}
\tilde{U}=\begin{pmatrix}
\tilde{U}^{AA} & 0 \\
0 & \tilde{U}^{BB}
\end{pmatrix}
\end{align}

Since this class of gauges is constraint to be block diagonal (basis reordering does not lead to coupling of the $A$ and $B$ spaces) and therefore proportional to identity in block matrix space the statement of the proof remains valid.

+++ {"user_expressions": []}

(sec:kpm)=
## Extanding the method to non diagonal Hamiltonians

While in section {ref}`sec:derivation` we assumed that $\mathcal{H}_0$ is already diagonalized, and therefore the spectrum and eigenbasis known in entirety, here we drop this constraint. The full problem can still be solved for an effective lower dimensional Hamiltionian than the original one. The sacrifice for this convenience is however the loss of meaning of the $BB$ block of the effective Hamiltonian.

Consider $\mathcal{H}_0\in\mathbb{C}^{N \times N}$ where $N>>1$ and we only posses parts of the spectrum, $\mathcal{H}_0|i\rangle=E_i|i\rangle$ where $i\in\mathcal{N}_A\subset\{i:i\in \mathbb{N}^+, i\leq N\}$. Even only possessing a limited set 


+++ {"user_expressions": []}

## References
(bravyi_divincenzo_loss)=
[1] S. Bravyi, D. DiVincenzo, and D. Loss; Schrieffer–Wolff transformation for quantum many-body systems, [Annals of Physics Vol. 326 10 (2011)](https://doi.org/10.1016/j.aop.2011.06.004)

```{code-cell} ipython3

```
