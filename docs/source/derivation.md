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
## Unitarity condition
We use $U=W+V$ where $W^\dagger=W$ is hermitian, $V^\dagger=-V$ is anti hermitian, and
$W_n,V_n\propto \lambda^n$. Unitarity requires
$(W+V)^\dagger (W+V)=1$ yielding $W_0=1$, $V=0$, and

```{math}
:label: unitarity
\begin{align}
\forall (n \geq 1):\quad \sum_{i=0}^n (W_{n-i} - V_{n-i})(W_i + V_i) &= 0\\
\Rightarrow \sum_{i=0}^n \left[(W_{n-i} - V_{n-i})(W_i + V_i) +
(W_{n-i} + V_{n-i})(W_i - V_i)\right] &= 0\\
\end{align}
```

Solving the latter for $W_n$ results on the constraint stated in THE INDEX. 
The result is convenient since $W_n$ only consists of terms of lower order
than itself.

The condition for $V_n$ follows from EFFECTIVEH and

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

where the structure of $W_n$ is a consequence of eq. UNITARITY and the structure
of the $V_n$.
Performing the block matrix multiplication for $\tilde{H}^{AB}_n$ and requiring 
$\tilde{H}^{AB}_n=0$ leads to Sylvester's equation for $V_n$ reading

```{math}
:label: v_condition
\begin{align}
 H_0^{AA} V_n^{AB} - V_n^{AB} H_0^{BB}=&-
\sum_{i=1}^{n-1}\left[W_{n-i}^{AA}H_0^{AA}V_i^{AB}-V_{n-i}^{AB}
H_0^{BB}W_i^{BB}\right] \\
&-\sum_{i=0}^{n-1}\bigg[W_{n-i-1}^{AA}H_p^{AA}V_i^{AB}+W_{n-i-1}^{AA}
H_p^{AB}W_i^{BB}
-V_{n-i-1}^{AB}(H_p^{AB})^\dagger V_i^{AB} -V_{n-i-1}^{AB}
H_p^{BB}W_i^{BB}\bigg]\\

\end{align}
```

where the terms on the right-hand side of the equation are combined to $Y_n$ in eq. SYLVESTERS EQUATION

+++ {"user_expressions": []}


## The polynomial and Schrieffer-Wolff representation

Contrary to the present polynomial parametrization of the unitary $U$,
the most commonly seen representation of $U$ makes use of the fact that it
realizes  a rotation in block matrix space {ref}`[1]<bravyi_divincenzo_loss>`.
Therefore, often times, $U$ is chosen as
$U=e^\mathcal{S}$, with $\mathcal{S}^\dagger=-\mathcal{S}$ anti hermitian,
where $\mathcal{S}$ is a generator of said rotation
{ref}`[1]<bravyi_divincenzo_loss>`. As we will now show, both parametrizations,
albeit different in their formulation generate the same rotation up to an
overall phase in block matrix space.

### Proof:

Assume $\mathcal{S}\in\mathbb{C}^{N\times N}$ solves {eq}`w_conditions` and
eq. {eq}`v_condition` for some $\mathcal{H}$. We rewrite

```{math}
:label: exp_s_expansion
\begin{align}
U=\exp{\left(\mathcal{S}\right)}=\exp{\left(\sum_{i=0}^\infty
\mathcal{S}_n\right)}=1+\sum_{n=1}^\infty \left[\frac{1}{n!}
\left(\sum_{j=1}^\infty \mathcal{S}_n\right)^n\right]
\end{align}
```

where $\mathcal{S}_n\in\mathbb{C}^{N\times N}$ inherits the property
$\mathcal{S}_n^\dagger=-\mathcal{S}_n$. We truncate the series at some finite
$n$.
To establish coincidence of the two formulations it suffices to show that a
truncated series of eq. {eq}`exp_s_expansion` only containes terms that are
block diagonal and hermitian or block offdiagonal and anti hermitian as
outlined in the parametrization presented above.
Observe that, through the multinomial theorem, each generated term of the
truncated expansion of {eq}`exp_s_expansion` at a given order consists out of
itself and its order reversed partner.
Furthermore observe

```{math}
:label: s_relation
\begin{align}
\prod_{\sum_i k_i=N}\mathcal{S}_{k_i} = (-1)^N\left(\prod_{\sum_ik_i=N}
S_{k_{N-i}}\right)^\dagger
\end{align}
```

for all $n\in\mathbb{N}$. The indices in the previous equation refer to all
vectors $\vec{k}\in\{\mathbb{N}^N:\sum_ik_i=N\}$ that are permissible by the
multinomial theorem. To realize that eq. {eq}`s_relation` is true note that the
adjoint operation, on one hand, maps $k_i\rightarrow k_{N-i}$ reversing the
order of the terms, and, on the other hand, because of the anti hermiticity
property we collect a minus for each factor in the product. In conjunction with
the earlier observation that each term comes with its reversed partner and the
fact that even number products of purely block offdiagonal matrices lead to a
purely block diagonal matrix we can conclude that a truncation of the series
{eq}`exp_s_expansion` only contains purely block diagonal unitaries or purely
block offdiagonal anti hermitian matrices. Since at $\lambda=0$ both
parametrizations must be proportional to the identity we can conclude
coincidence of both forumulations up to a trivial global phase of the unitary
$U$.

We want to point out that this proof establishes coincidence of the two
parametrizations given the same basis ordering of the original Hamiltonian
$\mathcal{H}$. Basis reordering pertains to gauges in block matrix space of the
form

\begin{align}
\tilde{U}=\begin{pmatrix}
\tilde{U}^{AA} & 0 \\
0 & \tilde{U}^{BB}
\end{pmatrix}
\end{align}

Since this class of gauges is constraint to be block diagonal (basis reordering
does not lead to coupling of the $A$ and $B$ spaces) and therefore proportional
to identity in block matrix space the statement of the proof remains valid.

+++ {"tags": [], "user_expressions": []}

(sec:kpm)=
## Extanding the method to non diagonal Hamiltonians

While in section {ref}`sec:derivation` we assumed that $\mathcal{H}_0$ is
already diagonalized, and therefore the spectrum and eigenbasis known in
entirety, here we drop this constraint. The full problem can still be solved
for an effective lower dimensional Hamiltionian than the original one.
The sacrifice for this convenience is however the loss of meaning of the $BB$
block of the effective Hamiltonian.

Consider $\mathcal{H}_0\in\mathbb{C}^{N \times N}$ where $N>>1$ and we only
posses parts of the spectrum, $\mathcal{H}_0|i\rangle=E_i|i\rangle$ where
$i\in\mathcal{N}_A\subset\{i:i\in \mathbb{N}^+, i\leq N\}$.
Even only possessing a limited set

+++ {"user_expressions": []}


## References
(bravyi_divincenzo_loss)=
[1] S. Bravyi, D. DiVincenzo, and D. Loss; Schrieffer–Wolff transformation for
quantum many-body systems,
[Annals of Physics Vol. 326 10 (2011)](https://doi.org/10.1016/j.aop.2011.06.004)

```{code-cell} ipython3

```

```{code-cell} ipython3

```
