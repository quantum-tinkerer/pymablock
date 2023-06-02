# Polynomial form of the block diagonalizing transformation

(sec:derivation)=
## Unitarity condition
One constraint imposed on the transformation $U$ is its unitarity
which we made use of. Consider $U=W+V$, where $W^\dagger=W$
and $V^\dagger=-V$. Also, we let $W_n,V_n\propto \lambda^n$.
Unitarity then requires $(W+V)^\dagger (W+V)=1$. Expanding in
orders this reads

```{math}
:label: unitarity
\begin{align}
\forall (n \geq 1):\quad \sum_{i=0}^n (W_{n-i} - V_{n-i})(W_i + V_i) &= 0\\
\Rightarrow \sum_{i=0}^n \left[(W_{n-i} - V_{n-i})(W_i + V_i) +
(W_{n-i} + V_{n-i})(W_i - V_i)\right] &= 0.\\
\end{align}
```
We observe that $W_0=1$, $V=0$. Using this observation and solving Eq.
{eq}`unitarity` for $W_n$ results in the constraint stated in THE INDEX.
The result is convenient since $W_n$ only consists of terms of lower order
than itself.


## Sylvester's equation
The second condition constraining $U$ comes from the block diagonalization
itself, namely that the off diagonal block are meant to vanish.
This leads to constraints on $V_n$ that follow from EFFECTIVEH and writing

```{math}
:label: W_V_block
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
```

where the structure of $W_n$ is a consequence of Eq. UNITARITY and the structure
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

where the terms on the right-hand side of the equation are combined to $Y_n$ in Eq. SYLVESTERS EQUATION

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

Assume $\mathcal{S}$ solves {eq}`unitarity` and
Eq. {eq}`v_condition` for some $H$. We rewrite

```{math}
:label: exp_s_expansion
\begin{align}
U=\exp{\left(\mathcal{S}\right)}=\exp{\left(\sum_{i=0}^\infty
\mathcal{S}_n\right)}=1+\sum_{n=1}^\infty \left[\frac{1}{n!}
\left(\sum_{j=1}^\infty \mathcal{S}_n\right)^n\right]
\end{align}
```

where $\mathcal{S}_n$ inherits the anti Hermiticity from $S$. We truncate the
series at some finite $n$.

To establish coincidence of the two formulations it suffices to show that said
truncated series only containes terms that are either block diagonal and
hermitian or block offdiagonal and anti hermitian as presented in Eq.
{eq}`W_V_block`. Through the multinomial theorem, each generated term of the
truncated expansion at a given order consists out of itself and its order 
reversed partner.
Furthermore observe how

```{math}
:label: s_relation
\begin{align}
\prod_{\sum_i k_i=N}\mathcal{S}_{k_i} = (-1)^N\left(\prod_{\sum_ik_i=N}
S_{k_{N-i}}\right)^\dagger,
\end{align}
```

for all $n\in\mathbb{N}$. The indexation refers to all
vectors $\vec{k}\in\{\mathbb{N}^N:\sum_ik_i=N\}$ that are permissible by the
multinomial theorem. 

To see that Eq. {eq}`s_relation` is true observe that the
adjoint operation, on one hand, maps $k_i\rightarrow k_{N-i}$ reversing the
order of the terms, and, on the other hand, leads to a minus
for each factor in the product due to the anti Hermiticity. Since each term
comes with its reversed partner and even number products of purely block
offdiagonal matrices yield a purely block diagonal matrix, we conclude that 
a truncation of the series {eq}`exp_s_expansion` only contains purely block
diagonal unitaries or purely block offdiagonal anti hermitian matrices.
Since at $\lambda=0$ both parametrizations must be proportional to the 
identity we can conclude coincidence of both forumulations up to a trivial
global phase of the unitary $U$.

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

(sec:kpm)=
## Extanding the method to non diagonal Hamiltonians

Consider $H_0|i\rangle=E_i|i\rangle$ where 
$|i\rangle \in \mathcal{N}=\mathcal{N}_A\cup \mathcal{N}_B$ and
$\mathcal{N}_A\subset \mathcal{N}$, i.e. we do not posses the 
entire spectrum of $H_0$. Yet, we can still obtain an effective
Hamiltonian from the states that are known.

Let us define
```{math}
:label: pojectors
\begin{align}
\mathcal{P}_B &= (1-\mathcal{P}_A) = 
\left(1 - \sum_{i\in\mathcal{N}_A}| i \rangle\langle i |\right)
\end{align}
```

We can now recast the block diagonalization problem to solve
the ammended Hamiltonian
```{math}
:label: accounting_hamiltonian
\begin{align}
\begin{pmatrix}
 H^{AA} & H_A \mathcal{P}_B \\
\mathcal{P}_B H_A^\dagger   &
\mathcal{P}_B H \mathcal{P}_B 
\end{pmatrix}
\end{align}
```
where $H^{AA}=\sum_{i\in\mathcal{N}_A}E_i|i \rangle\langle i|$
and $H_A=\sum_{i\in\mathcal{N}_A} \langle i|H$ is projected
into the $\mathcal{N}_A$ states from the left. Block 
diagonalizing this ammended representation of the problem
still yields a valid effective Hamiltonian in the $AA$ subspace.
Furthermore, since $H$ is typically sparse and $\mathcal{P}_A$
is small compared to the full problem, the solution
to this problem can be implemented very efficiently.

## References
(bravyi_divincenzo_loss)=
[1] S. Bravyi, D. DiVincenzo, and D. Loss; Schrieffer–Wolff transformation for
quantum many-body systems,
[Annals of Physics Vol. 326 10 (2011)](https://doi.org/10.1016/j.aop.2011.06.004)

```{code-cell} ipython3

```

```{code-cell} ipython3

```
