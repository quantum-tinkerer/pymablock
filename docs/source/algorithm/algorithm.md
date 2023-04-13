```{toctree}
:hidden:
:caption: Algorithms

general.md
expanded.md
kpm.md
```
# Algorithms

Lowdin block-diagonalizes Hamiltonians perturbatively by applying three different
algorithms, [general](general.md), [expanded](expanded.md), and [KPM](kpm.md).
While each is better suited for different applications, the results are equivalent, and
the underlying algorithm is common to all of them.

## Traditional Löwdin perturbation theory
Löwdin perturbation theory block-diagonalizes a perturbed Hamiltonian {math}`H_0 + H_p`,
where {math}`H_p \propto \lambda` the perturbation strength, by doing

```{math}
\tilde{H} = \exp(-S) H \exp(S), \quad S = \sum S_n,
\quad S_n = -S_n^\dagger, \quad S_n \propto \lambda^n.
```

The procedure goes as follows:
1. The Baker-Campbell-Hausdorff formula is applied up to a certain order to obtain the
transformed Hamiltonian, with nested commutators expanded recursively. This generates
terms also above the desired order, which are discarded.
2. The {math}`S_n` is solved for order by order.
3. The transformed diagonal blocks of the Hamiltonian are kept.
4. At the end, the desired operators {math}`O` are obtained as a series expansion of
{math}`\exp(-S) O \exp(S)`. This also generates terms of the higher orders than desired.

## Polynomial Löwdin perturbation theory
We apply an alternative polynomial aproximation of the same unitary transformation, and
demonstrate that it yields simpler expressions, while yielding identical results.

We avoid the exponents by using

```{math}
\tilde{H} = (U + V)^\dagger H (U + V),\quad U = \sum_{i=0}^\infty U_n, \quad V = \sum_{i=0}^\infty V_n,
```

where {math}`U_n = U_n^\dagger` and {math}`V_n = -V_n^\dagger`.
We also require that {math}`(U + V)^\dagger (U + V) = 1` and
{math}`U_n, V_n \propto \lambda^n`, where {math}`\lambda` is the perturbation strength.
This yields {math}`U_0 = 1`, {math}`V_0 = 0`, ands

```{math}
\begin{multline}
\forall (n \geq 1):\quad \sum_{i=0}^n (U_{n-i} - V_{n-i})(U_i + V_i) = 0\\
\Rightarrow \sum_{i=0}^n \left[(U_{n-i} - V_{n-i})(U_i + V_i) + (U_{n-i} + V_{n-i})(U_i - V_i)\right] = 0\\
\Rightarrow 2 U_n = -\sum_{i=1}^{n-1}(U_{n-i}U_i - V_{n-i}V_i).
\end{multline}
```

Here to get to the second identity we added the Hermitian conjugate of the equation to
itself.
In other words, the Hermitian part $U_n$ of the basis transformation is completely fixed
by lower orders, but the anti-Hermitian part $V_n$ is free for us to choose.
We compute {math}`n`-th order of the expansion of the transformed Hamiltonian as

```{math}
\tilde{H}^{(n)} = \sum_{i=0}^n (U_{n-i} - V_{n-i}) H_0 (U_i + V_i) +
\sum_{i=0}^{n-1} (U_{n-i-1} - V_{n-i-1}) H_p (U_i + V_i).
```

Our goal is to find {math}`U` and {math}`V` such that $H$ is block-diagonalized at each
order in {math}`\lambda`. For that we introduce the block form of two subspaces {math}`A`
and {math}`B`.
This yields

```{math}
H_0=
\begin{pmatrix}
H_0^{AA} & 0 \\
0 & H_0^{BB}
\end{pmatrix},
\quad H_p=
\begin{pmatrix}
H_p^{AA} & H_p^{AB} \\
H_p^{BA} & H_p^{BB}
\end{pmatrix},
```
and similarly
```{math}
V_n=
\begin{pmatrix}
0 & V_n^{AB} \\
-(V_n^{AB})^\dagger & 0
\end{pmatrix}.
```

Here we require that {math}`V_n` is block off-diagonal, which produces the transformation
identical to Löwdin perturbation theory (proof pending).
Because {math}`V_n` is block-offdiagonal, its squares will always enter the unitarity
condition as block-diagonal, which, together with {math}`U_0 = 1`, means that {math}`U_n`
is block-diagonal:

```{math}
U_n=
\begin{pmatrix}
U_n^{AA} & 0 \\
0 & U_n^{BB}
\end{pmatrix}.
```

Using the block structure of {math}`U` and {math}`V` and the unitarity condition, we
obtain the following expressions for the blocks of `U_n`:

```{math}
\begin{align}
-2U_n^{AA} &= \sum_{i=1}^{n-1} \left(U_{n-i}^{AA}U_i^{AA} + V_{n-i}^{AB}(V_i^{AB})^\dagger\right)\\
-2U_n^{BB} &= \sum_{i=1}^{n-1} \left(U_{n-i}^{BB}U_i^{BB} + ( V_{n-i}^{AB})^\dagger V_{i}^{AB}\right)
\end{align}.
```

To find {math}`V_n`, we compute the {math}`AB` block of {math}`\tilde{H}^{(n)}` and
require that it vanishes.
Substituting all block froms of {math}`H_0`, {math}`H_p`, {math}`U_n`, and {math}`V_n`
into the definition of {math}`\tilde{H}^{(n)}` yields

```{math}
\begin{align}
0 = \tilde{H}^{(n)AB}&=[H_0,V_n]^{AB} +\sum_{i=1}^{n-1}\left[U_{n-i}^{AA}H_0^{AA}V_i^{AB}-V_{n-i}^{AB} H_0^{BB}U_i^{BB}\right] \\
&+\sum_{i=0}^{n-1}\bigg[U_{n-i-1}^{AA}H_p^{AA}V_i^{AB} +U_{n-i-1}^{AA}H_p^{AB}U_i^{BB}\\
&-V_{n-i-1}^{AB}(H_p^{AB})^\dagger V_i^{AB} -V_{n-i-1}^{AB} H_p^{BB}U_i^{BB}\bigg]\\
&\equiv H_0^{AA} V_n^{AB} - V_n^{AB} H_0^{BB} - Y_n.
\end{align}
```

### Algorithm
This gives us the full step by step algorithm for determining {math}`U_n` and {math}`V_n`.
Here's how it goes.
1. Keep track of all previously computed {math}`U_n` and {math}`V_n` up to a certain order
{math}`n` in a block form (so {math}`U_n^{AA}`, {math}`U_n^{BB}`, and {math}`V_n^{AB}`).
2. Compute {math}`U_{n+1}` using the unitarity condition.
3. Gather all the terms into {math}`Y_{n+1}` (that's {math}`\mathcal{O}(n)` terms).
4. Solve {math}`H_0^{AA} V_{n+1}^{AB} - V_{n+1}^{AB} H_0^{BB} = Y_{n+1}` for
{math}`V_{n+1}^{AB}`, namely set {math}`(V_{n+1}^{AB})_{x,y} = (Y_{n+1})_{x,y} / (E_x - E_y)`,
where {math}`E_x` is an eigenvalue from the {math}`A` block, and {math}`E_y` is an
eigenvalue from the {math}`B` block.
5. Repeat from step 1. until the desired order is reached.
