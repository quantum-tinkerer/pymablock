# Algorithm definition

Compared to the notation so far we will use a slightly different parametrization of $\mathcal{U} = 1 + \mathcal{U}' = 1 + \mathcal{W} + \mathcal{V}$, where $\mathcal{W} = \mathcal{W}^\dagger$ and $\mathcal{V} = -\mathcal{V}^\dagger$ as before. This ensures that $(\mathcal{W}^2)_n$ does not depend on $W_n$.

Unitarity reads
$$
(1 + \mathcal{W} - \mathcal{V}) (1 + \mathcal{W} + \mathcal{V}) = 1,
$$
so that after adding $\mathcal{U}\mathcal{U}^\dagger$ like before we get
$$
2\mathcal{W} + \mathcal{W}^2 - \mathcal{V}^2 = 0,
$$
which we knew. Subtracting the $\mathcal{U}\mathcal{U}^\dagger$, on the other hand, gives $[\mathcal{W}, \mathcal{V}] = 0$, which is a bit of a new insight.

For the actual implementation we favor matrix notation to rely on `cauchy_dot_product`. This gives
$$
\mathcal{W} = (\mathcal{U}' + \mathcal{U}'^\dagger)/2 = -\mathcal{U}'^\dagger\mathcal{U}'/2.
$$

Alternatively, we could express $\mathcal{W}$ through the rest of the terms:
$$
\mathcal{W} = \sqrt{1 + \mathcal{V}^2} - 1 \equiv f(\mathcal{V}) \equiv \sum_n a_n \mathcal{V}^{2n}.
$$
Notice that evaluating $2\mathcal{W} = \mathcal{V}^2 - \mathcal{W}^2$ is cheap and recursion causes no problems because $\mathcal{W}$ has no 0th order term.
Computing a series expansion, on the other hand, has a higher scaling of its complexity.

Now let us look at the unitary transformation of the Hamiltonian
$$
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U} = (1 + \mathcal{W} - \mathcal{V}) (H_0 + \mathcal{H'}) (1 + \mathcal{W} + \mathcal{V}).
$$
Our goal is to manipulate the above expression so that the only terms containing $H_0$ are of the form $[\mathcal{V}, H_0] \equiv \mathcal{Y}$ ($\mathcal{Y}$ is off-diagonal and Hermitian).
We start by collecting the terms containing $H_0$ in the above expression
$$
\tilde{\mathcal{H}_0} = H_0 - \mathcal{Y} + \{\mathcal{W}, H_0\} + (\mathcal{W} - \mathcal{V})H_0(\mathcal{W} + \mathcal{V}).
$$
We want to eliminate $H_0$ from the last two terms. In other words, we want to find an expression for these that depends on $\mathcal{Y}$ and maybe other series that depend on $\mathcal{Y}$, $\mathcal{W}$, or $\mathcal{V}$, but do not contain $H_0$.

In the code we eliminated $H_0$ by commuting it to the right. Let us do so formally by introducing $\mathcal{Z} \equiv [\mathcal{W}, H_0]$ ($\mathcal{Z}$ is diagonal and anti-Hermitian).
Commuting $H_0$ to the right we get:
$$
\begin{align*}
\tilde{\mathcal{H}}_0 &= H_0 - \mathcal{Y} + 2\mathcal{W}H_0 - \mathcal{Z} + \mathcal{W}(\mathcal{W}H_0 - \mathcal{Z}) - \mathcal{V}(\mathcal{W}H_0 - \mathcal{Z}) \\
&-\mathcal{V}(\mathcal{V} H_0 - \mathcal{Y}) + \mathcal{W}(\mathcal{V} H_0 - \mathcal{Y}).
\end{align*}
$$
Examining all the terms that still contain $H_0$ we observe that they cancel by unitarity:
$$
\left(2\mathcal{W} + \mathcal{W}^2 - \mathcal{V}\mathcal{W} - \mathcal{V}^2 + \mathcal{W}\mathcal{V}\right)H_0 = 0.
$$
Therefore we get
$$
\tilde{\mathcal{H}}_0 = (H_0 - \mathcal{W}\mathcal{Z} + \mathcal{V}\mathcal{Y} - \mathcal{Z}) + (-\mathcal{Y} + \mathcal{V}\mathcal{Z} - \mathcal{W}\mathcal{Y}),
$$
where the first terms are block-diagonal, and therefore do not enter the equation for $\mathcal{Y}$. The terms $\mathcal{V}\mathcal{Z}$ and $\mathcal{W}\mathcal{Y}$ don't depend on $Y_n$, and therefore are fine to use in the recursive scheme.

We still have a problem of cheaply evaluating $\mathcal{Z}$ without multiplications by $H_0$. To do this we apply the same trick as we did for unitarity: we substitute the recursive definition for $\mathcal{W}$ into the definition of $\mathcal{Z}$:
$$
-2\mathcal{Z} = -2[\mathcal{W}, H_0] = [\mathcal{W}^2 - \mathcal{V}^2, H_0] = \mathcal{W}^2H_0 - H_0\mathcal{W}^2 - \mathcal{V}^2H_0 + H_0\mathcal{V}^2 = \mathcal{W}(H_0 \mathcal{W} + \mathcal{Z})-
(\mathcal{W} H_0 - \mathcal{Z})\mathcal{W}\\
- \mathcal{V}(H_0 \mathcal{V} - \mathcal{Y}) - (\mathcal{V} H_0 + \mathcal{Y})\mathcal{V} = \{\mathcal{W},\mathcal{Z}\} - \{\mathcal{V}, \mathcal{Y}\}.
$$
The right hand side has no terms that contain $Z_n$ or $Y_n$, and therefore it is a cheap definition of $\mathcal{Z}$, similar to a cheap definition of $\mathcal{W}$.

The expression for $\tilde{\mathcal{H}}_0$ is not Hermitian by construction, but this is an artifact of the way we eliminated $H_0$.
The equivalent Hermitian form would be
$$
\tilde{\mathcal{H}}_0 = (H_0 + [\mathcal{Z}, \mathcal{W}]/2 + [\mathcal{V}, \mathcal{Y}]/2) + (-\mathcal{Y} + \{\mathcal{Z}, \mathcal{V}\}/2 - \{\mathcal{W}, \mathcal{Y}\}/2),
$$
which is more symmetric, but otherwise not computationally advantageous.
Turning to the remaining terms in $\tilde{\mathcal{H}}$ we get
$$
\tilde{\mathcal{H}}' = \mathcal{U}^\dagger \mathcal{H}' \mathcal{U}.
$$
We don't need to expand this expression further because it does not contain $H_0$.

To bring the above to matrix notation (once again, we want to rely on `cauchy_dot_product`) we first define
$$
\mathcal{X} = \mathcal{Y} + \mathcal{Z} = [\mathcal{U}, H_0] = [\mathcal{U}', H_0].
$$
Its diagonal blocks are $\mathcal{Z}$, while its off-diagonal blocks are $\mathcal{Y}$ (no $-$ sign between the blocks because $\mathcal{Y}$ is Hermitian!), similar to the blocks of $\mathcal{U}'$ being $\mathcal{W}$ and $\mathcal{V}$. In terms of $\mathcal{X}$, $\tilde{\mathcal{H}}_0$ becomes
$$
\tilde{\mathcal{H}}_0 = H_0 - \mathcal{U}'^\dagger \mathcal{X} - \mathcal{X}.
$$

The above defines the full algorithm.
1. Define series $\mathcal{U}$, $\mathcal{U}'$, $\mathcal{X}$.
2. Use their block and hermiticity structure to define their adjoints.
3. Use $\mathcal{W} = -\mathcal{U}'^\dagger\mathcal{U}'/2$ to define diagonal blocks of $\mathcal{U}.
4. Use $\mathcal{V} = S(\mathcal{Y})$ (where $S$ is the solution of the Sylvester equation) for offdiagonal blocks of $\mathcal{U}$.
5. Use $\mathcal{Z} = (\mathcal{U}'^\dagger\mathcal{X} - h.c.)/2$ for diagonal blocks of $\mathcal{X}$ (note: I've obtained this by messing around with the blockwise expression, there may be an easy way to see this).
6. Use $\mathcal{Y} = (-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{U}^\dagger\mathcal{H}'\mathcal{U})_{\textrm{offdiag}}$
7. Use $\tilde{\mathcal{H}} = H_0 + (- \mathcal{U}'^\dagger \mathcal{X} - \mathcal{X} + \mathcal{U}^\dagger\mathcal{H}'\mathcal{U})_\mathcal{diag}$.
