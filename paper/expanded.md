# How does `expanded` work

This is a derivation of the expresisons in the `expanded` algorithm. Compared to the notation so far we will use a slightly different parametrization of $\mathcal{U} = 1 + \mathcal{W} + \mathcal{V}$, where $\mathcal{W} = \mathcal{W}^\dagger$ and $\mathcal{V} = -\mathcal{V}^\dagger$ as before. This ensures that $(\mathcal{W}^2)_n$ does not contain $W_n$.

Unitarity reads
$$
(1 + \mathcal{W} - \mathcal{V}) (1 + \mathcal{W} + \mathcal{V}) = 1,
$$
so that after adding the conjugate like before we get
$$
2\mathcal{W} + \mathcal{W}^2 - \mathcal{V}^2 = 0,
$$
which we knew. Subtracting the conjugate, on the other hand, gives $[\mathcal{W}, \mathcal{V}] = 0$, which is a bit of a new insight.

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
Our goal is to manipulate the above expression so that the only terms containing $H_0$ are of the form $[\mathcal{V}, H_0] \equiv \mathcal{Y}$.
We start by collecting the terms containing $H_0$ in the above expression
$$
\tilde{\mathcal{H}_0} = H_0 + \mathcal{Y} + \{\mathcal{W}, H_0\} + (\mathcal{W} - \mathcal{V})H_0(\mathcal{W} + \mathcal{V}).
$$
We want to eliminate $H_0$ from the last two terms. In other words, we want to find an expression for these that depends on $\mathcal{Y}$ and maybe other series that depend on $\mathcal{Y}$, $\mathcal{W}$, or $\mathcal{V}$, but do not contain $H_0$.

In the code we eliminated $H_0$ by commuting it to the right. Let us do so formally by introducing $\mathcal{Z} \equiv [\mathcal{W}, H_0]$.
Commuting $H_0$ to the right we get:
$$
\begin{align*}
\tilde{\mathcal{H}}_0 &= H_0 + \mathcal{Y} + 2\mathcal{W}H_0 + \mathcal{Z} + \mathcal{W}(\mathcal{W}H_0 + \mathcal{Z}) - \mathcal{V}(\mathcal{W}H_0 + \mathcal{Z}) \\
&-\mathcal{V}(\mathcal{V} H_0 - \mathcal{Y}) + \mathcal{W}(\mathcal{V} H_0 - \mathcal{Y}).
\end{align*}
$$
Examining all the terms that still contain $H_0$ we observe that they cancel:
$$
\left(2\mathcal{W} + \mathcal{W}^2 - \mathcal{V}\mathcal{W} - \mathcal{V}^2 + \mathcal{W}\mathcal{V}\right)H_0 = 0
$$
by unitarity.
Therefore we get
$$
\tilde{\mathcal{H}}_0 = (H_0 + \mathcal{W}\mathcal{Z} -\mathcal{V}\mathcal{Y} + \mathcal{Z}) + (\mathcal{Y}  - \mathcal{V}\mathcal{Z} - \mathcal{W}\mathcal{Y}),
$$
where the first terms are block-diagonal, and therefore do not enter the equation for $\mathcal{Y}$. The terms $\mathcal{V}\mathcal{Z}$ and $\mathcal{W}\mathcal{Y}$ don't depend on $Y_n$, and therefore are fine to use in the recursive scheme.

We still have a problem of cheaply evaluating $\mathcal{Z}$ without multiplications by $H_0$. To do this we apply the same trick as we did for unitarity: we substitute the recursive definition for $\mathcal{W}$ into the definition of $\mathcal{Z}$:
$$
-2\mathcal{Z} = -2[\mathcal{W}, H_0] = [\mathcal{W}^2 - \mathcal{V}^2, H_0] = \mathcal{W}^2H_0 - H_0\mathcal{W}^2 - \mathcal{V}^2H_0 + H_0\mathcal{V}^2 = \mathcal{W}(H_0 \mathcal{W} - \mathcal{Z})-
(\mathcal{W} H_0 + \mathcal{Z})\mathcal{W}\\
- \mathcal{V}(H_0 \mathcal{V} + \mathcal{Y}) - (\mathcal{V} H_0 - \mathcal{Y})\mathcal{V} = \{\mathcal{W},\mathcal{Z}\} - \{\mathcal{V}, \mathcal{Y}\}.
$$
The right hand side has no terms that contain $Z_n$ or $Y_n$, and therefore it is a cheap definition of $\mathcal{Z}$.
