# How does `expanded` work

This is a derivation of the expresisons in the `expanded` algorithm. Compared to the notation so far we will use a slightly different parametrization of $\mathcal{U} = 1 + \mathcal{W} + \mathcal{V}$, where $\mathcal{W} = \mathcal{W}^\dagger$ and $\mathcal{V} = -\mathcal{V}^\dagger$ as before. This ensures that $(\mathcal{W}^2)_n$ does not depend on $W_n$.

Unitarity reads
$$
(1 + \mathcal{W} - \mathcal{V}) (1 + \mathcal{W} + \mathcal{V}) = 1,
$$
so that after adding the conjugate like before we get
$$
2\mathcal{W} + \mathcal{W}^2 - \mathcal{V}^2 = 0,
$$
which we knew. But also this means
$$
\mathcal{W} = \sqrt{1 + \mathcal{V}^2} - 1 \equiv f(V) \equiv \sum_n a_n V^{2n}.
$$
We will need this expression for a series expansion later. Additionally we see that $[\mathcal{W}, \mathcal{V}] = 0$

Now let us look at the unitary transformation of the Hamiltonian
$$
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U} = (1 + \mathcal{W} - \mathcal{V}) (H_0 + \mathcal{H'}) (1 + \mathcal{W} + \mathcal{V}).
$$
Our goal is to manipulate the above expression so that the only terms containing $H_0$ are of the form $[\mathcal{V}, H_0] \equiv \mathcal{Y}$. Because $\mathcal{Y}$ has no 0th order term, an expression containing higher powers of $\mathcal{Y}$ is evaluatable recursively without any problem.

We start by collecting the terms containing $H_0$ in the above expression
$$
\tilde{\mathcal{H}_0} = H_0 + \mathcal{Y} + \{\mathcal{W}, H_0\} + (\mathcal{W} - \mathcal{V})H_0(\mathcal{W} + \mathcal{V}).
$$
We want to eliminate $H_0$ from the last two terms. In the algorithm we did it by commuting $H_0$ to the right, let's do the same and apply the series expansion of $\mathcal{W}$ and the definition of $Y$ to do so.
$$
H_0 \mathcal{W} = \sum_n a_n H_0 \mathcal{V}^{2n}
$$
Let's examine $H_0 V^{2n}$. By applying $H_0 \mathcal{V} = \mathcal{V} H_0 - \mathcal{Y}$ $2n$ times we get
$$
H_0 \mathcal{V}^{2n} = (\mathcal{V} H_0 - \mathcal{Y})V^{2n-1}=(\mathcal{V}^2 H_0-\mathcal{V}\mathcal{Y}-\mathcal{Y}\mathcal{V})\mathcal{V}^{2n-2}=\ldots=(\mathcal{V}^{2n}H_0 - \sum_{i=0}^{2n-1}\mathcal{V}^{2n-i-1}\mathcal{Y}\mathcal{V}^{i})
$$
We can now substitute this expression into the above series expansion of $H_0 \mathcal{W}$ to get
$$
\begin{align*}
H_0 \mathcal{W} &= \sum_n a_n (\mathcal{V}^{2n}H_0 - \sum_{i=0}^{2n-1}\mathcal{V}^{2n-i-1}\mathcal{Y}\mathcal{V}^{i}) \equiv \mathcal{W} H_0 + \mathcal{F}.
\end{align*}
$$
We now get back to $\tilde{\mathcal{H}}_0$:
$$
\begin{align*}
\tilde{\mathcal{H}}_0 &= H_0 + \mathcal{Y} + 2\mathcal{W}H_0 + \mathcal{F} + \mathcal{W}(\mathcal{W}H_0 + \mathcal{F}) - \mathcal{V}(\mathcal{W}H_0 + \mathcal{F}) \\
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
\tilde{\mathcal{H}}_0 = (H_0 + \mathcal{W}\mathcal{F} -\mathcal{V}\mathcal{Y} + \mathcal{F}) + (\mathcal{Y}  - \mathcal{V}\mathcal{F} - \mathcal{W}\mathcal{Y}),
$$
where the first terms are block-diagonal, and therefore do not enter the equation for $\mathcal{Y}$.

Can we evaluate $\mathcal{F}$ quickly by using a recursive definition similar to $\mathcal{W}$? Let's try the recursive definition for $\mathcal{W}$ and substitute it into the definition of $\mathcal{F}$:
$$
-2[\mathcal{W}, H_0] = [\mathcal{W}^2 - \mathcal{V}^2, H_0] = \mathcal{W}^2H_0 - H_0\mathcal{W}^2 - \mathcal{V}^2H_0 + H_0\mathcal{V}^2 = \mathcal{W}(H_0 \mathcal{W} - \mathcal{F})-
(\mathcal{W} H_0 + \mathcal{F})\mathcal{W}\\
- \mathcal{V}(H_0 \mathcal{V} + \mathcal{Y}) - (\mathcal{V} H_0 - \mathcal{Y})\mathcal{V} = \{\mathcal{W},\mathcal{F}\} - \{\mathcal{V}, \mathcal{Y}\} = -2\mathcal{F}
$$
