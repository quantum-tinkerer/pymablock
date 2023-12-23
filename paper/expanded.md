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

For the actual implementation we favor not separating $\mathcal{U}$ into different blocks to rely on `cauchy_dot_product`. This gives
$$
\mathcal{W} = (\mathcal{U}' + \mathcal{U}'^\dagger)/2 = -\mathcal{U}'^\dagger\mathcal{U}'/2\textrm{\quad or\quad} \mathcal{U}' + \mathcal{U}'^\dagger + \mathcal{U}'^\dagger\mathcal{U}' = 0.
$$

Alternatively, we could express $\mathcal{W}$ through $\mathcal{V}$:
$$
\mathcal{W} = \sqrt{1 + \mathcal{V}^2} - 1 \equiv f(\mathcal{V}) \equiv \sum_n a_n \mathcal{V}^{2n}.
$$
Notice that evaluating $2\mathcal{W} = -\mathcal{U}'^2$ is cheap and recursion causes no problems because $\mathcal{W}$ has no 0th order term.
Computing a series expansion, on the other hand, has a higher scaling of its complexity.

Now let us look at the unitary transformation of the Hamiltonian
$$
\tilde{\mathcal{H}} = \mathcal{U}^\dagger \mathcal{H} \mathcal{U} = (1 + \mathcal{U}'^\dagger) (H_0 + \mathcal{H'}) (1 + \mathcal{U}').
$$
Our goal is to define a recursive definition for $\mathcal{U}'$ that does not involve $H_0$ explicitly except for solving Sylvester equation.
We do so by defining the commutator $\mathcal{X} \equiv [\mathcal{U}', H_0]$ as an auxiliary varitable.
Somewhat similarly to $\mathcal{U}'$, $\mathcal{X} = \mathcal{Y} + \mathcal{Z}$, where $\mathcal{Y}$ is Hermitian offdiagonal and $\mathcal{Z}$ is anti-Hermitian diagonal.
Want to eliminate products of $H_0$ from $\tilde{\mathcal{H}}$. In other words, we want to find an expression for these that depends on $\mathcal{X}$ and $\mathcal{U}'$, but do not contain $H_0$.

In the code we eliminated $H_0$ by commuting it to the right, let us try doing the same here:
$$
\begin{align*}
\mathcal{U}^\dagger H_0 \mathcal{U}
&= H_0 + \mathcal{U}'^\dagger H_0 + H_0 \mathcal{U}' + \mathcal{U}'^\dagger H_0 \mathcal{U}'\\
&= H_0 + \mathcal{U}'^\dagger H_0 + \mathcal{U}'H_0 - \mathcal{X} + \mathcal{U}'^\dagger (\mathcal{U}' H_0 - \mathcal{X})\\
&= H_0 + (\mathcal{U}'^\dagger + \mathcal{U}' + \mathcal{U}'^\dagger \mathcal{U}')H_0 - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X}\\
&= H_0 - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X},
\end{align*}
$$
Where the terms multiplied by $H_0$ cancel by unitarity.
The expression for $\tilde{\mathcal{H}}_0$ does not appear Hermitian by construction, although this is an artifact of the way we eliminated $H_0$, and has no practical consequences, although it may be symmetrized for numerical stability.

The offdiagonal blocks of $\mathcal{X}$ are defined by $\tilde{\mathcal{H}}$ being block-diagonal, so
$$
\mathcal{Y} = (\mathcal{U}^\dagger \mathcal{H}' \mathcal{U} - \mathcal{U}'^\dagger \mathcal{X})_{\textrm{offdiag}}.
$$

We still have a problem of cheaply evaluating $\mathcal{Z}$ without multiplications by $H_0$.
Here is the solution:
$$
\begin{align*}
2 \mathcal{Z}
&= \mathcal{X} - \mathcal{X}^\dagger = \mathcal{U}'H_0 - H_0\mathcal{U}' + \mathcal{U}'^\dagger H_0 - H_0 \mathcal{U}'^\dagger\\
&=
    -\frac{1}{2}(\mathcal{U}'^\dagger\mathcal{U}' + \mathcal{U}'\mathcal{U}'^\dagger)H_0
    + \frac{1}{2} H_0(\mathcal{U}'\mathcal{U}'^\dagger + \mathcal{U}'^\dagger\mathcal{U}')\\
&=
    -\frac{1}{2}(\mathcal{U}'^\dagger\mathcal{U}'H_0 - H_0\mathcal{U}'\mathcal{U}'^\dagger)
    + \frac{1}{2} (-\mathcal{U}'\mathcal{U}'^\dagger H_0 + H_0\mathcal{U}'^\dagger\mathcal{U}')\\
&=
    -\frac{1}{2}(\mathcal{U}'^\dagger\mathcal{U}'H_0 - H_0\mathcal{U}'\mathcal{U}'^\dagger)
    + \frac{1}{2} (-H_0\mathcal{U}'\mathcal{U}'^\dagger + \mathcal{U}'^\dagger\mathcal{U}'H_0)^\dagger\\
&=
    -\frac{1}{2}(\mathcal{U}'^\dagger[H_0\mathcal{U}'-\mathcal{X}] - [\mathcal{U}'H_0 + \mathcal{X}]\mathcal{U}'^\dagger)
    + \frac{1}{2} (-[\mathcal{U}'H_0 + \mathcal{X}]\mathcal{U}'^\dagger + \mathcal{U}'^\dagger[H_0\mathcal{U}'-\mathcal{X}])^\dagger\\
&=
    \frac{1}{2}\{\mathcal{U}'^\dagger,\mathcal{X}\} - \textrm{h.c.}
\end{align*}
$$
Here for the third identity we used unitarity, and in the follow-up identities we manipulated the expressions to bring $H_0$ to the middle.
This derivation is unintuitive, but combines ideas from the recursive definition of $\mathcal{U}'$ and the elimination of $H_0$ from $\tilde{\mathcal{H}}$.

The above defines the full algorithm.
1. Define series $\mathcal{U}$, $\mathcal{U}'$, $\mathcal{X}$.
2. Use their block and hermiticity structure to define their adjoints.
3. Use $\mathcal{W} = -\mathcal{U}'^\dagger\mathcal{U}'/2$ to define diagonal blocks of $\mathcal{U}.
4. Use $\mathcal{V} = S(\mathcal{Y})$ (where $S$ is the solution of the Sylvester equation) for offdiagonal blocks of $\mathcal{U}$.
5. Use $\mathcal{Z} = (\{\mathcal{U}'^\dagger,\mathcal{X}\} - h.c.)/4$ for diagonal blocks of $\mathcal{X}$.
6. Use $\mathcal{Y} = (-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{U}^\dagger\mathcal{H}'\mathcal{U})_{\textrm{offdiag}}$
7. Use $\tilde{\mathcal{H}} = H_0 + (- \mathcal{U}'^\dagger \mathcal{X} - \mathcal{X} + \mathcal{U}^\dagger\mathcal{H}'\mathcal{U})$.
