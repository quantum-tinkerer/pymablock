# Further optimization attempt

Pymablock algorithm is as follows

- Define series $\mathcal{U}'$ and $\mathcal{X}$ and make use of their block structure and Hermiticity.
- To define the diagonal blocks of $\mathcal{U}'$, use $\mathcal{W} = -\mathcal{U}'^\dagger\mathcal{U}'/2$.
- To find the off-diagonal blocks of $\mathcal{U}'$, solve Sylvester's equation $\mathcal{V}^{AB}H_0^{BB} - H_0^{AA}\mathcal{V}^{AB} = \mathcal{Y}^{AB}$. This requires $\mathcal{X}$.
- To find the diagonal blocks of $\mathcal{X}$, define $\mathcal{Z} = (-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{X}^\dagger\mathcal{U}')/2$.
- For the off-diagonal blocks of $\mathcal{X}$, use $\mathcal{Y}^{AB} = (-\mathcal{U}'^\dagger\mathcal{X} + \mathcal{U}^\dagger\mathcal{H}'\mathcal{U})^{AB}$.
- Compute the effective Hamiltonian as $\tilde{\mathcal{H}}_{\textrm{diag}} = H_0 - \mathcal{X} - \mathcal{U}'^\dagger \mathcal{X} + \mathcal{U}^\dagger\mathcal{H'}\mathcal{U}$.

The more complete expressions for the series are

$$
\mathcal{A} = \mathcal{H}'_\textrm{diag} \mathcal{U}', \quad
\mathcal{B} = \mathcal{H}'_\textrm{offdiag} \mathcal{U}', \quad
\mathcal{C} = \mathcal{X} - \mathcal{H}'_\textrm{offdiag}.
$$

This gives an updated expression for $\mathcal{Z}$:

$$
\mathcal{Z} = \frac{1}{2}(\mathcal{B}^\dagger - \mathcal{U}^\dagger\mathcal{C}) - \textrm{h.c.},
$$

and finally $\tilde{\mathcal{H}}$:

$$
\tilde{\mathcal{H}} = H_0 + \mathcal{A} + \mathcal{A}^\dagger + (\mathcal{B} + \mathcal{B}^\dagger)/2 + \mathcal{U}'^\dagger (\mathcal{A} + \mathcal{B}) - (\mathcal{U}^\dagger \mathcal{C} + \textrm{h.c.})/2.
$$

We observe that with one perturbation $W_2$ drops out of $\tilde{H}_3$ as well as additional terms.
Instead, the expression for $\tilde{H}_3$ simplifies to

$$
\tilde{H}_3 = (\mathcal{B} + \mathcal{B}^\dagger)/2.
$$

To identify the source of the cancellation, we work out various terms in $\tilde{H}_3$.

$$
\begin{aligned}
\mathcal{A} &=
\end{aligned}
$$
