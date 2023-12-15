# Benchmark

## Benchmark against other algorithms

**The main computational advantage of Pymablock vs SW comes from the
parametrization of the unitary transformation.**
Despite the similarity between Pymablock's algorithms and a Schrieffer-Wolff
transformation, choosing a polynomial parametrization for the unitary
transformation brings a significant computational advantage.
In the former, the transformed Hamiltonian is given by:

:::{math}
\begin{equation}
\tilde{H} = e^S H e^{-S}
\end{equation}
:::
where $S$ is a polynomial in the perturbative parameter.
As a consequence, every new order of $S$ is determined by a recursive relation
whose terms contain nested commutators, making the number of matrix products
grow exponentially with the order.
Moreover, the transformed Hamiltonian is also given by a series of nested
commutators
:::{math}
\begin{equation}
\tilde{H} = \sum_{j=0}^\infty \frac{1}{j!} [H, S]^{(j)},
\end{equation}
:::
replicating the same problem.
This expression also requires truncating the series at the same order
to which $S$ is computed, which is a waste of computational resources.
Finally, generalizing the Schrieffer-Wolff transformation to multiple
perturbations is only straightforward if the perturbations are bundled
together.
However, this makes it impossible to request specific orders, making it
necessary to compute more terms than needed.

**These concerns were shared by others, but neither of the proposed solutions
is as efficient and versatile as Pymablock.**

...
Finally, it is common (????) for many-body Hamiltonians to need a generator
for the unitary transformation, something that Pymablock does not require.
Instead, Pymablock can use a custom `solve_sylvester` function with
second quantization operators to carry out the block-diagonalization.
(This is not because of the parametrization, but should probably mention it
somewhere)

Alternatives to Pymablock:
- Boxi's paper
- Density matrix PT (?)


## Time scaling

Showing scaling for large implicit Hamiltonians.

**To demonstrate the efficiency of the implicit algorithm, we show its time
scaling as a function of Hamiltonian size.**
Do we plot the Kwant tutorial here? Is there a way to count matrix products for
example? maybe using a counter in the code?


## Error scaling

Show error accumulation, show that the inverse of the transformation holds to numerical precision.
