# Benchmark

## Benchmark against other algorithms

**Pymablock's algorithm does not use the Schrieffer-Wolff transformation,
because the former is inefficient.**
The difference between Pymablock's algorithm and the Schrieffer-Wolff
lies in the parametrization of the unitary transformation.
In the former, the transformed Hamiltonian is given by:

:::{math}
\begin{equation}
\tilde{H} = e^S H e^{-S}
\end{equation}
:::
where $S$ is a polynomial series in the perturbative parameter.
As a consequence, every new order of $S$ is determined by a recursive relation
whose terms contain nested commutators,
making the number of matrix products grow exponentially with the order.
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
However, this makes it impossible to request individual order combinations,
making it necessary to compute more terms than needed.

**The main computational advantage of Pymablock comes from the
parametrization of the unitary transformation, but the parametrization
comes at a cost.**
Despite the similarity between Pymablock's algorithms and a Schrieffer-Wolff
transformation, choosing a polynomial parametrization for the unitary
transformation reduces the scaling from exponential to linear.
The effective Hamiltonian is equivalent to that of a Schrieffer-Wolff
transformation, but the terms that compose it are more than those strictly
required.
This is because Sylvester's equation is solved more than once per new order,
requiring additional numerators that cancel out in the final expression.
This cancellation can be costly for symbolic computations, and it is for
this reason that Pymablock also has an `expanded` algorithm, which
only solves Sylvester's equation once per order.
We leave for future work finding a parametrization that achieves the same
computational efficiency as the `general` algorithm, but that does not require
simplifications of the transformed Hamiltonian.

**Pymablock is not only efficient, but its implementation has potential
to be expanded to other settings, like time-dependent Hamiltonians, many-body
Hamiltonians, and continuum Hamiltonians.** _maybe??_ or maybe in conclusion??
Still, Pymablock supports inputs a variety of inputs, including like fermionic
and bosonic second quantization operators without using matrix representations.
Therefore, Pymablock can compute effective Hamiltonians for interacting
systems, infinitely-sized Hilbert spaces, and Hamiltonians with continuum
degrees of freedom, by defining a custom `solve_sylvester` function.
This is advantageous over other methods, which are limited to pre-defining
an appropriate generator for the unitary transformation.
Similarly, this flexibility allows Pymablock to work with time-dependent
Hamiltonians, an extension that we leave for future work.

## Time scaling

**To demonstrate the efficiency of the implicit algorithm, we show its time
scaling as a function of Hamiltonian size.**
Do we plot the Kwant tutorial here? Is there a way to count matrix products for
example? maybe using a counter in the code?
Showing scaling for large implicit Hamiltonians.


## Error scaling

Show error accumulation, show that the inverse of the transformation holds to numerical precision.
