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
whose terms contain nested commutators ,
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

**A SW transformation is not the only available alternative to quasi-degenerate
perturbation theory, but the others miss versatility and efficiency too.**
The reformulation of the Schrieffer-Wolff transformation into alternative
parametrizations is not new [Klein1974][10.1063/1.440050],
[Suzuki1983](doi:10.1143/PTP.70.439) [Shavitt2008](doi:10.1063/1.440050), and
the Schrieffer-Wolff transformation itself is known as the canonical Van Vleck
formalism [VanVleck1929](doi:10.1103/PhysRev.33.467), or Löwdin partitioning
[Lowdn1964](doi:10.1063/1.1724312).
However, these approaches were tailored to specific settings,
like many-body Hamiltonians, where a generator for the unitary transformation
or a Green's function is often needed, or to answer general questions about
convergence of the resulting series and equivalence to other methods.
Recently, an alternative recursive procedure was introduced by Ref.
[Li2022](doi:10.1103/PRXQuantum.3.030313), with the goal of improving the
scaling of diagonalization as a function of the perturbative order.
Even though, ....

**Pymablock is not only efficient, but its implementation has potential
to be expanded to other settings, like time-dependent Hamiltonians, many-body
Hamiltonians, and continuum Hamiltonians.** _maybe??_ or maybe in conclusion??

## Time scaling

Showing scaling for large implicit Hamiltonians.

**To demonstrate the efficiency of the implicit algorithm, we show its time
scaling as a function of Hamiltonian size.**
Do we plot the Kwant tutorial here? Is there a way to count matrix products for
example? maybe using a counter in the code?


## Error scaling

Show error accumulation, show that the inverse of the transformation holds to numerical precision.
