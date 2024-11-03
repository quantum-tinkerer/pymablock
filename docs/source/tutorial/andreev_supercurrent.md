---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Andreev Supercurrent

In this tutorial, we demonstrate how to use Pymablock to compute complicated analytical expressions.
Directly running Pymablock on a large symbolic Hamiltonian can computationally expensive, and simplifying the inputs and outputs is crucial to obtaining interpretable results in a reasonable amount of time.
Both of these steps benefit from physical insight and advanced manipulation of symbolic expressions.

As an example, we compute supercurrent between two superconductors weakly coupled through a quantum dot.

![Two superconductors and a quantum dot](superconductors_quantum_dot.svg)

We will compute the supercurrent by treating the tunneling to the quantum dot as a perturbation.
This requires calculating and manipulating fourth-order corrections to the ground state energy [^1^].

[^1^]: [Glazman et al.](http://jetpletters.ru/ps/1121/article_16988.pdf)

## Define the Hamiltonian

The Hamiltonian of the system is given by

$$
H = H_{\textrm{SC}}+ H_{\textrm{dot}} + H_{T}
$$

where the Hamiltonians of the superconductors, of the quantum dot, and of the tunnel coupling are

$$
H_{\textrm{SC}} =  \sum_{\alpha=L, R} \xi_{\alpha} \left(
    n_{\alpha \uparrow} + n_{\alpha \downarrow} \right)
+ \Gamma_{\alpha} \left( c_{\alpha, \uparrow}^\dagger c_{\alpha \downarrow}^\dagger + c_{\alpha \downarrow} c_{\alpha, \uparrow} \right), \\
H_{\textrm{dot}} = \frac{U}{2} \left( n_{\uparrow} + n_{\downarrow} - N \right)^2. \\
H_{T} = \sum_{\alpha=L,R} t_\alpha \left( c_{\alpha \uparrow}^\dagger d_{\uparrow} + c_{\alpha \downarrow}^\dagger d_{\downarrow} \right) + \textrm{h.c.}.
$$

Here $c_{\alpha, \sigma}$ and $d_{\sigma}$ are the annihilation operators of electrons in the superconductors and quantum dot, respectively, with $\sigma = \uparrow, \downarrow$.
The superconductors have a superconducting pairing $\Gamma_{\alpha}$ and onsite energy $\xi_{\alpha}$.
They are weakly coupled to the quantum dot, which has charging energy $U$ and offset number of electrons $N$.
The couplings, $t_{L}$ and $t_{R}$, include the phase difference $\phi$ between the superconductors, $t_{L} = \lvert t_{L} \rvert e^{i \phi}$ and $t_{R} = \lvert t_{R} \rvert$.
We treat both of them as independent perturbations.

We start by importing the necessary libraries and defining the symbols and operators in the Hamiltonian.

```{code-cell} ipython3
from IPython.display import display
import numpy as np
import sympy
from sympy import exp, I, symbols, Symbol, Eq
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger

from pymablock import block_diagonalize

# Define symbols
U, N = symbols(r"U N", positive=True)
xis = xi_L, xi_R = symbols(r"\xi_L \xi_R", positive=True)
Gammas = Gamma_L, Gamma_R = symbols(r"\Gamma_L \Gamma_R", positive=True)
ts = t_L_complex, t_R = Symbol("t_Lc", real=False), Symbol("t_R", real=True)

# Dot operators
c_up, c_down = FermionOp(r'c_{d, \uparrow}'), FermionOp(r'c_{d, \downarrow}')

# Superconductor operators
d_ups = FermionOp(r'd_{L, \uparrow}'), FermionOp(r'd_{R, \uparrow}')
d_downs = FermionOp(r'd_{L, \downarrow}'), FermionOp(r'd_{R, \downarrow}')

# Only used for printing. Because sympy forces lexicographic ordering,
# we prepend {} for h.c. to appear last.
hc = sympy.Symbol(r"{}\textrm{h.c.}")

def n(op):
    """Shorthand for Dagger(op) * op"""
    return Dagger(op) * op

def display_eq(title, expr):
    """Print a sympy expression as an equality."""
    display(Eq(sympy.Symbol(title), expr))
```

Here we use `t_L_complex` to avoid complications with simplification routines in
`sympy`, so that $\exp i \phi$ is not expanded into a sine and cosine.

Next, we define the Hamiltonians of quantum dot and tunneling.

```{code-cell} ipython3
# Quantum Dot Hamiltonian
H_dot = U * (n(c_up) + n(c_down) - N)**2 / 2

# Tunneling Hamiltonian
H_T = sum(t * (Dagger(c_up) * d_up + Dagger(c_down) * d_down)
    for t, d_up, d_down in zip(ts, d_ups, d_downs)
)  # + h.c. added later
```

```{code-cell} ipython3
:tags: [hide-cell]

display_eq("H_{dot}", H_dot)
display_eq("H_{T}", H_T + hc)
```

In this basis $H_{\textrm{dot}}$ is diagonal and its subspaces may be directly constructed, without performing a symbolic diagonalization.
This is, however, not the case for $H_{SC}$.
Therefore, we apply the Bogoliubov transformation on $H_{SC}$, such that the entire unperturbed Hamiltonian $H_0 = H_{SC} + H_{\textrm{dot}}$ is diagonal.

### Apply the Bogoliubov transformation

We define the superconductors' Hamiltonian using the Bogoliubov quasi-particle operators $f_{\alpha, \sigma}$, which are related to the original operators $c_{\alpha, \sigma}$ by the [Bogoliubov transformation](https://en.wikipedia.org/wiki/Bogoliubov_transformation):

$$
f_{\alpha, \uparrow} = u_\alpha c_{\alpha, \uparrow} + v_\alpha c_{\alpha, \downarrow} \\
f_{\alpha, \downarrow} = u_\alpha c_{\alpha, \downarrow} - v_\alpha c_{\alpha, \uparrow}
$$

where $u_\alpha$ and $v_\alpha$ are complex coefficients that satisfy $u_\alpha^2 + v_\alpha^2 = 1$.
As a result,

$$
H_{SC} = \sum_{\alpha=L, R} \xi_{\alpha} - E_{\alpha} + E_{\alpha} \left(
f_{\alpha, \uparrow}^\dagger f_{\alpha, \uparrow} + f_{\alpha,
\downarrow}^\dagger f_{\alpha, \downarrow} \right),
$$

where $E_{\alpha} = \sqrt{\Gamma_{\alpha}^2 + \xi_{\alpha}^2}$ are the Andreev bound state energies, and $\lvert u_{\alpha} \rvert = \sqrt{\frac{E_{\alpha} + \xi_{\alpha}}{2 E_{\alpha}}}$ and $\lvert v_{\alpha} \rvert = \sqrt{\frac{E_{\alpha} - \xi_{\alpha}}{2 E_{\alpha}}}$ are the Bogoliubov coefficients.

```{code-cell} ipython3
# Superconductors' energies
Es = E_L, E_R = symbols(r"E_L E_R", positive=True)

# Bogoliubov quasiparticle operators
f_ups = FermionOp('f_{L, \\uparrow}'), FermionOp('f_{R, \\uparrow}')
f_downs = FermionOp('f_{L, \\downarrow}'), FermionOp('f_{R, \\downarrow}')

# Superconductors' Hamiltonian
H_sc = sum(
    xi - E + E * n(f_up) + E * n(f_down)
    for xi, E, f_up, f_down in zip(xis, Es, f_ups, f_downs)
)
```

```{code-cell} ipython3
:tags: [hide-cell]

display_eq("H_{SC}", H_sc)
```

:::{admonition} Avoid symbolic diagonalization
:class: dropdown tip

Diagonalizing a large symbolic Hamiltonian is computationally expensive, and in many cases impossible.
To alleviate this, we have used physical insight: the Bogoliubov transformation allows us to get a diagonal unperturbed Hamiltonian.
:::

Similarly, we apply the Bogoliubov transformation to the tunneling Hamiltonian.

```{code-cell} ipython3
# Bogoliubov coefficients
us = u_L, u_R = symbols(r"u_L u_R", real=True)
vs = v_L, v_R = symbols(r"v_L v_R", real=True)

# Bogoliubov transformation from d operators to f operators
d_subs = {}
for u, v, d_down, d_up, f_down, f_up in zip(us, vs, d_downs, d_ups, f_downs, f_ups):
    d_subs[d_up] = u * f_up - v * Dagger(f_down)
    d_subs[d_down] = u * f_down + v * Dagger(f_up)
    d_subs[Dagger(d_up)] = Dagger(d_subs[d_up])
    d_subs[Dagger(d_down)] = Dagger(d_subs[d_down])

# Substitute d operators with f operators
H_T = H_T.subs(d_subs).expand()  # + h.c., expand to open up parentheses

# Total Hamiltonian
H = H_sc + H_dot + H_T + Dagger(H_T)
```

```{code-cell} ipython3
:tags: [hide-cell]

display_eq("H_{T}", H_T + hc)
```

We have now defined the total Hamiltonian $H$ in second quantization form.

:::{admonition} Avoid square roots
:class: dropdown tip

Using square roots can lead to complicated expressions, because assumptions about the arguments of square roots are not automatically inferred by sympy.
For example, $\sqrt{a^2}$ is not equivalent to $a$, but rather $\lvert a \rvert$.
To avoid lengthy expressions from unsimplified expressions with square roots, we have directly replaced thems with the sympy symbols $E_{\alpha}$, $u_{\alpha}$, and $v_{\alpha}$.
These will appear in the effective Hamiltonian, and we will substitute their values at the end of the calculation.
:::

### Convert the Hamiltonian to a matrix

Pymablock does not yet support working directly with operators. In the [Jaynes–Cummings tutorial](jaynes_cummings.md) we implemented a custom `solve_sylvester` function. Here the simplest option is to convert the Hamiltonian to a matrix representation.
The following code cell defines a function `to_matrix(...)` that computes the matrix representation of a Hamiltonian `H` with fermionic operators and the corresponding `basis`.
The details of the implementation are hidden for brevity.

```{code-cell} ipython3
:tags: [hide-cell]

from itertools import combinations

def to_matrix(H):
    """Compute a matrix representation of a sympy expression with fermion operators."""
    # Add an identity operator to all symbols so that we always work with operators
    H = H.subs({
        s: sympy.physics.quantum.IdentityOperator() * s for s in H.free_symbols
        if not isinstance(s, sympy.physics.quantum.Operator)
    })
    # Choose an order of fermionic operators
    fermions = [
        s for s in H.free_symbols
        if (
            isinstance(s, sympy.physics.quantum.fermion.FermionOp)
            and s.is_annihilation
        )
    ]
    fermions.sort(key=lambda f: f.name.name) # Sort by label to ensure consistent order
    # Compute matrix representations
    s_minus = sympy.Matrix([[0, 1], [0, 0]])
    s_z = sympy.Matrix([[1, 0], [0, -1]])
    s_0 = sympy.eye(2)
    matrix_subs = {
        op: sympy.kronecker_product(*(i * [s_z] + [s_minus] + (len(fermions) - i - 1) * [s_0]))
        for i, op in enumerate(fermions)
    }
    matrix_subs.update({Dagger(op): Dagger(mat) for op, mat in matrix_subs.items()})
    matrix_subs[sympy.physics.quantum.IdentityOperator()] = sympy.eye(2**len(fermions))

    # Generate basis
    basis = [(sympy.S.One,)]
    for n in range(len(fermions)):
        basis.extend(list(combinations(fermions, n + 1)))
    reversed_basis = list(reversed(basis))
    reversed_basis[-1] = (sympy.S.Zero,)

    basis_matrices = []
    for b, nb in zip(basis, reversed_basis):
        expr = [Dagger(op) * op for op in b]
        expr.extend([sympy.physics.quantum.IdentityOperator()-Dagger(op) * op for op in nb])
        basis_matrices.append(sympy.Mul(*expr).expand())
    basis_matrices = [b.subs(matrix_subs, simultaneous=True).expand() for b in basis_matrices]
    basis_order = [np.nonzero(np.array(b.diagonal(), dtype=int)[0])[0][0] for b in basis_matrices]
    basis = [sympy.Mul(*basis[i]) for i in np.argsort(basis_order)]
    return H.subs(matrix_subs, simultaneous=True).expand(), basis
```

Next, we obtain the matrix Hamiltonian and its basis.

```{code-cell} ipython3
%%time
# Compute Hamiltonian in matrix form
H, basis = to_matrix(H)
```

At this point, we are ready to feed the $64 \times 64$ symbolic Hamiltonian to the block-diagonalization routine of Pymablock.
However, we anticipate that the diagonal elements of the Hamiltonian will appear in the denominators in the effective Hamiltonian.
These elements correspond to the dot energies $E_n = U (N - n)^2 / 2$ and the energies of the superconductors $E_{\alpha}$.
To make the denominators simpler, we replace the dot energies with $E_n$ right away.

```{code-cell} ipython3
t_L, phi, dphi = symbols(r"t_L \phi \delta\phi", positive=True)
H = H.subs({t_L_complex: t_L * exp(I * (phi + dphi))})

E_0, E_1, E_2 = symbols(r"E_0 E_1 E_2", positive=True)
E_0_value, E_1_value, E_2_value = [(U * (N - i)**2 / 2).expand() for i in range(3)]
H = H.subs({E_2_value: E_2}).subs({E_1_value: E_1}).subs({E_0_value: E_0})
```

```{code-cell} ipython3
odd = [bool(len(i.free_symbols) % 2) for i in basis]
even = [not i for i in odd]
basis_even = [i for i, parity in zip(basis, even) if parity]
basis_odd = [i for i, parity in zip(basis, odd) if parity]
H_even = H[even, even]
H_odd = H[odd, odd]
```

Finally, the Hamiltonian is ready for further analysis.

### Identify the ground states

Before computing the effective Hamiltonian, we need to identify the ground states of the unperturbed Hamiltonian $H_0$.
Numerically, we observe that the ground states depend on the number of electrons $N$ in the quantum dot.

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

# Values for the parameters
values = {
    U: 10,
    Gamma_L: 0.2,
    Gamma_R: 0.25,
    t_L: 0.1,
    t_R: 0.1,
    xi_L: 0.9,
    xi_R: -0.7,
    E_0: E_0_value,
    E_1: E_1_value,
    E_2: E_2_value,
    phi: np.pi/4,
    dphi: 0,
}

# Extract eigenvalues of the unperturbed Hamiltonian
N_values = np.linspace(-0.5, 2.5, 30)
eigenvalues = []
for N_value in N_values:
    values.update({
        N: N_value,
        E_L: np.sqrt(values[Gamma_L]**2 + values[xi_L]**2),
        E_R: np.sqrt(values[Gamma_R]**2 + values[xi_R]**2)
    })
    eigenvalues.append(np.array(H.diagonal().subs(values), dtype=complex)[0])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(N_values, np.array(eigenvalues).real, '-', alpha=0.5);
ax.set_ylim(-2, 4)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([r'$0$', r'$1$', r'$2$'])
ax.set_yticks([0, 2])
ax.set_yticklabels([r'$0$', r'$2$'])
ax.set_ylabel('$E$')
ax.set_xlabel(r'$N$')
ax.set_title(r'$H_0$ band structure')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
```

Therefore, we define $3$ ground states for the $N=0, 1, 2$ occupation numbers of the quantum dot.

## Compute the supercurrent perturbatively

Because the supercurrent is defined as

$$
I = \frac{e}{\hbar} \frac{dE}{d\phi}
$$

we need to find the perturbed ground state energy $E(\phi)$.
To do so, we block-diagonalize the Hamiltonian and decouple the subspace spanned by the ground states of the quantum dot.
We start by finding the corrections to the ground state for $N=0$, which is the vacuum state $\lvert 0\rangle$ of the quantum dot.

```{code-cell} ipython3
%%time

ground_state_n0 = [sympy.S.One]  # vacuum state
subspace_indices = [int(not(element in ground_state_n0)) for element in basis_even]
H_tilde = block_diagonalize(H_even, subspace_indices=subspace_indices, symbols=[t_L, t_R, dphi])[0]
```

Here `subspace_indices` is a list of $0$ and $1$ that labels the diagonal elements of $H_0$, so that the ground state subspace is labeled by $0$ and the rest by $1$.

Now we are ready to compute the perturbed ground state energy, of which several orders vanish.
The first non-zero correction to the ground state energy is given by the terms proportional to $t_L^2 t_R^2$.

```{code-cell} ipython3
%%time
current = sympy.trace(H_tilde[0, 0, 2, 2, 1]).doit().subs({dphi: 1})
```

We can now compute the supercurrent $I_c$ by taking the derivative of the ground state energy with respect to the phase difference $\phi$.

+++

Here we have computed the trace of `H_22` as a way to obtain the sum of the eigenvalues.
This is a $1 \times 1$ matrix, and the trace is the only element of this matrix, but we use the `sympy.Trace` function to make the code generalizable to larger matrices.
The result, however, is complicated and requires simplification.

```{code-cell} ipython3
display(Eq(Symbol('I_c(n=0)'), current))
```

:::{admonition} Do not call `simplify()` on large expressions
:class: dropdown

Sympy provides several simplification routines, such as `simplify()`, `expand()`, `factor()`, and `collect()`, among [others](https://docs.sympy.org/latest/tutorials/intro-tutorial/simplification.html).
The most general simplification routine is `simplify()`, which tries a combination of simplification routines to the expression.
However, this routine can be unnecessarily slow and it is not guaranteed to simplify the expression to the desired form.
Therefore, we analyze instead the expression and identify common patterns to simplify it manually.
:::

### Simplify the expression

To simplify the supercurrent expression, we first identify common patterns:

+ The expression is formed by a sum of fractions.
+ Terms share common prefactors, which are good to factor out.
+ The numerators contain products of $u_{\alpha} v_{\alpha}$, $u_{\alpha}^2$, and $v_{\alpha}^2$, all of which are free of square roots.

Therefore, we can simplify the expression by factoring out the prefactor, replacing the products of $u_{\alpha}$ and $v_{\alpha}$ with their expressions, and grouping the fractions by their denominators to then simplify the numerators.

```{code-cell} ipython3
# Define Bogoliubov substitutions
uv_subs = {u * v : Gamma / (2 * E) for u, v, Gamma, E in zip(us, vs, Gammas, Es)}
uu_subs = {u**2: (1 + xi / E) / 2 for u, xi, E in zip(us, xis, Es)}
vv_subs = {v**2: (1 - xi / E) / 2 for v, xi, E in zip(vs, xis, Es)}

def simplify_current(expr):
    """Simplification routine tailored for the perturbative calculation of current."""
    # Expand complex exponents to get rid of the imaginary part:
    expr = expr.expand(complex=True)
    # Factor individual terms first, then simplify and substitute the eigenvector definitions
    terms = [
        term.factor().simplify().subs(uv_subs | uu_subs | vv_subs) for term in expr.as_ordered_terms()
    ]
    common = set.intersection(*[set(term.as_ordered_factors()) for term in terms])
    prefactor = sympy.Mul(*common)
    return prefactor * sympy.Add(*[
        sympy.Mul(*(set(term.as_ordered_factors()) - common))
        for term in terms
    ])
```

Now we simplify the expression for the supercurrent.

```{code-cell} ipython3
%%time
current = simplify_current(current)

display_eq('I(n=0)', current)
```

Applying the same procedure to the other two ground states, we can compute the supercurrent for the $n=1$ and $n=2$ ground states.

```{code-cell} ipython3
%%time
currents = [current]
for i, (H, basis, ground_state) in enumerate([(H_odd, basis_odd, [c_up, c_down]), (H_even, basis_even, [c_down * c_up])]): # n=1, n=2
    subspace_indices = [element not in ground_state for element in basis]
    H_tilde = block_diagonalize(
        H,
        subspace_indices=subspace_indices,
        symbols=[t_L, t_R, dphi]
    )[0]
    current = sympy.Trace(H_tilde[0, 0, 2, 2, 1]).subs({dphi: 1}).doit()
    current = simplify_current(current)
    currents.append(current)
    display_eq(f"I(n={i + 1})", current)
```

## Visualize the results

Finally, we plot the critical current, $I_{c, N} = I_N(\phi=\pi/4)$, as a function of the number of electrons $N$.

```{code-cell} ipython3
:tags: [hide-input]

num = 60
N_values = np.linspace(-0.5, 2.5, 3 * num)
current_values = []
for values[N] in N_values:
    if N_value < 0.5:
        current = currents[0]
    elif N_value < 1.5:
        current = currents[1]
    else:
        current = currents[2]
    current_values.append(current.subs(values))

current_values = np.array(current_values, dtype=float)

fig, ax = plt.subplots()
ax.plot(N_values[0:num], current_values[0:num], '-', color='k')
ax.plot(N_values[num: 2 * num], current_values[num: 2 * num], '-', color='r')
ax.plot(N_values[2 * num:3 * num], current_values[2 * num: 3 * num], '-', color='b')
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$I_c$')
ax.set_title(r'Critical current')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([r'$0$', r'$1$', r'$2$'])
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# ax.set_ylim(-0.5, 0.5)
```
