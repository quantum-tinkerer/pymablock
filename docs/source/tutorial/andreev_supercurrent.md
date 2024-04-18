---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Andreev Supercurrent

In this tutorial, we demonstrate how to use Pymablock to compute complicated
analytical expressions and obtain interpretable results.
In many cases, directly running Pymablock on a large symbolic Hamiltonian is
computationally expensive due to the inherent cost of manipulating symbolic
matrices, diagonalizing them, and simplifying results that contain many terms
and factors.
Simplifying the input Hamiltonian is crucial to obtaining results in a
reasonable amount of time, and simplifying the output helps to understand the
results.
Both of these steps benefit from physical insight and advanced manipulation
of symbolic expressions.

As an example, we consider two superconductors weakly coupled to a quantum dot,
between which a supercurrent flows.

![Two superconductors and a quantum dot](superconductors_quantum_dot.svg)

Our goal will be to compute the supercurrent perturbatively in the tunneling
amplitudes, which requires calculating and manipulating fourth-order
corrections to the ground state energy [^1^].

[^1^]: [Glazman et al.](http://jetpletters.ru/ps/1121/article_16988.pdf)

## Define the Hamiltonian

The Hamiltonian of the system is given by

$$
H = H_{\textrm{SC}}+ H_{\textrm{dot}} + H_{T}
$$

where the Hamiltonians of the superconductors, quantum dot, and tunneling are
defined as

$$
H_{\textrm{SC}} =  \sum_{\alpha=L, R} \xi_{\alpha} \left(
    n_{\alpha \uparrow}^\dagger + n_{\alpha \downarrow} \right)
+ \Delta \left( c_{\alpha, \uparrow}^\dagger c_{\alpha \downarrow}^\dagger + c_{\alpha \downarrow} c_{\alpha, \uparrow} \right), \\
H_{\textrm{dot}} = \frac{U}{2} \left( n_{\uparrow} + n_{\downarrow} - N \right)^2. \\
H_{T} = \sum_{\alpha=L,R} t_\alpha \left( c_{\alpha \uparrow}^\dagger d_{\uparrow} + c_{\alpha \downarrow}^\dagger d_{\downarrow} \right) + \textrm{h.c.}.
$$

Here $c_{\alpha, \sigma}$ and $d_{\sigma}$ are the annihilation operators of
electrons in the superconductors and quantum dot, respectively, with $\sigma =
\uparrow, \downarrow$.
Both superconductors share a superconducting gap $\Delta$ and have different
onsite energies $\xi_{\alpha}$.
They are weakly coupled to the quantum dot, which has charging energy $U$ and
offset number of electrons $N$.
The couplings, $t_{L}$ and $t_{R}$, include the phase difference $\phi$ between
the superconductors, $t_{L} = \lvert t_{L} \rvert e^{i \phi}$ and $t_{R} = \lvert t_{R} \rvert$.
We treat both of them as independent perturbations.

We start by importing the necessary libraries and defining the symbols
and operators in the Hamiltonian.

```{code-cell} ipython3
import numpy as np
import sympy
from sympy import cos, sin, I, symbols, Symbol
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger

from pymablock import block_diagonalize

# Define symbols
U, N, t_L, t_R, phi, Delta_L, Delta_R, xi_L, xi_R, E_L, E_R = symbols(
    r"U N t_L t_R \phi \Delta_L \Delta_R \xi_L \xi_R E_L E_R",
    real=True,
    commutative=True,
    positive=True
)

t_L_complex = symbols(r"t_Lc", real=False, commutative=True)  # t_L = |t_L| e^{i \phi}

ts = t_L_complex, t_R
Deltas = Delta_L, Delta_R
xis = xi_L, xi_R
Es = E_L, E_R

# Dot operators
c_up, c_down = FermionOp('c_{\\uparrow}'), FermionOp('c_{\\downarrow}')

# Superconductor operators
d_ups = FermionOp('d_{L, \\uparrow}'), FermionOp('d_{R, \\uparrow}')
d_downs = FermionOp('d_{L, \\downarrow}'), FermionOp('d_{R, \\downarrow}')
```

Here we use `t_L_complex` to avoid complications with simplification routines in
`sympy`, so that the exponential is not decomposed into sines and cosines.

Next, we define the Hamiltonians of quantum dot and tunneling.

```{code-cell} ipython3
# Quantum Dot Hamiltonian
H_dot = U * (Dagger(c_up) * c_up + Dagger(c_down) * c_down - N)**2 / 2

# Tunneling Hamiltonian
H_T = sympy.Add(*[t * (Dagger(c_up) * d_up + Dagger(c_down) * d_down)
    for t, d_up, d_down in zip(ts, d_ups, d_downs)])  # + h.c. added later
```

```{code-cell} ipython3
:tags: [hide-cell]

display(sympy.Eq(Symbol('H_{dot}'), H_dot))
display(sympy.Eq(Symbol('H_{T}'), (H_T) + Symbol('h.c.')))
```

In this basis $H_{\textrm{dot}}$ is diagonal and its subspaces may be directly
constructed, without performing a symbolic diagonalization.
This is, however, not the case for $H_{SC}$.
Therefore, we apply the Bogoliubov transformation on $H_{SC}$, such that the
entire unperturbed Hamiltonian $H_0 = H_{SC} + H_{\textrm{dot}}$ is diagonal.

### Apply the Bogoliubov transformation

We define the superconductors' Hamiltonian using the Bogoliubov quasi-particle
operators $f_{\alpha, \sigma}$, which are related to the original operators
$c_{\alpha, \sigma}$ by the [Bogoliubov
transformation](https://en.wikipedia.org/wiki/Bogoliubov_transformation#Single_bosonic_mode_example):

$$
f_{\alpha, \uparrow} = u_\alpha c_{\alpha, \uparrow} + v_\alpha c_{\alpha, \downarrow} \\
f_{\alpha, \downarrow} = u_\alpha c_{\alpha, \downarrow} - v_\alpha c_{\alpha, \uparrow}
$$

where $u_\alpha$ and $v_\alpha$ are complex coefficients that satisfy
$u_\alpha^2 + v_\alpha^2 = 1$.
As a result,

$$
H_{SC} = \sum_{\alpha=L, R} \xi_{\alpha} - E_{\alpha} + E_{\alpha} \left(
f_{\alpha, \uparrow}^\dagger f_{\alpha, \uparrow} + f_{\alpha,
\downarrow}^\dagger f_{\alpha, \downarrow} \right),
$$

where $E_{\alpha} = \sqrt{\Delta_{\alpha}^2 + \xi_{\alpha}^2}$ are the energies
of the superconductors, $u_{\alpha} = \sqrt{\frac{1 + \xi_{\alpha} /
E_{\alpha}}{2}}$, and $v_{\alpha} = \sqrt{\frac{1 - \xi_{\alpha} /
E_{\alpha}}{2}}$.

```{code-cell} ipython3
# Bogoliubov quasi-particle operators
f_ups = FermionOp('f_{L, \\uparrow}'), FermionOp('f_{R, \\uparrow}')
f_downs = FermionOp('f_{L, \\downarrow}'), FermionOp('f_{R, \\downarrow}')

# Superconductors' Hamiltonian
H_sc = sympy.Add(*[
    xi - E + E * Dagger(f_up) * f_up + E * Dagger(f_down) * f_down
    for xi, E, f_up, f_down in zip(xis, Es, f_ups, f_downs)
])
```

```{code-cell} ipython3
:tags: [hide-cell]

display(sympy.Eq(Symbol('H_{SC}'), H_sc))
```

:::{important}
:class: dropdown tip
Diagonalizing a large symbolic Hamiltonian is computationally expensive,
and in many cases impossible.
To alleviate this, we have used the Bogoliubov transformation, which allows us to
get a diagonal unperturbed Hamiltonian.
In many cases, using physical insight to simplify the Hamiltonian is crucial
to obtaining results in a reasonable amount of time.
:::

Similarly, we apply the Bogoliubov transformation to the tunneling Hamiltonian.

```{code-cell} ipython3
# Bogoliubov coefficients
us = u_L, u_R = symbols(r"u_L u_R", real=True, commutative=True, positive=True)
vs = v_L, v_R = symbols(r"v_L v_R", real=True, commutative=True, positive=True)

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

display(sympy.Eq(Symbol('H_{T}'), (H_T) + Symbol('h.c.')))
```

We have now defined the total Hamiltonian $H$ in second quantization form.

:::{important}
:class: dropdown tip

Using symbols for u and v avoids introducing square roots, which can complicate
simplification routines in `sympy`.
:::

### Convert the Hamiltonian to a matrix

To exploit the diagonal structure of the unperturbed Hamiltonian, we convert the
Hamiltonian to a matrix representation.
To do this, we choose an order of the fermionic operators and compute the matrix
elements of the Hamiltonian in this basis.
The following code cell defines a function `to_matrix(...)` that computes the
matrix representation of a Hamiltonian `H` with fermionic operators in the basis
`effective_basis`.
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
    effective_basis = [sympy.Mul(*basis[i]) for i in np.argsort(basis_order)]

    return H.subs(matrix_subs, simultaneous=True).expand(), effective_basis
```

Next, we obtain the matrix Hamiltonian and its basis.

```{code-cell} ipython3
%%time
# Compute Hamiltonian in matrix form
H_matrix, basis = to_matrix(H)
print(f'Matrix shape of the Hamiltonian:{H_matrix.shape}')
```

At this point, we are ready to feed the $64 \times 64$ symbolic Hamiltonian to
the block-diagonalization routine of Pymablock.
However, we anticipate that the diagonal elements of the Hamiltonian will appear
in the denominators in the effective Hamiltonian.
These elements correspond to the dot energies $E_n = U (N - n)^2 / 2$ and the
energies of the superconductors $E_{\alpha}$.
To make the denominators simpler, we replace the dot energies with $E_n$
right away.

```{code-cell} ipython3
H_matrix = H_matrix.subs({t_L_complex: t_L * cos(phi) + t_L * I * sin(phi)})

E_0, E_1, E_2 = symbols(r"E_0 E_1 E_2", real=True, commutative=True, positive=True)
E_0_value, E_1_value, E_2_value = [(U * (N - i)**2 / 2).expand() for i in range(3)]
H_matrix = H_matrix.subs({E_2_value: E_2}).subs({E_1_value: E_1}).subs({E_0_value: E_0})
```

:::{admonition} Replace energy to simplify denominators
:class: dropdown tip
This is a key aspect of the procedure.
:::

Finally, the Hamiltonian is ready for further analysis.


```{code-cell} ipython3
:tags: [hide-input]
import matplotlib.pyplot as plt

# Values for the parameters
values = {
    U: 5,
    Delta_L: 1,
    Delta_R: 1,
    t_L: 0.2,
    t_R: 0.1,
    xi_L: 2,
    xi_R: 1,
    E_0: E_0_value,
    E_1: E_1_value,
    E_2: E_2_value,
    phi: np.pi/4
}

# Extract eigenvalues of the unperturbed Hamiltonian
N_values = np.linspace(0, 2, 20)
eigenvalues = []
for N_value in N_values:
    values.update({
        N: N_value,
        E_R: np.sqrt(values[Delta_R]**2 + values[xi_R]**2),
        E_L: np.sqrt(values[Delta_L]**2 + values[xi_L]**2)
    })
    eigenvalues.append(np.array(H_matrix.diagonal().subs(values), dtype=complex)[0])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(N_values, np.array(eigenvalues).real, '-', alpha=0.5);
ax.set_ylim(-1, 12.5)
ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.set_xticklabels([r'$0$', r'$1/2$', r'$1$', r'$3/2$', r'$2$'])
ax.set_yticks([0, 5, 10])
ax.set_yticklabels([r'$0$', r'$5$', r'$10$'])
ax.set_ylabel('$E_0$')
ax.set_xlabel(r'$N$')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
```

Therefore, we define the following ground states for the $n=0, 1, 2$ occupation
numbers of the quantum dot.

## Compute the supercurrent perturbatively

Because the supercurrent is defined as

$$
I_c = \frac{e}{\hbar} \frac{dE}{d\phi}
$$
we need to find the perturbed ground state energy $E(\phi)$.
To do so, we need to block-diagonalize the Hamiltonian in the subspace spanned by the ground states of the quantum dot.
We start by finding the corrections to the ground state $n=0$.

```{code-cell} ipython3
%%time

ground_state_n0 = [sympy.S.One]
subspace_indices = [int(not(element in ground_state_n0)) for element in basis]
H_tilde = block_diagonalize(H_matrix, subspace_indices=subspace_indices, symbols=[t_L, t_R])[0]
```

```{code-cell} ipython3
%%time
E_0 = H_tilde[0, 0, 2, 2] # Ground state energy for n=0
```

We can now compute the supercurrent $I_c$.

```{code-cell} ipython3
%%time
current = sympy.diff(sympy.Trace(E_0), phi)
prefactor = Delta_L * Delta_R * t_L ** 2 * t_R ** 2 / (E_L * E_R) * 2 * sin(phi) * cos(phi)
result = (current / prefactor).expand()
display(result)
```

### Simplify the expression

The expression for the supercurrent is quite complicated, and we can simplify it by grouping terms with common denominators and simplifying the numerators.

```{code-cell} ipython3
# Define Bogoliubov substitutions
uv_subs = {u * v : Delta / (2 * E) for u, v, Delta, E in zip(us, vs, Deltas, Es)}
uu_subs = {u**2: (1 + xi / E) / 2 for u, xi, E in zip(us, xis, Es)}
vv_subs = {v**2: (1 - xi / E) / 2 for v, xi, E in zip(vs, xis, Es)}

def simplify_fraction(expr):
    """ Groups fractions by denominators and simplifies numerators."""
    expr = expr.expand()
    result = sympy.Add(*[term.factor() for term in expr.as_ordered_terms()])

    # Group fractions with common denominator
    fractions = {}
    for term in result.as_ordered_terms():
        numerator, denominator = sympy.fraction(term)
        numerator = numerator.subs(uv_subs).subs(uu_subs).subs(vv_subs)
        if denominator in fractions:
            fractions[denominator] += numerator
        else:
            fractions[denominator] = numerator

    # Put sum back together after simplifying numerator
    result = sum([(numerator.factor() / denominator) for denominator, numerator in fractions.items()])
    return result
```

Now we simplify the expression for the supercurrent.

```{code-cell} ipython3
%%time
simplified_result = simplify_fraction(result)
display(prefactor * simplified_result)
```

Applying the same procedure to the other two ground states, we can compute the supercurrent for the $n=1$ and $n=2$ ground states.

```{code-cell} ipython3
%%time
for ground_state in [[c_up, c_down], [c_up * c_down]]: # n=1, n=2
    subspace_indices = [int(not(element in ground_state)) for element in basis]
    H_tilde = block_diagonalize(H_matrix, subspace_indices=subspace_indices, symbols=[t_L, t_R])[0]
    current = sympy.diff(sympy.Trace(H_tilde[0, 0, 2, 2]), phi)
    simplified_result = simplify_fraction((current / prefactor).expand())
    display(sympy.Eq(Symbol(f"I_c(n={len(ground_state)})"), prefactor * simplified_result))
```

## Visualize the results

```{code-cell} ipython3
:tags: [hide-input]
%%time
numerical_prefactor = prefactor.subs(values)

N_values = np.linspace(0, 2, 300)
supercurrent_values = []
for N_value in N_values:
    values.update({N: N_value})
    supercurrent_values.append(simplified_result.subs(values) * numerical_prefactor)

supercurrent_values = np.array(supercurrent_values, dtype=float)

fig, ax = plt.subplots()
ax.plot(N_values, (supercurrent_values))
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$I_c$')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_ylim(-0.5, 0.5)
```
