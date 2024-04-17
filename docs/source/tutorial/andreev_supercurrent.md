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

The superconductors' creation and annihilation operators are $c_{\alpha,
\sigma}^\dagger$ and $c_{\alpha, \sigma}$, respectively, for spin $\sigma =
\uparrow, \downarrow$.
The energy of each superconductor is $\xi_{\alpha}$, and $\Delta$ is their gap.
The creation and annihilation operators of the quantum dot are $d_{\sigma}^\dagger$
and $d_{\sigma}$, respectively, $U$ is its charging energy, and $N$ is its offset
number of electrons.
The tunneling amplitudes between the dot and superconductors are $t_{L}$ and
$t_{R}$, which we treat perturbatively.
The phase difference between the superconductors is $\phi$, and its effect is
included in the tunneling amplitude $t_{L} = \lvert t_{L} \rvert e^{i \phi}$.

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
U, N, t_L, t_R, phi, Gamma_L, Gamma_R, xi_L, xi_R, E_L, E_R = symbols(
    r"U N t_L t_R \phi \Gamma_L \Gamma_R \xi_L \xi_R E_L E_R",
    real=True,
    commutative=True,
    positive=True
)

t_L_complex = symbols(r"t_Lc", real=False, commutative=True) #  trick

ts = t_L_complex, t_R
Gammas = Gamma_L, Gamma_R
xis = xi_L, xi_R
Es = E_L, E_R

# Dot operators
c_up, c_down = FermionOp('c_{\\uparrow}'), FermionOp('c_{\\downarrow}')
n = Dagger(c_up) * c_up + Dagger(c_down) * c_down

# Superconductor operators
d_ups = FermionOp('d_{L, \\uparrow}'), FermionOp('d_{R, \\uparrow}')
d_downs = FermionOp('d_{L, \\downarrow}'), FermionOp('d_{R, \\downarrow}')
```

Here `t_L_complex` represents the tunneling amplitude $t_L = \lvert t_L \rvert e^{i \phi}$.
We will use it to avoid complications with the simplification routines in
`sympy`, which may attempt to simplify the phase factor $e^{i \phi}$ as a sum
of sines and cosines, which is not what we want.

Next, we define the Hamiltonians of quantum dot and tunneling.

```{code-cell} ipython3
# Quantum Dot Hamiltonian
H_dot = U * (n - N)**2 / 2
display(sympy.Eq(Symbol('H_{dot}'), H_dot))

# Tunneling Hamiltonian
H_T = sympy.Add(*[t * (Dagger(c_up) * d_up + Dagger(c_down) * d_down)
    for t, d_up, d_down in zip(ts, d_ups, d_downs)]) # + h.c. is added later
display(sympy.Eq(Symbol('H_{T}'), (H_T) + Symbol('h.c.')))
```

### Apply the Bogoliubov transformation

We note that $H_0$ is not diagonal in the basis of ...
Performing this symbolic diagonalization is computationally expensive.

```{code-cell} ipython3
# Define 2nd quantization operators
# ABS operators, will do Bogoliubov transformation
f_ups = FermionOp('f_{L, \\uparrow}'), FermionOp('f_{R, \\uparrow}')
f_downs = FermionOp('f_{L, \\downarrow}'), FermionOp('f_{R, \\downarrow}')

# ABS Hamiltonian
H_abs = 0
for xi, E, f_up, f_down in zip(xis, Es, f_ups, f_downs):
    H_abs += xi - E + E * Dagger(f_up) * f_up + E * Dagger(f_down) * f_down
display(sympy.Eq(Symbol('H_{ABS}'), H_abs))
```

Therefore, we use the Bogoliubov transformation to bring $H_0$ to a diagonal form,
by defining the following transformation:

$$
d_{\alpha, \uparrow} = u_\alpha f_{\alpha, \uparrow} + v_\alpha f_{\alpha, \downarrow} \\
d_{\alpha, \downarrow} = u_\alpha f_{\alpha, \downarrow} - v_\alpha f_{\alpha, \uparrow}
$$

where $u_\alpha$ and $v_\alpha$ are the Bogoliubov coefficients.

```{code-cell} ipython3
# Bogoliubov coefficients
us = u_L, u_R = symbols(r"u_L u_R", real=True, commutative=True, positive=True)
vs = v_L, v_R = symbols(r"v_L v_R", real=True, commutative=True, positive=True)

# Bogoliubov transformation from d operators to f operators
transformation_substitution = {d_down: u * f_down + v * Dagger(f_up)
    for u, v, d_down, d_up, f_down, f_up in zip(us, vs, d_downs, d_ups, f_downs, f_ups)}

transformation_substitution.update({d_up: u * f_up - v * Dagger(f_down)
    for u, v, d_down, d_up, f_down, f_up in zip(us, vs, d_downs, d_ups, f_downs, f_ups)})

transformation_substitution.update({Dagger(key): Dagger(value) for key, value in transformation_substitution.items()})

# Substitute coefficients so that H_T is in ABS basis
H_T = H_T.subs(transformation_substitution).expand()
display(sympy.Eq(Symbol('H_{T}'), (H_T) + Symbol('h.c.')))
H_T += Dagger(H_T)

# Total Hamiltonian
H = H_abs + H_dot + H_T
```

:::{important}
:class: dropdown tip
Diagonalizing a large symbolic Hamiltonian is computationally expensive,
and in many cases impossible.
To alleviate this, we have used the Bogoliubov transformation, which allows us to
get a diagonal unperturbed Hamiltonian.
In many cases, using physical insight to simplify the Hamiltonian is crucial.
:::

```{code-cell} ipython3
:tags: [hide-input]
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

```{code-cell} ipython3
%%time
# Compute Hamiltonian in matrix form
H_matrix, basis = to_matrix(H)
```

```{code-cell} ipython3
H_matrix = H_matrix.subs({t_L_complex: t_L * cos(phi) + t_L * I * sin(phi)})

E_0, E_1, E_2 = symbols(r"E_0 E_1 E_2", real=True, commutative=True, positive=True)
E_0_value, E_1_value, E_2_value = [(U * (N - i)**2 / 2).expand() for i in range(3)]
H_matrix = H_matrix.subs({E_2_value: E_2}).subs({E_1_value: E_1}).subs({E_0_value: E_0})
```

where $E_n = U (N - n)^2 / 2$ are the energies of the quantum dot for $n=0, 1, 2$.

:::{admonition} Replace energy to simplify denominators
:class: dropdown tip
This is a key aspect of the procedure.
:::


```{code-cell} ipython3
:tags: [hide-input]
import matplotlib.pyplot as plt

# Values for the parameters
values = {
    U: 5,
    Gamma_L: 1,
    Gamma_R: 1,
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
        E_R: np.sqrt(values[Gamma_R]**2 + values[xi_R]**2),
        E_L: np.sqrt(values[Gamma_L]**2 + values[xi_L]**2)
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
numbers of the quantum dot:

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
prefactor = Gamma_L * Gamma_R * t_L ** 2 * t_R ** 2 / (E_L * E_R) * 2 * sin(phi) * cos(phi)
result = (current / prefactor).expand()
display(result)
```

### Simplify the expression

The expression for the supercurrent is quite complicated, and we can simplify it by grouping terms with common denominators and simplifying the numerators.

```{code-cell} ipython3
# Define Bogoliubov substitutions
uv_subs = {u * v : Gamma / (2 * E) for u, v, Gamma, E in zip(us, vs, Gammas, Es)}
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
