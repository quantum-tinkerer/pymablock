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

```{code-cell} ipython3
import numpy as np
import sympy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
import matplotlib.pyplot as plt

symbols = sympy.symbols(
    r"U n_C t_L t_R \phi \Gamma_L \Gamma_R \xi_L \xi_R E_L E_R",
    real=True,
    commutative=True,
    positive=True
)

U, n_C, t_L, t_R, phi, Gamma_L, Gamma_R, xi_L, xi_R, E_L, E_R = symbols
t_L_complex = sympy.symbols(r"t_Lc", real=False, commutative=True) # will replace t_L * exp(I * phi)

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

# Quantum Dot Hamiltonian
H_dot = U * (n - n_C)**2 / 2

# Interaction term with fermionic basis
H_T = 0
for t, d_up, d_down in zip(ts, d_ups, d_downs):
    H_T += t * (Dagger(c_up) * d_up + Dagger(c_down) * d_down)
H_T += Dagger(H_T)

display(sympy.Eq(sympy.Symbol('H_{dot}'), H_dot))
display(sympy.Eq(sympy.Symbol('H_{T}'), H_T))
```

```{code-cell} ipython3
# Define 2nd quantization operators
# ABS operators, will do Bogoliubov transformation
f_ups = FermionOp('f_{L, \\uparrow}'), FermionOp('f_{R, \\uparrow}')
f_downs = FermionOp('f_{L, \\downarrow}'), FermionOp('f_{R, \\downarrow}')

# ABS Hamiltonian
H_abs = 0
for xi, E, f_up, f_down in zip(xis, Es, f_ups, f_downs):
    H_abs += xi - E + E * Dagger(f_up) * f_up + E * Dagger(f_down) * f_down
display(sympy.Eq(sympy.Symbol('H_{ABS}'), H_abs))

# Bogoliubov coefficients
us = u_L, u_R = sympy.symbols(r"u_L u_R", real=True, commutative=True, positive=True)
vs = v_L, v_R = sympy.symbols(r"v_L v_R", real=True, commutative=True, positive=True)

# Bogoliubov transformation from d operators to f operators
transformation_substitution = {d_down: u * f_down + v * Dagger(f_up)
    for u, v, d_down, d_up, f_down, f_up in zip(us, vs, d_downs, d_ups, f_downs, f_ups)}

transformation_substitution.update({d_up: u * f_up - v * Dagger(f_down)
    for u, v, d_down, d_up, f_down, f_up in zip(us, vs, d_downs, d_ups, f_downs, f_ups)})

transformation_substitution.update({Dagger(key): Dagger(value) for key, value in transformation_substitution.items()})

# Substitute coefficients so that H_T is in ABS basis
H_T = H_T.subs(transformation_substitution).expand()
display(sympy.Eq(sympy.Symbol('H_{T}'), H_T))

# Total Hamiltonian
H = H_abs + H_dot + H_T
```

```{code-cell} ipython3
:tags: [hide-input]
%%time
from itertools import combinations

def to_matrix(H):
    """Compute a matrix representation of a sympy expression with fermion operators."""
    # Add an identity operator to all symbols so that we always work with operators
    H = H.subs({
        s: sympy.physics.quantum.IdentityOperator() * s for s in H.free_symbols
        if not isinstance(s, sympy.physics.quantum.Operator)
    })
    # Choose an order of Fermionic operators (or should we ask for it?
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
# Compute Hamiltonian in matrix form
H_matrix, basis = to_matrix(H)

H_matrix = H_matrix.subs({t_L_complex: t_L * sympy.cos(phi) + t_L * sympy.I * sympy.sin(phi)})

E_0, E_1, E_2 = sympy.symbols(r"E_0 E_1 E_2", real=True, commutative=True, positive=True)

E_0_value = U * n_C**2 / 2
E_1_value = (U * (1 - n_C)**2 / 2).expand()
E_2_value = (U * (2 - n_C)**2 / 2).expand()

H_matrix = H_matrix.subs({E_2_value: E_2})
H_matrix = H_matrix.subs({E_1_value: E_1})
H_matrix = H_matrix.subs({E_0_value: E_0})
```

```{code-cell} ipython3
# Spectrum of the Hamiltonian
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
    n_C: 1,
    phi: np.pi/4
}

n_C_values = np.linspace(0, 2, 10)
eigenvalues = []
for n_C_value in n_C_values:
    values.update({
        n_C: n_C_value,
        E_R: np.sqrt(values[Gamma_R]**2 + values[xi_R]**2),
        E_L: np.sqrt(values[Gamma_L]**2 + values[xi_L]**2)
    })
    eigenvalues.append(np.array(H_matrix.diagonal().subs(values), dtype=complex)[0])

fig, ax = plt.subplots()
ax.plot(n_C_values, np.array(eigenvalues).real, '-', alpha=0.5);
ax.set_ylabel('$E_0$')
ax.set_xlabel(r'$n_C$')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
```

```{code-cell} ipython3
# Possible ground states
# For n=0
ground_state_n0 = [sympy.S.One]
# For n=1
ground_state_n1 = [c_up, c_down]
# For n=2
ground_state_n2 = [c_up * c_down]
```


```{code-cell} ipython3
%%time

from pymablock import block_diagonalize
subspace_indices = [int(not(element in ground_state_n0)) for element in basis]
H_tilde = block_diagonalize(H_matrix, subspace_indices=subspace_indices, symbols=[t_L, t_R])[0]
```

```{code-cell} ipython3
%%time
order_L = 2
order_R = 2
correction = H_tilde[0, 0, order_L, order_R]
```

```{code-cell} ipython3
%%time
trace = sympy.Add(*[correction[i, i].expand() for i in range(correction.shape[0])])
current = sympy.diff(trace, phi)
prefactor = Gamma_L * Gamma_R * t_L ** order_L * t_R ** order_R / (E_L * E_R) * 2 * sympy.sin(phi) * sympy.cos(phi)
result = (current / prefactor).expand()
```

```{code-cell} ipython3
def simplify_fraction(expr):
    """ Groups fractions by denominators and simplifies numerators."""
    expr = expr.expand()
    result = sympy.Add(*[term.factor() for term in expr.as_ordered_terms()])

    uv_subs = {u * v : Gamma / (2 * E) for u, v, Gamma, E in zip(us, vs, Gammas, Es)}
    u_subs = {u**2: (1 + xi / E) / 2 for u, xi, E in zip(us, xis, Es)}
    v_subs = {v**2: (1 - xi / E) / 2 for v, xi, E in zip(vs, xis, Es)}

    # Group fractions with common denominator
    fractions = {}
    for term in result.as_ordered_terms():
        numerator, denominator = sympy.fraction(term)
        numerator = numerator.subs(uv_subs).subs(u_subs).subs(v_subs)
        if denominator in fractions:
            fractions[denominator] += numerator
        else:
            fractions[denominator] = numerator

    # Put sum back together after simplifying numerator
    result = sum([(numerator.factor() / denominator) for denominator, numerator in fractions.items()])
    return result
```

```{code-cell} ipython3
%%time
simplified_result = simplify_fraction(result)
display(prefactor * simplified_result)
```

```{code-cell} ipython3
:tags: [hide-input]
%%time
numerical_prefactor = prefactor.subs(values)

n_C_values = np.linspace(0, 2, 300)
supercurrent_values = []
for n_C_value in n_C_values:
    values.update({n_C: n_C_value})
    supercurrent_values.append(simplified_result.subs(values) * numerical_prefactor)

supercurrent_values = np.array(supercurrent_values, dtype=float)

fig, ax = plt.subplots()
ax.plot(n_C_values, (supercurrent_values))
ax.set_xlabel(r'$n_C$')
ax.set_ylabel(r'$I_c$')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_ylim(-0.5, 0.5)
```
