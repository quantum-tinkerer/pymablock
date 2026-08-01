---
jupytext:
  formats: md:myst,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Implicit MPS perturbation theory

Conceptually, this tutorial demonstrates implicit perturbation theory when the low-energy space is spanned by only a few MPSs.
Pymablock stores the response of those states rather than a transformation MPO acting on the full Hilbert space.
Physically, we compute the two lowest energies of the transverse-field Ising chain as functions of the field, separating their common perturbative shift from the much smaller tunneling splitting.

The analytical second-order coefficient and exact finite-chain spectrum provide independent checks.

## Model and analytical result

We use the notation of [the algorithm](../algorithms.md) and consider $L$ spin-$\frac12$ sites with open boundaries,

$$
\mathcal H(g)=H_0+\mathcal H'(g),
\qquad
\mathcal H'(g)=gH'_1,
\qquad
H_0=-J\sum_{i=1}^{L-1}Z_iZ_{i+1},
\qquad
H'_1=-\sum_{i=1}^{L}X_i.
$$

Here $X_i$ and $Z_i$ are Pauli operators on site $i$, $J>0$ is the ferromagnetic coupling, and the transverse field $g$ is the perturbative parameter.
The Ising term favors alignment along $z$, while $H'_1$ flips individual spins and introduces quantum fluctuations.
For $g\ll J$, the low-energy physics is a nearly degenerate ferromagnetic doublet separated from spin-flip excitations by a finite gap.
Perturbation theory describes both the common energy shift of this doublet and the much weaker tunneling between its two states.

At $g=0$, the states
$\lvert\Uparrow\rangle=\lvert\uparrow\cdots\uparrow\rangle$ and
$\lvert\Downarrow\rangle=\lvert\downarrow\cdots\downarrow\rangle$
are degenerate ground states with energy $E_0=-J(L-1)$.
They form the columns of $\Psi_E$ and span the explicit subspace $E$ whose transformed Hamiltonian block we seek.
The projector onto the implicit subspace $I$ is
$P_I=1-\Psi_E\Psi_E^\dagger$.

We will compute its $2\times2$ effective Hamiltonian and plot the two corresponding energies as functions of $g/J$.
Comparison with the exact Jordan--Wigner spectrum will distinguish the second-order energy shift from the higher-order splitting.

At second order, the intermediate states contain one flipped spin.
An edge flip costs $2J$, while an interior flip costs $4J$, so

$$
\tilde{\mathcal H}^{EE}(g)
=E_0 I_2+g^2\tilde H_2^{EE}+\cdots,
\qquad
\tilde H_2^{EE}
=-\left(\frac{2}{2J}+\frac{L-2}{4J}\right)I_2
=-\frac{L+2}{4J}I_2.
$$

Connecting $\lvert\Uparrow\rangle$ to $\lvert\Downarrow\rangle$ requires flipping every spin, so tunneling first appears at order $g^L$.

```{code-cell}
%%time

from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy_implicit_backend import TenpyImplicitBackend
from pymablock.backends.tenpy import product_mpo

from pymablock.implicit import block_diagonalize_implicit
```

## Build the MPO and explicit subspace

We use $L=8$ and $J=0.7$.
In the code, `h_0` represents $H_0$, `perturbation` represents $H'_1$, and `references` contains the two MPS basis states of $E$.
The implicit subspace $I$ is handled through applications of $P_I$, without constructing a basis for it.

```{code-cell}
%%time

L = 8
J = 0.7

sites = [SpinHalfSite(conserve=None) for _ in range(L)]
identity = np.eye(2)
x = np.array([[0.0, 1.0], [1.0, 0.0]])
z = np.diag([1.0, -1.0])

backend = TenpyImplicitBackend(
    chi_max=32,
    svd_min=1e-12,
    max_sweeps=10,
    solver_tolerance=1e-9,
)


def product_term(operators):
    """Construct a Pauli string from ``site: local_operator`` entries."""
    return product_mpo(
        sites,
        [operators.get(site, identity) for site in range(L)],
    )


def add_all(operators):
    """Add a nonempty sequence of MPOs."""
    return reduce(backend.mpo_backend.add, operators)


h_0 = backend.mpo_backend.scale(
    add_all([product_term({site: z, site + 1: z}) for site in range(L - 1)]),
    -J,
)
perturbation = backend.mpo_backend.scale(
    add_all([product_term({site: x}) for site in range(L)]),
    -1,
)
references = [
    MPS.from_product_state(
        sites,
        [orientation] * L,
        bc="finite",
        unit_cell_width=L,
    )
    for orientation in ("up", "down")
]
```

## Compute the perturbative coefficient

Pymablock constructs the perturbative series lazily.
Because the explicit states diagonalize $H_0$, applying the $IE$ block of the Sylvester equation to each one gives an independent response equation at order $\mathbf n$,

$$
P_I(H_0-E_a)P_I\lvert v_{\mathbf n,a}\rangle
=\lvert f_{\mathbf n,a}\rangle,
\qquad
\langle\phi_b\vert v_{\mathbf n,a}\rangle=0.
$$

Here $\lvert\phi_a\rangle$ is either explicit ferromagnetic state, $E_a=E_0$ is its unperturbed energy,
$\lvert v_{\mathbf n,a}\rangle=P_IV_{\mathbf n}^{IE}\lvert\phi_a\rangle$
is the unknown response MPS, and
$\lvert f_{\mathbf n,a}\rangle=P_IF_{\mathbf n}^{IE}\lvert\phi_a\rangle$
is the source assembled from lower perturbative orders.
We introduce $F_{\mathbf n}^{IE}$ for the known $IE$-block right-hand side passed to the Sylvester solver.
At first order, $H'_1$ makes each source a superposition of one-spin-flip states.

The excitation gap makes $P_I(H_0-E_0)P_I$ positive definite.
The backend solves this response problem with two-site variational sweeps, enforcing orthogonality to both reference states and truncating the MPS after each update.
After every sweep, it recomputes the global residual.
The sweep pattern resembles DMRG, but its target is a linear response state rather than a ground state.

```{code-cell}
%%time

h_tilde, unitary, _ = block_diagonalize_implicit(
    [h_0, perturbation],
    references,
    backend,
    backend.solve_shifted,
    max_relative_residual=1e-8,
)
second_order = h_tilde[0, 0, 2].dense
first_order_columns = unitary[1, 0, 1].states
```

Here `second_order` is $\tilde H_2^{EE}$, while `first_order_columns` stores
$U_1^{IE}\lvert\phi_a\rangle=V_1^{IE}\lvert\phi_a\rangle$ because $\mathcal W_1=0$.
Compare the resulting $2\times2$ coefficient with the spin-flip
denominators. This checks both shifted solves and the overlap products that
feed the effective Hamiltonian.

```{code-cell}
%%time

expected_second_order = -(L + 2) / (4 * J) * np.eye(2)
np.testing.assert_allclose(second_order, expected_second_order, atol=1e-10)

maximum_residual = max(record.relative_residuals[-1] for record in backend.solver_records)
maximum_orthogonality_error = max(
    record.orthogonality_errors[-1] for record in backend.solver_records
)
first_order_maximum_bond_dimension = max(
    max(column.chi) for column in first_order_columns
)

assert maximum_residual < 1e-8
assert maximum_orthogonality_error < 1e-8
assert first_order_maximum_bond_dimension <= backend.chi_max

{
    "computed second-order coefficient": second_order,
    "maximum global residual": maximum_residual,
    "maximum reference overlap": maximum_orthogonality_error,
    "first-order MPS bond dimension": first_order_maximum_bond_dimension,
}
```

The reported bond dimension describes the first-order response MPSs.
Respecting `chi_max` verifies the cap, but convergence still requires repeating the calculation with tighter truncation parameters.

## Compare with the exact free-fermion energy

For the open chain, the exact ground-state energy is minus the sum of the singular values of the bidiagonal Jordan--Wigner matrix with diagonal $g$ and subdiagonal $J$.
For $L=8$, the leading omitted diagonal term is fourth order, while tunneling starts at eighth order.
The error of the second-order result should therefore scale as $g^4$: halving $g$ reduces it by approximately $2^4=16$, so `observed_orders` should approach $4$.

```{code-cell}
%%time


def exact_low_energy_levels(field):
    """Return the two lowest finite-chain energies."""
    jordan_wigner = np.diag(np.full(L, field))
    jordan_wigner += np.diag(np.full(L - 1, J), k=-1)
    singular_values = np.linalg.svd(jordan_wigner, compute_uv=False)
    ground = -singular_values.sum()
    return ground, ground + 2 * singular_values.min()


fields = np.array([0.2, 0.1, 0.05, 0.025])
unperturbed_energy = -J * (L - 1)
second_order_energy = second_order[0, 0]
errors = np.array(
    [
        abs(
            exact_low_energy_levels(field)[0]
            - (unperturbed_energy + field**2 * second_order_energy)
        )
        for field in fields
    ]
)
observed_orders = np.log2(errors[:-1] / errors[1:])

assert observed_orders[-1] > 3.99

{
    "fields": fields,
    "absolute errors": errors,
    "observed orders": observed_orders,
}
```

## Interpret the low-energy doublet

The Jordan--Wigner singular values also give the excitation energies.
The smallest singular value therefore determines the splitting between the two lowest states.
We compare these exact levels with the degenerate pair predicted by the second-order effective Hamiltonian.

```{code-cell}
%%time

plot_fields = np.linspace(0, 0.8 * J, 101)
exact_levels = np.array([exact_low_energy_levels(field) for field in plot_fields])
second_order_level = unperturbed_energy + plot_fields**2 * np.real(second_order_energy)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    plot_fields / J,
    (exact_levels[:, 0] - unperturbed_energy) / J,
    label="exact ground state",
)
ax.plot(
    plot_fields / J,
    (exact_levels[:, 1] - unperturbed_energy) / J,
    label="exact first excited state",
)
ax.plot(
    plot_fields / J,
    (second_order_level - unperturbed_energy) / J,
    "k--",
    label="second order (both states)",
)
ax.set(
    xlabel=r"$g/J$",
    ylabel=r"$(E-E_0)/J$",
    xlim=(0, 0.8),
)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
plt.show()
```

At weak field, both exact levels follow the common second-order curve: virtual one-spin flips lower the ferromagnetic doublet without splitting it at this order.
The exact levels separate by an amount of order $g^L$, reflecting tunneling that requires all $L$ spins to flip.
Their eventual departure from the dashed curve shows where a second-order expansion ceases to be quantitatively accurate.

## Conclusion

This calculation obtains the second-order coefficient $\tilde H_2^{EE}$ while storing only the two response MPSs $V_1^{IE}\lvert\phi_a\rangle$ instead of a full transformation MPO.
Its coefficient matches the result derived from the spin-flip gaps, and inserting the computed coefficient into the exact finite-chain energy leaves the expected fourth-order error.
The level plot separates the common perturbative energy shift from the much smaller high-order tunneling splitting.

This example validates the implicit formulation; it is not a scaling benchmark.
For larger systems, increase `chi_max`, lower the SVD cutoff, and tighten the residual tolerance until the requested coefficient remains stable.
