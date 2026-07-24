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

This tutorial demonstrates implicit perturbation theory when the low-energy space is spanned by a few MPSs.
We compute the $2\times2$ effective Hamiltonian of the two ferromagnetic ground states of the transverse-field Ising chain.
Pymablock stores only their response states as MPSs, rather than a transformation MPO acting on the full Hilbert space.
The analytical second-order coefficient and exact finite-chain energy provide independent checks.

## Model and analytical result

We consider $L$ spin-$\frac12$ sites with open boundaries,

$$
H(g)=H_0+gV,
\qquad
H_0=-J\sum_{i=1}^{L-1}Z_iZ_{i+1},
\qquad
V=-\sum_{i=1}^{L}X_i.
$$

Here $X_i$ and $Z_i$ are Pauli operators on site $i$, $J>0$ is the ferromagnetic coupling, and the transverse field $g$ is the perturbative parameter.

At $g=0$, the states
$\lvert\Uparrow\rangle=\lvert\uparrow\cdots\uparrow\rangle$ and
$\lvert\Downarrow\rangle=\lvert\downarrow\cdots\downarrow\rangle$
are degenerate ground states with energy $E_0=-J(L-1)$.
They span the low-energy model space $P$ whose effective Hamiltonian we seek.

At second order, the intermediate states contain one flipped spin.
An edge flip costs $2J$, while an interior flip costs $4J$, so

$$
H_\mathrm{eff}
=E_0 I_2+g^2H_\mathrm{eff}^{(2)}+\cdots,
\qquad
H_\mathrm{eff}^{(2)}
=-\left(\frac{2}{2J}+\frac{L-2}{4J}\right)I_2
=-\frac{L+2}{4J}I_2.
$$

Connecting $\lvert\Uparrow\rangle$ to $\lvert\Downarrow\rangle$ requires flipping every spin, so tunneling first appears at order $g^L$.

```{code-cell}
%%time

from functools import reduce

import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy_implicit_backend import TenpyImplicitBackend
from tenpy_mpo_backend import product_mpo

from pymablock.implicit import block_diagonalize_implicit
```

## Build the MPO and model space

We use $L=8$ and $J=0.7$.
In the code, `h_0` represents $H_0$, `perturbation` represents $V$, and `references` contains the two MPS basis states of $P$.
The complementary space $Q=1-P$ is handled through projected MPO applications, without constructing its basis.

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
Because the retained states diagonalize $H_0$, applying the operator Sylvester equation to each one gives an independent response equation,

$$
Q(H_0-E_a)Q\lvert\eta_a\rangle=\lvert S_a\rangle,
\qquad
\langle\phi_b\vert\eta_a\rangle=0.
$$

Here $\lvert\phi_a\rangle$ is either retained ferromagnetic state, $E_a=E_0$ is its unperturbed energy, $\lvert\eta_a\rangle$ is the unknown response MPS, and $\lvert S_a\rangle$ is the source assembled from lower perturbative orders.
At first order, $V$ makes each source a superposition of one-spin-flip states.

The excitation gap makes $Q(H_0-E_0)Q$ positive definite.
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


def exact_ground_energy(field):
    """Return the finite-chain free-fermion ground-state energy."""
    jordan_wigner = np.diag(np.full(L, field))
    jordan_wigner += np.diag(np.full(L - 1, J), k=-1)
    return -np.linalg.svd(jordan_wigner, compute_uv=False).sum()


fields = np.array([0.2, 0.1, 0.05, 0.025])
unperturbed_energy = -J * (L - 1)
second_order_energy = second_order[0, 0]
errors = np.array(
    [
        abs(
            exact_ground_energy(field)
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

## Conclusion

This calculation obtains the second-order $2\times2$ effective Hamiltonian while storing only two response MPSs instead of a full transformation MPO.
Its coefficient matches the result derived from the spin-flip gaps, and inserting the computed coefficient into the exact finite-chain energy leaves the expected fourth-order error.

This example validates the implicit formulation; it is not a scaling benchmark.
For larger systems, increase `chi_max`, lower the SVD cutoff, and tighten the residual tolerance until the requested coefficient remains stable.
