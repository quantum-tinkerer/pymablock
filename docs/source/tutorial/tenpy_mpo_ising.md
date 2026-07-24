---
jupytext:
  formats: md:myst,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Analytical Ising-chain benchmark

This tutorial applies MPO perturbation theory to the finite longitudinal-field Ising chain, a standard low-bond-dimension MPO.
We couple the chain to a detuned two-level sector in a way that has an exact operator solution.
The example therefore checks both the tensor-network calculation and the expected perturbative error without using dense Pymablock as a reference.

We use the full-MPO formulation because the desired answer is the operator $-M_z^2/\Delta$, not its action on a few selected states.
Once constructed, this effective interaction can be reused in a later tensor-network calculation.

## Model and analytical result

For an open chain, define

$$
H_\mathrm{I}
=-J\sum_{i=1}^{L-1}Z_iZ_{i+1}
-h\sum_{i=1}^{L}Z_i,
\qquad
M_z=\sum_{i=1}^{L}Z_i.
$$

We use these operators in the block Hamiltonian

$$
H(g)=
\begin{pmatrix}
H_\mathrm{I} & gM_z\\
gM_z & H_\mathrm{I}+\Delta I
\end{pmatrix}.
$$

Because $H_\mathrm{I}$ and $M_z$ commute, every Ising configuration reduces this Hamiltonian to an ordinary $2\times2$ matrix.
For $\Delta>0$, the exact lower block after diagonalization is consequently

$$
H_\mathrm{exact}^{AA}(g)
=H_\mathrm{I}
+\frac{\Delta-\sqrt{\Delta^2+4g^2M_z^2}}{2}.
$$

Expanding the square root gives

$$
H_\mathrm{exact}^{AA}(g)
=H_\mathrm{I}
-\frac{g^2}{\Delta}M_z^2
+\frac{g^4}{\Delta^3}M_z^4
+\mathcal{O}(g^6).
$$

We will ask Pymablock for the second-order coefficient and verify that it equals $-M_z^2/\Delta$.

```{code-cell} ipython3
%%time
from functools import reduce
from itertools import product

import numpy as np
from tenpy.networks.site import SpinHalfSite

from tenpy_mpo_backend import TenpyMPOBackend, mpo_to_dense, product_mpo
from pymablock import block_diagonalize
from pymablock.mpo import BackendMPO, make_mpo_sylvester_solver
```

## Build the Ising MPO

We use six sites, which is large enough to exercise repeated MPO addition and multiplication while keeping the analytical check quick.
The helper `product_term` constructs one Pauli string, and `add_all` forms a compressed sum of such strings.

```{code-cell} ipython3
%%time
L = 6
J = 0.7
h = 0.2
Delta = 5.0

sites = [SpinHalfSite(conserve=None) for _ in range(L)]
identity = np.eye(2)
z = np.diag([1.0, -1.0])

backend = TenpyMPOBackend(
    chi_max=64,
    svd_min=1e-12,
    krylov_dimension=8,
    max_restarts=10,
    solver_tolerance=1e-9,
)


def product_term(operators):
    """Construct a product operator from ``site: local_operator`` entries."""
    return product_mpo(
        sites,
        [operators.get(site, identity) for site in range(L)],
    )


def add_all(operators):
    """Add and compress a nonempty sequence of MPOs."""
    return reduce(backend.add, operators)


magnetization = add_all(
    [product_term({site: z}) for site in range(L)]
)
ising_bonds = add_all(
    [product_term({site: z, site + 1: z}) for site in range(L - 1)]
)
ising = backend.add(
    backend.scale(ising_bonds, -J),
    backend.scale(magnetization, -h),
)
identity_mpo = product_term({})
upper_block = backend.add(
    ising,
    backend.scale(identity_mpo, Delta),
)
```

## Compute the effective Hamiltonian

Pymablock treats $g$ as the formal perturbative parameter, so the off-diagonal input is $M_z$ rather than $gM_z$.
Requesting the second-order term triggers both the MPO Sylvester solve and the product that generates the effective interaction.

```{code-cell} ipython3
%%time
wrapped_ising = BackendMPO(ising, backend)
wrapped_upper = BackendMPO(upper_block, backend)
wrapped_magnetization = BackendMPO(magnetization, backend)

solve_sylvester = make_mpo_sylvester_solver(
    [wrapped_ising, wrapped_upper],
    backend.solve_sylvester,
    max_relative_residual=1e-8,
)

H_tilde, _, _ = block_diagonalize(
    [
        [[wrapped_ising, 0], [0, wrapped_upper]],
        [
            [0, wrapped_magnetization],
            [wrapped_magnetization, 0],
        ],
    ],
    solve_sylvester=solve_sylvester,
)
H_AA_2 = H_tilde[0, 0, 2]
```

We next compare the computed coefficient with the analytical MPO $-M_z^2/\Delta$.
Dense contraction is used only to measure the error of this small benchmark; neither Pymablock nor the Sylvester solver uses it.

```{code-cell} ipython3
%%time
expected_h_aa_2 = backend.scale(
    backend.matmul(magnetization, magnetization),
    -1 / Delta,
)
computed_dense = mpo_to_dense(H_AA_2.operator)
expected_dense = mpo_to_dense(expected_h_aa_2)
coefficient_relative_error = (
    np.linalg.norm(computed_dense - expected_dense)
    / np.linalg.norm(expected_dense)
)
maximum_bond_dimension = max(
    max(record.output_bond_dimensions)
    for record in backend.compression_records
)
sylvester_residual = backend.solver_records[-1].relative_residuals[-1]

assert coefficient_relative_error < 1e-8
assert sylvester_residual < 1e-8
assert maximum_bond_dimension <= backend.chi_max

{
    "coefficient relative error": coefficient_relative_error,
    "maximum bond dimension": maximum_bond_dimension,
    "Sylvester residual": sylvester_residual,
}
```

## Check the perturbative error

The analytical solution also tells us how the truncated effective Hamiltonian should fail.
For every Ising configuration with energy $E$ and magnetization $m$, we compare

$$
E+\frac{\Delta-\sqrt{\Delta^2+4g^2m^2}}{2}
\quad\text{with}\quad
E-\frac{g^2m^2}{\Delta}.
$$

Halving $g$ should reduce the leading fourth-order error by a factor approaching $2^4=16$.

```{code-cell} ipython3
%%time
configurations = np.asarray(list(product((1.0, -1.0), repeat=L)))
magnetizations = configurations.sum(axis=1)
ising_energies = (
    -J
    * np.sum(
        configurations[:, :-1] * configurations[:, 1:],
        axis=1,
    )
    - h * magnetizations
)


def maximum_energy_error(coupling):
    """Return the largest second-order energy error over all spin states."""
    exact = ising_energies + (
        Delta
        - np.sqrt(
            Delta**2
            + 4 * coupling**2 * magnetizations**2
        )
    ) / 2
    second_order = (
        ising_energies
        - coupling**2 * magnetizations**2 / Delta
    )
    return np.max(np.abs(exact - second_order))


couplings = np.array([0.4, 0.2, 0.1, 0.05])
energy_errors = np.array(
    [maximum_energy_error(coupling) for coupling in couplings]
)
observed_orders = np.log2(energy_errors[:-1] / energy_errors[1:])

assert observed_orders[-1] > 3.9

{
    "couplings": couplings,
    "maximum energy errors": energy_errors,
    "observed orders": observed_orders,
}
```

The coefficient agreement verifies the MPO perturbation calculation, while the approach to fourth-order error verifies its physical interpretation.
For larger chains the same calculation remains useful only while the measured bond dimensions and Sylvester iterations grow slowly enough to beat a dense or state-targeting calculation.
