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

# MPO perturbation theory with TeNPy

This tutorial implements finite-chain MPO perturbation theory by combining Pymablock's backend-neutral hooks with TeNPy.
We use two spin-$\frac12$ sites so that the tensor-network calculation remains visible and can be checked independently with dense matrices.
The MPO algorithm itself never uses the dense representation.

We choose the full-MPO formulation because this tutorial computes complete coefficients of both the transformation and the effective Hamiltonian.
This would be useful if those operators were later applied to many states or observables.
“Full” refers to representing the unknown operator as an MPO: the Sylvester solver remains matrix-free and never assembles its exponentially large matrix.

## Construct the MPO blocks

We first import the TeNPy backend defined alongside this executable tutorial.
The example uses no conserved charges because the adapter deliberately keeps its first implementation focused on the MPO algebra and Sylvester solver.

```{code-cell} ipython3
%%time
import numpy as np
from tenpy.networks.site import SpinHalfSite

from tenpy_mpo_backend import (
    TenpyMPOBackend,
    mpo_to_dense,
    product_mpo,
)
from pymablock import block_diagonalize
from pymablock.mpo import BackendMPO, make_mpo_sylvester_solver
```

We choose two unperturbed blocks

$$
A=0.3(Z_1+Z_2),\qquad B=A+3I,
$$

and an off-diagonal perturbation

$$
T=X_1+0.2X_2.
$$

The offset of $3$ makes the spectra of $A$ and $B$ disjoint.

```{code-cell} ipython3
%%time
sites = [SpinHalfSite(conserve=None) for _ in range(2)]
identity = np.eye(2)
x = np.array([[0.0, 1.0], [1.0, 0.0]])
z = np.diag([1.0, -1.0])

backend = TenpyMPOBackend(
    chi_max=64,
    svd_min=1e-12,
    krylov_dimension=8,
    max_restarts=10,
    solver_tolerance=1e-9,
)

identity_mpo = product_mpo(sites, [identity, identity])
z_1 = product_mpo(sites, [z, identity])
z_2 = product_mpo(sites, [identity, z])
x_1 = product_mpo(sites, [x, identity])
x_2 = product_mpo(sites, [identity, x])

A = backend.scale(backend.add(z_1, z_2), 0.3)
B = backend.add(A, backend.scale(identity_mpo, 3))
T = backend.add(x_1, backend.scale(x_2, 0.2))
```

## Connect the backend to Pymablock

We next wrap each native TeNPy MPO in {autolink}`~pymablock.mpo.BackendMPO`.
The wrapper makes TeNPy's compressed arithmetic available to Pymablock without adding TeNPy as a core dependency.

```{code-cell} ipython3
%%time
wrapped_a = BackendMPO(A, backend)
wrapped_b = BackendMPO(B, backend)
wrapped_t = BackendMPO(T, backend)

solve_sylvester = make_mpo_sylvester_solver(
    [wrapped_a, wrapped_b],
    backend.solve_sylvester,
    max_relative_residual=1e-8,
)

H_tilde_mpo, U_mpo, _ = block_diagonalize(
    [
        [[wrapped_a, 0], [0, wrapped_b]],
        [[0, wrapped_t], [wrapped_t.adjoint(), 0]],
    ],
    solve_sylvester=solve_sylvester,
)
```

Pymablock is lazy, so the MPO products and the Sylvester solve occur only when we request perturbative coefficients.
The first-order off-diagonal transformation exercises the Sylvester solver, while the second-order effective Hamiltonian additionally exercises MPO multiplication.

```{code-cell} ipython3
%%time
U_AB_1 = U_mpo[0, 1, 1]
H_AA_2 = H_tilde_mpo[0, 0, 2]

backend.solver_records[-1]
```

## Verify the minimal calculation

We finally contract these two-site MPOs to dense matrices only for validation.
An independent dense Pymablock calculation provides the reference coefficients.

```{code-cell} ipython3
%%time
dense_a = mpo_to_dense(A)
dense_b = mpo_to_dense(B)
dense_t = mpo_to_dense(T)

H_tilde_dense, U_dense, _ = block_diagonalize(
    [
        [[dense_a, 0], [0, dense_b]],
        [[0, dense_t], [dense_t.conj().T, 0]],
    ]
)

first_order_error = np.linalg.norm(
    mpo_to_dense(U_AB_1.operator) - U_dense[0, 1, 1]
)
second_order_error = np.linalg.norm(
    mpo_to_dense(H_AA_2.operator) - H_tilde_dense[0, 0, 2]
)
maximum_bond_dimension = max(
    max(record.output_bond_dimensions)
    for record in backend.compression_records
)

assert first_order_error < 1e-8
assert second_order_error < 1e-8
assert maximum_bond_dimension <= backend.chi_max

{
    "first-order error": first_order_error,
    "second-order error": second_order_error,
    "maximum bond dimension": maximum_bond_dimension,
    "Sylvester residual": backend.solver_records[-1].relative_residuals[-1],
}
```

The agreement checks multiplication and the solution of Sylvester's equation in one minimal calculation.
For a larger system, dense contraction is unavailable, so convergence must instead be established by tightening `svd_min`, increasing `chi_max`, and lowering the accepted Sylvester residual until the requested effective terms stop changing.
