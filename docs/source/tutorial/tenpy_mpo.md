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

# Minimal full-MPO validation

This minimal tutorial demonstrates the complete interface between Pymablock and a tensor-network backend.
TeNPy supplies MPO arithmetic and the Sylvester solver; Pymablock uses them to compute a first-order transformation coefficient and a second-order effective-Hamiltonian coefficient.
We use two spin-$\frac12$ sites and an auxiliary two-level sector so that both operator-valued results can be checked against dense Pymablock.

The calculation uses the full-MPO formulation because it represents the coefficients as operators, not only their action on selected states.
The MPO calculation itself never uses a dense representation.
Our interesting subspace is sector $A$: we eliminate the detuned sector $B$ and find the effective Hamiltonian acting within $A$.

## Construct the MPO blocks

We first import the TeNPy backend defined alongside this executable tutorial.
The example uses no conserved charges because the adapter deliberately keeps its first implementation focused on the MPO algebra and Sylvester solver.

```{code-cell}
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

We study two spins and an additional sector label.
Sector $A$ is the low-energy subspace whose effective Hamiltonian we want; the energetically separated sector $B$ will be eliminated.
Both sectors contain the same four-dimensional spin Hilbert space.

We expand in the dimensionless coupling $\lambda$:

$$
H(\lambda)=H_0+\lambda V,\qquad
H_0=
\begin{pmatrix}
A&0\\
0&B
\end{pmatrix},
\qquad
V=
\begin{pmatrix}
0&T\\
T^\dagger&0
\end{pmatrix},
$$

with

$$
A=0.3(Z_1+Z_2),\qquad B=A+3I,
\qquad
T=X_1+0.2X_2.
$$

Here $X_i$ and $Z_i$ are Pauli operators on spin $i$, $I$ is the two-spin identity, and all coefficients use the same energy unit.
The two sectors have the same longitudinal field, but $B$ lies three energy units above $A$.
The perturbation changes sector while flipping spin 1 with amplitude $1$ or spin 2 with amplitude $0.2$.

At first order, Pymablock obtains the off-diagonal transformation from

$$
A U_{AB}^{(1)}-U_{AB}^{(1)}B=-T.
$$

The spectra of $A$ and $B$ are disjoint, so this Sylvester equation has a unique solution.
At second order, $T$ takes a state from $A$ to $B$ and $T^\dagger$ brings it back.
These virtual excursions produce $\widetilde H_{AA}^{(2)}$ even though $V$ has no matrix element within $A$.

```{code-cell}
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
The backend aims for a relative Sylvester residual below $10^{-9}$, while the adapter independently rejects any result above $10^{-8}$.
This second threshold prevents an unconverged compressed result from entering the perturbation series.
For a generic equation $A\mathcal X-\mathcal X B=Y$, both thresholds use

$$
\frac{\lVert Y-(A\mathcal X-\mathcal X B)\rVert_F}{\lVert Y\rVert_F}.
$$

```{code-cell}
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

Pymablock stores each power of $\lambda$ separately, so $\lambda$ does not appear as a numerical variable in the code.
In `U_mpo[0, 1, 1]`, the first two indices select the $AB$ block and the last selects first order.
Likewise, `H_tilde_mpo[0, 0, 2]` selects the second-order coefficient in the target $AA$ block.
Requesting the first term runs the Sylvester solver; requesting the second also requires MPO multiplication.

```{code-cell}
%%time
U_AB_1 = U_mpo[0, 1, 1]
H_AA_2 = H_tilde_mpo[0, 0, 2]
```

## Verify the minimal calculation

We finally contract these two-site MPOs to dense matrices only for validation.
An independent dense Pymablock calculation provides the reference coefficients.

```{code-cell}
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

first_order_error = np.linalg.norm(mpo_to_dense(U_AB_1.operator) - U_dense[0, 1, 1])
second_order_error = np.linalg.norm(
    mpo_to_dense(H_AA_2.operator) - H_tilde_dense[0, 0, 2]
)

assert first_order_error < 1e-8
assert second_order_error < 1e-8

{
    "first-order error": first_order_error,
    "second-order error": second_order_error,
    "Sylvester residual": backend.solver_records[-1].relative_residuals[-1],
}
```

## Conclusion

This calculation establishes the minimal end-to-end path from TeNPy MPOs to a Pymablock perturbation series.
Agreement of $U_{AB}^{(1)}$ validates the Sylvester solve, while agreement of $\widetilde H_{AA}^{(2)}$ also validates MPO multiplication.

The model is deliberately too small to demonstrate a scaling advantage.
The [large-chain tutorial](tenpy_mpo_ising.md) shows the regime where a dense operator is impossible but the perturbative coefficients remain compact MPOs.
