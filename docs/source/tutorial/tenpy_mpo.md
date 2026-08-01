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

# MPO perturbation theory from small to large systems

Full-MPO perturbation theory should reproduce familiar dense calculations and remain usable after dense operators become too large to store.
We test both claims with the same Ising chain coupled to a detuned two-level ancilla.
The ancilla couples to the total magnetization, so virtual ancilla excitations mediate long-range interactions between the spins.

```{figure} ising_ancilla_schematic.svg
:alt: An open Ising chain with nearest-neighbor coupling J and longitudinal field h, collectively coupled through its total magnetization to a two-level ancilla with detuning Delta.
:width: 70%

An open Ising chain couples collectively to a detuned ancilla through $gM_z$.
```

We compute the second- and fourth-order coefficients of the effective Hamiltonian within the ancilla ground sector.
For four spins, we verify both coefficients against analytical expressions and dense Pymablock.
For 24 spins, we compute the same coefficients without dense matrices and use them to obtain the magnetization-dependent energy shift.

## Ising chain coupled to a detuned ancilla

We consider an open chain of $L$ spin-$\frac12$ sites with Hamiltonian

$$
H_\mathrm{I}
=-J\sum_{i=1}^{L-1}Z_iZ_{i+1}
-h\sum_{i=1}^{L}Z_i,
\qquad
M_z=\sum_{i=1}^{L}Z_i.
$$

Here $J>0$ is the ferromagnetic coupling, $h$ is a longitudinal field, and $M_z$ is the total magnetization.
The chain couples to a two-level ancilla with ground sector $A$ and excited sector $B$.
Exciting the ancilla costs an energy $\Delta$, while the coupling $gM_z$ flips it:

$$
\mathcal H(g)=H_0+\mathcal H'(g),
\qquad
H_0+\mathcal H'(g)=
\begin{pmatrix}
H_\mathrm{I} & gM_z\\
gM_z & H_\mathrm{I}+\Delta I
\end{pmatrix}.
$$

We treat $g$ perturbatively, retain sector $A$, and eliminate virtual visits to $B$.
Because $H_\mathrm{I}$ and $M_z$ contain only $Z_i$, they commute.
For a spin state with Ising energy $E$ and magnetization $m$, the problem reduces to a $2\times2$ ancilla Hamiltonian with lower eigenvalue

$$
E+\frac{\Delta-\sqrt{\Delta^2+4g^2m^2}}{2}.
$$

This eigenvalue applies independently to every common eigenstate of $H_\mathrm{I}$ and $M_z$.
Replacing $E$ by $H_\mathrm{I}$ and $m$ by $M_z$ therefore gives the block that becomes sector $A$ at $g=0$ and its perturbative expansion:

$$
\tilde{\mathcal H}^{AA}_\mathrm{exact}(g)
=H_\mathrm{I}
+\frac{\Delta-\sqrt{\Delta^2+4g^2M_z^2}}{2}
=H_\mathrm{I}
-\frac{g^2}{\Delta}M_z^2
+\frac{g^4}{\Delta^3}M_z^4
+\mathcal O(g^6).
$$

No step in this reduction assumes a particular chain length.
For every finite $L$, the exact block and the coefficients $-M_z^2/\Delta$ and $M_z^4/\Delta^3$ apply unchanged.
The length changes the allowed magnetizations $m=-L,-L+2,\ldots,L$ and the largest expansion parameter; convergence for every spin state requires $2|g|L/\Delta<1$.
The numerical full-MPO calculation also chooses $\Delta$ large enough to keep the complete spectra of sectors $A$ and $B$ disjoint; we check this length-dependent condition explicitly below.

Since $Z_i^2=I$, already the second-order term contains

$$
M_z^2=LI+2\sum_{i<j}Z_iZ_j.
$$

Virtual ancilla excitations therefore generate interactions between every pair of spins, even though the original Ising coupling is nearest-neighbor.

## Represent the operators as MPOs

We represent the chain operators as MPOs and choose TeNPy as the tensor-network library for this tutorial.
Pymablock organizes the perturbation series, and {autolink}`~pymablock.backends.tenpy.TenpyMPOBackend` connects it to TeNPy.
`TenpyMPOBackend` contains no Hamiltonian itself; it provides MPO arithmetic, compression, and the linear solves that Pymablock needs.

```{code-cell}
%%time
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

from pymablock import block_diagonalize
from pymablock.backends.tenpy import (
    TenpyMPOBackend,
    mpo_to_dense,
    mpo_to_mps,
    product_mpo,
)
from pymablock.mpo import BackendMPO, make_mpo_sylvester_solver
```

We create one backend and share it between all MPOs in both parts of the tutorial.
`chi_max` caps the retained bond dimension and `svd_min` discards smaller singular values, following [TeNPy's truncation options](https://tenpy.readthedocs.io/en/stable/reference/tenpy.linalg.truncation.truncate.html).
The remaining parameters set the Krylov-space dimension, restart limit, and residual tolerance of the restarted Sylvester solver.

```{code-cell}
%%time
backend = TenpyMPOBackend(
    chi_max=64,
    svd_min=1e-12,
    krylov_dimension=8,
    max_restarts=10,
    solver_tolerance=1e-9,
)
```

The following functions contain the full MPO workflow.
The first constructs native TeNPy MPOs for $H_\mathrm{I}$, $M_z$, and the detuned block $H_\mathrm{I}+\Delta I$.
The second pairs each MPO with the shared backend, adapts its Sylvester solver, and returns the lazy transformed Hamiltonian series.

```{code-cell}
def build_ising_mpos(L, J, h, Delta):
    """Construct the three MPOs in the block Hamiltonian."""
    sites = [SpinHalfSite(conserve=None) for _ in range(L)]
    identity = np.eye(2)
    z = np.diag([1.0, -1.0])

    def product_term(operators):
        return product_mpo(
            sites,
            [operators.get(site, identity) for site in range(L)],
        )

    magnetization = reduce(
        backend.add,
        [product_term({site: z}) for site in range(L)],
    )
    ising_bonds = reduce(
        backend.add,
        [product_term({site: z, site + 1: z}) for site in range(L - 1)],
    )
    ising = backend.add(
        backend.scale(ising_bonds, -J),
        backend.scale(magnetization, -h),
    )
    identity_mpo = product_term({})
    detuned = backend.add(
        ising,
        backend.scale(identity_mpo, Delta),
    )
    return sites, ising, detuned, magnetization


def effective_hamiltonian(ising, detuned, magnetization):
    """Return the transformed Hamiltonian as a lazy perturbative series."""
    wrapped_ising = BackendMPO(ising, backend)
    wrapped_detuned = BackendMPO(detuned, backend)
    wrapped_magnetization = BackendMPO(magnetization, backend)

    solve_sylvester = make_mpo_sylvester_solver(
        [wrapped_ising, wrapped_detuned],
        backend.solve_sylvester,
        max_relative_residual=1e-8,
    )
    H_tilde, _, _ = block_diagonalize(
        [
            [[wrapped_ising, 0], [0, wrapped_detuned]],
            [[0, wrapped_magnetization], [wrapped_magnetization, 0]],
        ],
        solve_sylvester=solve_sylvester,
    )
    return H_tilde
```

The two nested $2\times2$ lists in `block_diagonalize` reproduce the unperturbed and first-order block matrices in the model above.
Pymablock treats $g$ as a formal parameter, so we supply $M_z$ as the coefficient of $g$ rather than choosing a numerical coupling.
Requesting fourth order evaluates all required lower orders lazily.
The Sylvester equations used internally are derived on [the algorithm page](../algorithms.md); here the backend solves them directly with compressed MPO arithmetic.

::::{admonition} Using another MPO backend
:class: dropdown tip

Pymablock is not tied to TeNPy.
Any native MPO type can be wrapped in {autolink}`~pymablock.mpo.BackendMPO` when its backend implements the {autolink}`~pymablock.mpo.MPOBackend` operations `add`, `scale`, `matmul`, and `adjoint`.
Full-MPO perturbation theory also needs a compatible Sylvester solver, supplied through {autolink}`~pymablock.mpo.make_mpo_sylvester_solver`.
Replacing the TeNPy MPO construction and backend leaves the `block_diagonalize` call and the indexing of `H_tilde` unchanged.
::::

## Part 1: verify the calculation on a small chain

We begin with $L=4$, where the chain operators are only $16\times16$ matrices.
We fix $J$, $h$, and $\Delta$ here and reuse them for the 24-site calculation; the chosen detuning keeps the two unperturbed sectors separated at both lengths.
The MPO workflow is already the one used for the large system, but at this size we can still contract every operator to a dense matrix.

```{code-cell}
%%time
J = 0.07
h = 0.02
Delta = 5.0
L_small = 4

small_sites, small_ising, small_detuned, small_magnetization = build_ising_mpos(
    L_small,
    J,
    h,
    Delta,
)
H_tilde_small = effective_hamiltonian(
    small_ising,
    small_detuned,
    small_magnetization,
)
H_AA_2_small = H_tilde_small[0, 0, 2]
H_AA_4_small = H_tilde_small[0, 0, 4]

{
    "Hilbert-space dimension": 2**L_small,
    "Ising MPO bond dimension": max(small_ising.chi),
    "second-order bond dimension": max(H_AA_2_small.operator.chi),
    "fourth-order bond dimension": max(H_AA_4_small.operator.chi),
}
```

In `H_tilde_small[0, 0, n]`, the first two indices select the retained $AA$ block and `n` selects the perturbative order.
Requesting `n=4` triggers the lower-order operations that fourth order depends on and caches their results.

### Compare with dense Pymablock and the analytical result

We now compute the same coefficients in three ways:

1. with the TeNPy MPO backend above;
2. with standard dense Pymablock;
3. from the analytical coefficients $-M_z^2/\Delta$ and $M_z^4/\Delta^3$.

The dense route repeats the same Pymablock algorithm with ordinary arrays, while the analytical coefficients provide an independent target.
We first build both references.

```{code-cell}
%%time
dense_ising = mpo_to_dense(small_ising)
dense_detuned = mpo_to_dense(small_detuned)
dense_magnetization = mpo_to_dense(small_magnetization)

H_tilde_dense, _, _ = block_diagonalize(
    [
        [[dense_ising, 0], [0, dense_detuned]],
        [[0, dense_magnetization], [dense_magnetization, 0]],
    ]
)
H_AA_2_dense = H_tilde_dense[0, 0, 2]
H_AA_4_dense = H_tilde_dense[0, 0, 4]

magnetization_squared = dense_magnetization @ dense_magnetization
H_AA_2_analytical = -magnetization_squared / Delta
H_AA_4_analytical = magnetization_squared @ magnetization_squared / Delta**3
```

The dense coefficients test whether substituting MPO arithmetic changes Pymablock's result.
The analytical coefficients test both implementations against the closed-form solution.
We now contract only the two computed MPO coefficients and measure their relative Frobenius errors against both references.

```{code-cell}
%%time
def relative_matrix_error(computed, expected):
    """Return the relative Frobenius error between two matrices."""
    return np.linalg.norm(computed - expected) / np.linalg.norm(expected)


H_AA_2_mpo = mpo_to_dense(H_AA_2_small.operator)
H_AA_4_mpo = mpo_to_dense(H_AA_4_small.operator)
small_chain_errors = {
    "MPO vs dense Pymablock, second order": relative_matrix_error(
        H_AA_2_mpo,
        H_AA_2_dense,
    ),
    "MPO vs dense Pymablock, fourth order": relative_matrix_error(
        H_AA_4_mpo,
        H_AA_4_dense,
    ),
    "MPO vs analytical, second order": relative_matrix_error(
        H_AA_2_mpo,
        H_AA_2_analytical,
    ),
    "MPO vs analytical, fourth order": relative_matrix_error(
        H_AA_4_mpo,
        H_AA_4_analytical,
    ),
}

assert max(small_chain_errors.values()) < 1e-8
small_chain_errors
```

The errors are near floating-point precision, so all three routes give the same effective operators.
The small chain establishes correctness before dense matrices disappear from the calculation.
The cell timings also show that dense Pymablock is faster for this four-site problem.
Here every operator is only $16\times16$, so optimized dense linear algebra finishes quickly, while the MPO calculation still pays the fixed cost of tensor bookkeeping, compression, and iterative Sylvester solves.
MPOs become useful as $L$ grows: dense operator storage increases exponentially, whereas the operators in this example retain small bond dimensions.

## Part 2: continue where dense operators no longer fit

We now change only the chain length, from $L=4$ to $L=24$; the model, parameters, backend, and perturbative orders remain the same.
The spin Hilbert space has $2^{24}$ states, and one complex dense operator would require $4$ PiB of memory.

Before solving, we check that the unperturbed spectra of sectors $A$ and $B$ cannot overlap.
The bound $\lVert H_\mathrm{I}\rVert\leq J(L-1)+hL$ places the Ising spectrum in an interval of width at most $2[J(L-1)+hL]$, which remains smaller than $\Delta$ for our parameters.

```{code-cell}
%%time
L_large = 24
large_records_start = len(backend.solver_records)

large_sites, large_ising, large_detuned, large_magnetization = build_ising_mpos(
    L_large,
    J,
    h,
    Delta,
)
H_tilde_large = effective_hamiltonian(
    large_ising,
    large_detuned,
    large_magnetization,
)
H_AA_2_large = H_tilde_large[0, 0, 2]
H_AA_4_large = H_tilde_large[0, 0, 4]

hilbert_dimension = 2**L_large
dense_storage_pib = np.dtype(complex).itemsize * hilbert_dimension**2 / 2**50
spectral_width_bound = 2 * (J * (L_large - 1) + h * L_large)
assert Delta > spectral_width_bound

{
    "Hilbert-space dimension": hilbert_dimension,
    "one dense complex operator [PiB]": dense_storage_pib,
    "Ising MPO bond dimension": max(large_ising.chi),
    "second-order bond dimension": max(H_AA_2_large.operator.chi),
    "fourth-order bond dimension": max(H_AA_4_large.operator.chi),
}
```

The contrast with dense storage is now visible: the Hilbert-space dimension is more than sixteen million, while the reported MPO bond dimensions remain in the single digits.

### Verify the large system without dense matrices

For the large chain, dense contraction is no longer available, so we compare with direct MPO representations of the analytical coefficients.
Vectorizing an MPO as an MPS preserves its Frobenius norm, so we can measure the operator error without dense contraction.

```{code-cell}
%%time
large_magnetization_squared = backend.matmul(
    large_magnetization,
    large_magnetization,
)
H_AA_2_expected = backend.scale(
    large_magnetization_squared,
    -1 / Delta,
)
H_AA_4_expected = backend.scale(
    backend.matmul(large_magnetization_squared, large_magnetization_squared),
    1 / Delta**3,
)


def relative_frobenius_error(computed, expected):
    """Return ``||computed - expected||_F / ||expected||_F`` for MPOs."""
    computed_vector = mpo_to_mps(computed)
    expected_vector = mpo_to_mps(expected)
    difference = computed_vector.add(
        expected_vector,
        alpha=1,
        beta=-1,
        cutoff=0,
    )
    difference_norm_squared = float(np.real(difference.overlap(difference)))
    expected_norm_squared = float(np.real(expected_vector.overlap(expected_vector)))
    return np.sqrt(max(difference_norm_squared, 0) / expected_norm_squared)


large_chain_errors = {
    "second order": relative_frobenius_error(
        H_AA_2_large.operator,
        H_AA_2_expected,
    ),
    "fourth order": relative_frobenius_error(
        H_AA_4_large.operator,
        H_AA_4_expected,
    ),
}
large_solver_records = backend.solver_records[large_records_start:]
maximum_sylvester_residual = max(
    record.relative_residuals[-1] for record in large_solver_records
)

assert max(large_chain_errors.values()) < 1e-8
assert maximum_sylvester_residual < 1e-8
{
    "relative errors": large_chain_errors,
    "maximum Sylvester residual": maximum_sylvester_residual,
}
```

The relative Frobenius errors test the final effective operators, while the reported residual independently checks the compressed Sylvester solves used to construct them.
Both diagnostics are below the requested tolerance.
The effective operators stay compact because $M_z^2$ and $M_z^4$ have small MPO representations despite containing long-range interactions.

### Read the induced interaction from the effective MPO

For a $Z$-basis product state with total magnetization $m$, the ancilla induces the energy shift

$$
\delta E_\mathrm{exact}(m)
=\frac{\Delta-\sqrt{\Delta^2+4g^2m^2}}{2}.
$$

Both the exact shift and its perturbative approximation depend only on $m$.
All product states with the same magnetization therefore give the same value, so one representative product MPS per allowed $m$ is sufficient.
The expansion converges for every spin state when $2|g|L/\Delta<1$.
We choose $g=0.08$, which lies inside this radius while making the fourth-order improvement visible.

```{code-cell}
%%time
plot_coupling = 0.08
magnetizations = np.arange(-L_large, L_large + 1, 2)
product_states = [
    MPS.from_product_state(
        large_sites,
        ["up"] * ((L_large + magnetization) // 2)
        + ["down"] * ((L_large - magnetization) // 2),
        bc="finite",
        unit_cell_width=L_large,
    )
    for magnetization in magnetizations
]


def real_expectation(operator, state):
    """Evaluate a Hermitian MPO on an MPS."""
    value = operator.expectation_value(state)
    assert abs(np.imag(value)) < 1e-10
    return float(np.real(value))


second_order_shifts = plot_coupling**2 * np.array(
    [real_expectation(H_AA_2_large.operator, state) for state in product_states]
)
fourth_order_shifts = second_order_shifts + plot_coupling**4 * np.array(
    [real_expectation(H_AA_4_large.operator, state) for state in product_states]
)
exact_shifts = (Delta - np.sqrt(Delta**2 + 4 * plot_coupling**2 * magnetizations**2)) / 2

expansion_parameter = 2 * plot_coupling * L_large / Delta
second_order_max_error = np.max(np.abs(second_order_shifts - exact_shifts))
fourth_order_max_error = np.max(np.abs(fourth_order_shifts - exact_shifts))
assert expansion_parameter < 1
assert fourth_order_max_error < second_order_max_error

figure, axis = plt.subplots(figsize=(6, 4))
axis.plot(
    magnetizations / L_large,
    exact_shifts / L_large,
    color="black",
    label="exact",
)
axis.plot(
    magnetizations / L_large,
    second_order_shifts / L_large,
    "o--",
    label="second order",
)
axis.plot(
    magnetizations / L_large,
    fourth_order_shifts / L_large,
    "s:",
    label="fourth order",
)
axis.set(
    xlabel=r"Magnetization density $m/L$",
    ylabel=r"Induced energy shift $\delta E/L$",
)
axis.legend()
figure.tight_layout()
plt.show()

{
    "expansion parameter": expansion_parameter,
    "maximum second-order error": second_order_max_error,
    "maximum fourth-order error": fourth_order_max_error,
}
```

Virtual ancilla excitations lower the energy most strongly for states with large $|m|$.
The second-order term overestimates this lowering, while the positive fourth-order term bends the result toward the exact curve and reduces the maximum error.

## Conclusion

The four-site calculation establishes that MPO Pymablock, dense Pymablock, and the analytical expansion produce the same effective operators.
The 24-site calculation then applies the same workflow where dense operator algebra would require pebibytes of memory.
The result remains a reusable effective Hamiltonian: its compact MPO coefficients describe every magnetization sector and reproduce the analytical second- and fourth-order interactions.
This commuting model is deliberately favorable, so it validates the workflow rather than generic bond-dimension scaling.
For a noncommuting model, repeat the calculation with larger `chi_max`, smaller `svd_min`, and tighter solver tolerances until the requested coefficients are stable.
