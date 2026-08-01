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

# Reusing perturbation theory across a DMRG parameter sweep

This tutorial has two goals:

- **What you will learn:** how to compute a ground-state energy response once
  with MPO perturbation theory, reuse it across many coupling values, and make
  a fair timing and accuracy comparison with warm-started DMRG.
- **Physics motivation:** how weak Ising exchange lowers the energy of a
  field-polarized spin chain through virtual neighboring-spin flips, before
  the chain approaches its quantum critical point.

The useful regime is deliberately specific. We want a weak-coupling response
at many parameter values, not a single nonperturbative ground state. In this
setting perturbation theory does one linear-response calculation, while a
DMRG sweep must solve a new variational problem at every coupling even when
each run starts from the previous ground state.

## Model and perturbative prediction

Consider an open spin-$\frac12$ chain,

$$
\mathcal H(\lambda)=H_0+\mathcal H'(\lambda),
\qquad
\mathcal H'(\lambda)=\lambda H'_1,
\qquad
H_0=-h\sum_{i=1}^{L}Z_i,
\qquad
H'_1=-\sum_{i=1}^{L-1}X_iX_{i+1}.
$$

At $\lambda=0$, the unique ground state
$\lvert\phi_0\rangle=\lvert\uparrow\cdots\uparrow\rangle$ has energy
$E_0=-hL$. It is the only column of $\Psi_E$ and spans the explicit subspace
$E$, with $P_I=1-\Psi_E\Psi_E^\dagger$ projecting onto the implicit subspace
$I$.
Each term in $H'_1$ flips two neighboring spins and creates an
excitation of energy $4h$. The first-order correction vanishes, while the
$L-1$ orthogonal two-flip states give

$$
E(\lambda)
=-hL-\frac{L-1}{4h}\lambda^2+\mathcal{O}(\lambda^4).
$$

After rotating the spin axes, this is the transverse-field Ising chain. Its
critical point is at $\lambda/h=1$ in the thermodynamic limit. We stay at
$\lambda/h\leq0.3$, where the polarized phase is gapped and second order is
accurate.

```{code-cell}
%%time
from functools import reduce
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from tenpy.algorithms import dmrg
from tenpy.models.lattice import Chain
from tenpy.models.model import MPOModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy_implicit_backend import TenpyImplicitBackend
from pymablock.backends.tenpy import product_mpo

from pymablock.implicit import block_diagonalize_implicit
```

## Build one MPO for both methods

Both calculations below use exactly the same TeNPy MPOs. This matters for the
comparison: the only difference is whether we solve one perturbative response
problem or repeatedly minimize $H_0+\lambda H'_1$.

```{code-cell}
%%time
L = 32
h = 1.0

site = SpinHalfSite(conserve=None)
sites = [site] * L
identity = np.eye(2)
x = np.array([[0.0, 1.0], [1.0, 0.0]])
z = np.diag([1.0, -1.0])

backend = TenpyImplicitBackend(
    chi_max=32,
    svd_min=1e-12,
    max_sweeps=10,
    solver_tolerance=1e-10,
)


def product_term(operators):
    """Construct a Pauli string from ``site: local_operator`` entries."""
    return product_mpo(
        sites,
        [operators.get(position, identity) for position in range(L)],
    )


def add_all(operators):
    """Add and compress a nonempty collection of MPOs."""
    return reduce(backend.mpo_backend.add, operators)


h_0 = backend.mpo_backend.scale(
    add_all([product_term({position: z}) for position in range(L)]),
    -h,
)
perturbation = backend.mpo_backend.scale(
    add_all([product_term({position: x, position + 1: x}) for position in range(L - 1)]),
    -1,
)
reference = MPS.from_product_state(
    sites,
    ["up"] * L,
    bc="finite",
    unit_cell_width=L,
)

{
    "sites": L,
    "Hilbert-space dimension": 2**L,
    "unperturbed MPO bond dimension": max(h_0.chi),
    "perturbation MPO bond dimension": max(perturbation.chi),
}
```

## Compute the response once

The implicit MPO formulation never constructs an explicit basis for the
implicit subspace $I$ of the $2^L$-dimensional Hilbert space.
It applies the Hamiltonian MPO to MPS response states and solves

$$
P_I(H_0-E_0)P_I\lvert v_{1,0}\rangle
=\lvert f_{1,0}\rangle,
\qquad
\lvert f_{1,0}\rangle
=P_IF_1^{IE}\lvert\phi_0\rangle
=-P_IH'_1\lvert\phi_0\rangle,
$$

where
$\lvert v_{1,0}\rangle=P_IV_1^{IE}\lvert\phi_0\rangle$
and $F_1^{IE}$ denotes the known right-hand side passed to the Sylvester solver.
The variational two-site sweeps solve for this first-order coefficient of the
anti-Hermitian series $\mathcal V$. Pymablock then contracts it with
lower-order terms to obtain the scalar coefficient $\tilde H_2^{EE}$ in the
explicit one-state subspace.

Timing begins after the common MPO construction. Requesting second order
evaluates all required lower orders lazily.

```{code-cell}
%%time
pt_start = perf_counter()
h_tilde, unitary, _ = block_diagonalize_implicit(
    [h_0, perturbation],
    [reference],
    backend,
    backend.solve_shifted,
    max_relative_residual=1e-9,
)
second_order_coefficient = float(np.real(h_tilde[0, 0, 2].dense[0, 0]))
pt_time = perf_counter() - pt_start

expected_coefficient = -(L - 1) / (4 * h)
np.testing.assert_allclose(
    second_order_coefficient,
    expected_coefficient,
    rtol=1e-10,
    atol=1e-10,
)

maximum_residual = max(record.relative_residuals[-1] for record in backend.solver_records)
maximum_reference_overlap = max(
    record.orthogonality_errors[-1] for record in backend.solver_records
)
response_bond_dimension = max(unitary[1, 0, 1].states[0].chi)

assert maximum_residual < 1e-9
assert maximum_reference_overlap < 1e-9

{
    "computed coefficient": second_order_coefficient,
    "analytical coefficient": expected_coefficient,
    "response MPS bond dimension": response_bond_dimension,
    "maximum global residual": maximum_residual,
    "maximum reference overlap": maximum_reference_overlap,
    "PT wall time [s]": pt_time,
}
```

The coefficient is the zero-coupling energy curvature divided by two:
$E''(0)/2=-(L-1)/(4h)$. Once it is known, evaluating the second-order energy at
any number of couplings is just scalar arithmetic.

## Warm-start a DMRG sweep

For the comparison, we increase $\lambda$ monotonically and pass the optimized
MPS from each point to the next DMRG run. This is the standard warm-start
strategy and avoids restarting from a product state. We use two-site DMRG,
briefly enable the mixer for the first point, and switch it off for the
already-entangled warm starts. We retain sweep statistics so that convergence
is checked rather than inferred from the final energy alone.

The timer covers only DMRG itself, excluding even the inexpensive
parameter-dependent MPO addition. This choice favors the DMRG baseline.

```{code-cell}
%%time
couplings = np.linspace(0.025, 0.3, 12) * h
chain = Chain(L, site, bc="open", bc_MPS="finite")
warm_state = reference.copy()
dmrg_options = {
    "active_sites": 2,
    "max_sweeps": 8,
    "N_sweeps_check": 1,
    "max_E_err": 1e-11,
    "max_S_err": 1e-9,
    "trunc_params": {
        "chi_max": 32,
        "svd_min": 1e-12,
    },
}

dmrg_energies = []
dmrg_point_times = []
dmrg_sweeps = []
dmrg_max_chi = []
dmrg_final_relative_energy_changes = []
dmrg_final_entropy_changes = []
dmrg_final_norm_errors = []
dmrg_final_truncation_errors = []

for point, coupling in enumerate(couplings):
    hamiltonian = backend.mpo_backend.add(
        h_0,
        backend.mpo_backend.scale(perturbation, coupling),
    )
    model = MPOModel(chain, hamiltonian)
    point_options = dmrg_options | {"mixer": False}
    if point == 0:
        point_options |= {
            "mixer": True,
            "mixer_params": {
                "amplitude": 1e-5,
                "decay": 2.0,
                "disable_after": 2,
            },
        }

    point_start = perf_counter()
    info = dmrg.run(warm_state, model, point_options)
    dmrg_point_times.append(perf_counter() - point_start)

    stats = info["sweep_statistics"]
    dmrg_energies.append(float(np.real(info["E"])))
    dmrg_sweeps.append(int(stats["sweep"][-1]))
    dmrg_max_chi.append(int(stats["max_chi"][-1]))
    dmrg_final_relative_energy_changes.append(
        abs(stats["Delta_E"][-1] / max(abs(stats["E"][-1]), 1.0))
    )
    dmrg_final_entropy_changes.append(abs(stats["Delta_S"][-1]))
    dmrg_final_norm_errors.append(float(stats["norm_err"][-1]))
    dmrg_final_truncation_errors.append(float(stats["max_trunc_err"][-1]))

dmrg_energies = np.asarray(dmrg_energies)
dmrg_point_times = np.asarray(dmrg_point_times)
dmrg_cumulative_times = np.cumsum(dmrg_point_times)

assert max(dmrg_final_relative_energy_changes) < dmrg_options["max_E_err"]
assert max(dmrg_final_entropy_changes) < dmrg_options["max_S_err"]
assert max(dmrg_final_norm_errors) < 1e-8
assert max(dmrg_final_truncation_errors) < 1e-10

{
    "total DMRG wall time [s]": dmrg_cumulative_times[-1],
    "PT wall time [s]": pt_time,
    "DMRG/PT time ratio": dmrg_cumulative_times[-1] / pt_time,
    "DMRG sweeps per point": dmrg_sweeps,
    "largest DMRG bond dimension": max(dmrg_max_chi),
    "largest final norm error": max(dmrg_final_norm_errors),
    "largest final truncation error": max(dmrg_final_truncation_errors),
}
```

## Check accuracy against the exact chain

For open boundaries, the exact ground-state energy is minus the sum of the
singular values of a bidiagonal Jordan--Wigner matrix with diagonal $h$ and
subdiagonal $\lambda$. This gives an independent reference for both methods.

```{code-cell}
%%time
def exact_ground_energy(coupling):
    """Return the exact open-chain ground-state energy."""
    jordan_wigner = np.diag(np.full(L, h))
    jordan_wigner += np.diag(np.full(L - 1, coupling), k=-1)
    return -np.linalg.svd(jordan_wigner, compute_uv=False).sum()


exact_energies = np.array([exact_ground_energy(coupling) for coupling in couplings])
pt_energies = -h * L + couplings**2 * second_order_coefficient

pt_errors = np.abs(pt_energies - exact_energies)
dmrg_errors = np.abs(dmrg_energies - exact_energies)
energy_lowering = np.abs(exact_energies + h * L)
pt_relative_response_errors = pt_errors / energy_lowering

assert np.max(pt_relative_response_errors) < 0.025
assert np.max(dmrg_errors) < 1e-8

{
    "largest PT error in energy lowering": np.max(pt_relative_response_errors),
    "largest absolute DMRG error": np.max(dmrg_errors),
    "number of reused PT evaluations": len(couplings),
}
```

## Accuracy and cost

The left panel asks whether second order resolves the desired physics; the
right panel shows the cost of sampling more coupling values. The PT line is
horizontal because the response coefficient is computed once. The DMRG line
accumulates one warm-started optimization per point.

```{code-cell}
%%time
figure, (energy_axis, time_axis) = plt.subplots(1, 2, figsize=(10, 4))

energy_axis.plot(
    couplings / h,
    (exact_energies + h * L) / L,
    color="black",
    label="exact",
)
energy_axis.plot(
    couplings / h,
    (pt_energies + h * L) / L,
    "o--",
    label="second-order MPO PT",
)
energy_axis.plot(
    couplings / h,
    (dmrg_energies + h * L) / L,
    "x",
    label="warm-start DMRG",
)
energy_axis.set(
    xlabel=r"Exchange $\lambda/h$",
    ylabel=r"Energy lowering $(E-E_0)/L$",
)
energy_axis.grid(alpha=0.25)
energy_axis.legend()

number_of_points = np.arange(1, len(couplings) + 1)
time_axis.plot(
    number_of_points,
    dmrg_cumulative_times,
    "o-",
    label="warm-start DMRG",
)
time_axis.axhline(
    pt_time,
    color="C1",
    linestyle="--",
    label="one MPO PT solve",
)
time_axis.set(
    xlabel="Number of coupling values",
    ylabel="Cumulative wall time [s]",
    xticks=[1, 4, 8, 12],
)
time_axis.grid(alpha=0.25)
time_axis.legend()

figure.tight_layout()
plt.show()
```

Across this weak-coupling interval, second-order perturbation theory captures
the energy lowering to within 2.5%, while DMRG reproduces the exact finite-chain
energies to the requested numerical precision. The methods answer different
numerical questions: DMRG is more accurate at each sampled point, but it pays
for that accuracy repeatedly. MPO perturbation theory computes the response
coefficient once and reuses it.

The measured time ratio is hardware dependent, but the scaling distinction is
not. For $N_\lambda$ sample points, this tutorial performs one PT response
solve versus $N_\lambda$ DMRG minimizations. The timing panel exposes the
resulting break-even point on the machine that executes the notebook.

## When this is—and is not—the efficient approach

Use the perturbative route when:

- the unperturbed state or low-energy subspace is known;
- a gap separates it from eliminated excitations;
- the goal is a response coefficient or a dense scan over weak couplings; and
- residuals, bond dimensions, and perturbative order are converged.

Use warm-started DMRG instead when a single high-accuracy ground state is
needed, the coupling is not small, the gap closes, or observables depend on
nonperturbative changes of the wavefunction. In this model, the second-order
curve must not be extrapolated toward the critical point $\lambda/h=1$.

## Conclusion

The virtual process in this chain is simple: the exchange flips a neighboring
pair, pays an energy $4h$, and lowers the polarized ground-state energy by
$-(L-1)\lambda^2/(4h)$. Pymablock obtains that coefficient from one MPO/MPS
response calculation without constructing the complementary Hilbert space.

For a weak-coupling parameter sweep, that coefficient is the reusable result.
Warm-starting makes consecutive DMRG runs cheaper, but it does not remove the
need for one variational optimization per coupling. The tutorial therefore
demonstrates the intended gain: when fixed-order accuracy is sufficient, MPO
perturbation theory turns a repeated ground-state search into one response
solve followed by negligible-cost evaluations.
