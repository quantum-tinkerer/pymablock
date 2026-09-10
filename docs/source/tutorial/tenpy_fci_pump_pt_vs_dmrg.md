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

# Local charge pumping in the Haldane FCI model

This tutorial has two goals:

- **What you will learn:** how to obtain the linear pumped-charge response
  from one MPO perturbation-theory solve and reuse it across a fine flux grid,
  then compare its accuracy and cost with sequential warm-started DMRG.
- **Physics motivation:** how Laughlin flux insertion transfers charge across
  a cylinder in the hardcore-boson Haldane model used as a minimal fractional
  Chern insulator (FCI).

We adapt TeNPy's
[Haldane FCI example](https://tenpy.readthedocs.io/en/latest/examples/chern_insulators/haldane_FCI.html),
which follows the ground state while threading flux through an infinite
cylinder. The original calculation is the right tool for a complete pump
cycle. Here we ask a narrower question: what is the initial charge-transfer
slope, and how cheaply can we sample the weak-flux curve?

The executable benchmark uses a $2\times2$-unit-cell finite cylinder with
eight hardcore-boson sites. This small system makes the comparison fast and
allows exact diagonalization, but it is not large enough to establish
thermodynamic FCI order or a quantized pumped charge. It tests the numerical
strategy on the same interacting Haldane Hamiltonian.

## Flux insertion and pumped charge

The Hamiltonian is

$$
H(f)
=\sum_{ij} t_{ij}(f)b_i^\dagger b_j+\mathrm{H.c.},
\qquad f=\frac{\Phi_y}{2\pi}.
$$

Nearest-neighbor hopping has amplitude $t_1$. Complex next-nearest-neighbor
hopping $t_2$ breaks time-reversal symmetry and produces a nearly flat Chern
band. TeNPy inserts flux by multiplying every hopping that crosses the
periodic $y$ boundary by $\exp(\pm 2\pi i f)$.

We split the finite cylinder at its center and define

$$
N_L=\sum_{i\in\mathrm{left}}n_i,
\qquad
\Delta Q_L(f)=\langle N_L\rangle_f-\langle N_L\rangle_0.
$$

Positive $\Delta Q_L$ means charge is pumped into the left half. Around
$f=0$,

$$
\Delta Q_L(f)=f\,\chi_Q+\mathcal{O}(f^2),
\qquad
\chi_Q=2\,\mathrm{Re}\langle\psi_0|N_L|\psi_1\rangle,
$$

where $|\psi_1\rangle$ is the first-order ground-state response. One
perturbative solve determines $\chi_Q$ for every point in the local flux
window.

```{code-cell}
%%time
from functools import reduce
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from tenpy.algorithms import dmrg
from tenpy.models.haldane import BosonicHaldaneModel
from tenpy.models.model import MPOModel
from tenpy.networks.mps import MPS
from tenpy_implicit_backend import TenpyImplicitBackend

from pymablock.backends.tenpy import (
    mpo_to_dense,
    open_boundary_mpo,
    product_mpo,
)
from pymablock.implicit import block_diagonalize_implicit
```

## Build the finite-cylinder Hamiltonian

The original example uses charge conservation to fix quarter filling.
The educational implicit backend currently supports only finite,
charge-unconstrained MPSs. We therefore add

$$
H_{\mathrm{sector}}=\kappa(N-N_\mathrm{target})^2.
$$

Since $[H(f),N]=0$, this penalty vanishes identically in the target sector and
does not change its eigenstates or energies. It only lifts states with the
wrong total particle number. We verify the resulting particle number below.

```{code-cell}
%%time
Lx = 2
Ly = 2
number_of_sites = 2 * Lx * Ly
target_particles = number_of_sites // 4
sector_penalty = 5.0

t1 = 1.0
haldane_phase = np.arccos(3 * np.sqrt(3 / 43))
t2 = (np.sqrt(129) / 36) * t1 * np.exp(1j * haldane_phase)

model_parameters = {
    "conserve": None,
    "t1": t1,
    "t2": t2,
    "mu": 0,
    "V": 0,
    "bc_MPS": "finite",
    "order": "default",
    "Lx": Lx,
    "Ly": Ly,
    "bc_y": "cylinder",
}


def model_at_flux(flux):
    """Construct the Haldane model at ``flux = Phi_y / (2 pi)``."""
    return BosonicHaldaneModel(model_parameters | {"phi_ext": flux})


models = {flux: model_at_flux(flux) for flux in (0.0, 0.25, 0.5)}
flux_mpos = {flux: open_boundary_mpo(model.H_MPO) for flux, model in models.items()}

backend = TenpyImplicitBackend(
    chi_max=32,
    svd_min=1e-12,
    max_sweeps=10,
    solver_tolerance=1e-8,
)
sites = flux_mpos[0.0].sites
identity = np.eye(sites[0].dim)
number = sites[0].get_op("N").to_ndarray()


def product_term(operators):
    """Construct a product operator from ``site: local_operator`` entries."""
    return product_mpo(
        sites,
        [operators.get(position, identity) for position in range(number_of_sites)],
    )


def add_all(operators):
    """Add and compress a nonempty collection of MPOs."""
    return reduce(backend.mpo_backend.add, operators)


identity_mpo = product_term({})
total_number = add_all(
    [product_term({position: number}) for position in range(number_of_sites)]
)
number_offset = backend.mpo_backend.add(
    total_number,
    backend.mpo_backend.scale(identity_mpo, -target_particles),
)
number_penalty = backend.mpo_backend.matmul(number_offset, number_offset)
```

Every flux-dependent hopping contains only the harmonics
$\exp(\pm2\pi i f)$. Three model evaluations therefore reconstruct the exact
flux dependence:

$$
H_{\mathrm{Haldane}}(f)
=H_{\mathrm{bulk}}
+\cos(2\pi f)H_{\cos}
+\sin(2\pi f)H_{\sin}.
$$

The perturbation entering first-order PT is
$H_1=\partial_f H|_{f=0}=2\pi H_{\sin}$.

```{code-cell}
%%time
haldane_zero = flux_mpos[0.0]
haldane_half = flux_mpos[0.5]
haldane_quarter = flux_mpos[0.25]

h_bulk = backend.mpo_backend.scale(
    backend.mpo_backend.add(haldane_zero, haldane_half),
    0.5,
)
h_cos = backend.mpo_backend.scale(
    backend.mpo_backend.add(
        haldane_zero,
        backend.mpo_backend.scale(haldane_half, -1),
    ),
    0.5,
)
h_sin = backend.mpo_backend.add(
    haldane_quarter,
    backend.mpo_backend.scale(h_bulk, -1),
)
h_sector = backend.mpo_backend.scale(number_penalty, sector_penalty)
h_0 = backend.mpo_backend.add(haldane_zero, h_sector)
h_1 = backend.mpo_backend.scale(h_sin, 2 * np.pi)

left_positions = [
    position
    for position in range(number_of_sites)
    if models[0.0].lat.mps2lat_idx(position)[0] < Lx / 2
]
left_number = add_all([product_term({position: number}) for position in left_positions])

{
    "sites": number_of_sites,
    "target particles": target_particles,
    "Hilbert-space dimension": 2**number_of_sites,
    "H(0) MPO bond dimension": max(h_0.chi),
    "flux derivative MPO bond dimension": max(h_1.chi),
    "left-half sites in MPS order": left_positions,
}
```

## Obtain the reference state

Both methods need the ground state at $f=0$. We compute it once with two-site
DMRG, canonicalize it before measuring observables, and exclude this common
setup cost from the timing comparison.

The small cylinder also permits exact diagonalization. We use it to check the
reference energy, the many-body gap, and every pumped-charge value in the
later scan.

```{code-cell}
%%time
product_state = [1 if position % 4 == 0 else 0 for position in range(number_of_sites)]
reference = MPS.from_product_state(
    sites,
    product_state,
    bc="finite",
    unit_cell_width=number_of_sites,
)
reference_model = MPOModel(models[0.0].lat, h_0)
reference_options = {
    "active_sites": 2,
    "mixer": True,
    "mixer_params": {
        "amplitude": 1e-5,
        "decay": 2.0,
        "disable_after": 2,
    },
    "trunc_params": {
        "chi_max": 32,
        "svd_min": 1e-12,
    },
    "N_sweeps_check": 1,
    "max_E_err": 1e-11,
    "max_S_err": 1e-9,
    "max_sweeps": 20,
}

reference_start = perf_counter()
reference_info = dmrg.run(reference, reference_model, reference_options)
reference_time = perf_counter() - reference_start
reference.canonical_form()

reference_energy = float(np.real(reference_info["E"]))
reference_stats = reference_info["sweep_statistics"]
reference_variance = abs(h_0.variance(reference, exp_val=reference_energy))
measured_particles = float(np.real(total_number.expectation_value(reference)))
reference_left_charge = float(np.real(left_number.expectation_value(reference)))

h_0_dense = mpo_to_dense(h_0)
left_number_dense = mpo_to_dense(left_number)
reference_eigenvalues, reference_eigenvectors = np.linalg.eigh(h_0_dense)
exact_reference_energy = reference_eigenvalues[0]
exact_reference = reference_eigenvectors[:, 0]
many_body_gap = reference_eigenvalues[1] - reference_eigenvalues[0]
exact_reference_left_charge = float(
    np.real(exact_reference.conj() @ left_number_dense @ exact_reference)
)

assert reference_stats["sweep"][-1] < reference_options["max_sweeps"]
assert (
    abs(reference_stats["Delta_E"][-1] / max(abs(reference_stats["E"][-1]), 1.0))
    < reference_options["max_E_err"]
)
assert abs(reference_stats["Delta_S"][-1]) < reference_options["max_S_err"]
assert np.linalg.norm(reference.norm_test()) < 1e-8
assert reference_variance < 1e-10
np.testing.assert_allclose(measured_particles, target_particles, atol=1e-10)
np.testing.assert_allclose(
    reference_energy,
    exact_reference_energy,
    atol=1e-10,
)

{
    "reference DMRG wall time [s]": reference_time,
    "reference sweeps": int(reference_stats["sweep"][-1]),
    "reference bond dimension": max(reference.chi),
    "energy variance": reference_variance,
    "measured particle number": measured_particles,
    "exact many-body gap": many_body_gap,
}
```

## Compute the pumped-charge slope once

Pymablock solves the projected response equation

$$
Q(H_0-E_0)Q|\psi_1\rangle=-QH_1|\psi_0\rangle,
\qquad
Q=1-|\psi_0\rangle\langle\psi_0|.
$$

The gap found above makes this linear problem well defined. The backend stores
$|\psi_1\rangle$ as an MPS and checks its global residual and overlap with the
reference state.

```{code-cell}
%%time
pt_start = perf_counter()
_, unitary, _ = block_diagonalize_implicit(
    [h_0, h_1],
    [reference],
    backend,
    backend.solve_shifted,
    max_relative_residual=1e-7,
    reference_tolerance=1e-7,
)
first_order_response = unitary[1, 0, 1].states[0]
charge_slope = 2 * np.real(
    backend.inner(
        reference,
        backend.apply(left_number, first_order_response),
    )
)
pt_time = perf_counter() - pt_start

maximum_pt_residual = max(
    record.relative_residuals[-1] for record in backend.solver_records
)
maximum_pt_overlap = max(
    record.orthogonality_errors[-1] for record in backend.solver_records
)

assert maximum_pt_residual < 1e-7
assert maximum_pt_overlap < 1e-7

{
    "pumped-charge slope dQ_L / df": charge_slope,
    "response MPS bond dimension": max(first_order_response.chi),
    "maximum global residual": maximum_pt_residual,
    "maximum reference overlap": maximum_pt_overlap,
    "PT wall time [s]": pt_time,
}
```

Evaluating the PT curve at additional fluxes is now scalar multiplication:
$\Delta Q_L^\mathrm{PT}(f)=f\chi_Q$. No new tensor-network optimization is
required.

## Warm-start DMRG across a fine flux grid

We compare with 40 positive flux values. Each DMRG run starts from the
optimized MPS at the preceding flux. Since the reference calculation already
generated entanglement and disabled the mixer, the scan uses no mixer.

As in the preceding PT-versus-DMRG tutorial, the DMRG timer excludes
construction of the parameter-dependent MPO. This favors the DMRG baseline.

```{code-cell}
%%time
fluxes = np.linspace(0.001, 0.04, 40)
warm_state = reference.copy()
scan_options = {
    "active_sites": 2,
    "mixer": False,
    "trunc_params": {
        "chi_max": 32,
        "svd_min": 1e-12,
    },
    "N_sweeps_check": 1,
    "max_E_err": 1e-11,
    "max_S_err": 1e-9,
    "max_sweeps": 8,
}

dmrg_pumped_charges = []
dmrg_point_times = []
dmrg_sweeps = []
dmrg_bond_dimensions = []
dmrg_relative_energy_changes = []
dmrg_entropy_changes = []
dmrg_norm_errors = []
dmrg_truncation_errors = []

for flux in fluxes:
    hamiltonian = add_all(
        [
            h_bulk,
            backend.mpo_backend.scale(h_cos, np.cos(2 * np.pi * flux)),
            backend.mpo_backend.scale(h_sin, np.sin(2 * np.pi * flux)),
            h_sector,
        ]
    )
    model = MPOModel(models[0.0].lat, hamiltonian)

    point_start = perf_counter()
    info = dmrg.run(warm_state, model, scan_options)
    dmrg_point_times.append(perf_counter() - point_start)

    warm_state.canonical_form()
    stats = info["sweep_statistics"]
    dmrg_pumped_charges.append(
        float(np.real(left_number.expectation_value(warm_state))) - reference_left_charge
    )
    dmrg_sweeps.append(int(stats["sweep"][-1]))
    dmrg_bond_dimensions.append(int(stats["max_chi"][-1]))
    dmrg_relative_energy_changes.append(
        abs(stats["Delta_E"][-1] / max(abs(stats["E"][-1]), 1.0))
    )
    dmrg_entropy_changes.append(abs(stats["Delta_S"][-1]))
    dmrg_norm_errors.append(float(stats["norm_err"][-1]))
    dmrg_truncation_errors.append(float(stats["max_trunc_err"][-1]))

dmrg_pumped_charges = np.asarray(dmrg_pumped_charges)
dmrg_point_times = np.asarray(dmrg_point_times)
dmrg_cumulative_times = np.cumsum(dmrg_point_times)

assert max(dmrg_sweeps) < scan_options["max_sweeps"]
assert max(dmrg_relative_energy_changes) < scan_options["max_E_err"]
assert max(dmrg_entropy_changes) < scan_options["max_S_err"]
assert max(dmrg_norm_errors) < 1e-8
assert max(dmrg_truncation_errors) < 1e-10

{
    "total DMRG wall time [s]": dmrg_cumulative_times[-1],
    "PT wall time [s]": pt_time,
    "DMRG/PT time ratio": dmrg_cumulative_times[-1] / pt_time,
    "DMRG sweeps per point": dmrg_sweeps,
    "largest DMRG bond dimension": max(dmrg_bond_dimensions),
    "largest final norm error": max(dmrg_norm_errors),
    "largest final truncation error": max(dmrg_truncation_errors),
}
```

## Check the charge response against exact diagonalization

We diagonalize the 256-dimensional Hamiltonian at every flux. This check is
specific to the tiny cylinder; it would be impossible in the larger systems
where MPO perturbation theory and DMRG are useful.

```{code-cell}
%%time
h_bulk_dense = mpo_to_dense(h_bulk)
h_cos_dense = mpo_to_dense(h_cos)
h_sin_dense = mpo_to_dense(h_sin)
h_sector_dense = mpo_to_dense(h_sector)

exact_pumped_charges = []
for flux in fluxes:
    hamiltonian_dense = (
        h_bulk_dense
        + np.cos(2 * np.pi * flux) * h_cos_dense
        + np.sin(2 * np.pi * flux) * h_sin_dense
        + h_sector_dense
    )
    _, eigenvectors = np.linalg.eigh(hamiltonian_dense)
    ground_state = eigenvectors[:, 0]
    exact_left_charge = np.real(ground_state.conj() @ left_number_dense @ ground_state)
    exact_pumped_charges.append(float(exact_left_charge - exact_reference_left_charge))

exact_pumped_charges = np.asarray(exact_pumped_charges)
pt_pumped_charges = charge_slope * fluxes
dmrg_errors = np.abs(dmrg_pumped_charges - exact_pumped_charges)
pt_errors = np.abs(pt_pumped_charges - exact_pumped_charges)
pt_relative_errors = pt_errors / np.abs(exact_pumped_charges)

assert np.max(dmrg_errors) < 1e-9
assert np.max(pt_relative_errors) < 0.02

{
    "largest absolute DMRG error": np.max(dmrg_errors),
    "largest relative PT error": np.max(pt_relative_errors),
    "number of reused PT evaluations": len(fluxes),
}
```

## Accuracy and cost

The left panel shows the local pumped-charge curve. The right panel compares
the one-time PT response solve with the accumulated cost of warm-started DMRG.

```{code-cell}
%%time
figure, (charge_axis, time_axis) = plt.subplots(1, 2, figsize=(10, 4))

charge_axis.plot(
    fluxes,
    exact_pumped_charges,
    color="black",
    label="exact",
)
charge_axis.plot(
    fluxes,
    pt_pumped_charges,
    "--",
    label="first-order MPO PT",
)
charge_axis.plot(
    fluxes,
    dmrg_pumped_charges,
    "x",
    markersize=4,
    label="warm-start DMRG",
)
charge_axis.set(
    xlabel=r"Threaded flux $f=\Phi_y/(2\pi)$",
    ylabel=r"Pumped charge $\Delta Q_L$",
)
charge_axis.grid(alpha=0.25)
charge_axis.legend()

number_of_points = np.arange(1, len(fluxes) + 1)
time_axis.plot(
    number_of_points,
    dmrg_cumulative_times,
    label="warm-start DMRG",
)
time_axis.axhline(
    pt_time,
    color="C1",
    linestyle="--",
    label="one MPO PT solve",
)
time_axis.set(
    xlabel="Number of flux values",
    ylabel="Cumulative wall time [s]",
)
time_axis.grid(alpha=0.25)
time_axis.legend()

figure.tight_layout()
plt.show()
```

First-order PT resolves the charge transferred in this weak-flux interval to
within 2%. Its cost is independent of the number of plotted flux values after
the response state has been computed. Warm-starting makes each DMRG point
inexpensive, but the cumulative time still grows with the size of the grid.

## Scope of the gain

Use this perturbative route when the desired quantity is a local Hall
response, a flux derivative, or a densely sampled curve inside a window where
the Taylor error is controlled. Check the response residual, MPS bond
dimension, perturbative window, and the target observable itself.

Use sequential DMRG for the complete FCI pump. The original tutorial scans
$0\leq f\leq2$ so that the entanglement spectrum flows and the pumped charge
reveals the fractional topological response. A Taylor expansion about $f=0$
is neither periodic in flux nor guaranteed to remain on the adiabatic branch
through the full cycle. It cannot establish charge quantization.

## Conclusion

Flux insertion changes only the hopping terms that cross the cylinder seam.
Pymablock solves once for the resulting first-order MPS response and turns the
pumped charge near zero flux into the reusable coefficient
$\chi_Q=d\langle N_L\rangle/df$. A fine local scan then requires only scalar
evaluations.

The comparison isolates the useful gain: for a response coefficient or dense
weak-flux curve, MPO perturbation theory avoids one ground-state optimization
per point. Warm-started DMRG remains the appropriate method when the physics
depends on nonperturbative spectral flow across the full flux cycle.
