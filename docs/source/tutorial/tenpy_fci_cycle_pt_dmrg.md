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

# Reconstruct a full flux cycle with PT-assisted DMRG

This tutorial has two goals:

- **What you will learn:** how to use local MPO perturbation theory as a
  predictor and derivative calculator, correct only a sparse set of flux
  anchors with DMRG, and reconstruct a dense $2\pi$ flux cycle.
- **Physics motivation:** how charge rearranges during Laughlin flux insertion
  in the interacting hardcore-boson Haldane model, and why a finite unique
  ground state must be distinguished from the fractional pump between
  topological sectors on an infinite cylinder.

We compare two calculations at the same accuracy:

1. warm-started DMRG at every requested flux;
2. six first-order MPO-PT patches, each followed by a DMRG correction at the
   next anchor.

The PT response also supplies the charge derivative at every anchor. Cubic
Hermite interpolation uses the anchor charges and these derivatives to fill
the dense output grid without more ground-state optimizations.

```{code-cell}
%%time
from functools import reduce
from itertools import pairwise
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from tenpy.algorithms import dmrg
from tenpy.models.haldane import BosonicHaldaneModel
from tenpy.models.model import MPOModel
from tenpy.networks.mps import MPS
from tenpy_implicit_backend import TenpyImplicitBackend

from pymablock.backends.tenpy import mpo_to_dense, open_boundary_mpo, product_mpo
from pymablock.implicit import block_diagonalize_implicit
```

## The flux-dependent Haldane Hamiltonian

We write the threaded flux as

$$
f=\frac{\Phi_y}{2\pi}.
$$

A physical $2\pi$ cycle is therefore $0\leq f\leq1$. TeNPy multiplies
hoppings that cross the cylinder seam by $\exp(\pm2\pi i f)$. The Hamiltonian
consequently has the exact form

$$
H(f)=H_{\mathrm{bulk}}
+\cos(2\pi f)H_{\cos}
+\sin(2\pi f)H_{\sin}
+H_{\mathrm{sector}}.
$$

We use the same $2\times2$-unit-cell finite cylinder as the local-response
tutorial. Its eight-site Hilbert space permits exact diagonalization of every
plotted point. This makes the comparison reproducible, but the small cylinder
does not establish an FCI phase or a quantized pump.

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


sample_models = {flux: model_at_flux(flux) for flux in (0.0, 0.25, 0.5)}
sample_mpos = {
    flux: open_boundary_mpo(model.H_MPO) for flux, model in sample_models.items()
}

algebra = TenpyImplicitBackend(
    chi_max=32,
    svd_min=1e-12,
    max_sweeps=10,
    solver_tolerance=1e-8,
)
sites = sample_mpos[0.0].sites
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
    return reduce(algebra.mpo_backend.add, operators)


identity_mpo = product_term({})
total_number = add_all(
    [product_term({position: number}) for position in range(number_of_sites)]
)
number_offset = algebra.mpo_backend.add(
    total_number,
    algebra.mpo_backend.scale(identity_mpo, -target_particles),
)
number_penalty = algebra.mpo_backend.matmul(number_offset, number_offset)
h_sector = algebra.mpo_backend.scale(number_penalty, sector_penalty)

haldane_zero = sample_mpos[0.0]
haldane_quarter = sample_mpos[0.25]
haldane_half = sample_mpos[0.5]
h_bulk = algebra.mpo_backend.scale(
    algebra.mpo_backend.add(haldane_zero, haldane_half),
    0.5,
)
h_cos = algebra.mpo_backend.scale(
    algebra.mpo_backend.add(
        haldane_zero,
        algebra.mpo_backend.scale(haldane_half, -1),
    ),
    0.5,
)
h_sin = algebra.mpo_backend.add(
    haldane_quarter,
    algebra.mpo_backend.scale(h_bulk, -1),
)


def hamiltonian_at_flux(flux):
    """Evaluate the exact harmonic MPO at one flux."""
    return add_all(
        [
            h_bulk,
            algebra.mpo_backend.scale(h_cos, np.cos(2 * np.pi * flux)),
            algebra.mpo_backend.scale(h_sin, np.sin(2 * np.pi * flux)),
            h_sector,
        ]
    )


def flux_derivative_at(flux):
    """Evaluate ``dH / df`` at one flux."""
    return algebra.mpo_backend.add(
        algebra.mpo_backend.scale(
            h_cos,
            -2 * np.pi * np.sin(2 * np.pi * flux),
        ),
        algebra.mpo_backend.scale(
            h_sin,
            2 * np.pi * np.cos(2 * np.pi * flux),
        ),
    )


left_positions = [
    position
    for position in range(number_of_sites)
    if sample_models[0.0].lat.mps2lat_idx(position)[0] < Lx / 2
]
left_number = add_all([product_term({position: number}) for position in left_positions])

{
    "sites": number_of_sites,
    "target particles": target_particles,
    "Hilbert-space dimension": 2**number_of_sites,
    "left-half sites in MPS order": left_positions,
    "H(0) MPO bond dimension": max(hamiltonian_at_flux(0).chi),
}
```

The charge displacement across the central cut is

$$
\Delta Q_L(f)
=\langle N_L\rangle_f-\langle N_L\rangle_0,
\qquad
N_L=\sum_{i\in\mathrm{left}}n_i.
$$

The sector penalty

$$
H_{\mathrm{sector}}=\kappa(N-N_{\mathrm{target}})^2
$$

lets the educational backend use charge-unconstrained MPSs. Since the Haldane
Hamiltonian commutes with $N$, the penalty vanishes in the target sector and
leaves its eigenstates unchanged. We verify the particle number below.

## A common reference state

Both methods start from the same DMRG ground state at $f=0$. We exclude this
shared calculation from the timing comparison.

```{code-cell}
%%time
product_state = [1 if position % 4 == 0 else 0 for position in range(number_of_sites)]
reference = MPS.from_product_state(
    sites,
    product_state,
    bc="finite",
    unit_cell_width=number_of_sites,
)
reference_hamiltonian = hamiltonian_at_flux(0.0)
reference_model = MPOModel(sample_models[0.0].lat, reference_hamiltonian)
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
reference_variance = abs(
    reference_hamiltonian.variance(reference, exp_val=reference_energy)
)
reference_particles = float(np.real(total_number.expectation_value(reference)))
reference_left_charge = float(np.real(left_number.expectation_value(reference)))

assert reference_stats["sweep"][-1] < reference_options["max_sweeps"]
assert (
    abs(reference_stats["Delta_E"][-1] / max(abs(reference_energy), 1.0))
    < reference_options["max_E_err"]
)
assert abs(reference_stats["Delta_S"][-1]) < reference_options["max_S_err"]
assert np.linalg.norm(reference.norm_test()) < 1e-8
assert reference_variance < 1e-10
np.testing.assert_allclose(reference_particles, target_particles, atol=1e-10)

{
    "reference DMRG wall time [s]": reference_time,
    "reference sweeps": int(reference_stats["sweep"][-1]),
    "reference bond dimension": max(reference.chi),
    "energy variance": reference_variance,
    "measured particle number": reference_particles,
}
```

## Baseline: DMRG at every flux

The baseline evaluates 60 new flux values. Each optimization starts from the
preceding ground state, which is the usual warm-start strategy in a pump
calculation. The mixer remains off because the common reference state already
has the required entanglement.

```{code-cell}
%%time
fluxes = np.linspace(0, 1, 61)
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
    "max_sweeps": 12,
}

warm_state = reference.copy()
warm_charges = [0.0]
warm_sweeps = []
warm_relative_energy_changes = []
warm_entropy_changes = []
warm_norm_errors = []
warm_truncation_errors = []

warm_start = perf_counter()
for flux in fluxes[1:]:
    model = MPOModel(
        sample_models[0.0].lat,
        hamiltonian_at_flux(flux),
    )
    info = dmrg.run(warm_state, model, scan_options)
    warm_state.canonical_form()
    stats = info["sweep_statistics"]

    warm_charges.append(
        float(np.real(left_number.expectation_value(warm_state))) - reference_left_charge
    )
    warm_sweeps.append(int(stats["sweep"][-1]))
    warm_relative_energy_changes.append(
        abs(stats["Delta_E"][-1] / max(abs(stats["E"][-1]), 1.0))
    )
    warm_entropy_changes.append(abs(stats["Delta_S"][-1]))
    warm_norm_errors.append(float(stats["norm_err"][-1]))
    warm_truncation_errors.append(float(stats["max_trunc_err"][-1]))
warm_time = perf_counter() - warm_start
warm_charges = np.asarray(warm_charges)

assert max(warm_sweeps) < scan_options["max_sweeps"]
assert max(warm_relative_energy_changes) < scan_options["max_E_err"]
assert max(warm_entropy_changes) < scan_options["max_S_err"]
assert max(warm_norm_errors) < 1e-8
assert max(warm_truncation_errors) < 1e-10

{
    "warm-start DMRG wall time [s]": warm_time,
    "DMRG optimizations": len(fluxes) - 1,
    "total DMRG sweeps": sum(warm_sweeps),
    "sweeps per point": warm_sweeps,
}
```

## PT predictor and DMRG corrector

At an anchor $f_k$, Pymablock solves

$$
Q_k(H_k-E_k)Q_k|\partial_f\psi_k\rangle
=-Q_k(\partial_fH_k)|\psi_k\rangle,
\qquad
Q_k=1-|\psi_k\rangle\langle\psi_k|.
$$

The tangent state predicts the next anchor,

$$
|\psi_{k+1}^{\mathrm{pred}}\rangle
\propto
|\psi_k\rangle
+(f_{k+1}-f_k)|\partial_f\psi_k\rangle.
$$

DMRG then corrects this prediction to the ground state of $H(f_{k+1})$.
Since DMRG removes the accumulated state error, the predictor solve only needs
a moderate residual. We deliberately limit it to one variational sweep and
record its global residual rather than presenting it as a converged
stand-alone response.

The same tangent gives the derivative needed for interpolation:

$$
\frac{d\langle N_L\rangle}{df}
=2\,\mathrm{Re}
\langle\psi_k|N_L|\partial_f\psi_k\rangle.
$$

```{code-cell}
%%time
anchors = np.linspace(0, 1, 7)
anchor_state = reference.copy()
anchor_charges = [0.0]
anchor_slopes = []
anchor_sweeps = []
anchor_relative_energy_changes = []
anchor_entropy_changes = []
anchor_norm_errors = []
anchor_truncation_errors = []
predictor_residuals = []
predictor_overlaps = []
response_bond_dimensions = []

assisted_start = perf_counter()
for left_anchor, right_anchor in pairwise(anchors):
    response_backend = TenpyImplicitBackend(
        chi_max=32,
        svd_min=1e-12,
        max_sweeps=1,
        solver_tolerance=5e-2,
    )
    _, unitary, _ = block_diagonalize_implicit(
        [
            hamiltonian_at_flux(left_anchor),
            flux_derivative_at(left_anchor),
        ],
        [anchor_state],
        response_backend,
        response_backend.solve_shifted,
        max_relative_residual=5e-2,
        reference_tolerance=1e-7,
    )
    tangent = unitary[1, 0, 1].states[0]
    record = response_backend.solver_records[-1]

    predictor_residuals.append(record.relative_residuals[-1])
    response_bond_dimensions.append(max(tangent.chi))
    anchor_slopes.append(
        2
        * np.real(
            algebra.inner(
                anchor_state,
                algebra.apply(left_number, tangent),
            )
        )
    )

    predictor = algebra.add_states(
        anchor_state,
        algebra.scale_state(tangent, right_anchor - left_anchor),
    )
    predictor.canonical_form(renormalize=True)
    predictor = algebra.scale_state(
        predictor,
        1 / np.sqrt(np.real(algebra.inner(predictor, predictor))),
    )
    uncorrected_predictor = predictor.copy()

    model = MPOModel(
        sample_models[0.0].lat,
        hamiltonian_at_flux(right_anchor),
    )
    info = dmrg.run(predictor, model, scan_options)
    predictor.canonical_form()
    predictor = algebra.scale_state(
        predictor,
        1 / np.sqrt(np.real(algebra.inner(predictor, predictor))),
    )
    stats = info["sweep_statistics"]

    anchor_state = predictor
    predictor_overlaps.append(abs(uncorrected_predictor.overlap(anchor_state)))
    anchor_charges.append(
        float(np.real(left_number.expectation_value(anchor_state)))
        - reference_left_charge
    )
    anchor_sweeps.append(int(stats["sweep"][-1]))
    anchor_relative_energy_changes.append(
        abs(stats["Delta_E"][-1] / max(abs(stats["E"][-1]), 1.0))
    )
    anchor_entropy_changes.append(abs(stats["Delta_S"][-1]))
    anchor_norm_errors.append(float(stats["norm_err"][-1]))
    anchor_truncation_errors.append(float(stats["max_trunc_err"][-1]))

# H(1) = H(0), and the corrected finite-system ground state returns to the
# reference state. The endpoint derivative therefore equals the initial one.
anchor_slopes.append(anchor_slopes[0])
assisted_time = perf_counter() - assisted_start
anchor_charges = np.asarray(anchor_charges)
anchor_slopes = np.asarray(anchor_slopes)

assert max(predictor_residuals) < 5e-2
assert max(anchor_sweeps) < scan_options["max_sweeps"]
assert max(anchor_relative_energy_changes) < scan_options["max_E_err"]
assert max(anchor_entropy_changes) < scan_options["max_S_err"]
assert max(anchor_norm_errors) < 1e-8
assert max(anchor_truncation_errors) < 1e-10
assert abs(reference.overlap(anchor_state)) > 1 - 1e-8

{
    "PT-assisted wall time [s]": assisted_time,
    "PT response solves": len(anchors) - 1,
    "DMRG corrections": len(anchors) - 1,
    "total correction sweeps": sum(anchor_sweeps),
    "largest predictor residual": max(predictor_residuals),
    "smallest predictor/corrected-state overlap": min(predictor_overlaps),
    "largest response bond dimension": max(response_bond_dimensions),
}
```

## Fill the cycle with Hermite interpolation

On each interval, cubic Hermite interpolation is fixed by two anchor charges
and two PT derivatives. It is local: adding an anchor changes only its
neighboring intervals.

```{code-cell}
def cubic_hermite(flux):
    """Interpolate the charge from the PT value and slope at each anchor."""
    interval = np.searchsorted(anchors, flux, side="right") - 1
    interval = min(max(interval, 0), len(anchors) - 2)
    left_flux, right_flux = anchors[interval : interval + 2]
    left_charge, right_charge = anchor_charges[interval : interval + 2]
    left_slope, right_slope = anchor_slopes[interval : interval + 2]
    step = right_flux - left_flux
    coordinate = (flux - left_flux) / step

    return (
        (2 * coordinate**3 - 3 * coordinate**2 + 1) * left_charge
        + (coordinate**3 - 2 * coordinate**2 + coordinate) * step * left_slope
        + (-2 * coordinate**3 + 3 * coordinate**2) * right_charge
        + (coordinate**3 - coordinate**2) * step * right_slope
    )


assisted_charges = np.asarray([cubic_hermite(flux) for flux in fluxes])
```

## Validate the entire cycle

Exact diagonalization of the 256-dimensional Hilbert space checks the DMRG
states and the interpolated charge curve. It is a validation tool for this
tutorial, not part of either timed method.

```{code-cell}
%%time
h_bulk_dense = mpo_to_dense(h_bulk)
h_cos_dense = mpo_to_dense(h_cos)
h_sin_dense = mpo_to_dense(h_sin)
h_sector_dense = mpo_to_dense(h_sector)
left_number_dense = mpo_to_dense(left_number)

exact_charges = []
exact_gaps = []
exact_reference_left_charge = None
for flux in fluxes:
    dense_hamiltonian = (
        h_bulk_dense
        + np.cos(2 * np.pi * flux) * h_cos_dense
        + np.sin(2 * np.pi * flux) * h_sin_dense
        + h_sector_dense
    )
    eigenvalues, eigenvectors = np.linalg.eigh(dense_hamiltonian)
    ground_state = eigenvectors[:, 0]
    left_charge = float(np.real(ground_state.conj() @ left_number_dense @ ground_state))
    if exact_reference_left_charge is None:
        exact_reference_left_charge = left_charge
    exact_charges.append(left_charge - exact_reference_left_charge)
    exact_gaps.append(eigenvalues[1] - eigenvalues[0])

exact_charges = np.asarray(exact_charges)
warm_errors = np.abs(warm_charges - exact_charges)
assisted_errors = np.abs(assisted_charges - exact_charges)
anchor_indices = [np.argmin(abs(fluxes - anchor)) for anchor in anchors]
anchor_errors = np.abs(anchor_charges - exact_charges[anchor_indices])

assert np.max(warm_errors) < 1e-9
assert np.max(anchor_errors) < 1e-9
assert np.max(assisted_errors) < 3e-3
assert min(exact_gaps) > 0.1

{
    "largest warm-start DMRG error": np.max(warm_errors),
    "largest corrected-anchor error": np.max(anchor_errors),
    "largest PT-assisted interpolation error": np.max(assisted_errors),
    "RMS PT-assisted interpolation error": np.sqrt(np.mean(assisted_errors**2)),
    "minimum many-body gap": min(exact_gaps),
    "warm-start/assisted time ratio": warm_time / assisted_time,
    "full-cycle endpoint charge": assisted_charges[-1],
}
```

## Accuracy and cost

The left panel shows the reconstructed cycle. The center panel separates
interpolation error from the converged DMRG error. The right panel includes
every PT solve and DMRG correction in the assisted time.

```{code-cell}
%%time
figure, (charge_axis, error_axis, time_axis) = plt.subplots(
    1,
    3,
    figsize=(13, 4),
)

charge_axis.plot(fluxes, exact_charges, color="black", label="exact")
charge_axis.plot(
    fluxes,
    assisted_charges,
    "--",
    color="C0",
    label="PT + Hermite",
)
charge_axis.plot(
    fluxes,
    warm_charges,
    "x",
    color="C1",
    markersize=4,
    label="warm-start DMRG",
)
charge_axis.plot(
    anchors,
    anchor_charges,
    "o",
    color="C0",
    markersize=5,
    label="DMRG anchors",
)
charge_axis.set(
    xlabel=r"Threaded flux $f=\Phi_y/(2\pi)$",
    ylabel=r"Charge displacement $\Delta Q_L$",
)
charge_axis.grid(alpha=0.25)
charge_axis.legend(
    fontsize="x-small",
    loc="upper right",
)

error_axis.semilogy(
    fluxes,
    np.maximum(warm_errors, 1e-15),
    color="C1",
    label="warm-start DMRG",
)
error_axis.semilogy(
    fluxes,
    np.maximum(assisted_errors, 1e-15),
    color="C0",
    label="PT-assisted reconstruction",
)
error_axis.set(
    xlabel=r"Threaded flux $f$",
    ylabel="Absolute charge error",
)
error_axis.grid(alpha=0.25)

time_axis.barh(
    ["PT + DMRG\n(6 patches)", "warm DMRG\n(60 points)"],
    [assisted_time, warm_time],
    color=["C0", "C1"],
)
time_axis.set(xlabel="Wall time [s]")
time_axis.grid(axis="x", alpha=0.25)

figure.tight_layout()
plt.show()
```

The assisted calculation replaces 60 DMRG optimizations with six response
solves and six DMRG corrections. On this system it reduces the wall time by
about a factor of two while keeping the maximum charge error below $3\times
10^{-3}$. The PT predictor saves little work inside each correction; the main
gain comes from the derivatives, which let a sparse set of corrected anchors
represent the dense curve.

Plain interpolation between sparse DMRG anchors can also be cheap for a curve
known in advance to be smooth. PT adds two pieces of evidence that an
unconstrained interpolation lacks: the Hamiltonian-derived slope and the
residual of the tangent-state equation. An adaptive production calculation
should add anchors where the response norm grows, the residual deteriorates,
or the gap becomes small.

## What this cycle does—and does not—show

The finite cylinder has a unique ground state and a nonzero gap throughout
the cycle. Because $H(1)=H(0)$, continuously following that ground state
returns to the initial state:

$$
\Delta Q_L(1)=0.
$$

The nonzero values at intermediate flux measure reversible charge
redistribution. They are not a fractional pumped charge.

A $\nu=1/2$ FCI on an infinite cylinder behaves differently. One $2\pi$
insertion carries the state into its partner topological sector and transfers
approximately half a boson across the entanglement cut; a second insertion
returns to the initial sector. Following that process requires an infinite,
charge-conserving MPS backend and transport of the low-energy topological
manifold. The educational MPO backend used here supports finite,
charge-unconstrained MPSs, so this tutorial does not claim that calculation.

This limitation is also the branch-tracking test for any future extension:
minimizing the unique ground-state energy independently at every flux would
close the loop and incorrectly erase the fractional pump. A full FCI
implementation must track the topological sector by overlap or transport the
complete quasi-degenerate manifold.

## Conclusion

Local MPO perturbation theory can contribute to a full parameter cycle
without extrapolating one Taylor series through $2\pi$. We re-expand at six
anchors, use each tangent state to predict the next DMRG state, and use the
same response to reconstruct the charge between anchors. This produces a
controlled full-cycle curve more cheaply than optimizing every requested
point.

The method accelerates a dense smooth scan. It does not turn a finite
unique-ground-state calculation into a topological pump. Demonstrating the
fractional Haldane FCI pump still requires the infinite-cylinder,
charge-sector-resolved extension described above.
