# Applications

These advanced examples derive effective Hamiltonians for interacting quantum
systems using [structured embeddings](../structured_embeddings.md). Each develops
a model, identifies its retained degrees of freedom, and interprets the resulting
couplings or observables. The calculations assume familiarity with
[second quantization](../second_quantization.md) and perturbation theory and can
be read independently. References on each page identify the source model and
separate reproduced results from further calculations.

| Application | Retained degrees of freedom | Effective description |
| --- | --- | --- |
| [Supercurrent through an interacting dot](supercurrent.md) | Two dot fermions | Symbolic fourth-order current in all charge sectors |
| [Crépel–Fu interactions](crepel_fu_interactions.md) | Equal-spin dopants on the B sublattice | Analytic assisted hopping and density interactions |
| [Tunable coupler](tunable_coupler.md) | Two qubits | Exchange and fourth-order ZZ coupling, including counterrotating processes |
| [Artificial cavity spin](cavity_spin.md) | A three- or four-state matrix | Engineered ladder amplitudes and second-order virtual corrections |

## Reproducing the calculations

The pages are executable MyST notebooks. From the repository root,
`pixi run -e docs docs-build` builds the documentation with their outputs,
reusing cached calculations where available.

To execute a single document and save its output as a notebook:

```bash
pixi run -e docs python docs/source/applications/reproduce.py supercurrent \
    --output-dir /tmp/pymablock-applications
```

Replace `supercurrent` with `crepel_fu_interactions`, `tunable_coupler`, or
`cavity_spin` to run another example. The
{download}`reproduction script <reproduce.py>` uses the current Python environment.
