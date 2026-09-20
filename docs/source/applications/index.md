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
| [Crépel–Fu interactions](crepel_fu_interactions.md) | Equal-spin dopants on the B sublattice | Assisted hopping and density interactions; a fourth-order hopping path |
| [Tunable coupler](tunable_coupler.md) | Two qubits | Exchange including counterrotating processes and the fourth-order Hamiltonian |
| [Artificial cavity spin](cavity_spin.md) | A three- or four-state matrix | Engineered ladder amplitudes and second-order virtual corrections |

## Reproducing the calculations

The pages are executable MyST notebooks. From the repository root,
`pixi run -e docs docs-build` builds the documentation with their outputs,
reusing cached calculations where available. The two-star Crépel–Fu cluster is
the longest calculation; allow a few minutes on a CPU.

To execute a single document and save its output as a notebook:

```bash
pixi run -e docs python docs/source/applications/reproduce.py supercurrent \
    --output-dir /tmp/pymablock-applications
```

Replace `supercurrent` with `crepel_fu_interactions`, `tunable_coupler`, or
`cavity_spin` to run another example. The
{download}`reproduction script <reproduce.py>` uses the current Python environment.
Keep the documents alongside {download}`validation.py <validation.py>`, which
supplies finite occupation matrices for the numerical comparisons. Each page
states the parameter choices and truncations used in those comparisons.
