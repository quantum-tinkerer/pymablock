# Applications

These executable documents construct effective Hamiltonians for four research
models using [structured embeddings](../structured_embeddings.md). Each gives
the microscopic Hamiltonian, retained generators, perturbative result, and a
finite-matrix or analytical check. The collection is separate from the
introductory tutorials and can be read in any order.

```{toctree}
:maxdepth: 1

supercurrent.md
crepel_fu_interactions.md
tunable_coupler.md
cavity_spin.md
```

| Application | Retained representation | Results reproduced |
| --- | --- | --- |
| Supercurrent through an interacting dot | Two dot fermions | Symbolic fourth-order currents in all charge sectors and their gate dependence |
| Crépel–Fu interactions | Doped fermions on the B sublattice | Assisted hopping and density interactions at second order; connected hopping at fourth order |
| Tunable coupler | Two spin-one-half lowering operators | Exchange including counterrotating processes; the fourth-order effective matrix |
| Artificial cavity spin | A three- or four-state matrix from a reference list | Engineered ladder amplitudes and second-order cavity/Floquet corrections |

## Reproducing the calculations

The pages are MyST notebooks. From a checkout containing the structured-embedding
API, `pixi run -e docs docs-build` executes them and includes their outputs in the HTML
documentation. Execution results are cached by notebook content. The connected
TMD cluster is the longest calculation: allow a few minutes on a CPU.

To execute one document separately from the repository root:

```bash
pixi run -e docs python docs/source/applications/reproduce.py supercurrent \
    --output-dir /tmp/pymablock-applications
```

The {download}`reproduction script <reproduce.py>` uses the current Python
environment and saves an executed notebook. The document and the small
{download}`validation.py <validation.py>` helper must remain together. The helper
constructs finite occupation matrices and evaluates
source expressions with ordinary matrix arithmetic. It does not use the
embedding compiler. Every model Hamiltonian and embedding is defined in its
own document; the documents do not import test fixtures.

The checks establish the displayed coefficients and the stated finite-order
comparisons. Each page specifies its parameter choices, truncation check, and
which parts of a full material or device calculation are represented.
