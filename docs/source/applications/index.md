# Applications

These pages derive effective models of interacting quantum systems with
[structured embeddings](../structured_embeddings.md). Each page states its
question and result first, then builds the full Hamiltonian, chooses the states
to keep, and checks the result against the literature, a limiting case, or a
symmetry. The pages assume familiarity with
[second quantization](../second_quantization.md) and perturbation theory, and
can be read in any order.

- [Supercurrent through an interacting dot](supercurrent.md): how does the
  Josephson current depend on the dot charge? The effective model keeps the two
  dot spin orbitals.
- [Crépel–Fu interactions](crepel_fu_interactions.md): how do dopants in a
  charge-transfer insulator hop and interact? The effective model has
  spin-polarized fermions on the B sublattice.
- [Tunable coupler](tunable_coupler.md): how does a coupler switch the exchange
  and $ZZ$ coupling between two qubits? The effective model keeps two levels of
  each qubit.
- [Artificial cavity spin](cavity_spin.md): can cavity Fock states driven
  through an ancilla act as a large spin? The effective model is a $(2s+1)$-level
  spin matrix.

Each page is an executable notebook; the documentation build runs all of them.
