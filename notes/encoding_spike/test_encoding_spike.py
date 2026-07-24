"""Physical and representation checks for the occupation-encoding spike."""

from __future__ import annotations

from itertools import product

import numpy as np
import sympy
from sympy.physics.quantum.fermion import FermionOp

from .encoding import CompiledOperator, FermionOperator, block_diagonalize
from .models import crepel_fu_triangle, fermionic_ring_exchange, tunable_coupler


def _assert_zero_matrix(matrix) -> None:
    assert all(sympy.expand(value) == 0 for value in matrix)


def test_coupler_encoding_and_effective_hamiltonian() -> None:
    """The three-boson problem closes to 44 states and yields the known exchange."""
    model = tunable_coupler()
    H_tilde, *_ = block_diagonalize(
        [model.H0, model.V], encoding=model.encoding, to_order=4
    )
    assert H_tilde.info.retained_states == 4
    assert H_tilde.info.total_states == 44

    for state in model.encoding.target.states:
        assert model.encoding.decode(model.encoding.encode(state)) == state

    H1 = H_tilde[1]
    H2 = H_tilde[2]
    H4 = H_tilde[4]
    _assert_zero_matrix(H4.matrix - H4.matrix.adjoint())

    omega1, omega2, omega_c = model.frequencies
    g12, g1c, g2c = model.couplings
    # Target state order is |00>, |01>, |10>, |11>.
    assert H1.matrix_element((0, 1), (1, 0)) == -g12
    expected_mediated_exchange = (
        g1c
        * g2c
        * sympy.Rational(1, 2)
        * (
            1 / (omega1 - omega_c)
            + 1 / (omega2 - omega_c)
            - 1 / (omega1 + omega_c)
            - 1 / (omega2 + omega_c)
        )
    )
    assert (
        sympy.factor(H2.matrix_element((0, 1), (1, 0)) - expected_mediated_exchange) == 0
    )

    pauli = H4.pauli()
    _assert_zero_matrix(pauli.as_matrix() - H4.matrix)
    # Four times the ZZ Pauli coefficient is the conventional cross-Kerr energy
    # combination E00 - E01 - E10 + E11.
    cross_kerr = (
        H4.matrix_element((0, 0), (0, 0))
        - H4.matrix_element((0, 1), (0, 1))
        - H4.matrix_element((1, 0), (1, 0))
        + H4.matrix_element((1, 1), (1, 1))
    )
    assert sympy.expand(cross_kerr - 4 * pauli.coefficient("ZZ")) == 0


def test_fermionic_ring_exchange() -> None:
    """The encoded Hubbard square gives K = 20 t⁴/U³."""
    model = fermionic_ring_exchange()
    H_tilde, *_ = block_diagonalize(
        [model.H0, model.V], encoding=model.encoding, to_order=4
    )
    assert H_tilde.info.retained_states == 16
    assert H_tilde.info.total_states == 70

    neel = (1, 0, 1, 0)
    reversed_neel = (0, 1, 0, 1)
    amplitude = H_tilde[4].matrix_element(reversed_neel, neel)
    assert sympy.factor(amplitude - 40 * model.t**4 / model.U**3) == 0
    ring_coefficient = amplitude / 2
    assert sympy.factor(ring_coefficient - 20 * model.t**4 / model.U**3) == 0

    # The effective result is also a target-spin operator, not only a matrix with
    # unexplained source-state indices.
    pauli = H_tilde[4].pauli()
    _assert_zero_matrix(pauli.as_matrix() - H_tilde[4].matrix)
    assert any(sum(axis != "I" for axis in string) == 4 for string in pauli.terms)


def test_direct_fermion_embedding_includes_fock_phases() -> None:
    """A frozen occupied mode does not change the mapped target generator."""
    from .encoding import fermion_embedding

    fixed = FermionOp("a_fixed")
    source = FermionOp("b_source")
    target = FermionOp("f_target")
    embedding = fermion_embedding({fixed: 1, source: target})

    projected = embedding.project(source)
    target_operator = embedding.target.operator(target)
    assert projected.generators == (target,)
    assert projected.matrix == target_operator.matrix


def test_crepel_fu_published_second_order() -> None:
    """The one-A cluster reproduces the published SW result."""
    model = crepel_fu_triangle()
    H_tilde, *_ = block_diagonalize(
        [model.H0, model.V], encoding=model.encoding, to_order=2
    )
    assert isinstance(H_tilde[2], FermionOperator)
    assert H_tilde[2].generators == model.target_fermions
    assert H_tilde.info.retained_states == 7
    assert H_tilde.info.total_states == 35

    f0, f1, f2 = model.target_fermions
    state = model.encoding.target.state
    bare_hopping = H_tilde[2].matrix_element(state([f1]), state([f0]))
    assisted_hopping = H_tilde[2].matrix_element(state([f1, f2]), state([f0, f2]))
    expected_bare = model.t0**2 / (model.Delta + model.V0)
    expected_assisted = model.t0**2 / model.Delta
    assert sympy.factor(bare_hopping - expected_bare) == 0
    assert sympy.factor(assisted_hopping - expected_assisted) == 0
    assert (
        sympy.factor(
            assisted_hopping
            - bare_hopping
            - model.t0**2 * (1 / model.Delta - 1 / (model.Delta + model.V0))
        )
        == 0
    )


def test_local_crepel_fu_fourth_order_matches_exact_spectrum() -> None:
    """The fourth-order recurrence is exact through O(t0^4) on one A star."""
    model = crepel_fu_triangle()
    H_tilde, *_ = block_diagonalize(
        [model.H0, model.V], encoding=model.encoding, to_order=4
    )
    series = sympy.zeros(H_tilde.target.dimension)
    for order in range(5):
        series += H_tilde[order].matrix

    up_indices = [
        index
        for index, mode in enumerate(model.encoding.operators)
        if str(mode).endswith("_up")
    ]
    down_indices = [
        index
        for index, mode in enumerate(model.encoding.operators)
        if str(mode).endswith("_down")
    ]
    parameters = {
        model.Delta: 10,
        model.V0: 2,
        model.UA: 7,
        model.UB: 11,
    }

    def spectral_error(t0: float) -> float:
        substitution = {**parameters, model.t0: t0}
        exact_operator = CompiledOperator(
            (model.H0 + model.V).subs(substitution), model.encoding.operators
        )
        errors = []
        for number in (0, 1, 2):
            target_indices = [
                index
                for index, state in enumerate(H_tilde.target.states)
                if sum(state) == number
            ]
            effective = np.array(
                series.extract(target_indices, target_indices).subs(substitution),
                dtype=float,
            )
            source_states = [
                state
                for state in product((0, 1), repeat=len(model.encoding.operators))
                if sum(state[index] for index in up_indices) == number + 1
                and sum(state[index] for index in down_indices) == 1
            ]
            source_index = {state: index for index, state in enumerate(source_states)}
            exact = np.zeros((len(source_states), len(source_states)))
            for column, source_state in enumerate(source_states):
                for target_state, coefficient in exact_operator.apply(
                    source_state
                ).items():
                    row = source_index.get(target_state)
                    if row is not None:
                        exact[row, column] += float(coefficient)
            exact_low = np.linalg.eigvalsh(exact)[: len(target_indices)]
            effective_low = np.linalg.eigvalsh(effective)
            errors.append(float(np.max(np.abs(exact_low - effective_low))))
        return max(errors)

    coarse_error = spectral_error(0.08)
    fine_error = spectral_error(0.04)
    assert coarse_error < 3e-10
    assert fine_error < coarse_error / 50
