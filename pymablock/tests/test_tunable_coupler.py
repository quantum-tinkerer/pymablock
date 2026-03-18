import numpy as np
import pytest
import scipy as sp
import sympy
from scipy.sparse import kron
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator
from pymablock.series import zero


def _tunable_coupler_parameters():
    return {
        "e1": -206.0,
        "e2": -202.0,
        "ec": -254.0,
        "g1c": 76.9,
        "g2c": 76.9,
        "g12": 6.74,
        "w1": 4952.0,
        "w2": 4917.0,
    }


def _symbolic_tunable_coupler_order4_xi_zz(wc_value):
    parameters = _tunable_coupler_parameters()
    a_1, a_2, a_c = BosonOp("a_1"), BosonOp("a_2"), BosonOp("a_c")
    n_1, n_2, n_c = (NumberOperator(op) for op in (a_1, a_2, a_c))

    H_0 = (
        parameters["w1"] * n_1
        + parameters["w2"] * n_2
        + wc_value * n_c
        + parameters["e1"] * n_1 * (n_1 - 1) / 2
        + parameters["e2"] * n_2 * (n_2 - 1) / 2
        + parameters["ec"] * n_c * (n_c - 1) / 2
    )
    H_1 = sympy.S.Zero
    for g, a_l, a_r in (
        (parameters["g1c"], a_1, a_c),
        (parameters["g2c"], a_2, a_c),
        (parameters["g12"], a_1, a_2),
    ):
        H_1 += g * (
            a_l * Dagger(a_r) + Dagger(a_l) * a_r - a_l * a_r - Dagger(a_l) * Dagger(a_r)
        )

    H_tilde, *_ = block_diagonalize([H_0, H_1])
    H_eff = sum((H_tilde[0, 0, order] for order in range(5)), start=sympy.S.Zero)

    def energy(occupations):
        value = H_eff.subs(
            {n_1: occupations[0], n_2: occupations[1], n_c: occupations[2]}
        )
        return float(sympy.N(value.as_expr()))

    return energy((0, 0, 0)) + energy((1, 1, 0)) - energy((1, 0, 0)) - energy((0, 1, 0))


def _state_to_index(state, highest_state):
    return state[0] * highest_state**2 + state[1] * highest_state + state[2]


def _finite_mode_operator(highest_state):
    return sp.sparse.diags_array(np.sqrt(np.arange(1, highest_state)), offsets=1)


def _numeric_tunable_coupler_order4_xi_zz(wc_value, highest_state=4):
    parameters = _tunable_coupler_parameters()

    identity = sp.sparse.eye_array(highest_state)
    boson_operator = _finite_mode_operator(highest_state)

    a_1 = kron(kron(boson_operator, identity), identity)
    a_2 = kron(kron(identity, boson_operator), identity)
    a_c = kron(kron(identity, identity), boson_operator)

    H_0 = 0
    for e, omega, a in (
        (parameters["e1"], parameters["w1"], a_1),
        (parameters["e2"], parameters["w2"], a_2),
        (parameters["ec"], wc_value, a_c),
    ):
        adag = a.T.conjugate()
        H_0 = H_0 + omega * (adag @ a) + (e / 2) * ((adag @ adag) @ (a @ a))

    H_1 = 0
    for g, a_l, a_r in (
        (parameters["g1c"], a_1, a_c),
        (parameters["g2c"], a_2, a_c),
        (parameters["g12"], a_1, a_2),
    ):
        adag_l = a_l.T.conjugate()
        adag_r = a_r.T.conjugate()
        H_1 = H_1 + g * (a_l @ adag_r + adag_l @ a_r - a_l @ a_r - adag_l @ adag_r)

    tracked_states = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    state_indices = [_state_to_index(state, highest_state) for state in tracked_states]
    tracked_by_index = {index: block for block, index in enumerate(state_indices)}
    subspace_indices = [
        tracked_by_index.get(index, len(tracked_states)) for index in range(a_1.shape[0])
    ]

    H_tilde, *_ = block_diagonalize([H_0, H_1], subspace_indices=subspace_indices)

    def block_entry(order, block):
        value = H_tilde[block, block, order]
        if value is zero:
            return 0.0
        return float(value[0, 0])

    energies = []
    for block in range(4):
        energies.append(sum(block_entry(order, block) for order in range(5)))
    return float(energies[0] + energies[3] - energies[1] - energies[2])


@pytest.mark.no_cover
def test_tunable_coupler_order4_matches_finite_truncation():
    wc_value = 5300.0

    symbolic_xi_zz = _symbolic_tunable_coupler_order4_xi_zz(wc_value)
    numeric_xi_zz = _numeric_tunable_coupler_order4_xi_zz(wc_value)

    assert np.isclose(symbolic_xi_zz, numeric_xi_zz, atol=1e-8)
