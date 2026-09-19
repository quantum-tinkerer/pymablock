"""Coupler, ring-exchange, and dressed-spin models with physical reference checks."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize
from pymablock.number_ordered_form import (
    LadderOp,
    NumberOperator,
    NumberOrderedForm,
    SpinOp,
)
from pymablock.operator_embedding import Embedding


@dataclass
class CouplerModel:
    """Three nonlinear bosons with pairwise capacitive coupling."""

    H0: sympy.Expr
    V: sympy.Expr
    encoding: Embedding
    target_coordinates: tuple[SigmaMinus, SigmaMinus]
    frequencies: tuple[sympy.Symbol, sympy.Symbol, sympy.Symbol]
    anharmonicities: tuple[sympy.Symbol, sympy.Symbol, sympy.Symbol]
    couplings: tuple[sympy.Symbol, sympy.Symbol, sympy.Symbol]


def tunable_coupler() -> CouplerModel:
    """Return the three-nonlinear-boson tunable-coupler benchmark.

    The capacitive interaction uses ``g_ij (a_i† - a_i) (a_j† - a_j)`` and includes
    the direct qubit-qubit capacitance.  The target contains the lowest two levels of
    the two computational modes, while the coupler starts in its ground state.
    """
    a1, a2, ac = sympy.symbols("a1 a2 ac", cls=BosonOp)
    q1, q2 = SigmaMinus("q1"), SigmaMinus("q2")
    omega1, omega2, omega_c = sympy.symbols(
        "omega1 omega2 omega_c", nonzero=True, real=True
    )
    alpha1, alpha2, alpha_c = sympy.symbols(
        "alpha1 alpha2 alpha_c", nonzero=True, real=True
    )
    g12, g1c, g2c = sympy.symbols("g12 g1c g2c", real=True)

    modes = (a1, a2, ac)
    frequencies = (omega1, omega2, omega_c)
    anharmonicities = (alpha1, alpha2, alpha_c)
    H0 = sympy.Add(
        *(
            omega * NumberOperator(mode)
            + alpha * NumberOperator(mode) * (NumberOperator(mode) - 1) / 2
            for mode, omega, alpha in zip(
                modes, frequencies, anharmonicities, strict=True
            )
        )
    )
    V = sympy.Add(
        *(
            coupling * (Dagger(left) - left) * (Dagger(right) - right)
            for left, right, coupling in (
                (a1, a2, g12),
                (a1, ac, g1c),
                (a2, ac, g2c),
            )
        )
    )
    encoding = Embedding(
        {q1: a1, q2: a2},
        reference={a1: 0, a2: 0, ac: 0},
    )
    return CouplerModel(
        H0,
        V,
        encoding,
        (q1, q2),
        frequencies,
        anharmonicities,
        (g12, g1c, g2c),
    )


@dataclass
class RingExchangeModel:
    """Half-filled four-site Hubbard plaquette."""

    H0: sympy.Expr
    V: sympy.Expr
    encoding: Embedding
    target_coordinates: tuple[SigmaMinus, ...]
    U: sympy.Symbol
    t: sympy.Symbol


def fermionic_ring_exchange() -> RingExchangeModel:
    """Return a half-filled Hubbard square encoded as four target spins."""
    sites = range(4)
    up = tuple(FermionOp(f"c{i}_up") for i in sites)
    down = tuple(FermionOp(f"c{i}_down") for i in sites)
    spins = tuple(SigmaMinus(f"s{i}") for i in sites)
    U, t = sympy.symbols("U t", nonzero=True, real=True)

    H0 = U * sympy.Add(*(NumberOperator(up[i]) * NumberOperator(down[i]) for i in sites))
    V = -t * sympy.Add(
        *(
            Dagger(modes[i]) * modes[j] + Dagger(modes[j]) * modes[i]
            for modes in (up, down)
            for i, j in ((0, 1), (1, 2), (2, 3), (3, 0))
        )
    )
    encoding = Embedding(
        {spins[i]: Dagger(down[i]) * up[i] for i in sites},
        reference={**dict.fromkeys(up, 0), **dict.fromkeys(down, 1)},
    )
    return RingExchangeModel(H0, V, encoding, spins, U, t)


@dataclass
class SyntheticSpinFloquetModel:
    """One-rotation Floquet formulation of the synthetic-spin experiment."""

    H0: sympy.Expr
    V: sympy.Expr
    encoding: Embedding
    cavity: BosonOp
    floquet: LadderOp
    dressed_ancilla: SigmaMinus
    spin: sympy.Rational
    phases: tuple[sympy.Symbol, ...]
    cavity_phase: sympy.Symbol
    chi: sympy.Symbol
    Omega: sympy.Symbol
    epsilon: sympy.Symbol

    def spin_phase_substitutions(self) -> dict[sympy.Symbol, sympy.Expr]:
        """Return the drive phases producing the paper's spin generator."""
        maximum = len(self.phases) - 1
        substitutions = {self.phases[0]: sympy.S.Zero}
        phase = sympy.S.Zero
        for occupation in range(1, maximum + 1):
            phase += 2 * sympy.acos(
                sympy.sqrt(sympy.Rational(maximum + 1 - occupation, maximum))
            )
            substitutions[self.phases[occupation]] = phase
        return substitutions

    def expected_first_order(self) -> sympy.ImmutableMatrix:
        """Return the matrix-element-modified drive from Eq. (A13)."""
        size = len(self.phases)
        result = sympy.MutableSparseMatrix(size, size, {})
        for occupation in range(1, size):
            amplitude = (
                self.epsilon
                * sympy.sqrt(occupation)
                * sympy.cos((self.phases[occupation] - self.phases[occupation - 1]) / 2)
                * sympy.exp(sympy.I * self.cavity_phase)
                / 2
            )
            result[occupation - 1, occupation] = amplitude
            result[occupation, occupation - 1] = sympy.conjugate(amplitude)
        return sympy.ImmutableMatrix(result)


def _number_projector(operator, occupation: int, cutoff: int) -> sympy.Expr:
    number = NumberOperator(operator)
    return sympy.prod(
        (number - other) / (occupation - other)
        for other in range(cutoff + 1)
        if other != occupation
    )


def _binary_matrix_operator(matrix: sympy.MatrixBase, operator) -> sympy.Expr:
    number = NumberOperator(operator)
    return (
        matrix[0, 0] * (1 - number)
        + matrix[1, 1] * number
        + matrix[0, 1] * operator
        + matrix[1, 0] * Dagger(operator)
    )


def _floquet_shift(operator: LadderOp, shift: int) -> sympy.Expr:
    if shift > 0:
        return Dagger(operator) ** shift
    if shift < 0:
        return operator ** (-shift)
    return sympy.S.One


def synthetic_spin_floquet(
    spin: sympy.Rational = sympy.Rational(1),
    *,
    virtual_shells: int = 1,
) -> SyntheticSpinFloquetModel:
    """Return the carrier-rotated Sambe Hamiltonian of Roy et al.

    The source modes are the cavity, one bilateral Floquet ladder, and the dressed
    ancilla. The retained target is one dressed branch with cavity occupations
    ``0 .. 2s``. ``virtual_shells`` only sets how many higher cavity occupations are
    represented in the finite polynomial projector identities; one shell suffices
    through second order.
    """
    twice_spin = sympy.sympify(2 * spin)
    if not twice_spin.is_Integer or twice_spin < 1:
        raise ValueError("spin must be a positive integer or half-integer")
    if virtual_shells < 1:
        raise ValueError("virtual_shells must be positive")
    maximum = int(twice_spin)
    cutoff = maximum + virtual_shells

    cavity = BosonOp("c")
    floquet = LadderOp("ell")
    dressed = SigmaMinus(sympy.Symbol("d"))
    phases = tuple(sympy.symbols(f"phi_0:{maximum + 1}", real=True))
    cavity_phase = sympy.Symbol("varphi", real=True)
    chi, Omega, epsilon = sympy.symbols("chi Omega epsilon", nonzero=True, real=True)

    projectors = tuple(
        _number_projector(cavity, occupation, cutoff) for occupation in range(cutoff + 1)
    )
    block_projector = sympy.Add(*projectors[: maximum + 1])
    H0 = (
        chi * NumberOperator(floquet)
        + Omega * block_projector * (1 - 2 * NumberOperator(dressed)) / 2
    )

    identity = sympy.eye(2)
    bare_ground = sympy.diag(1, 0)
    bare_excited = sympy.diag(0, 1)
    sigma_minus = sympy.Matrix([[0, 1], [0, 0]])

    def rotation(occupation: int) -> sympy.MatrixBase:
        if occupation > maximum:
            return identity
        half_phase = phases[occupation] / 2
        return sympy.Matrix(
            [
                [
                    sympy.exp(sympy.I * half_phase),
                    -sympy.exp(sympy.I * half_phase),
                ],
                [
                    sympy.exp(-sympy.I * half_phase),
                    sympy.exp(-sympy.I * half_phase),
                ],
            ]
        ) / sympy.sqrt(2)

    qubit_terms = []
    for occupation, projector in enumerate(projectors):
        basis_rotation = rotation(occupation)
        rotated_lowering = basis_rotation.adjoint() * sigma_minus * basis_rotation
        for tooth, phase in enumerate(phases):
            if occupation <= maximum and tooth == occupation:
                continue
            term = (
                Omega
                * projector
                * sympy.exp(sympy.I * phase)
                * _binary_matrix_operator(rotated_lowering, dressed)
                * _floquet_shift(floquet, tooth - occupation)
                / 2
            )
            qubit_terms.extend((term, Dagger(term)))

    cavity_terms = []
    for occupation in range(1, cutoff + 1):
        initial_rotation = rotation(occupation)
        final_rotation = rotation(occupation - 1)
        ground_path = final_rotation.adjoint() * bare_ground * initial_rotation
        excited_path = final_rotation.adjoint() * bare_excited * initial_rotation
        dressed_paths = _binary_matrix_operator(ground_path, dressed) * (
            1 + Dagger(floquet)
        ) + _binary_matrix_operator(excited_path, dressed) * (1 + floquet)
        term = (
            epsilon
            * sympy.exp(sympy.I * cavity_phase)
            * cavity
            * projectors[occupation]
            * dressed_paths
            / 2
        )
        cavity_terms.extend((term, Dagger(term)))

    encoding = Embedding(
        {SpinOp("S", spin): sympy.sqrt(maximum - NumberOperator(cavity)) * cavity},
        reference={cavity: 0, floquet: 0, dressed: 0},
    )
    return SyntheticSpinFloquetModel(
        H0,
        sympy.Add(*qubit_terms, *cavity_terms),
        encoding,
        cavity,
        floquet,
        dressed,
        sympy.Rational(twice_spin, 2),
        phases,
        cavity_phase,
        chi,
        Omega,
        epsilon,
    )


def test_coupler_exchange_and_fourth_order():
    """The two-qubit result retains the analytic mediated exchange."""
    model = tunable_coupler()
    effective, *_ = block_diagonalize(
        [model.H0, model.V], subspace_eigenvectors=model.encoding
    )
    w1, w2, wc = model.frequencies
    g12, g1c, g2c = model.couplings
    expected = (
        g1c * g2c / 2 * (1 / (w1 - wc) + 1 / (w2 - wc) - 1 / (w1 + wc) - 1 / (w2 + wc))
    )
    assert sympy.simplify(effective[0, 0, 1].terms[(1, -1)] + g12) == 0
    assert sympy.simplify(effective[0, 0, 2].terms[(1, -1)] - expected) == 0
    fourth = effective[0, 0, 4]
    assert tuple(fourth.operators) == model.encoding.target.operators
    assert all(sympy.simplify(c) == 0 for c in (fourth - fourth.adjoint()).terms.values())


def test_hubbard_ring_coefficient():
    """Both circulating paths contribute 40 t^4/U^3 to the alternating spin flip."""
    model = fermionic_ring_exchange()
    effective, *_ = block_diagonalize(
        [model.H0, model.V], subspace_eigenvectors=model.encoding
    )
    ring = effective[0, 0, 4].terms[(1, -1, 1, -1)]
    assert sympy.factor(ring - 40 * model.t**4 / model.U**3) == 0


@pytest.mark.parametrize("spin", [sympy.S.One, sympy.Rational(3, 2)])
def test_dressed_spin_one_first_order(spin):
    """The finite-spin output agrees with the analytic dressed drive."""
    model = synthetic_spin_floquet(spin)
    effective, *_ = block_diagonalize(
        [model.H0, model.V], subspace_eigenvectors=model.encoding
    )
    difference = effective[0, 0, 1].to_matrix() - model.expected_first_order()
    assert difference.applyfunc(
        lambda x: sympy.trigsimp(sympy.expand_complex(x))
    ) == sympy.zeros(int(2 * spin + 1))


@pytest.mark.parametrize("spin", [sympy.S.One, sympy.Rational(3, 2)])
def test_dressed_spin_one_second_order_against_finite_matrices(spin):
    """A finite occupation-matrix oracle independently checks virtual processes."""
    from itertools import product

    import numpy as np

    from pymablock.tests.second_quantization_helpers import (
        occupation_matrices,
        operator_matrix,
    )

    model = synthetic_spin_floquet(spin)
    substitutions = {
        **{phase: sympy.pi * i / (i + 2) for i, phase in enumerate(model.phases)},
        model.cavity_phase: sympy.pi / 7,
        model.chi: 11,
        model.Omega: 3,
        model.epsilon: 1,
    }
    operators = (model.cavity, model.floquet, model.dressed_ancilla)
    dimension = int(2 * spin + 1)
    occupations = (range(dimension + 1), range(-dimension, dimension + 1), range(2))
    matrices = occupation_matrices(operators, occupations)
    matrices[sympy.I] = 1j * matrices[sympy.S.One]
    # These ranges include every intermediate state reached by one application
    # of V from the three retained states, hence suffice at second order.
    source = [x.subs(substitutions) for x in (model.H0, model.V)]
    h0 = operator_matrix(source[0].evalf(), matrices).toarray()
    perturbation = operator_matrix(source[1].evalf(), matrices).toarray()
    states = list(product(*occupations))
    retained = [states.index((n, 0, 0)) for n in range(dimension)]
    complement = [i for i in range(len(states)) if i not in retained]
    energy = np.diag(h0)
    denominator = energy[retained[0]] - energy[complement]
    assert np.all(np.abs(denominator) > 0)
    reference = (perturbation[np.ix_(retained, complement)] / denominator) @ perturbation[
        np.ix_(complement, retained)
    ]
    effective, *_ = block_diagonalize(source, subspace_eigenvectors=model.encoding)
    actual = np.array(effective[0, 0, 2].to_matrix().evalf(), dtype=complex)
    np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)


@dataclass(frozen=True)
class CrepelFuModel:
    """Connected spinful ionic-Hubbard cluster from Crépel and Fu."""

    H0: sympy.Expr
    V: sympy.Expr
    encoding: Embedding
    target_fermions: tuple[FermionOp, ...]
    source_a: tuple[tuple[FermionOp, FermionOp], ...]
    source_b: tuple[tuple[FermionOp, FermionOp], ...]
    edges: tuple[tuple[int, int], ...]
    Delta: sympy.Symbol
    V0: sympy.Symbol
    UA: sympy.Symbol
    UB: sympy.Symbol
    t0: sympy.Symbol


def _crepel_fu_cluster(
    edges: tuple[tuple[int, int], ...],
    *,
    num_a: int,
    num_b: int,
) -> CrepelFuModel:
    """Build a connected honeycomb cluster with filled A and doped B sites."""
    up, down = range(2)
    source_a = tuple(
        (FermionOp(f"a{site}_up"), FermionOp(f"a{site}_down")) for site in range(num_a)
    )
    source_b = tuple(
        (FermionOp(f"b{site}_up"), FermionOp(f"b{site}_down")) for site in range(num_b)
    )
    target = tuple(FermionOp(f"f{site}") for site in range(num_b))
    Delta, V0, UA, UB, t0 = sympy.symbols("Delta V0 UA UB t0", positive=True, real=True)

    n_a = tuple(sympy.Add(*(NumberOperator(mode) for mode in site)) for site in source_a)
    n_b = tuple(sympy.Add(*(NumberOperator(mode) for mode in site)) for site in source_b)
    b_degrees = tuple(sum(b_site == site for _, b_site in edges) for site in range(num_b))
    delta0 = Delta + UA - 3 * V0
    H0 = (
        delta0 * (sympy.Add(*n_b) - sympy.Add(*n_a)) / 2
        + UA
        * sympy.Add(
            *(NumberOperator(site[up]) * NumberOperator(site[down]) for site in source_a)
        )
        + UB
        * sympy.Add(
            *(NumberOperator(site[up]) * NumberOperator(site[down]) for site in source_b)
        )
        + V0 * sympy.Add(*(n_a[a_site] * n_b[b_site] for a_site, b_site in edges))
        # Every omitted A neighbor remains doubly occupied. Its density interaction
        # is therefore a static 2 V0 n_B boundary term.
        + 2
        * V0
        * sympy.Add(*((3 - degree) * number for degree, number in zip(b_degrees, n_b)))
    )
    V = -t0 * sympy.Add(
        *(
            Dagger(source_b[b_site][spin]) * source_a[a_site][spin]
            + Dagger(source_a[a_site][spin]) * source_b[b_site][spin]
            for a_site, b_site in edges
            for spin in (up, down)
        )
    )
    encoding = Embedding(
        {target[site]: source_b[site][up] for site in range(num_b)},
        reference={
            **{mode: 1 for site in source_a for mode in site},
            **{mode: 0 for site in source_b for mode in site},
        },
    )
    return CrepelFuModel(
        H0,
        V,
        encoding,
        target,
        source_a,
        source_b,
        edges,
        Delta,
        V0,
        UA,
        UB,
        t0,
    )


def crepel_fu_triangle() -> CrepelFuModel:
    """Return the one-A cluster used for the published second-order process.

    One doubly occupied A orbital is coupled to its three B neighbors. The two
    external filled A neighbors of every B site are included as their static Hartree
    energy. This is the smallest cluster reproducing the bare and
    spectator-assisted second-order hopping denominators.
    """
    return _crepel_fu_cluster(
        ((0, 0), (0, 1), (0, 2)),
        num_a=1,
        num_b=3,
    )


def crepel_fu_two_star() -> CrepelFuModel:
    """Return two overlapping A stars needed by connected fourth-order paths.

    The stars share B1. A four-hop path can connect B0 to B3 through
    ``B0-A0-B1-A1-B3``. The side sites B2 and B4 retain the complete spectator
    dependence of both local triangles.
    """
    return _crepel_fu_cluster(
        (
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 3),
            (1, 4),
        ),
        num_a=2,
        num_b=5,
    )


def test_crepel_fu_interaction_and_assisted_hopping():
    """Retain B fermions and reproduce the spectator-dependent denominators."""
    from pymablock.operator_embedding import _number_symbols

    model = crepel_fu_triangle()
    h, *_ = block_diagonalize([model.H0, model.V], subspace_eigenvectors=model.encoding)
    second = h[0, 0, 2]
    f0, f1, f2 = model.encoding.target.operators
    n0, n1, n2 = _number_symbols((f0, f1, f2))
    hopping = second.terms[(1, -1, 0)]
    assert sympy.factor(hopping.subs(n2, 0) - model.t0**2 / (model.Delta + model.V0)) == 0
    assert sympy.factor(hopping.subs(n2, 1) - model.t0**2 / model.Delta) == 0
    # Check the connected density interaction from diagonal virtual processes
    # against direct second-order sums in the full spinful Fock space.
    import numpy as np

    from pymablock.tests.second_quantization_helpers import (
        occupation_matrices,
        operator_matrix,
    )

    parameters = {model.Delta: 10, model.V0: 2, model.UA: 7, model.UB: 11, model.t0: 1}
    operators = model.encoding.operators
    matrices = occupation_matrices(operators, [range(2)] * len(operators))
    e = operator_matrix(model.H0.subs(parameters), matrices).diagonal().real
    v = operator_matrix(model.V.subs(parameters), matrices).toarray()
    shifts = []
    for occupations in ((0, 0, 0), (1, 0, 0), (1, 1, 0)):
        # A is doubly occupied; every B down mode is empty. Build indices
        # directly from the model, without the embedding's state compiler.
        state = {mode: 1 for site in model.source_a for mode in site}
        state.update({mode: 0 for site in model.source_b for mode in site})
        state.update(
            {
                site[0]: occupation
                for site, occupation in zip(model.source_b, occupations, strict=True)
            }
        )
        index = sum(
            int(state[op]) << (len(operators) - i - 1) for i, op in enumerate(operators)
        )
        coupled = np.flatnonzero(np.abs(v[:, index]) > 0)
        expected = np.sum(np.abs(v[coupled, index]) ** 2 / (e[index] - e[coupled]))
        actual = (
            second.terms[(0, 0, 0)]
            .subs(parameters)
            .subs(dict(zip((n0, n1, n2), occupations, strict=True)))
        )
        assert abs(float(actual) - expected) < 1e-12
        shifts.append(float(actual))
    assert abs(shifts[2] - 2 * shifts[1] + shifts[0]) > 1e-4


def test_crepel_fu_connected_fourth_order():
    """Two overlapping A stars generate hopping absent at second order."""
    from pymablock.operator_embedding import _number_symbols

    model = crepel_fu_two_star()
    parameters = {model.Delta: 10, model.V0: 2, model.UA: 7, model.UB: 11, model.t0: 1}
    h, *_ = block_diagonalize(
        [model.H0.subs(parameters), model.V.subs(parameters)],
        subspace_eigenvectors=model.encoding,
    )
    powers = (1, 0, 0, -1, 0)
    assert h[0, 0, 2].terms.get(powers, 0) == 0
    coefficient = h[0, 0, 4].terms[powers]
    actual = coefficient.subs(
        dict.fromkeys(_number_symbols(model.encoding.target.operators), 0)
    )
    expected = (
        -(model.t0**4)
        * (2 * model.Delta**2 + 4 * model.Delta * model.V0 + model.V0**2)
        / (2 * (model.Delta + model.V0) ** 3 * (model.Delta + 2 * model.V0) ** 2)
    )
    assert sympy.factor(actual - expected.subs(parameters)) == 0


@dataclass(frozen=True)
class SupercurrentModel:
    """Second-quantized superconducting-dot Hamiltonian and its parameters."""

    H0: NumberOrderedForm
    V: NumberOrderedForm
    source_operators: tuple[FermionOp, ...]
    dot: tuple[FermionOp, FermionOp]
    quasiparticles: tuple[FermionOp, ...]
    U: sympy.Symbol
    N: sympy.Symbol
    xi_L: sympy.Symbol  # noqa: N815
    xi_R: sympy.Symbol  # noqa: N815
    E_L: sympy.Symbol
    E_R: sympy.Symbol
    u_L: sympy.Symbol  # noqa: N815
    u_R: sympy.Symbol  # noqa: N815
    v_L: sympy.Symbol  # noqa: N815
    v_R: sympy.Symbol  # noqa: N815
    t_L: sympy.Symbol  # noqa: N815
    t_R: sympy.Symbol  # noqa: N815
    phi: sympy.Symbol


def make_supercurrent_model() -> SupercurrentModel:
    """Construct the tutorial Hamiltonian as two NOF perturbation coefficients."""
    U, N = sympy.symbols("U N", positive=True)
    xi_L, xi_R, E_L, E_R = sympy.symbols("xi_L xi_R E_L E_R", positive=True)
    u_L, u_R, v_L, v_R = sympy.symbols("u_L u_R v_L v_R", real=True)
    t_L, t_R, phi = sympy.symbols("t_L t_R phi", real=True)

    d_up = FermionOp("d_up")
    d_down = FermionOp("d_down")
    f_Lup = FermionOp("f_Lup")
    f_Ldown = FermionOp("f_Ldown")
    f_Rup = FermionOp("f_Rup")
    f_Rdown = FermionOp("f_Rdown")
    source_operators = tuple(
        sorted(
            (d_up, d_down, f_Lup, f_Ldown, f_Rup, f_Rdown),
            key=lambda operator: str(operator.name),
        )
    )

    def number(operator):
        return Dagger(operator) * operator

    H_dot = U * (number(d_up) + number(d_down) - N) ** 2 / 2
    H_sc = sum(
        xi - energy + energy * number(f_up) + energy * number(f_down)
        for xi, energy, f_up, f_down in (
            (xi_L, E_L, f_Lup, f_Ldown),
            (xi_R, E_R, f_Rup, f_Rdown),
        )
    )

    tunneling = sympy.S.Zero
    for amplitude, u, v, f_up, f_down in (
        (t_L * sympy.exp(sympy.I * phi), u_L, v_L, f_Lup, f_Ldown),
        (t_R, u_R, v_R, f_Rup, f_Rdown),
    ):
        forward = amplitude * (
            Dagger(d_up) * (u * f_up - v * Dagger(f_down))
            + Dagger(d_down) * (u * f_down + v * Dagger(f_up))
        )
        tunneling += forward + Dagger(forward)

    return SupercurrentModel(
        H0=NumberOrderedForm.from_expr(H_dot + H_sc, operators=source_operators),
        V=NumberOrderedForm.from_expr(tunneling, operators=source_operators),
        source_operators=source_operators,
        dot=tuple(
            operator for operator in source_operators if operator in (d_up, d_down)
        ),
        quasiparticles=tuple(
            operator for operator in source_operators if operator not in (d_up, d_down)
        ),
        U=U,
        N=N,
        xi_L=xi_L,
        xi_R=xi_R,
        E_L=E_L,
        E_R=E_R,
        u_L=u_L,
        u_R=u_R,
        v_L=v_L,
        v_R=v_R,
        t_L=t_L,
        t_R=t_R,
        phi=phi,
    )


def sample_values(model: SupercurrentModel) -> dict[sympy.Symbol, sympy.Expr]:
    """Return a generic nondegenerate point for finite-reference validation."""
    return {
        model.U: 10,
        model.N: sympy.Rational(1, 5),
        model.xi_L: sympy.Rational(3, 10),
        model.xi_R: sympy.Rational(2, 5),
        model.E_L: sympy.Rational(11, 10),
        model.E_R: sympy.Rational(7, 5),
        model.u_L: sympy.Rational(3, 5),
        model.v_L: sympy.Rational(4, 5),
        model.u_R: sympy.Rational(5, 13),
        model.v_R: sympy.Rational(12, 13),
        model.t_L: sympy.Rational(1, 5),
        model.t_R: sympy.Rational(3, 20),
        model.phi: sympy.Rational(37, 100),
    }


def test_supercurrent_fourth_order_and_phase_derivative():
    """Check all dot charge sectors against full-Fock-space resolvent sums."""
    from itertools import product

    import numpy as np

    from pymablock.operator_embedding import _number_symbols
    from pymablock.tests.second_quantization_helpers import (
        occupation_matrices,
        operator_matrix,
    )

    model = make_supercurrent_model()
    parameters = sample_values(model)
    phi_value = parameters.pop(model.phi)
    reference = dict.fromkeys(model.source_operators, 0)
    embedding = Embedding({mode: mode for mode in model.dot}, reference=reference)
    h0 = model.H0.as_expr().subs(parameters)
    v = model.V.as_expr().subs(parameters)
    h, *_ = block_diagonalize([h0, v], subspace_eigenvectors=embedding)
    h2, h4 = h[0, 0, 2], h[0, 0, 4]
    target_matrices = occupation_matrices(model.dot, [range(2)] * 2)
    e_target = operator_matrix(h[0, 0, 0], target_matrices).diagonal().real
    states = tuple(product((0, 1), repeat=len(model.source_operators)))
    kept = [
        i
        for i, state in enumerate(states)
        if all(
            state[model.source_operators.index(op)] == 0 for op in model.quasiparticles
        )
    ]
    source_matrices = occupation_matrices(model.source_operators, [range(2)] * 6)
    source_matrices[sympy.I] = 1j * source_matrices[sympy.S.One]
    energy = operator_matrix(h0, source_matrices).diagonal().real
    charge_groups = ((0,), (1, 2), (3,))

    def reference_fourth(phi):
        perturbation = operator_matrix(
            v.subs(model.phi, phi).evalf(), source_matrices
        ).toarray()
        result = []
        for group in charge_groups:
            p = [kept[i] for i in group]
            q = [i for i in range(64) if i not in p]
            gap = energy[p[0]] - energy[q]
            assert np.min(np.abs(gap)) > 0.01
            pq = perturbation[np.ix_(p, q)]
            qq = perturbation[np.ix_(q, q)]
            r = np.diag(1 / gap)
            second = pq @ r @ pq.conj().T
            norm = pq @ r @ r @ pq.conj().T
            fourth = (
                pq @ r @ qq @ r @ qq @ r @ pq.conj().T
                - (norm @ second + second @ norm) / 2
            )
            result.extend(np.diag(fourth).real)
        return np.array(result)

    numbers = _number_symbols(model.dot)
    energies = []
    # The two even charge sectors also acquire a fourth-order energy shift
    # from their second-order pairing inside the retained four-state block.
    for row, state in enumerate(product((0, 1), repeat=2)):
        value = h4.terms.get((0, 0), 0).subs(dict(zip(numbers, state, strict=True)))
        if row in (0, 3):
            off_diagonal = h2.terms[(-1, -1)]
            off_diagonal = off_diagonal.subs(dict.fromkeys(numbers, 0))
            other = 3 - row
            value += (
                off_diagonal
                * sympy.conjugate(off_diagonal)
                / sympy.Rational(str(e_target[row] - e_target[other]))
            )
        energies.append(sympy.simplify(value))
    actual = np.array(
        [complex(value.subs(model.phi, phi_value).evalf()).real for value in energies]
    )
    expected = reference_fourth(phi_value)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
    current = np.array(
        [
            complex(sympy.diff(value, model.phi).subs(model.phi, phi_value).evalf()).real
            for value in energies
        ]
    )
    step = 1e-5
    reference_current = (
        reference_fourth(float(phi_value) + step)
        - reference_fourth(float(phi_value) - step)
    ) / (2 * step)
    np.testing.assert_allclose(current, reference_current, atol=1e-11, rtol=1e-7)
    assert np.max(np.abs(current)) > 1e-7
    assert abs(current[1] - current[2]) < 1e-12


def test_coupler_fourth_order_against_converged_matrices():
    """Check the full fourth-order qubit matrix and oscillator cutoff closure."""
    from itertools import product

    import numpy as np

    from pymablock.tests.second_quantization_helpers import (
        occupation_matrices,
        operator_matrix,
    )

    model = tunable_coupler()
    parameters = dict(
        zip(
            model.frequencies + model.anharmonicities + model.couplings,
            (
                3,
                5,
                9,
                sympy.Rational(-1, 5),
                sympy.Rational(-1, 4),
                sympy.Rational(-1, 7),
                sympy.Rational(1, 11),
                sympy.Rational(1, 5),
                sympy.Rational(1, 6),
            ),
            strict=True,
        )
    )
    source = [value.subs(parameters) for value in (model.H0, model.V)]
    effective = block_diagonalize(source, subspace_eigenvectors=model.encoding)[0]
    target_matrices = occupation_matrices(model.encoding.target.operators, [range(2)] * 2)
    actual = operator_matrix(effective[0, 0, 4], target_matrices).toarray()
    for cutoff in (4, 5):
        operators = model.encoding.operators
        matrices = occupation_matrices(operators, [range(cutoff)] * 3)
        h0, v = (operator_matrix(value, matrices).toarray() for value in source)
        states = list(product(range(cutoff), repeat=3))
        # Canonical source order is a1, a2, ac. No generated columns are used
        # in this independent finite-matrix construction.
        kept = [states.index((n1, n2, 0)) for n1, n2 in product(range(2), repeat=2)]
        labels = np.ones(len(states), dtype=int)
        labels[kept] = 0
        reference = block_diagonalize([h0, v], subspace_indices=labels)[0][0, 0, 4]
        np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-10)
