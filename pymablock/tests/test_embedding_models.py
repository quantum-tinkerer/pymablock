"""Physical fixtures and analytic checks independent of NOF storage.

Model definitions adapted from the embedding spike in commit 182b54c. Only the
physical Hamiltonians are retained here, without its alternate storage backends.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus
from sympy.physics.quantum.spin import JminusOp, JzOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import LadderOp, NumberOperator
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
        target=(q1, q2),
        occupations={a1: NumberOperator(q1), a2: NumberOperator(q2), ac: 0},
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
        target=spins,
        occupations={
            operator: occupation
            for i in sites
            for operator, occupation in (
                (up[i], NumberOperator(spins[i])),
                (down[i], 1 - NumberOperator(spins[i])),
            )
        },
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
    level = JzOp("S") + spin
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
        target={JminusOp("S"): maximum + 1},
        occupations={cavity: level, floquet: 0, dressed: 0},
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


def test_dressed_spin_one_first_order():
    """The finite-spin output agrees with the analytic dressed drive."""
    model = synthetic_spin_floquet()
    effective, *_ = block_diagonalize(
        [model.H0, model.V], subspace_eigenvectors=model.encoding
    )
    difference = effective[0, 0, 1] - model.expected_first_order()
    assert difference.applyfunc(
        lambda x: sympy.trigsimp(sympy.expand_complex(x))
    ) == sympy.zeros(3)


def test_dressed_spin_one_second_order_against_finite_matrices():
    """A finite occupation-matrix oracle independently checks virtual processes."""
    from itertools import product

    import numpy as np

    from pymablock.tests.second_quantization_helpers import (
        occupation_matrices,
        operator_matrix,
    )

    model = synthetic_spin_floquet()
    substitutions = {
        model.phases[0]: 0,
        model.phases[1]: sympy.pi / 3,
        model.phases[2]: sympy.pi / 2,
        model.cavity_phase: sympy.pi / 7,
        model.chi: 11,
        model.Omega: 3,
        model.epsilon: 1,
    }
    operators = (model.cavity, model.floquet, model.dressed_ancilla)
    occupations = (range(4), range(-3, 4), range(2))
    matrices = occupation_matrices(operators, occupations)
    matrices[sympy.I] = 1j * matrices[sympy.S.One]
    # These ranges include every intermediate state reached by one application
    # of V from the three retained states, hence suffice at second order.
    source = [x.subs(substitutions) for x in (model.H0, model.V)]
    h0 = operator_matrix(source[0].evalf(), matrices).toarray()
    perturbation = operator_matrix(source[1].evalf(), matrices).toarray()
    states = list(product(*occupations))
    retained = [states.index((n, 0, 0)) for n in range(3)]
    complement = [i for i in range(len(states)) if i not in retained]
    energy = np.diag(h0)
    denominator = energy[retained[0]] - energy[complement]
    assert np.all(np.abs(denominator) > 0)
    reference = (perturbation[np.ix_(retained, complement)] / denominator) @ perturbation[
        np.ix_(complement, retained)
    ]
    effective, *_ = block_diagonalize(source, subspace_eigenvectors=model.encoding)
    actual = np.array(effective[0, 0, 2].evalf(), dtype=complex)
    np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)
