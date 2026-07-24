"""Physical benchmark models for the encoding spike."""

from __future__ import annotations

from dataclasses import dataclass

import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock.number_ordered_form import LadderOp, NumberOperator

from .encoding import (
    Coordinate,
    FermionEmbedding,
    OccupationEncoding,
    fermion_embedding,
    levels,
    occupation_map,
)


@dataclass(frozen=True)
class CouplerModel:
    """Three nonlinear bosons with pairwise capacitive coupling."""

    H0: sympy.Expr
    V: sympy.Expr
    encoding: OccupationEncoding
    target_coordinates: tuple[Coordinate, Coordinate]
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
    q1, q2 = levels("q1 q2", 2)
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
    encoding = occupation_map({a1: q1, a2: q2, ac: 0})
    return CouplerModel(
        H0,
        V,
        encoding,
        (q1, q2),
        frequencies,
        anharmonicities,
        (g12, g1c, g2c),
    )


@dataclass(frozen=True)
class RingExchangeModel:
    """Half-filled four-site Hubbard plaquette."""

    H0: sympy.Expr
    V: sympy.Expr
    encoding: OccupationEncoding
    target_coordinates: tuple[Coordinate, ...]
    U: sympy.Symbol
    t: sympy.Symbol


def fermionic_ring_exchange() -> RingExchangeModel:
    """Return a half-filled Hubbard square encoded as four target spins."""
    sites = range(4)
    up = tuple(FermionOp(f"c{i}_up") for i in sites)
    down = tuple(FermionOp(f"c{i}_down") for i in sites)
    spins = levels("s0:4", 2)
    U, t = sympy.symbols("U t", nonzero=True, real=True)

    H0 = U * sympy.Add(*(NumberOperator(up[i]) * NumberOperator(down[i]) for i in sites))
    V = -t * sympy.Add(
        *(
            Dagger(modes[i]) * modes[j] + Dagger(modes[j]) * modes[i]
            for modes in (up, down)
            for i, j in ((0, 1), (1, 2), (2, 3), (3, 0))
        )
    )
    encoding = occupation_map(
        {
            operator: occupation
            for i in sites
            for operator, occupation in (
                (up[i], spins[i]),
                (down[i], 1 - spins[i]),
            )
        }
    )
    return RingExchangeModel(H0, V, encoding, spins, U, t)


@dataclass(frozen=True)
class SyntheticSpinFloquetModel:
    """One-rotation Floquet formulation of the synthetic-spin experiment."""

    H0: sympy.Expr
    V: sympy.Expr
    encoding: OccupationEncoding
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
    level = levels("m", maximum + 1)
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

    encoding = occupation_map({cavity: level, floquet: 0, dressed: 0})
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


@dataclass(frozen=True)
class CrepelFuModel:
    """Connected spinful ionic-Hubbard cluster from Crépel and Fu."""

    H0: sympy.Expr
    V: sympy.Expr
    encoding: FermionEmbedding
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
    encoding = fermion_embedding(
        {
            **{mode: 1 for site in source_a for mode in site},
            **{
                source_b[site][spin]: (target[site] if spin == up else 0)
                for site in range(num_b)
                for spin in (up, down)
            },
        },
        number=(0, 1, 2),
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
