"""Run the physical examples and print their effective target-space readouts."""

from __future__ import annotations

from time import perf_counter

import sympy

from .algebraic import block_diagonalize_algebraic, block_diagonalize_maps
from .encoding import block_diagonalize
from .models import (
    crepel_fu_triangle,
    crepel_fu_two_star,
    fermionic_ring_exchange,
    synthetic_spin_floquet,
    tunable_coupler,
)


def run_coupler() -> None:
    """Compute the fourth-order two-spin coupler Hamiltonian."""
    model = tunable_coupler()
    start = perf_counter()
    H_tilde, *_ = block_diagonalize(
        [model.H0, model.V], encoding=model.encoding, to_order=4
    )
    H4 = H_tilde[4]
    elapsed = perf_counter() - start
    pauli = H4.pauli()
    cross_kerr = 4 * pauli.coefficient("ZZ")
    omega1, omega2, omega_c = model.frequencies
    alpha1, alpha2, alpha_c = model.anharmonicities
    sample = sympy.Poly(
        cross_kerr.subs(
            {
                omega1: 5,
                omega2: sympy.Rational(11, 2),
                omega_c: 7,
                alpha1: sympy.Rational(-1, 5),
                alpha2: sympy.Rational(-1, 4),
                alpha_c: sympy.Rational(-3, 10),
            }
        ),
        *model.couplings,
    )
    print(
        "coupler:",
        f"{H_tilde.info.retained_states} retained + "
        f"{H_tilde.info.virtual_states} virtual states,",
        f"H4 in {elapsed:.3f} s",
    )
    print(
        "  held fourth-order cross-Kerr:",
        f"{sympy.count_ops(cross_kerr)} scalar operations",
    )
    terms = []
    for powers, coefficient in sample.terms():
        monomial = sympy.prod(
            symbol**power for symbol, power in zip(model.couplings, powers, strict=True)
        )
        terms.append(f"{float(coefficient):+.6g}*{monomial}")
    polynomial = " ".join(terms).lstrip("+").replace("+-", "-")
    print("  sampled coefficient polynomial:", polynomial)


def run_ring_exchange() -> None:
    """Compute the fourth-order ring exchange of the half-filled Hubbard square."""
    model = fermionic_ring_exchange()
    start = perf_counter()
    H_tilde, *_ = block_diagonalize(
        [model.H0, model.V], encoding=model.encoding, to_order=4
    )
    amplitude = H_tilde[4].matrix_element((0, 1, 0, 1), (1, 0, 1, 0))
    coefficient = sympy.factor(amplitude / 2)
    elapsed = perf_counter() - start
    print(
        "ring exchange:",
        f"{H_tilde.info.retained_states} retained + "
        f"{H_tilde.info.virtual_states} virtual states,",
        f"H4 in {elapsed:.3f} s",
    )
    print("  K in K(C4 + C4†):", coefficient)


def run_crepel_fu() -> None:
    """Check the published process and compute a connected fourth-order hop."""
    triangle = crepel_fu_triangle()
    second_order, *_ = block_diagonalize(
        [triangle.H0, triangle.V], encoding=triangle.encoding, to_order=2
    )
    f0, f1, f2 = triangle.target_fermions
    state = triangle.encoding.target.state
    bare_second = second_order[2].matrix_element(state([f1]), state([f0]))
    assisted_second = second_order[2].matrix_element(state([f1, f2]), state([f0, f2]))

    model = crepel_fu_two_star()
    start = perf_counter()
    H_tilde, *_ = block_diagonalize_maps(
        [model.H0, model.V], encoding=model.encoding, to_order=4
    )
    elapsed = perf_counter() - start
    f0, _, _, f3, _ = model.target_fermions
    state = model.encoding.target.state
    range_two_fourth = H_tilde[4].matrix_element(state([f3]), state([f0]))
    sample = {
        model.Delta: 10,
        model.V0: 2,
        model.UA: 7,
        model.UB: 11,
        model.t0: 1,
    }
    print(
        "Crepel-Fu two-star cluster:",
        f"{len(model.encoding.operators)} source modes and "
        f"{model.encoding.target.dimension} retained states,",
        f"connected H4 in {elapsed:.3f} s",
    )
    print("  target fermions:", H_tilde[4].generators)
    print("  t^(2):", sympy.factor(bare_second))
    print("  lambda^(2):", sympy.factor(assisted_second - bare_second))
    print("  range-two t^(4):", sympy.factor(range_two_fourth))
    print("  sampled range-two t^(4):", sympy.factor(range_two_fourth.subs(sample)))


def run_synthetic_spin() -> None:
    """Compare all implementations for the finite spin-1 Floquet block."""
    model = synthetic_spin_floquet()
    results = {}
    for name, implementation in (
        ("finite reference", block_diagonalize),
        ("algebraic", block_diagonalize_algebraic),
        ("maps", block_diagonalize_maps),
    ):
        start = perf_counter()
        effective, *_ = implementation(
            [model.H0, model.V], encoding=model.encoding, to_order=2
        )
        first = (
            effective[1].matrix()
            if callable(effective[1].matrix)
            else effective[1].matrix
        )
        second = (
            effective[2].matrix()
            if callable(effective[2].matrix)
            else effective[2].matrix
        )
        results[name] = (first, second)
        print(f"synthetic spin-1 ({name}): H2 in {perf_counter() - start:.3f} s")

    substitutions = {
        model.phases[0]: 0,
        model.phases[1]: sympy.pi / 3,
        model.phases[2]: sympy.pi / 2,
        model.cavity_phase: sympy.pi / 7,
        model.chi: 11,
        model.Omega: 3,
        model.epsilon: 1,
    }
    for name in ("algebraic", "maps"):
        difference = (results[name][1] - results["finite reference"][1]).subs(
            substitutions
        )
        assert difference.applyfunc(sympy.simplify) == sympy.zeros(3)
    first = results["maps"][0].applyfunc(
        lambda value: sympy.trigsimp(sympy.expand_complex(value))
    )
    print("  first-order 3x3 block:", first)
    print("  second-order backends agree at an exact generic point")


if __name__ == "__main__":
    run_coupler()
    run_ring_exchange()
    run_crepel_fu()
    run_synthetic_spin()
