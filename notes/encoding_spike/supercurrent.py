"""Single-NOF supercurrent benchmark for the algebraic embedding backend.

The source Hamiltonian contains all six fermionic modes.  Each requested dot
occupation is embedded with the four Bogoliubov quasiparticle modes fixed to their
vacuum.  The algebraic backend evaluates the resulting ``Q X W`` columns without
constructing the 64-dimensional source matrix.
"""

# The left/right subscripts are the conventional notation of this model.
# ruff: noqa: N815

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOrderedForm

from .algebraic import block_diagonalize_algebraic
from .encoding import CompiledOperator, fermion_embedding, occupation_map


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
    xi_L: sympy.Symbol
    xi_R: sympy.Symbol
    E_L: sympy.Symbol
    E_R: sympy.Symbol
    u_L: sympy.Symbol
    u_R: sympy.Symbol
    v_L: sympy.Symbol
    v_R: sympy.Symbol
    t_L: sympy.Symbol
    t_R: sympy.Symbol
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


def fourth_order_energies(
    model: SupercurrentModel,
) -> tuple[dict[tuple[int, int], sympy.Expr], dict[tuple[int, int], float]]:
    """Compute all dot-occupation energies using algebraic scalar embeddings."""
    energies = {}
    timings = {}
    for dot_state in product((0, 1), repeat=2):
        occupations = dict.fromkeys(model.source_operators, 0)
        occupations.update(dict(zip(model.dot, dot_state, strict=True)))
        start = perf_counter()
        effective, *_ = block_diagonalize_algebraic(
            [model.H0, model.V],
            encoding=occupation_map(occupations),
            to_order=4,
        )
        energies[dot_state] = effective[4].as_expr()
        timings[dot_state] = perf_counter() - start
    return energies, timings


def effective_dot_hamiltonian(
    model: SupercurrentModel,
) -> tuple[dict[int, NumberOrderedForm], float]:
    """Retain both dot fermions and eliminate quasiparticle excitations at once."""
    modes = {
        operator: operator if operator in model.dot else 0
        for operator in model.source_operators
    }
    start = perf_counter()
    effective, *_ = block_diagonalize_algebraic(
        [model.H0, model.V],
        encoding=fermion_embedding(modes),
        to_order=4,
    )
    result = {order: effective[order].form for order in (0, 2, 4)}
    return result, perf_counter() - start


def diagonal_dot_nof(
    model: SupercurrentModel,
    values: dict[tuple[int, int], sympy.Expr],
) -> NumberOrderedForm:
    """Interpolate four occupation values into one diagonal two-fermion NOF."""
    identity = NumberOrderedForm(
        model.dot,
        {(0, 0): sympy.S.One},
        validate=False,
    )
    n_0, n_1 = identity._number_operator_placeholders
    value_00 = values[0, 0]
    value_10 = values[1, 0]
    value_01 = values[0, 1]
    value_11 = values[1, 1]
    coefficient = (
        value_00
        + (value_10 - value_00) * n_0
        + (value_01 - value_00) * n_1
        + (value_11 - value_10 - value_01 + value_00) * n_0 * n_1
    )
    return NumberOrderedForm(
        model.dot,
        {(0, 0): coefficient},
        validate=False,
    )


def josephson_current(
    model: SupercurrentModel,
    energies: dict[tuple[int, int], sympy.Expr],
) -> NumberOrderedForm:
    """Extract the phase response before interpolating the target NOF."""

    def mixed_current(coefficient):
        return sympy.diff(
            sympy.expand(coefficient).coeff(model.t_L, 2).coeff(model.t_R, 2),
            model.phi,
        )

    return diagonal_dot_nof(
        model,
        {state: mixed_current(value) for state, value in energies.items()},
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


def _finite_matrix(
    form: NumberOrderedForm,
    states: tuple[tuple[int, ...], ...],
    state_index: dict[tuple[int, ...], int],
    substitutions: dict[sympy.Symbol, sympy.Expr],
) -> np.ndarray:
    compiled = CompiledOperator(form.as_expr(), form.operators)
    matrix = np.zeros((len(states), len(states)), dtype=complex)
    for column, state in enumerate(states):
        for output, coefficient in compiled.apply(state).items():
            matrix[state_index[output], column] += complex(
                coefficient.subs(substitutions).evalf()
            )
    return matrix


def finite_reference_residuals(
    model: SupercurrentModel,
    energies: dict[tuple[int, int], sympy.Expr],
) -> dict[tuple[int, int], float]:
    """Compare nondegenerate charge states with the full 64-state calculation."""
    substitutions = sample_values(model)
    states = tuple(product((0, 1), repeat=len(model.source_operators)))
    state_index = {state: index for index, state in enumerate(states)}
    H0 = _finite_matrix(model.H0, states, state_index, substitutions)
    V = _finite_matrix(model.V, states, state_index, substitutions)

    residuals = {}
    for dot_state in ((0, 0), (1, 1)):
        occupation = dict.fromkeys(model.source_operators, 0)
        occupation.update(dict(zip(model.dot, dot_state, strict=True)))
        target = tuple(occupation[operator] for operator in model.source_operators)
        labels = np.ones(len(states), dtype=int)
        labels[state_index[target]] = 0
        finite = block_diagonalize(
            [H0, V],
            subspace_indices=labels,
        )[0][0, 0, 4][0, 0]
        algebraic = complex(energies[dot_state].subs(substitutions).evalf())
        residuals[dot_state] = abs(algebraic - finite)
    return residuals


def target_nof_residuals(
    model: SupercurrentModel,
    effective_dot: dict[int, NumberOrderedForm],
    energies: dict[tuple[int, int], sympy.Expr],
) -> dict[tuple[int, int], float]:
    """Validate that the target NOFs reproduce fully diagonalized energies."""
    substitutions = sample_values(model)
    states = tuple(product((0, 1), repeat=2))
    state_index = {state: index for index, state in enumerate(states)}
    matrices = {
        order: _finite_matrix(form, states, state_index, substitutions)
        for order, form in effective_dot.items()
    }
    H0, H2, H4 = (matrices[order] for order in (0, 2, 4))
    residuals = {}
    for state in states:
        row = state_index[state]
        correction = H4[row, row]
        for other in states:
            column = state_index[other]
            if column == row:
                continue
            numerator = H2[row, column] * H2[column, row]
            denominator = H0[row, row] - H0[column, column]
            if abs(denominator) < 1e-12:
                if abs(numerator) >= 1e-12:
                    raise ValueError("Degenerate target states mix at second order")
                continue
            correction += numerator / denominator
        reference = complex(energies[state].subs(substitutions).evalf())
        residuals[state] = abs(correction - reference)
    return residuals


def benchmark_supercurrent() -> dict[str, object]:
    """Run the algebraic calculation and its finite validation."""
    model = make_supercurrent_model()
    start = perf_counter()
    effective_dot, target_time = effective_dot_hamiltonian(model)
    energies, timings = fourth_order_energies(model)
    fourth_order = diagonal_dot_nof(model, energies)
    current = josephson_current(model, energies)
    total = perf_counter() - start
    return {
        "model": model,
        "energies": energies,
        "effective_dot": effective_dot,
        "fourth_order": fourth_order,
        "current": current,
        "target_time": target_time,
        "timings": timings,
        "total": total,
        "finite_residuals": finite_reference_residuals(model, energies),
        "target_residuals": target_nof_residuals(model, effective_dot, energies),
    }


if __name__ == "__main__":
    benchmark = benchmark_supercurrent()
    print(f"two-fermion target NOF: {benchmark['target_time']:.3f} s")
    print(
        "target terms:",
        {order: len(form.terms) for order, form in benchmark["effective_dot"].items()},
    )
    print("projection timings:")
    for state, elapsed in benchmark["timings"].items():
        print(f"  {state}: {elapsed:.3f} s")
    print(f"symbolic total: {benchmark['total']:.3f} s")
    print(f"current operations: {sympy.count_ops(benchmark['current'].as_expr())}")
    print(f"finite residuals: {benchmark['finite_residuals']}")
    print(f"target-NOF residuals: {benchmark['target_residuals']}")
