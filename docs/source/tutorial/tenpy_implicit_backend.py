"""Finite TeNPy backend for implicit state-space perturbation theory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tenpy.networks.mps import MPS

from pymablock.implicit import StateSolveResult

if TYPE_CHECKING:
    from numbers import Number

    from tenpy.networks.mpo import MPO

try:
    from .tenpy_mpo_backend import (
        TenpyMPOBackend,
        _add_mps,
        _mps_norm,
        _scale_mps,
        _validate_mpo,
        product_mpo,
    )
except ImportError:
    from tenpy_mpo_backend import (  # type: ignore[no-redef]
        TenpyMPOBackend,
        _add_mps,
        _mps_norm,
        _scale_mps,
        _validate_mpo,
        product_mpo,
    )


@dataclass(frozen=True)
class ImplicitSolverRecord:
    """Diagnostics for one two-site variational linear solve."""

    perturbative_index: tuple[int, ...]
    column: int
    relative_residuals: tuple[float, ...]
    orthogonality_errors: tuple[float, ...]
    maximum_bond_dimensions: tuple[int, ...]
    maximum_discarded_weights: tuple[float, ...]
    converged: bool


def _validate_state(state: MPS, operator: MPO | None = None) -> None:
    if not state.finite or state.bc != "finite":
        raise ValueError("Only finite MPSs are supported.")
    if any(site.leg.chinfo.qnumber for site in state.sites):
        raise ValueError("The TeNPy example requires `conserve=None`.")
    if operator is not None:
        _validate_mpo(operator)
        if state.L != operator.L:
            raise ValueError("The MPS and MPO have different lengths.")
        if [site.dim for site in state.sites] != [site.dim for site in operator.sites]:
            raise ValueError("The MPS and MPO have different local dimensions.")


def _state_cores(state: MPS) -> list[np.ndarray]:
    """Return ordinary MPS tensors carrying the complete state norm."""
    canonical = state.copy()
    canonical.canonical_form(renormalize=False)
    cores = [
        canonical.get_B(i, form=None).transpose(["vL", "p", "vR"]).to_ndarray()
        for i in range(canonical.L)
    ]
    cores[0] = canonical.norm * cores[0]
    return cores


def _from_state_cores(template: MPS, cores: list[np.ndarray]) -> MPS:
    """Construct a canonical finite MPS without discarding its norm."""
    overlap = np.ones((1, 1), dtype=complex)
    for core in cores:
        overlap = np.einsum(
            "ij,iar,jas->rs",
            overlap,
            core.conj(),
            core,
            optimize=True,
        )
    norm = float(np.sqrt(max(float(np.real(overlap[0, 0])), 0.0)))
    flat = [core.transpose(1, 0, 2) for core in cores]
    result = MPS.from_Bflat(
        template.sites,
        flat,
        bc="finite",
        permute=False,
        form=None,
        unit_cell_width=template.L,
    )
    result.norm = norm
    return result


def _mpo_cores(operator: MPO) -> list[np.ndarray]:
    """Return MPO tensors ordered as left, right, output, input."""
    return [
        operator.get_W(i).transpose(["wL", "wR", "p", "p*"]).to_ndarray()
        for i in range(operator.L)
    ]


def _left_operator_environment(
    bra: list[np.ndarray],
    operator: list[np.ndarray],
    ket: list[np.ndarray],
    stop: int,
) -> np.ndarray:
    environment = np.ones((1, 1, 1), dtype=complex)
    for site in range(stop):
        environment = np.einsum(
            "ijk,iar,jlab,kbs->rls",
            environment,
            bra[site].conj(),
            operator[site],
            ket[site],
            optimize=True,
        )
    return environment


def _right_operator_environment(
    bra: list[np.ndarray],
    operator: list[np.ndarray],
    ket: list[np.ndarray],
    start: int,
) -> np.ndarray:
    environment = np.ones((1, 1, 1), dtype=complex)
    for site in range(len(bra) - 1, start - 1, -1):
        environment = np.einsum(
            "iar,jlab,kbs,rls->ijk",
            bra[site].conj(),
            operator[site],
            ket[site],
            environment,
            optimize=True,
        )
    return environment


def _left_overlap_environment(
    bra: list[np.ndarray],
    ket: list[np.ndarray],
    stop: int,
) -> np.ndarray:
    environment = np.ones((1, 1), dtype=complex)
    for site in range(stop):
        environment = np.einsum(
            "ij,iar,jas->rs",
            environment,
            bra[site].conj(),
            ket[site],
            optimize=True,
        )
    return environment


def _right_overlap_environment(
    bra: list[np.ndarray],
    ket: list[np.ndarray],
    start: int,
) -> np.ndarray:
    environment = np.ones((1, 1), dtype=complex)
    for site in range(len(bra) - 1, start - 1, -1):
        environment = np.einsum(
            "iar,jas,rs->ij",
            bra[site].conj(),
            ket[site],
            environment,
            optimize=True,
        )
    return environment


def _two_site_operator(
    solution: list[np.ndarray],
    operator: list[np.ndarray],
    site: int,
) -> np.ndarray:
    left = _left_operator_environment(solution, operator, solution, site)
    right = _right_operator_environment(solution, operator, solution, site + 2)
    tensor = np.einsum(
        "ijk,jmab,mncd,rns->iacrkbds",
        left,
        operator[site],
        operator[site + 1],
        right,
        optimize=True,
    )
    dimension = int(np.prod(tensor.shape[:4]))
    return tensor.reshape(dimension, dimension)


def _two_site_source(
    solution: list[np.ndarray],
    source: list[np.ndarray],
    site: int,
) -> np.ndarray:
    left = _left_overlap_environment(solution, source, site)
    right = _right_overlap_environment(solution, source, site + 2)
    tensor = np.einsum(
        "ik,kam,mcs,rs->iacr",
        left,
        source[site],
        source[site + 1],
        right,
        optimize=True,
    )
    return tensor.reshape(-1)


def _two_site_constraint(
    reference: list[np.ndarray],
    solution: list[np.ndarray],
    site: int,
) -> np.ndarray:
    left = _left_overlap_environment(reference, solution, site)
    right = _right_overlap_environment(reference, solution, site + 2)
    tensor = np.einsum(
        "ik,iam,mcr,rs->kacs",
        left,
        reference[site].conj(),
        reference[site + 1].conj(),
        right,
        optimize=True,
    )
    return tensor.reshape(-1)


def _split_two_site(
    theta: np.ndarray,
    left_shape: tuple[int, int],
    right_shape: tuple[int, int],
    *,
    move_right: bool,
    chi_max: int,
    svd_min: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    left_bond, left_physical = left_shape
    right_physical, right_bond = right_shape
    matrix = theta.reshape(
        left_bond * left_physical,
        right_physical * right_bond,
    )
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    keep = min(chi_max, int(np.count_nonzero(singular_values >= svd_min)))
    keep = max(keep, 1)
    norm_squared = float(np.sum(singular_values**2))
    discarded = float(np.sum(singular_values[keep:] ** 2))
    discarded_weight = discarded / norm_squared if norm_squared else 0.0
    left = left[:, :keep]
    singular_values = singular_values[:keep]
    right = right[:keep]
    if move_right:
        left_core = left.reshape(left_bond, left_physical, keep)
        right_core = (singular_values[:, None] * right).reshape(
            keep, right_physical, right_bond
        )
    else:
        left_core = (left * singular_values).reshape(left_bond, left_physical, keep)
        right_core = right.reshape(keep, right_physical, right_bond)
    return left_core, right_core, discarded_weight


class TenpyImplicitBackend:
    """MPS algebra and a constrained two-site Hylleraas sweep."""

    def __init__(
        self,
        *,
        chi_max: int = 64,
        svd_min: float = 1e-12,
        max_sweeps: int = 20,
        solver_tolerance: float = 1e-9,
    ) -> None:
        """Configure MPS truncation and variational-sweep tolerances."""
        self.truncation_parameters = {
            "chi_max": chi_max,
            "svd_min": svd_min,
        }
        self.max_sweeps = max_sweeps
        self.solver_tolerance = solver_tolerance
        self.mpo_backend = TenpyMPOBackend(
            chi_max=chi_max,
            svd_min=svd_min,
        )
        self.solver_records: list[ImplicitSolverRecord] = []

    @property
    def chi_max(self) -> int:
        """Maximum retained MPS bond dimension."""
        return int(self.truncation_parameters["chi_max"])

    @property
    def svd_min(self) -> float:
        """Smallest retained two-site singular value."""
        return float(self.truncation_parameters["svd_min"])

    def apply(self, operator: MPO, state: MPS) -> MPS:
        """Apply and immediately compress an MPO."""
        _validate_state(state, operator)
        result = state.copy()
        operator.apply(
            result,
            {
                "compression_method": "SVD",
                "trunc_params": self.truncation_parameters,
            },
        )
        return result

    def add_states(self, left: MPS, right: MPS) -> MPS:
        """Add and immediately compress two finite MPSs."""
        _validate_state(left)
        _validate_state(right)
        if left.L != right.L:
            raise ValueError("The MPSs have different lengths.")
        if left.norm == 0:
            return right.copy()
        if right.norm == 0:
            return left.copy()
        return _add_mps(
            left,
            right,
            1,
            1,
            self.truncation_parameters,
        )

    def scale_state(self, state: MPS, factor: Number) -> MPS:
        """Return a scaled MPS without changing the input."""
        _validate_state(state)
        if factor == 0:
            result = state.copy()
            result.norm = 0.0
            return result
        return _scale_mps(state, factor)

    def inner(self, left: MPS, right: MPS) -> complex:
        """Return the MPS overlap."""
        _validate_state(left)
        _validate_state(right)
        return complex(left.overlap(right))

    def adjoint_operator(self, operator: MPO) -> MPO:
        """Return the MPO adjoint."""
        _validate_mpo(operator)
        return operator.dagger()

    def _project(self, state: MPS, references: tuple[MPS, ...]) -> MPS:
        result = state
        for reference in references:
            overlap = self.inner(reference, result)
            if overlap:
                result = self.add_states(
                    result,
                    self.scale_state(reference, -overlap),
                )
        return result

    def _relative_residual(
        self,
        h_0: MPO,
        energy: complex,
        solution: MPS,
        rhs: MPS,
        references: tuple[MPS, ...],
    ) -> tuple[float, float]:
        applied = self.add_states(
            self.apply(h_0, solution),
            self.scale_state(solution, -energy),
        )
        applied = self._project(applied, references)
        residual = self.add_states(applied, self.scale_state(rhs, -1))
        relative = _mps_norm(residual) / _mps_norm(rhs)
        orthogonality = max(
            (abs(self.inner(reference, solution)) for reference in references),
            default=0.0,
        )
        return relative, orthogonality

    def solve_shifted(
        self,
        h_0: MPO,
        energy: complex,
        rhs: MPS,
        references: tuple[MPS, ...],
        index: tuple[int, ...],
        column: int,
    ) -> StateSolveResult[MPS]:
        """Solve ``Q(H_0-energy)Q x=rhs`` with constrained two-site sweeps."""
        _validate_state(rhs, h_0)
        for reference in references:
            _validate_state(reference, h_0)
        rhs = self._project(rhs, references)
        rhs_norm = _mps_norm(rhs)
        if rhs_norm == 0:
            return StateSolveResult(
                self.scale_state(rhs, 0),
                0.0,
                True,
                0,
                "right-hand side is exactly zero",
            )

        identity = product_mpo(h_0.sites, [np.eye(site.dim) for site in h_0.sites])
        shifted = self.mpo_backend.add(
            h_0,
            self.mpo_backend.scale(identity, -energy),
        )
        shifted_cores = _mpo_cores(shifted)
        source_cores = _state_cores(rhs)
        reference_cores = [_state_cores(reference) for reference in references]
        solution = rhs

        residuals = []
        orthogonality_errors = []
        maximum_bond_dimensions = []
        discarded_weights = []
        converged = False

        for _sweep in range(1, self.max_sweeps + 1):
            solution_cores = _state_cores(solution)
            maximum_discarded = 0.0
            schedule = [
                *((site, True) for site in range(h_0.L - 1)),
                *((site, False) for site in range(h_0.L - 2, -1, -1)),
            ]
            for site, move_right in schedule:
                effective = _two_site_operator(
                    solution_cores,
                    shifted_cores,
                    site,
                )
                hermiticity_error = np.linalg.norm(effective - effective.conj().T) / max(
                    np.linalg.norm(effective), 1.0
                )
                if hermiticity_error > 1e-9:
                    raise RuntimeError(
                        "The local shifted Hamiltonian is not Hermitian; "
                        f"relative defect {hermiticity_error:.3e}."
                    )
                effective = (effective + effective.conj().T) / 2
                source = _two_site_source(solution_cores, source_cores, site)
                constraints = np.vstack(
                    [
                        _two_site_constraint(reference, solution_cores, site)
                        for reference in reference_cores
                    ]
                )
                dimension = effective.shape[0]
                augmented = np.block(
                    [
                        [effective, constraints.conj().T],
                        [
                            constraints,
                            np.zeros(
                                (len(references), len(references)),
                                dtype=complex,
                            ),
                        ],
                    ]
                )
                target = np.concatenate(
                    [source, np.zeros(len(references), dtype=complex)]
                )
                try:
                    theta = np.linalg.solve(augmented, target)[:dimension]
                except np.linalg.LinAlgError as error:
                    raise RuntimeError(
                        "A local constrained Hylleraas equation is singular. "
                        "No pseudoinverse or broadening is applied."
                    ) from error

                left = solution_cores[site]
                right = solution_cores[site + 1]
                left_core, right_core, discarded = _split_two_site(
                    theta,
                    (left.shape[0], left.shape[1]),
                    (right.shape[1], right.shape[2]),
                    move_right=move_right,
                    chi_max=self.chi_max,
                    svd_min=self.svd_min,
                )
                solution_cores[site] = left_core
                solution_cores[site + 1] = right_core
                maximum_discarded = max(maximum_discarded, discarded)

            solution = _from_state_cores(rhs, solution_cores)
            solution = self._project(solution, references)
            relative_residual, orthogonality = self._relative_residual(
                h_0,
                energy,
                solution,
                rhs,
                references,
            )
            residuals.append(relative_residual)
            orthogonality_errors.append(orthogonality)
            maximum_bond_dimensions.append(max(solution.chi))
            discarded_weights.append(maximum_discarded)
            if relative_residual <= self.solver_tolerance:
                converged = True
                break

        record = ImplicitSolverRecord(
            index,
            column,
            tuple(residuals),
            tuple(orthogonality_errors),
            tuple(maximum_bond_dimensions),
            tuple(discarded_weights),
            converged,
        )
        self.solver_records.append(record)
        message = None
        if not converged:
            message = f"two-site sweeps stopped at relative residual {residuals[-1]:.3e}"
        return StateSolveResult(
            solution,
            residuals[-1],
            converged,
            len(residuals),
            message,
        )
