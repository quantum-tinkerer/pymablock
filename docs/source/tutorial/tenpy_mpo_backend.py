"""Finite, charge-free TeNPy backend for the MPO perturbation tutorial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
from tenpy.networks.site import Site

from pymablock.mpo import SylvesterResult

if TYPE_CHECKING:
    from numbers import Number


@dataclass(frozen=True)
class CompressionRecord:
    """Diagnostics for one MPO compression."""

    operation: str
    input_bond_dimensions: tuple[int, ...]
    output_bond_dimensions: tuple[int, ...]
    truncation_error: float


@dataclass(frozen=True)
class SolverRecord:
    """Diagnostics for one restarted GMRES solve."""

    perturbative_index: tuple[int, ...]
    relative_residuals: tuple[float, ...]
    iterations: int
    converged: bool


def _cores(mpo: MPO) -> list[np.ndarray]:
    """Return MPO tensors ordered as output, input, left bond, right bond."""
    return [
        mpo.get_W(i).transpose(["p", "p*", "wL", "wR"]).to_ndarray() for i in range(mpo.L)
    ]


def _operator_sites(mpo: MPO) -> list[Site]:
    """Construct trivial-charge sites for vectorized local operators."""
    return [Site(npc.LegCharge.from_trivial(site.dim * site.dim)) for site in mpo.sites]


def _from_cores(sites: list[Site], cores: list[np.ndarray]) -> MPO:
    """Construct a finite MPO with one-dimensional boundary bonds."""
    if len(sites) != len(cores):
        raise ValueError("The number of MPO cores and sites must match.")
    chinfo = sites[0].leg.chinfo
    left_leg = npc.LegCharge.from_trivial(
        cores[0].shape[2],
        chinfo,
        qconj=1,
    )
    tensors = []
    for site, core in zip(sites, cores):
        right_leg = npc.LegCharge.from_trivial(
            core.shape[3],
            chinfo,
            qconj=-1,
        )
        tensor = npc.Array.from_ndarray(
            core,
            [site.leg, site.leg.conj(), left_leg, right_leg],
            qtotal=chinfo.make_valid(None),
            labels=["p", "p*", "wL", "wR"],
        )
        tensors.append(tensor)
        left_leg = right_leg.conj()
    return MPO(
        sites,
        tensors,
        bc="finite",
        IdL=0,
        IdR=0,
        mps_unit_cell_width=len(sites),
    )


def _zero_mpo(sites: list[Site]) -> MPO:
    """Construct a minimal exact-zero MPO."""
    cores = [np.eye(site.dim).reshape(site.dim, site.dim, 1, 1) for site in sites]
    cores[0] = np.zeros_like(cores[0])
    return _from_cores(sites, cores)


def _mpo_norm_squared(mpo: MPO) -> float:
    """Return the squared Frobenius norm of a finite MPO."""
    return float(np.real(np.real_if_close(mpo.overlap(mpo))))


def mpo_to_mps(mpo: MPO) -> MPS:
    """Vectorize a finite square MPO into an MPS using row-major order."""
    _validate_mpo(mpo)
    norm_squared = _mpo_norm_squared(mpo)
    if norm_squared <= 0:
        raise ValueError("Cannot vectorize a zero MPO.")

    flat_cores = [
        core.reshape(core.shape[0] * core.shape[1], *core.shape[2:])
        for core in _cores(mpo)
    ]
    vector = MPS.from_Bflat(
        _operator_sites(mpo),
        flat_cores,
        bc="finite",
        permute=False,
        form=None,
        unit_cell_width=mpo.L,
    )
    vector.canonical_form(renormalize=True)
    vector.norm = np.sqrt(norm_squared)
    return vector


def mps_to_mpo(vector: MPS, sites: list[Site]) -> MPO:
    """Undo :func:`mpo_to_mps` for square local operator spaces."""
    if not vector.finite or vector.bc != "finite":
        raise ValueError("Only finite MPS vectors are supported.")
    if vector.L != len(sites):
        raise ValueError("The vector and physical sites have different lengths.")

    cores = []
    for i, site in enumerate(sites):
        if vector.sites[i].dim != site.dim**2:
            raise ValueError("The vectorized and physical local dimensions do not match.")
        core = vector.get_B(i, form=None).transpose(["p", "vL", "vR"]).to_ndarray()
        core = core.reshape(site.dim, site.dim, *core.shape[1:])
        cores.append(core)
    cores[0] = vector.norm * cores[0]
    return _from_cores(sites, cores)


def mpo_to_dense(mpo: MPO) -> np.ndarray:
    """Contract a small finite MPO to a dense matrix for validation only."""
    _validate_mpo(mpo)
    cores = _cores(mpo)
    tensor = cores[0][:, :, 0, :]
    for core in cores[1:]:
        tensor = np.tensordot(tensor, core, axes=(-1, 2))
    tensor = np.squeeze(tensor, axis=-1)
    output_axes = list(range(0, 2 * mpo.L, 2))
    input_axes = list(range(1, 2 * mpo.L, 2))
    tensor = tensor.transpose(output_axes + input_axes)
    dimension = int(np.prod([site.dim for site in mpo.sites]))
    return tensor.reshape(dimension, dimension)


def product_mpo(sites: list[Site], operators: list[np.ndarray]) -> MPO:
    """Construct a bond-dimension-one product MPO."""
    if len(sites) != len(operators):
        raise ValueError("Each site needs one local operator.")
    cores = [
        np.asarray(operator).reshape(site.dim, site.dim, 1, 1)
        for site, operator in zip(sites, operators)
    ]
    return _from_cores(sites, cores)


def _validate_mpo(mpo: MPO) -> None:
    if not mpo.finite or mpo.bc != "finite":
        raise ValueError("Only finite MPOs are supported.")
    if mpo.L < 2:
        raise ValueError("The TeNPy example requires at least two sites.")
    if mpo.explicit_plus_hc:
        raise ValueError("`explicit_plus_hc` MPOs are not supported.")
    if any(site.leg.chinfo.qnumber for site in mpo.sites):
        raise ValueError("The TeNPy example requires `conserve=None`.")
    for core, site in zip(_cores(mpo), mpo.sites):
        if core.shape[:2] != (site.dim, site.dim):
            raise ValueError("Only square local operator spaces are supported.")


def _validate_pair(left: MPO, right: MPO) -> None:
    _validate_mpo(left)
    _validate_mpo(right)
    if left.L != right.L:
        raise ValueError("MPOs have different lengths.")
    if [site.dim for site in left.sites] != [site.dim for site in right.sites]:
        raise ValueError("MPOs have different local dimensions.")


def _scale_mps(vector: MPS, factor: complex) -> MPS:
    """Return a scaled, canonical copy of an MPS."""
    result = vector.copy()
    core = result.get_B(0, form=None).copy()
    core *= factor
    result.set_B(0, core, form=None)
    result.canonical_form(renormalize=False)
    return result


def _compress_mps(vector: MPS, truncation_parameters: dict) -> float:
    error = vector.compress_svd(truncation_parameters)
    return float(error.eps)


def _add_mps(
    left: MPS,
    right: MPS,
    alpha: complex,
    beta: complex,
    truncation_parameters: dict,
) -> MPS:
    result = left.add(right, alpha=alpha, beta=beta, cutoff=0)
    _compress_mps(result, truncation_parameters)
    return result


def _mps_norm(vector: MPS) -> float:
    norm_squared = np.real_if_close(vector.overlap(vector))
    return float(np.sqrt(max(float(np.real(norm_squared)), 0.0)))


class TenpyMPOBackend:
    """Compressed MPO algebra and Sylvester solves implemented with TeNPy."""

    def __init__(
        self,
        *,
        chi_max: int = 64,
        svd_min: float = 1e-12,
        krylov_dimension: int = 8,
        max_restarts: int = 10,
        solver_tolerance: float = 1e-9,
    ) -> None:
        """Configure truncation and restarted-GMRES tolerances."""
        self.truncation_parameters = {
            "chi_max": chi_max,
            "svd_min": svd_min,
        }
        self.krylov_dimension = krylov_dimension
        self.max_restarts = max_restarts
        self.solver_tolerance = solver_tolerance
        self.compression_records: list[CompressionRecord] = []
        self.solver_records: list[SolverRecord] = []

    @property
    def chi_max(self) -> int:
        """Maximum retained virtual bond dimension."""
        return int(self.truncation_parameters["chi_max"])

    def _compress(self, mpo: MPO, operation: str) -> MPO:
        _validate_mpo(mpo)
        input_chi = tuple(mpo.chi)
        if _mpo_norm_squared(mpo) == 0:
            result = _zero_mpo(mpo.sites)
            self.compression_records.append(
                CompressionRecord(
                    operation,
                    input_chi,
                    tuple(result.chi),
                    0.0,
                )
            )
            return result
        vector = mpo_to_mps(mpo)
        error = _compress_mps(vector, self.truncation_parameters)
        result = mps_to_mpo(vector, mpo.sites)
        self.compression_records.append(
            CompressionRecord(
                operation,
                input_chi,
                tuple(result.chi),
                error,
            )
        )
        return result

    def add(self, left: MPO, right: MPO) -> MPO:
        """Add MPOs by direct-summing their virtual bonds and compress."""
        _validate_pair(left, right)
        result = []
        for i, (left_core, right_core) in enumerate(zip(_cores(left), _cores(right))):
            if i == 0:
                core = np.concatenate((left_core, right_core), axis=3)
            elif i == left.L - 1:
                core = np.concatenate((left_core, right_core), axis=2)
            else:
                output_dim, input_dim = left_core.shape[:2]
                core = np.zeros(
                    (
                        output_dim,
                        input_dim,
                        left_core.shape[2] + right_core.shape[2],
                        left_core.shape[3] + right_core.shape[3],
                    ),
                    dtype=np.result_type(left_core, right_core),
                )
                core[
                    :,
                    :,
                    : left_core.shape[2],
                    : left_core.shape[3],
                ] = left_core
                core[
                    :,
                    :,
                    left_core.shape[2] :,
                    left_core.shape[3] :,
                ] = right_core
            result.append(core)
        return self._compress(_from_cores(left.sites, result), "add")

    def scale(self, operator: MPO, factor: Number) -> MPO:
        """Scale one boundary tensor without changing bond dimensions."""
        _validate_mpo(operator)
        cores = _cores(operator)
        cores[0] = factor * cores[0]
        return _from_cores(operator.sites, cores)

    def matmul(self, left: MPO, right: MPO) -> MPO:
        """Multiply MPOs locally, fuse their virtual bonds, and compress."""
        _validate_pair(left, right)
        result = []
        for left_core, right_core in zip(_cores(left), _cores(right)):
            core = np.einsum(
                "suab,utcd->stacbd",
                left_core,
                right_core,
                optimize=True,
            )
            result.append(
                core.reshape(
                    core.shape[0],
                    core.shape[1],
                    core.shape[2] * core.shape[3],
                    core.shape[4] * core.shape[5],
                )
            )
        return self._compress(_from_cores(left.sites, result), "matmul")

    def adjoint(self, operator: MPO) -> MPO:
        """Return the exact TeNPy adjoint."""
        _validate_mpo(operator)
        return operator.dagger()

    def sylvester_superoperator(self, left: MPO, right: MPO) -> MPO:
        """Construct ``left kron I - I kron right.T`` as an MPO."""
        _validate_pair(left, right)
        operator_sites = _operator_sites(left)
        left_cores = []
        right_cores = []
        for left_core, right_core in zip(_cores(left), _cores(right)):
            dimension = left_core.shape[0]
            identity = np.eye(dimension)
            left_action = np.einsum(
                "suab,tv->stuvab",
                left_core,
                identity,
                optimize=True,
            )
            right_action = np.einsum(
                "su,vtcd->stuvcd",
                identity,
                right_core,
                optimize=True,
            )
            left_cores.append(
                left_action.reshape(
                    dimension**2,
                    dimension**2,
                    left_core.shape[2],
                    left_core.shape[3],
                )
            )
            right_cores.append(
                right_action.reshape(
                    dimension**2,
                    dimension**2,
                    right_core.shape[2],
                    right_core.shape[3],
                )
            )
        left_action = _from_cores(operator_sites, left_cores)
        right_action = _from_cores(operator_sites, right_cores)
        return self.add(left_action, self.scale(right_action, -1))

    def _apply_superoperator(self, operator: MPO, vector: MPS) -> MPS:
        result = vector.copy()
        error = operator.apply(
            result,
            {
                "compression_method": "SVD",
                "trunc_params": self.truncation_parameters,
            },
        )
        self.compression_records.append(
            CompressionRecord(
                "superoperator_apply",
                tuple(left * right for left, right in zip(operator.chi, vector.chi)),
                tuple(result.chi),
                float(error.eps),
            )
        )
        return result

    def _linear_combination(
        self,
        vectors: list[MPS],
        coefficients: np.ndarray,
    ) -> MPS:
        nonzero = [
            (vector, coefficient)
            for vector, coefficient in zip(vectors, coefficients)
            if abs(coefficient) > np.finfo(float).eps
        ]
        if not nonzero:
            raise RuntimeError("GMRES produced a zero update.")
        result = _scale_mps(*nonzero[0])
        for vector, coefficient in nonzero[1:]:
            result = _add_mps(
                result,
                vector,
                1,
                coefficient,
                self.truncation_parameters,
            )
        return result

    def solve_sylvester(
        self,
        left: MPO,
        right: MPO,
        rhs: MPO,
        index: tuple[int, ...],
    ) -> SylvesterResult[MPO]:
        """Solve ``left @ X - X @ right = rhs`` with compressed GMRES."""
        _validate_pair(left, right)
        _validate_pair(left, rhs)
        if _mpo_norm_squared(rhs) == 0:
            self.solver_records.append(SolverRecord(index, (0.0,), 0, True))
            return SylvesterResult(
                _zero_mpo(rhs.sites),
                0.0,
                True,
                0,
                "right-hand side is exactly zero",
            )
        superoperator = self.sylvester_superoperator(left, right)
        source = mpo_to_mps(rhs)
        source_norm = _mps_norm(source)
        solution = None
        residuals = []
        iterations = 0

        for _restart in range(self.max_restarts):
            if solution is None:
                residual = source.copy()
            else:
                applied = self._apply_superoperator(superoperator, solution)
                residual = _add_mps(
                    source,
                    applied,
                    1,
                    -1,
                    self.truncation_parameters,
                )
            beta = _mps_norm(residual)
            relative_residual = beta / source_norm
            residuals.append(relative_residual)
            if relative_residual <= self.solver_tolerance:
                break

            basis = [_scale_mps(residual, 1 / beta)]
            hessenberg = np.zeros(
                (self.krylov_dimension + 1, self.krylov_dimension),
                dtype=complex,
            )
            krylov_size = 0
            for column in range(self.krylov_dimension):
                candidate = self._apply_superoperator(
                    superoperator,
                    basis[column],
                )
                for _pass in range(2):
                    for row, vector in enumerate(basis):
                        projection = vector.overlap(candidate)
                        hessenberg[row, column] += projection
                        candidate = _add_mps(
                            candidate,
                            vector,
                            1,
                            -projection,
                            self.truncation_parameters,
                        )
                next_norm = _mps_norm(candidate)
                hessenberg[column + 1, column] = next_norm
                krylov_size = column + 1
                iterations += 1
                if next_norm <= np.finfo(float).eps * source_norm:
                    break
                if column + 1 < self.krylov_dimension:
                    basis.append(_scale_mps(candidate, 1 / next_norm))

            target = np.zeros(krylov_size + 1, dtype=complex)
            target[0] = beta
            coefficients = np.linalg.lstsq(
                hessenberg[: krylov_size + 1, :krylov_size],
                target,
                rcond=None,
            )[0]
            update = self._linear_combination(
                basis[:krylov_size],
                coefficients,
            )
            solution = (
                update
                if solution is None
                else _add_mps(
                    solution,
                    update,
                    1,
                    1,
                    self.truncation_parameters,
                )
            )

        if solution is None:
            raise RuntimeError("GMRES stopped without constructing a solution.")

        applied = self._apply_superoperator(superoperator, solution)
        residual = _add_mps(
            source,
            applied,
            1,
            -1,
            self.truncation_parameters,
        )
        relative_residual = _mps_norm(residual) / source_norm
        if not residuals or not np.isclose(relative_residual, residuals[-1]):
            residuals.append(relative_residual)
        converged = relative_residual <= self.solver_tolerance
        self.solver_records.append(
            SolverRecord(index, tuple(residuals), iterations, converged)
        )
        message = None
        if not converged:
            message = (
                f"restarted GMRES stopped after {iterations} iterations "
                f"with relative residual {relative_residual:.3e}"
            )
        return SylvesterResult(
            mps_to_mpo(solution, rhs.sites),
            relative_residual,
            converged,
            iterations,
            message,
        )
