"""Implicit perturbation theory with a low-dimensional state subspace.

This module represents only the action of perturbative operators between a
small explicit subspace and its implicit complement.  The complement vectors
may be matrix product states; no basis for the complement is constructed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

from pymablock.block_diagonalization import block_diagonalize

__all__ = [
    "ImplicitBlock",
    "ImplicitStateBackend",
    "StateSolveResult",
    "block_diagonalize_implicit",
]

OperatorType = TypeVar("OperatorType")
StateType = TypeVar("StateType")


class ImplicitStateBackend(Protocol[OperatorType, StateType]):
    """Operations required for implicit perturbation theory.

    Every method must return a new object rather than mutating an input.  State
    addition and operator application may compress their result.
    """

    def apply(self, operator: OperatorType, state: StateType) -> StateType:
        """Apply an operator to a state."""

    def add_states(self, left: StateType, right: StateType) -> StateType:
        """Add two states."""

    def scale_state(self, state: StateType, factor: Number) -> StateType:
        """Scale a state."""

    def inner(self, left: StateType, right: StateType) -> complex:
        """Return ``<left|right>``."""

    def adjoint_operator(self, operator: OperatorType) -> OperatorType:
        """Return the adjoint of an operator."""


@dataclass(frozen=True)
class StateSolveResult(Generic[StateType]):
    """Result and diagnostics of a projected shifted linear solve."""

    state: StateType
    relative_residual: float
    converged: bool
    sweeps: int | None = None
    message: str | None = None


@dataclass(frozen=True, eq=False)
class _ImplicitSpace(Generic[OperatorType, StateType]):
    backend: ImplicitStateBackend[OperatorType, StateType]
    references: tuple[StateType, ...]

    def add(self, left: StateType, right: StateType) -> StateType:
        return self.backend.add_states(left, right)

    def scale(self, state: StateType, factor: Number) -> StateType:
        return self.backend.scale_state(state, factor)

    def inner(self, left: StateType, right: StateType) -> complex:
        return complex(self.backend.inner(left, right))

    def project(self, state: StateType) -> StateType:
        """Project a state onto the complement of the reference subspace."""
        overlaps = tuple(self.inner(reference, state) for reference in self.references)
        norm_squared = float(np.real(self.inner(state, state)))
        projected_norm_squared = norm_squared - sum(
            abs(overlap) ** 2 for overlap in overlaps
        )
        zero_threshold = (
            100 * np.finfo(float).eps * max(norm_squared, np.finfo(float).tiny)
        )
        if projected_norm_squared <= zero_threshold:
            return self.scale(state, 0)
        result = state
        for reference, overlap in zip(self.references, overlaps, strict=True):
            if overlap:
                result = self.add(result, self.scale(reference, -overlap))
        return result

    def norm(self, state: StateType) -> float:
        norm_squared = float(np.real(self.inner(state, state)))
        return float(np.sqrt(max(norm_squared, 0.0)))

    def combine(
        self,
        columns: tuple[StateType, ...],
        coefficients: np.ndarray,
    ) -> tuple[StateType, ...]:
        """Right-multiply a state bundle by a dense matrix."""
        coefficients = np.asarray(coefficients)
        if coefficients.ndim != 2 or coefficients.shape[0] != len(columns):
            raise ValueError("State-bundle coefficient dimensions do not match.")
        result = []
        for output in range(coefficients.shape[1]):
            terms = [
                (column, coefficients[input_, output])
                for input_, column in enumerate(columns)
                if coefficients[input_, output] != 0
            ]
            if not terms:
                result.append(self.scale(columns[0], 0))
                continue
            state = self.scale(terms[0][0], terms[0][1])
            for column, coefficient in terms[1:]:
                state = self.add(state, self.scale(column, coefficient))
            result.append(state)
        return tuple(result)


@dataclass(frozen=True, eq=False)
class _ComplementMap(Generic[OperatorType, StateType]):
    """A lazily composed linear map acting inside the implicit complement."""

    space: _ImplicitSpace[OperatorType, StateType]
    forward: Callable[[StateType], StateType]
    backward: Callable[[StateType], StateType]

    def apply(self, state: StateType) -> StateType:
        return self.forward(state)

    def adjoint(self) -> _ComplementMap[OperatorType, StateType]:
        return type(self)(self.space, self.backward, self.forward)

    def add(self, other: _ComplementMap) -> _ComplementMap:
        self._check_space(other)
        return type(self)(
            self.space,
            lambda state: self.space.add(self.forward(state), other.forward(state)),
            lambda state: self.space.add(self.backward(state), other.backward(state)),
        )

    def scale(self, factor: Number) -> _ComplementMap:
        return type(self)(
            self.space,
            lambda state: self.space.scale(self.forward(state), factor),
            lambda state: self.space.scale(self.backward(state), np.conj(factor)),
        )

    def compose(self, other: _ComplementMap) -> _ComplementMap:
        self._check_space(other)
        return type(self)(
            self.space,
            lambda state: self.forward(other.forward(state)),
            lambda state: other.backward(self.backward(state)),
        )

    def _check_space(self, other: _ComplementMap) -> None:
        if self.space is not other.space:
            raise ValueError("Implicit blocks belong to different state spaces.")


@dataclass(frozen=True, eq=False)
class ImplicitBlock(Generic[OperatorType, StateType]):
    """One block of an operator in an explicit-plus-implicit decomposition.

    ``kind`` is one of ``"dense"``, ``"column"``, ``"row"``, or ``"map"``.
    A column contains complement-space states, one for each explicit basis
    vector.  A row stores the states whose adjoints form that row.
    """

    kind: str
    payload: Any
    space: _ImplicitSpace[OperatorType, StateType]

    __array_priority__ = 1000

    @property
    def dense(self) -> np.ndarray:
        """Return the explicit-subspace matrix."""
        if self.kind != "dense":
            raise TypeError(f"A {self.kind!r} block does not contain a dense matrix.")
        return self.payload

    @property
    def states(self) -> tuple[StateType, ...]:
        """Return the states stored by a row or column block."""
        if self.kind not in {"column", "row"}:
            raise TypeError(f"A {self.kind!r} block does not contain states.")
        return self.payload

    def _check_space(self, other: ImplicitBlock) -> None:
        if self.space is not other.space:
            raise ValueError("Implicit blocks belong to different state spaces.")

    def __add__(self, other: object) -> ImplicitBlock:
        """Add compatible blocks."""
        if not isinstance(other, ImplicitBlock):
            return NotImplemented
        self._check_space(other)
        if self.kind != other.kind:
            raise TypeError(f"Cannot add {self.kind!r} and {other.kind!r} blocks.")
        if self.kind == "dense":
            payload = self.dense + other.dense
        elif self.kind in {"column", "row"}:
            if len(self.states) != len(other.states):
                raise ValueError("State bundles have different sizes.")
            payload = tuple(
                self.space.add(left, right)
                for left, right in zip(self.states, other.states, strict=True)
            )
        else:
            payload = self.payload.add(other.payload)
        return type(self)(self.kind, payload, self.space)

    def __radd__(self, other: object) -> ImplicitBlock:
        """Support the scalar zero used to initialize sums."""
        if isinstance(other, Number) and other == 0:
            return self
        return NotImplemented

    def __neg__(self) -> ImplicitBlock:
        """Negate the block."""
        return self * -1

    def __sub__(self, other: object) -> ImplicitBlock:
        """Subtract a compatible block."""
        if not isinstance(other, ImplicitBlock):
            return NotImplemented
        return self + (-other)

    def __mul__(self, factor: object) -> ImplicitBlock:
        """Scale the block."""
        if not isinstance(factor, Number):
            return NotImplemented
        if self.kind == "dense":
            payload = self.dense * factor
        elif self.kind in {"column", "row"}:
            state_factor = factor if self.kind == "column" else np.conj(factor)
            payload = tuple(
                self.space.scale(state, state_factor) for state in self.states
            )
        else:
            payload = self.payload.scale(factor)
        return type(self)(self.kind, payload, self.space)

    def __rmul__(self, factor: object) -> ImplicitBlock:
        """Scale the block with a scalar on the left."""
        return self * factor

    def __truediv__(self, divisor: object) -> ImplicitBlock:
        """Divide the block by a scalar."""
        if not isinstance(divisor, Number):
            return NotImplemented
        if divisor == 0:
            raise ZeroDivisionError("Cannot divide an implicit block by zero.")
        return self * (1 / divisor)

    def __matmul__(self, other: object) -> ImplicitBlock:
        """Compose two compatible blocks."""
        if not isinstance(other, ImplicitBlock):
            return NotImplemented
        self._check_space(other)
        left, right = self.kind, other.kind

        if (left, right) == ("dense", "dense"):
            return type(self)("dense", self.dense @ other.dense, self.space)
        if (left, right) == ("column", "dense"):
            return type(self)(
                "column",
                self.space.combine(self.states, other.dense),
                self.space,
            )
        if (left, right) == ("dense", "row"):
            columns = self.space.combine(other.states, self.dense.conj().T)
            return type(self)("row", columns, self.space)
        if (left, right) == ("row", "column"):
            matrix = np.array(
                [
                    [self.space.inner(bra, ket) for ket in other.states]
                    for bra in self.states
                ]
            )
            return type(self)("dense", matrix, self.space)
        if (left, right) == ("map", "column"):
            columns = tuple(self.payload.apply(state) for state in other.states)
            return type(self)("column", columns, self.space)
        if (left, right) == ("row", "map"):
            adjoint = other.payload.adjoint()
            columns = tuple(adjoint.apply(state) for state in self.states)
            return type(self)("row", columns, self.space)
        if (left, right) == ("map", "map"):
            return type(self)("map", self.payload.compose(other.payload), self.space)
        if (left, right) == ("column", "row"):
            columns_left = self.states
            columns_right = other.states

            def forward(state):
                coefficients = np.array(
                    [self.space.inner(column, state) for column in columns_right]
                )
                return self.space.combine(columns_left, coefficients.reshape(-1, 1))[0]

            def backward(state):
                coefficients = np.array(
                    [self.space.inner(column, state) for column in columns_left]
                )
                return self.space.combine(columns_right, coefficients.reshape(-1, 1))[0]

            return type(self)(
                "map",
                _ComplementMap(self.space, forward, backward),
                self.space,
            )
        raise TypeError(f"Cannot multiply {left!r} and {right!r} implicit blocks.")

    def adjoint(self) -> ImplicitBlock:
        """Return the adjoint block."""
        if self.kind == "dense":
            return type(self)("dense", self.dense.conj().T, self.space)
        if self.kind == "column":
            return type(self)("row", self.states, self.space)
        if self.kind == "row":
            return type(self)("column", self.states, self.space)
        return type(self)("map", self.payload.adjoint(), self.space)


def _operator_map(
    operator: OperatorType,
    space: _ImplicitSpace[OperatorType, StateType],
) -> _ComplementMap[OperatorType, StateType]:
    adjoint = space.backend.adjoint_operator(operator)
    return _ComplementMap(
        space,
        lambda state: space.project(space.backend.apply(operator, space.project(state))),
        lambda state: space.project(space.backend.apply(adjoint, space.project(state))),
    )


def _operator_blocks(
    operator: OperatorType,
    space: _ImplicitSpace[OperatorType, StateType],
) -> list[list[ImplicitBlock[OperatorType, StateType]]]:
    applied = tuple(space.backend.apply(operator, state) for state in space.references)
    dense = np.array(
        [[space.inner(left, right) for right in applied] for left in space.references]
    )
    columns = tuple(space.project(state) for state in applied)
    adjoint = space.backend.adjoint_operator(operator)
    adjoint_columns = tuple(
        space.project(space.backend.apply(adjoint, state)) for state in space.references
    )
    return [
        [
            ImplicitBlock("dense", dense, space),
            ImplicitBlock("row", adjoint_columns, space),
        ],
        [
            ImplicitBlock("column", columns, space),
            ImplicitBlock("map", _operator_map(operator, space), space),
        ],
    ]


def _validate_references(
    h_0: OperatorType,
    space: _ImplicitSpace[OperatorType, StateType],
    tolerance: float,
) -> np.ndarray:
    overlap = np.array(
        [
            [space.inner(left, right) for right in space.references]
            for left in space.references
        ]
    )
    if not np.allclose(overlap, np.eye(len(space.references)), atol=tolerance):
        raise ValueError("Reference states must be orthonormal.")

    applied = tuple(space.backend.apply(h_0, state) for state in space.references)
    projected = np.array(
        [[space.inner(left, right) for right in applied] for left in space.references]
    )
    if not np.allclose(projected, np.diag(np.diag(projected)), atol=tolerance):
        raise ValueError("Reference states must diagonalize the projected H_0.")

    energies = np.diag(projected)
    for state, energy in zip(space.references, energies, strict=True):
        applied = space.backend.apply(h_0, state)
        residual_norm_squared = (
            np.real(space.inner(applied, applied))
            + abs(energy) ** 2 * np.real(space.inner(state, state))
            - 2 * np.real(energy * space.inner(applied, state))
        )
        residual_norm = np.sqrt(max(float(residual_norm_squared), 0.0))
        scale = max(space.norm(applied), abs(energy), 1.0)
        if residual_norm / scale > tolerance:
            raise ValueError("Reference states must be eigenstates of H_0.")
    return energies


def _check_solve_result(
    result: StateSolveResult[StateType],
    *,
    max_relative_residual: float,
    index: tuple[int, ...],
    column: int,
) -> None:
    if not isinstance(result, StateSolveResult):
        raise TypeError("The backend solver must return a StateSolveResult.")
    residual = float(result.relative_residual)
    detail = f" at perturbative index {index}, column {column}"
    if result.sweeps is not None:
        detail += f" after {result.sweeps} sweeps"
    if result.message:
        detail += f": {result.message}"
    if not result.converged:
        raise RuntimeError(f"The implicit state solver did not converge{detail}")
    if not np.isfinite(residual):
        raise RuntimeError(
            f"The implicit state solver returned a non-finite residual{detail}"
        )
    if residual > max_relative_residual:
        raise RuntimeError(
            "The implicit state solver residual "
            f"{residual:.3e} exceeds {max_relative_residual:.3e}{detail}"
        )


def block_diagonalize_implicit(
    hamiltonian: Sequence[OperatorType] | Mapping[tuple[int, ...], OperatorType],
    reference_states: Sequence[StateType],
    backend: ImplicitStateBackend[OperatorType, StateType],
    solve_shifted: Callable[
        [OperatorType, complex, StateType, tuple[StateType, ...], tuple[int, ...], int],
        StateSolveResult[StateType],
    ],
    *,
    max_relative_residual: float = 1e-8,
    reference_tolerance: float = 1e-8,
):
    """Block diagonalize around a small state subspace without building its complement.

    The returned series contain :class:`ImplicitBlock` coefficients.  Their
    ``(0, 0)`` blocks expose ordinary small matrices through
    :attr:`ImplicitBlock.dense`; their ``(1, 0)`` blocks expose complement-space
    state columns through :attr:`ImplicitBlock.states`.

    ``solve_shifted`` solves

    ``Q (H_0 - energy) Q state = rhs``

    for one right-hand side.  It must enforce orthogonality to every reference
    state and report the true projected relative residual.
    """
    if not reference_states:
        raise ValueError("At least one reference state is required.")
    if not np.isfinite(max_relative_residual) or max_relative_residual < 0:
        raise ValueError("`max_relative_residual` must be finite and non-negative.")
    if not np.isfinite(reference_tolerance) or reference_tolerance < 0:
        raise ValueError("`reference_tolerance` must be finite and non-negative.")

    references = tuple(reference_states)
    space = _ImplicitSpace(backend, references)
    if isinstance(hamiltonian, Mapping):
        zero_order = (0,) * len(next(iter(hamiltonian)))
        try:
            h_0 = hamiltonian[zero_order]
        except KeyError as error:
            raise ValueError(
                "The Hamiltonian mapping must contain a zeroth order."
            ) from error
    else:
        if not hamiltonian:
            raise ValueError("The Hamiltonian must contain H_0.")
        h_0 = hamiltonian[0]

    energies = _validate_references(h_0, space, reference_tolerance)
    h_0_blocks = [
        [
            ImplicitBlock("dense", np.diag(energies), space),
            0,
        ],
        [
            0,
            ImplicitBlock("map", _operator_map(h_0, space), space),
        ],
    ]

    if isinstance(hamiltonian, Mapping):
        zero_order = (0,) * len(next(iter(hamiltonian)))
        projected_hamiltonian = {
            index: (
                h_0_blocks if index == zero_order else _operator_blocks(operator, space)
            )
            for index, operator in hamiltonian.items()
        }
    else:
        projected_hamiltonian = [
            h_0_blocks,
            *(_operator_blocks(operator, space) for operator in hamiltonian[1:]),
        ]

    def solve_sylvester(
        rhs: ImplicitBlock[OperatorType, StateType],
        index: tuple[int, ...],
    ) -> ImplicitBlock[OperatorType, StateType]:
        left_index, right_index = index[:2]
        if (left_index, right_index) == (1, 0):
            source = rhs
            sign = 1
            return_kind = "column"
        elif (left_index, right_index) == (0, 1):
            source = rhs.adjoint()
            sign = -1
            return_kind = "row"
        else:
            raise NotImplementedError(
                "The implicit solver only supports explicit-complement blocks."
            )
        if source.kind != "column":
            raise TypeError(
                "The implicit Sylvester right-hand side must be a state bundle."
            )

        states = []
        for column, (energy, state) in enumerate(
            zip(energies, source.states, strict=True)
        ):
            projected_source = space.project(space.scale(state, sign))
            result = solve_shifted(
                h_0,
                energy,
                projected_source,
                references,
                index,
                column,
            )
            _check_solve_result(
                result,
                max_relative_residual=max_relative_residual,
                index=index,
                column=column,
            )
            states.append(space.project(result.state))
        solved = ImplicitBlock("column", tuple(states), space)
        return solved if return_kind == "column" else solved.adjoint()

    return block_diagonalize(
        projected_hamiltonian,
        solve_sylvester=solve_sylvester,
    )
