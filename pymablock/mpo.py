"""Backend-neutral matrix product operator support."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import numpy as np

from pymablock.series import zero

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "BackendMPO",
    "MPOBackend",
    "SylvesterResult",
    "make_mpo_sylvester_solver",
]

MPOType = TypeVar("MPOType")


class MPOBackend(Protocol[MPOType]):
    """Library-specific MPO operations shared by wrapped operators.

    A backend contains no MPO itself. It implements the algebra for one native
    MPO type and is shared by multiple :class:`BackendMPO` objects. See
    :class:`~pymablock.backends.tenpy.TenpyMPOBackend` for a
    concrete implementation.

    Implementations return new operators instead of modifying either input.
    """

    def add(self, left: MPOType, right: MPOType) -> MPOType:
        """Add two operators."""

    def scale(self, operator: MPOType, factor: Number) -> MPOType:
        """Multiply an operator by a scalar."""

    def matmul(self, left: MPOType, right: MPOType) -> MPOType:
        """Multiply two operators."""

    def adjoint(self, operator: MPOType) -> MPOType:
        """Return the adjoint of an operator."""


@dataclass(frozen=True, eq=False)
class BackendMPO(Generic[MPOType]):
    """One library-native MPO paired with its :class:`MPOBackend`.

    ``operator`` stores the MPO, while the shared ``backend`` implements its
    addition, scaling, multiplication, and adjoint.

    Examples
    --------
    >>> H = BackendMPO(native_mpo, backend)  # doctest: +SKIP
    >>> H2 = H.adjoint() @ H  # doctest: +SKIP

    """

    operator: MPOType
    backend: MPOBackend[MPOType]

    __array_priority__ = 1000

    def _check_backend(self, other: BackendMPO[MPOType]) -> None:
        if self.backend is not other.backend:
            raise ValueError("Cannot combine MPOs that use different backends.")

    def __add__(self, other: object) -> BackendMPO[MPOType]:
        """Add another MPO using the backend."""
        if not isinstance(other, BackendMPO):
            return NotImplemented
        self._check_backend(other)
        return type(self)(
            self.backend.add(self.operator, other.operator),
            self.backend,
        )

    def __radd__(self, other: object) -> BackendMPO[MPOType]:
        """Support the scalar zero used to initialize sums."""
        if isinstance(other, Number) and other == 0:
            return self
        return NotImplemented

    def __sub__(self, other: object) -> BackendMPO[MPOType]:
        """Subtract another MPO using addition and scaling."""
        if not isinstance(other, BackendMPO):
            return NotImplemented
        self._check_backend(other)
        return self + (-other)

    def __neg__(self) -> BackendMPO[MPOType]:
        """Negate the MPO."""
        return type(self)(self.backend.scale(self.operator, -1), self.backend)

    def __mul__(self, factor: object) -> BackendMPO[MPOType]:
        """Scale the MPO."""
        if not isinstance(factor, Number):
            return NotImplemented
        return type(self)(self.backend.scale(self.operator, factor), self.backend)

    def __rmul__(self, factor: object) -> BackendMPO[MPOType]:
        """Scale the MPO with a scalar on the left."""
        return self * factor

    def __truediv__(self, divisor: object) -> BackendMPO[MPOType]:
        """Divide the MPO by a scalar."""
        if not isinstance(divisor, Number):
            return NotImplemented
        if divisor == 0:
            raise ZeroDivisionError("Cannot divide an MPO by zero.")
        return self * (1 / divisor)

    def __matmul__(self, other: object) -> BackendMPO[MPOType]:
        """Multiply two MPOs using the backend."""
        if not isinstance(other, BackendMPO):
            return NotImplemented
        self._check_backend(other)
        return type(self)(
            self.backend.matmul(self.operator, other.operator),
            self.backend,
        )

    def adjoint(self) -> BackendMPO[MPOType]:
        """Return the backend-provided adjoint."""
        return type(self)(self.backend.adjoint(self.operator), self.backend)


@dataclass(frozen=True)
class SylvesterResult(Generic[MPOType]):
    """Result and diagnostics of an approximate Sylvester solve."""

    operator: MPOType
    relative_residual: float
    converged: bool
    iterations: int | None = None
    message: str | None = None


def make_mpo_sylvester_solver(
    diagonal_blocks: Sequence[BackendMPO[MPOType]],
    solve: Callable[
        [MPOType, MPOType, MPOType, tuple[int, ...]],
        SylvesterResult[MPOType],
    ],
    *,
    max_relative_residual: float = 1e-8,
) -> Callable[[BackendMPO[MPOType], tuple[int, ...]], BackendMPO[MPOType]]:
    """Adapt a backend Sylvester solver to Pymablock's solver interface.

    A backend supplies MPO arithmetic for a tensor-network library; ``solve``
    is usually a method of that backend.

    At perturbative order ``n``, the backend solver must solve
    ``left @ V_n - V_n @ right = F_n`` and measure the residual of the
    returned, potentially compressed coefficient ``V_n``. Here ``F_n`` is
    supplied as ``rhs``.

    Examples
    --------
    >>> blocks = [BackendMPO(H_A, backend), BackendMPO(H_B, backend)]  # doctest: +SKIP
    >>> solver = make_mpo_sylvester_solver(  # doctest: +SKIP
    ...     blocks, backend.solve_sylvester
    ... )

    """
    diagonal_blocks = tuple(diagonal_blocks)
    if not diagonal_blocks:
        raise ValueError("At least one diagonal block is required.")
    if not np.isfinite(max_relative_residual) or max_relative_residual < 0:
        raise ValueError("`max_relative_residual` must be finite and non-negative.")

    backend = diagonal_blocks[0].backend
    if any(block.backend is not backend for block in diagonal_blocks):
        raise ValueError("All diagonal blocks must use the same backend.")

    def solve_sylvester(
        rhs: BackendMPO[MPOType],
        index: tuple[int, ...],
    ) -> BackendMPO[MPOType]:
        if len(index) < 2:
            raise ValueError("A Sylvester index must contain two block indices.")
        if rhs is zero:
            return zero
        left_index, right_index = index[:2]
        try:
            left = diagonal_blocks[left_index]
            right = diagonal_blocks[right_index]
        except IndexError as error:
            raise IndexError(
                f"Sylvester block indices {(left_index, right_index)} are out of range."
            ) from error

        if rhs.backend is not backend:
            raise ValueError("The Sylvester right-hand side uses a different backend.")

        result = solve(left.operator, right.operator, rhs.operator, index)
        if not isinstance(result, SylvesterResult):
            raise TypeError("The backend solver must return a SylvesterResult.")

        residual = float(result.relative_residual)
        detail = f" at perturbative index {index}"
        if result.iterations is not None:
            detail += f" after {result.iterations} iterations"
        if result.message:
            detail += f": {result.message}"
        if not result.converged:
            raise RuntimeError(f"The MPO Sylvester solver did not converge{detail}")
        if not np.isfinite(residual):
            raise RuntimeError(
                f"The MPO Sylvester solver returned a non-finite residual{detail}"
            )
        if residual > max_relative_residual:
            raise RuntimeError(
                "The MPO Sylvester solver residual "
                f"{residual:.3e} exceeds {max_relative_residual:.3e}{detail}"
            )

        return BackendMPO(result.operator, backend)

    return solve_sylvester
