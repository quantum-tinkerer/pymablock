"""Executable spike for occupation-map encodings.

The spike keeps the Hamiltonian in Pymablock's existing SymPy second-quantized input
format.  It compiles each number-ordered term to state action, constructs only the
source states that may occur on a returning path through the requested order, and
lowers the embedding to Pymablock's existing implicit-subspace recurrence.

The central restriction is deliberate: an encoding maps each target occupation state
to one source occupation state.  This covers the tunable-coupler and Hubbard-plaquette
problems without an explicit table of retained states.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import sympy
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize as _pymablock_block_diagonalize
from pymablock.block_diagonalization import solve_sylvester_diagonal
from pymablock.number_ordered_form import (
    LadderOp,
    NumberOrderedForm,
    generator_types,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

SourceState = tuple[int, ...]
TargetState = tuple[int, ...]


class _CoordinateArithmetic:
    """Arithmetic shared by target coordinates and coordinate expressions."""

    def _binary(self, other, operation: Callable, symbol: str):
        other = _as_coordinate_expression(other)
        return CoordinateExpression(
            lambda values: operation(self.evaluate(values), other.evaluate(values)),
            _unique((*self.coordinates, *other.coordinates)),
            f"({self!s} {symbol} {other!s})",
        )

    def __add__(self, other):
        return self._binary(other, lambda left, right: left + right, "+")

    def __radd__(self, other):
        return _as_coordinate_expression(other).__add__(self)

    def __sub__(self, other):
        return self._binary(other, lambda left, right: left - right, "-")

    def __rsub__(self, other):
        return _as_coordinate_expression(other).__sub__(self)

    def __mul__(self, other):
        return self._binary(other, lambda left, right: left * right, "*")

    def __rmul__(self, other):
        return _as_coordinate_expression(other).__mul__(self)

    def __neg__(self):
        return CoordinateExpression(
            lambda values: -self.evaluate(values),
            self.coordinates,
            f"-{self!s}",
        )


@dataclass(frozen=True)
class Coordinate(_CoordinateArithmetic):
    """One finite target-space coordinate."""

    name: str
    values: tuple[int, ...]

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        """Return this coordinate as a one-item dependency tuple."""
        return (self,)

    def evaluate(self, values: Mapping[Coordinate, int]) -> int:
        """Read this coordinate from a target-state mapping."""
        return values[self]

    def __str__(self) -> str:
        """Return the coordinate name."""
        return self.name


@dataclass(frozen=True)
class CoordinateExpression(_CoordinateArithmetic):
    """A callable integer expression in target coordinates."""

    function: Callable[[Mapping[Coordinate, int]], int]
    coordinates: tuple[Coordinate, ...]
    text: str

    def evaluate(self, values: Mapping[Coordinate, int]) -> int:
        """Evaluate the integer expression on a target-state mapping."""
        return int(self.function(values))

    def __str__(self) -> str:
        """Return a readable representation of the expression."""
        return self.text


def _as_coordinate_expression(value: int | _CoordinateArithmetic):
    if isinstance(value, _CoordinateArithmetic):
        return value
    integer = int(value)
    return CoordinateExpression(lambda _values: integer, (), str(integer))


def _unique(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def levels(names: str | Sequence[str], size: int):
    """Create finite target coordinates, following ``sympy.symbols`` style."""
    if size <= 0:
        raise ValueError("size must be positive")
    symbols = sympy.symbols(names)
    if isinstance(symbols, sympy.Symbol):
        return Coordinate(str(symbols), tuple(range(size)))
    return tuple(Coordinate(str(symbol), tuple(range(size))) for symbol in symbols)


def _operator_sort_key(operator) -> tuple[int, str]:
    return generator_types.index(type(operator)), str(operator.name)


@dataclass(frozen=True)
class TargetSpace:
    """Finite product target space inferred from an occupation map."""

    coordinates: tuple[Coordinate, ...]

    @cached_property
    def states(self) -> tuple[TargetState, ...]:
        """Enumerate target states in product order."""
        return tuple(product(*(coordinate.values for coordinate in self.coordinates)))

    @cached_property
    def index(self) -> dict[TargetState, int]:
        """Map each target state to its matrix index."""
        return {state: index for index, state in enumerate(self.states)}

    @property
    def dimension(self) -> int:
        """Return the target Hilbert-space dimension."""
        return len(self.states)

    def state(self, **values: int) -> TargetState:
        """Construct a target state by coordinate name."""
        unknown = set(values) - {coordinate.name for coordinate in self.coordinates}
        if unknown:
            raise KeyError(f"Unknown target coordinates: {sorted(unknown)}")
        return tuple(values[coordinate.name] for coordinate in self.coordinates)

    def pauli(self, coordinate: Coordinate | int, axis: str) -> TargetOperator:
        """Return a Pauli operator acting on one binary target coordinate."""
        coordinate_index = coordinate
        if not isinstance(coordinate, int):
            coordinate_index = self.coordinates.index(coordinate)
        if self.coordinates[coordinate_index].values != (0, 1):
            raise ValueError("Pauli operators require a binary coordinate")
        axis = axis.upper()
        if axis not in _PAULI:
            raise ValueError(f"Unknown Pauli axis: {axis}")
        factors = [sympy.eye(len(item.values)) for item in self.coordinates]
        factors[coordinate_index] = _PAULI[axis]
        return TargetOperator(self, sympy.kronecker_product(*factors))


@dataclass(frozen=True)
class FermionSpace:
    """A target Fock space generated by named fermionic modes."""

    modes: tuple[FermionOp, ...]
    numbers: frozenset[int] | None = None

    @cached_property
    def states(self) -> tuple[TargetState, ...]:
        """Enumerate occupation states in the selected number sectors."""
        states = product((0, 1), repeat=len(self.modes))
        if self.numbers is None:
            return tuple(states)
        return tuple(state for state in states if sum(state) in self.numbers)

    @cached_property
    def index(self) -> dict[TargetState, int]:
        """Map each target Fock state to its matrix index."""
        return {state: index for index, state in enumerate(self.states)}

    @property
    def dimension(self) -> int:
        """Return the selected target-space dimension."""
        return len(self.states)

    def state(self, occupied: Iterable[FermionOp] = ()) -> TargetState:
        """Construct a target state from its occupied generators."""
        occupied = set(occupied)
        unknown = occupied - set(self.modes)
        if unknown:
            raise KeyError(f"Unknown target fermions: {sorted(map(str, unknown))}")
        state = tuple(int(mode in occupied) for mode in self.modes)
        if state not in self.index:
            raise ValueError("The state lies outside the selected number sectors")
        return state

    def operator(self, expression: sympy.Expr) -> FermionOperator:
        """Represent a second-quantized expression on this target Fock space."""
        compiled = CompiledOperator(sympy.sympify(expression), self.modes)
        matrix = sympy.MutableSparseMatrix(self.dimension, self.dimension, {})
        for column, state in enumerate(self.states):
            for target, coefficient in compiled.apply(state).items():
                row = self.index.get(target)
                if row is not None:
                    matrix[row, column] += coefficient
        return FermionOperator(self, sympy.ImmutableSparseMatrix(matrix))


class OccupationEncoding:
    """An isometric occupation map from a finite target to source Fock states."""

    def __init__(self, occupations: Mapping[Any, int | _CoordinateArithmetic]):
        """Compile and validate a target-to-source occupation map."""
        if not occupations:
            raise ValueError("An occupation map may not be empty")
        self.operators = tuple(sorted(occupations, key=_operator_sort_key))
        self._expressions = {
            operator: _as_coordinate_expression(value)
            for operator, value in occupations.items()
        }
        coordinates = _unique(
            coordinate
            for value in occupations.values()
            for coordinate in _as_coordinate_expression(value).coordinates
        )
        self.target = TargetSpace(coordinates)

    @cached_property
    def _source_to_target(self) -> dict[SourceState, TargetState]:
        """Construct the inverse table only for the finite-state reference backend."""
        source_to_target: dict[SourceState, TargetState] = {}
        for target_state in self.target.states:
            source_state = self.encode(target_state)
            if source_state in source_to_target:
                other = source_to_target[source_state]
                raise ValueError(
                    "The occupation map is not injective: "
                    f"{other} and {target_state} have the same image"
                )
            source_to_target[source_state] = target_state
        return source_to_target

    def encode(self, state: TargetState) -> SourceState:
        """Apply W to a target occupation state."""
        state = tuple(state)
        if len(state) != len(self.target.coordinates):
            raise ValueError("Target state has the wrong number of coordinates")
        values = dict(zip(self.target.coordinates, state, strict=True))
        if any(
            value not in coordinate.values
            for coordinate, value in zip(self.target.coordinates, state, strict=True)
        ):
            raise ValueError(f"State {state} lies outside the target basis")

        occupations = tuple(
            self._expressions[operator].evaluate(values) for operator in self.operators
        )
        for operator, occupation in zip(self.operators, occupations, strict=True):
            if isinstance(operator, (BosonOp, FermionOp, SigmaMinus)) and occupation < 0:
                raise ValueError(f"Negative occupation {occupation} for {operator}")
            if isinstance(operator, (FermionOp, SigmaMinus)) and occupation not in (0, 1):
                raise ValueError(
                    f"Binary operator {operator} has occupation {occupation}"
                )
        return occupations

    def decode(self, state: SourceState) -> TargetState | None:
        """Apply W† to a source basis state, returning ``None`` outside the image."""
        return self._source_to_target.get(tuple(state))


def occupation_map(
    occupations: Mapping[Any, int | _CoordinateArithmetic],
) -> OccupationEncoding:
    """Construct an :class:`OccupationEncoding`."""
    return OccupationEncoding(occupations)


class FermionEmbedding:
    """An embedding from target fermionic generators into source modes.

    Each source mode is either fixed empty/occupied or identified with one target
    fermion. Occupation tuples are compiled internally; the public target remains a
    fermionic Fock algebra.
    """

    def __init__(
        self,
        modes: Mapping[FermionOp, FermionOp | int],
        *,
        number: int | Iterable[int] | None = None,
    ):
        """Compile a source-to-target generator map and optional number sector."""
        if not modes:
            raise ValueError("A fermion embedding may not be empty")
        if not all(isinstance(mode, FermionOp) for mode in modes):
            raise TypeError("All source generators must be fermions")
        if not all(
            isinstance(value, FermionOp) or value in (0, 1) for value in modes.values()
        ):
            raise TypeError("Source fermions map to a target fermion, zero, or one")

        self.operators = tuple(sorted(modes, key=_operator_sort_key))
        self._modes = dict(modes)
        target_modes = tuple(
            self._modes[source]
            for source in self.operators
            if isinstance(self._modes[source], FermionOp)
        )
        if len(target_modes) != len(set(target_modes)):
            raise ValueError("Each target fermion must map to one source mode")
        if number is None:
            numbers = None
        elif isinstance(number, int):
            numbers = frozenset((number,))
        else:
            numbers = frozenset(number)
        if numbers is not None and any(
            value < 0 or value > len(target_modes) for value in numbers
        ):
            raise ValueError("A target particle number lies outside the Fock space")
        self.target = FermionSpace(target_modes, numbers)
        self._target_index = {mode: index for index, mode in enumerate(self.target.modes)}
        fixed_before = {}
        occupied_fixed = 0
        for source in self.operators:
            target = self._modes[source]
            if isinstance(target, FermionOp):
                fixed_before[target] = occupied_fixed
            elif target == 1:
                occupied_fixed += 1
        self._fixed_before = fixed_before

    @cached_property
    def _source_to_target(self) -> dict[SourceState, TargetState]:
        """Compile the finite inverse only when a state-based backend requests it."""
        return {self.encode(state): state for state in self.target.states}

    @cached_property
    def phases(self) -> tuple[sympy.Integer, ...]:
        """Return finite embedding phases only when materialization requests them."""
        return tuple(
            sympy.S.NegativeOne
            if sum(
                occupation * self._fixed_before[mode]
                for mode, occupation in zip(self.target.modes, state, strict=True)
            )
            % 2
            else sympy.S.One
            for state in self.target.states
        )

    def encode(self, state: TargetState) -> SourceState:
        """Apply the Fock embedding to a target occupation state."""
        state = tuple(state)
        if state not in self.target.index:
            raise ValueError("The state lies outside the target Fock space")
        return tuple(
            (
                state[self._target_index[value]]
                if isinstance(value, FermionOp)
                else int(value)
            )
            for source in self.operators
            for value in (self._modes[source],)
        )

    def decode(self, state: SourceState) -> TargetState | None:
        """Apply the adjoint embedding, returning ``None`` outside its image."""
        return self._source_to_target.get(tuple(state))

    def project(self, expression: sympy.Expr) -> FermionOperator:
        """Project a source expression into the target fermion algebra."""
        compiled = CompiledOperator(sympy.sympify(expression), self.operators)
        matrix = sympy.MutableSparseMatrix(
            self.target.dimension, self.target.dimension, {}
        )
        for column, state in enumerate(self.target.states):
            source = self.encode(state)
            for source_target, coefficient in compiled.apply(source).items():
                target = self.decode(source_target)
                if target is None:
                    continue
                row = self.target.index[target]
                matrix[row, column] += (
                    self.phases[row] * self.phases[column] * coefficient
                )
        return FermionOperator(self.target, sympy.ImmutableSparseMatrix(matrix))


def fermion_embedding(
    modes: Mapping[FermionOp, FermionOp | int],
    *,
    number: int | Iterable[int] | None = None,
) -> FermionEmbedding:
    """Define a direct source-fermion to target-fermion embedding."""
    return FermionEmbedding(modes, number=number)


class CompiledOperator:
    """Number-ordered operator compiled to direct occupation-state action."""

    def __init__(self, expression: sympy.Expr, operators: Sequence[Any]):
        """Convert an expression to a number-ordered state-action form."""
        self.form = NumberOrderedForm.from_expr(expression, operators=operators)
        self.operators = tuple(operators)
        self._fermion_indices = tuple(
            index
            for index, operator in enumerate(self.operators)
            if isinstance(operator, FermionOp)
        )

    def apply(self, source: SourceState) -> dict[SourceState, sympy.Expr]:
        """Apply the operator to one source occupation state."""
        result: defaultdict[SourceState, sympy.Expr] = defaultdict(lambda: sympy.S.Zero)
        for powers, coefficient in self.form.terms.items():
            state = list(source)
            amplitude: sympy.Expr = sympy.S.One

            # The normal form is creators · coefficient(N) · annihilators.
            # Acting on a ket therefore applies annihilators in ascending mode order.
            valid = True
            for index, power in enumerate(powers):
                for _ in range(max(int(power), 0)):
                    action = self._apply_generator(state, index, annihilate=True)
                    if action is None:
                        valid = False
                        break
                    factor, state = action
                    amplitude *= factor
                if not valid:
                    break
            if not valid:
                continue

            substitutions = {
                placeholder: sympy.Integer(occupation)
                for placeholder, occupation in zip(
                    self.form._number_operator_placeholders, state, strict=True
                )
            }
            amplitude *= coefficient.xreplace(substitutions)
            if amplitude == 0:
                continue

            # Creators are on the left and therefore act in descending mode order.
            for index in reversed(range(len(powers))):
                for _ in range(max(-int(powers[index]), 0)):
                    action = self._apply_generator(state, index, annihilate=False)
                    if action is None:
                        valid = False
                        break
                    factor, state = action
                    amplitude *= factor
                if not valid:
                    break
            if not valid or amplitude == 0:
                continue

            target = tuple(state)
            result[target] += amplitude
            if result[target] == 0:
                del result[target]
        return dict(result)

    def diagonal(self, source: SourceState) -> sympy.Expr:
        """Return a diagonal matrix element, rejecting a non-diagonal operator."""
        action = self.apply(source)
        offdiagonal = set(action) - {tuple(source)}
        if offdiagonal:
            raise ValueError(
                "The unperturbed Hamiltonian must be diagonal in source states"
            )
        return action.get(tuple(source), sympy.S.Zero)

    def _apply_generator(
        self, state: list[int], index: int, *, annihilate: bool
    ) -> tuple[sympy.Expr, list[int]] | None:
        operator = self.operators[index]
        occupation = state[index]
        target = list(state)

        if isinstance(operator, BosonOp):
            if annihilate:
                if occupation == 0:
                    return None
                target[index] -= 1
                return sympy.sqrt(occupation), target
            target[index] += 1
            return sympy.sqrt(occupation + 1), target

        if isinstance(operator, LadderOp):
            target[index] += -1 if annihilate else 1
            return sympy.S.One, target

        if isinstance(operator, SigmaMinus):
            if annihilate:
                if occupation == 0:
                    return None
                target[index] = 0
            else:
                if occupation == 1:
                    return None
                target[index] = 1
            return sympy.S.One, target

        if isinstance(operator, FermionOp):
            if annihilate:
                if occupation == 0:
                    return None
                target[index] = 0
            else:
                if occupation == 1:
                    return None
                target[index] = 1
            parity = sum(state[i] for i in self._fermion_indices if i < index) % 2
            return (-sympy.S.One if parity else sympy.S.One), target

        raise TypeError(f"Unsupported source operator: {operator!r}")


@dataclass(frozen=True)
class CompilationInfo:
    """State-graph and timing information exposed by the spike."""

    retained_states: int
    virtual_states: int
    transition_edges: int
    closure_seconds: float
    matrix_seconds: float

    @property
    def total_states(self) -> int:
        """Return the total compiled source-space dimension."""
        return self.retained_states + self.virtual_states


_PAULI = {
    "I": sympy.eye(2),
    "X": sympy.Matrix([[0, 1], [1, 0]]),
    "Y": sympy.Matrix([[0, -sympy.I], [sympy.I, 0]]),
    "Z": sympy.diag(1, -1),
}


@dataclass(frozen=True)
class TargetOperator:
    """A finite operator whose rows and columns are labeled by target states."""

    target: TargetSpace
    matrix: sympy.MatrixBase

    def matrix_element(self, bra: TargetState, ket: TargetState) -> sympy.Expr:
        """Return a target-basis matrix element."""
        return self.matrix[self.target.index[tuple(bra)], self.target.index[tuple(ket)]]

    def pauli(self) -> PauliPolynomial:
        """Expand an all-binary target operator in Pauli strings."""
        if any(coordinate.values != (0, 1) for coordinate in self.target.coordinates):
            raise ValueError("Pauli expansion requires binary target coordinates")
        dimension = self.target.dimension
        terms: dict[str, sympy.Expr] = {}
        for axes in product("IXYZ", repeat=len(self.target.coordinates)):
            pauli_matrix = sympy.kronecker_product(*(_PAULI[axis] for axis in axes))
            coefficient = (
                sympy.Add(
                    *(
                        pauli_matrix[row, column] * self.matrix[column, row]
                        for row in range(dimension)
                        for column in range(dimension)
                        if pauli_matrix[row, column] != 0
                        and self.matrix[column, row] != 0
                    )
                )
                / dimension
            )
            if coefficient != 0:
                terms["".join(axes)] = coefficient
        return PauliPolynomial(self.target, terms)


@dataclass(frozen=True)
class FermionOperator:
    """An operator acting on a target Fock space with named generators."""

    target: FermionSpace
    matrix: sympy.MatrixBase

    @property
    def generators(self) -> tuple[FermionOp, ...]:
        """Return the fermionic generators of the target algebra."""
        return self.target.modes

    def matrix_element(self, bra: TargetState, ket: TargetState) -> sympy.Expr:
        """Return a target-Fock-basis matrix element."""
        return self.matrix[self.target.index[tuple(bra)], self.target.index[tuple(ket)]]


@dataclass(frozen=True)
class PauliPolynomial:
    """Sparse Pauli-string coordinates of a finite target operator."""

    target: TargetSpace
    terms: Mapping[str, sympy.Expr]

    def coefficient(self, pauli_string: str) -> sympy.Expr:
        """Return one Pauli-string coefficient, or zero when absent."""
        return self.terms.get(pauli_string.upper(), sympy.S.Zero)

    def as_matrix(self) -> sympy.MatrixBase:
        """Reconstruct the target-space matrix."""
        result = sympy.zeros(self.target.dimension)
        for axes, coefficient in self.terms.items():
            result += coefficient * sympy.kronecker_product(
                *(_PAULI[axis] for axis in axes)
            )
        return sympy.ImmutableMatrix(result)


class EffectiveSeries:
    """Target-labeled view of Pymablock's retained effective block."""

    def __init__(
        self,
        series,
        target: TargetSpace | FermionSpace,
        to_order: int,
        info: CompilationInfo,
    ):
        """Wrap the retained block of an existing Pymablock series."""
        self._series = series
        self.target = target
        self.to_order = to_order
        self.info = info

    def __getitem__(self, order: int) -> TargetOperator | FermionOperator:
        """Return one perturbative coefficient as a target operator."""
        if not isinstance(order, int):
            raise TypeError("The spike currently accepts one integer perturbative order")
        if not 0 <= order <= self.to_order:
            raise IndexError(f"Order {order} lies outside 0..{self.to_order}")
        coefficient = self._series[0, 0, order]
        if isinstance(coefficient, np.ndarray):
            coefficient = sympy.ImmutableMatrix(coefficient.tolist())
        elif not isinstance(coefficient, sympy.MatrixBase):
            coefficient = sympy.ImmutableSparseMatrix(
                self.target.dimension, self.target.dimension, {}
            )
        operator_type = (
            FermionOperator if isinstance(self.target, FermionSpace) else TargetOperator
        )
        return operator_type(self.target, coefficient)


def _reachable_closure(
    retained: Sequence[SourceState], perturbation: CompiledOperator, radius: int
) -> tuple[list[SourceState], int]:
    distance = {state: 0 for state in retained}
    queue = deque(retained)
    edges = 0
    while queue:
        state = queue.popleft()
        if distance[state] == radius:
            continue
        for target in perturbation.apply(state):
            edges += 1
            if target not in distance:
                distance[target] = distance[state] + 1
                queue.append(target)
    virtual = sorted(set(distance) - set(retained))
    return [*retained, *virtual], edges


def block_diagonalize(
    hamiltonian: Sequence[sympy.Expr],
    *,
    encoding: OccupationEncoding | FermionEmbedding,
    to_order: int,
):
    """Block-diagonalize a first-order encoded second-quantized problem.

    Parameters mirror the intended extension of :func:`pymablock.block_diagonalize`.
    The spike supports ``[H0, V]`` with a diagonal ``H0`` and a Hermitian first-order
    perturbation.  Its source closure contains every state with graph distance at most
    ``to_order // 2`` from the retained image.  Such states are sufficient for every
    returning path of length ``to_order``.
    """
    if len(hamiltonian) != 2:
        raise NotImplementedError("The spike currently supports [H0, V]")
    if to_order < 0:
        raise ValueError("to_order must be non-negative")

    H0 = CompiledOperator(sympy.sympify(hamiltonian[0]), encoding.operators)
    V = CompiledOperator(sympy.sympify(hamiltonian[1]), encoding.operators)
    retained = [encoding.encode(state) for state in encoding.target.states]

    start = perf_counter()
    source_states, _explored_edges = _reachable_closure(retained, V, to_order // 2)
    closure_seconds = perf_counter() - start

    start = perf_counter()
    state_index = {state: index for index, state in enumerate(source_states)}
    dimension = len(source_states)
    energies = np.empty(dimension, dtype=object)
    perturbation_matrix = np.zeros((dimension, dimension), dtype=object)
    transition_edges = 0
    for column, state in enumerate(source_states):
        energies[column] = H0.diagonal(state)
        for target, coefficient in V.apply(state).items():
            row = state_index.get(target)
            if row is None:
                continue
            perturbation_matrix[row, column] += coefficient
            transition_edges += 1
    matrix_seconds = perf_counter() - start

    retained_dimension = len(retained)
    # W maps the target basis into the retained source states. Fermionic embeddings
    # include the graded Fock-space phase directly in W, so target readout is simply
    # the retained block returned by Pymablock.
    embedding_matrix = np.zeros((dimension, retained_dimension), dtype=int)
    phases = getattr(encoding, "phases", (sympy.S.One,) * retained_dimension)
    for column, phase in enumerate(phases):
        embedding_matrix[column, column] = int(phase)

    # H0 is diagonal on the compiled source basis. Supplying all virtual basis vectors
    # to the standard diagonal solver gives an exact implicit resolvent while the block
    # recurrence itself only knows P through W and represents Q as its complement.
    virtual_eigenvectors = np.eye(dimension, dtype=int)[:, retained_dimension:]
    solve_sylvester = solve_sylvester_diagonal(
        (energies[:retained_dimension], energies[retained_dimension:]),
        vecs_implicit=virtual_eigenvectors,
    )
    series = _pymablock_block_diagonalize(
        [np.diag(energies), perturbation_matrix],
        subspace_eigenvectors=(embedding_matrix,),
        solve_sylvester=solve_sylvester,
    )
    info = CompilationInfo(
        retained_states=retained_dimension,
        virtual_states=dimension - retained_dimension,
        transition_edges=transition_edges,
        closure_seconds=closure_seconds,
        matrix_seconds=matrix_seconds,
    )
    return (
        EffectiveSeries(
            series[0],
            encoding.target,
            to_order,
            info,
        ),
        *series[1:],
    )
