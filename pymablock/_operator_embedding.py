"""Block diagonalization using ordinary operators and fixed projectors.

P = W W† selects the retained representation and Q = 1 - P its complement.
The retained block is expressed in the target algebra. Other blocks are source
operators supported on P or Q; no source basis truncation is introduced.
"""

from dataclasses import dataclass
from functools import cache
from itertools import product

import numpy as np
import sympy
from sympy.physics.quantum.boson import BosonOp

from pymablock.algorithm_parsing import series_computation
from pymablock.algorithms import main
from pymablock.number_ordered_form import (
    LadderOp,
    NumberOrderedForm,
    _NOFTransition,
    _occupation_dimension,
)
from pymablock.operator_embedding import Embedding, _ReferenceBasis
from pymablock.series import BlockSeries, one, zero


def _occupation_projector(left, right):
    return sympy.Piecewise((1, sympy.Eq(left, right)), (0, True))


def _projectors(expression):
    return (
        p
        for p in expression.atoms(sympy.Piecewise)
        if len(p.args) == 2
        and p.args[0].expr == 1
        and p.args[1] == (0, sympy.true)
        and isinstance(p.args[0].cond, sympy.Equality)
    )


def _adjoint(value):
    """Return an algebra-native adjoint."""
    result = value.adjoint()
    if isinstance(value, NumberOrderedForm) and result.operators != value.operators:
        # SymPy may cache an equal expression with a different list of unused
        # generators. Restore the declared basis using the public conversion.
        return NumberOrderedForm.from_expr(result.as_expr(), operators=value.operators)
    return result


def _projector(basis, shape):
    """Compile the occupation indicator of the retained subspace."""
    numbers = basis._source_placeholders

    def diagonal(expression):
        return NumberOrderedForm(basis.operators, {(0,) * len(numbers): expression}) * 1

    def point(state):
        return sympy.prod(
            _occupation_projector(n, value) for n, value in zip(numbers, state)
        )

    if isinstance(basis, _ReferenceBasis):
        entries = {}
        for component, state in basis._references:
            entries[component, component] = entries.get(
                (component, component), 0
            ) + point(state)
        return sympy.ImmutableSparseMatrix(
            *shape, {key: diagonal(value) for key, value in entries.items()}
        )
    matrix = basis._occupation_matrix
    occupations = sympy.Matrix(numbers) - sympy.Matrix(basis.reference)
    target = basis._occupation_left_inverse * occupations
    indicator = sympy.prod(
        _occupation_projector(sympy.expand(value), 0)
        for value in occupations - matrix * target
    )
    for q, size, op in zip(target, basis._target_dimensions, basis._target_operators):
        if size is not None:
            indicator *= sum(_occupation_projector(q, i) for i in range(size))
        else:
            indicator *= _occupation_projector(q, sympy.floor(q))
            # Prove the target domain using the physical source occupations.
            physical = {
                n: sympy.Dummy(integer=True, nonnegative=True)
                for source, n in zip(basis.operators, numbers)
                if not isinstance(source, LadderOp)
            }
            if (
                isinstance(op, BosonOp)
                and q.xreplace(physical).is_nonnegative is not True
            ):
                raise NotImplementedError(
                    "This bosonic embedding requires an occupation inequality"
                )
    return diagonal(indicator)


def _lift(value, basis, p, shape):
    """Embed a retained operator using its normal-ordered monomials."""
    numbers = basis._source_placeholders
    if isinstance(basis, _ReferenceBasis):
        entries = {}
        for (i, j), amplitude in value.todok().items():
            row, final = basis._references[i]
            col, initial = basis._references[j]
            powers = tuple(a - b for a, b in zip(initial, final))
            monomial = NumberOrderedForm(basis.operators, {powers: sympy.S.One})
            (transition,) = _NOFTransition.from_form(monomial)
            weight = transition.apply(initial).weight
            mask = sympy.prod(
                _occupation_projector(n, a) for n, a in zip(numbers, initial)
            )
            column = NumberOrderedForm(basis.operators, {(0,) * len(numbers): mask})
            key = row, col
            entries[key] = entries.get(key, 0) + monomial * column * (amplitude / weight)
        return sympy.ImmutableSparseMatrix(*shape, entries)
    coordinates = basis._occupation_left_inverse * (
        sympy.Matrix(numbers) - sympy.Matrix(basis.reference)
    )
    substitutions = dict(zip(basis._target_placeholders, coordinates))
    result = NumberOrderedForm(basis.operators, {}, validate=False)
    for powers, coefficient in value.terms.items():
        term = NumberOrderedForm(
            basis.operators, {(0,) * len(numbers): coefficient.xreplace(substitutions)}
        )
        for generator, power in reversed(tuple(zip(basis._generators, powers))):
            if power > 0:
                term = term * generator.form**power
        for generator, power in reversed(tuple(zip(basis._generators, powers))):
            if power < 0:
                term = generator.form.adjoint() ** (-power) * term
        result += term
    return p * result * p


@dataclass(frozen=True)
class _Block:
    """A block index and its ordinary source or retained operator."""

    index: tuple[int, int]
    value: NumberOrderedForm | sympy.MatrixBase

    def __add__(self, other):
        return self if other is zero else _Block(self.index, self.value + other.value)

    def __neg__(self):
        return _Block(self.index, -self.value)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, divisor):
        return _Block(self.index, self.value / divisor)

    def adjoint(self):
        return _Block(self.index[::-1], _adjoint(self.value))


def block_diagonalize(
    hamiltonian: BlockSeries, embedding: Embedding
) -> tuple[BlockSeries, BlockSeries, BlockSeries]:
    """Run the standard recurrence, compressing only the retained block."""
    basis = embedding._basis
    if hamiltonian.shape:
        raise ValueError("Structured embeddings require an unseparated Hamiltonian.")
    origin = (0,) * hamiltonian.n_infinite
    h0 = basis._source_form(hamiltonian[origin])
    shape = h0.shape if isinstance(h0, sympy.MatrixBase) else None
    numbers = basis._source_placeholders
    dimensions = tuple(map(_occupation_dimension, basis.operators))
    unit = basis._source_form(sympy.eye(shape[0]) if shape else sympy.S.One)
    p = _projector(basis, shape)
    blocks = (p, unit - p)

    @cache
    def on_support(coefficient):
        """Reduce point projectors without expanding unrelated coefficient factors."""
        replacements, points = {}, {}
        for delta in _projectors(coefficient):
            variables = set(numbers) & delta.free_symbols
            if len(variables) != 1:
                continue
            (n,) = variables
            equation = sympy.expand(delta.args[0].cond.lhs - delta.args[0].cond.rhs)
            slope = equation.coeff(n)
            if not slope.is_number or not slope:
                continue
            value = sympy.cancel(n - equation / slope)
            if not value.is_number:
                continue
            if value.is_integer is False or (
                not isinstance(basis.operators[numbers.index(n)], LadderOp) and value < 0
            ):
                replacements[delta] = sympy.S.Zero
            else:
                replacements[delta] = _occupation_projector(n, value)
                points.setdefault(n, set()).add(value)
        coefficient = coefficient.xreplace(replacements)
        for n, values in sorted(
            points.items(), key=lambda item: sympy.default_sort_key(item[0])
        ):
            background = coefficient.xreplace(
                {_occupation_projector(n, v): sympy.S.Zero for v in values}
            )
            coefficient = background + sum(
                _occupation_projector(n, v)
                * (coefficient.xreplace({n: v}) - background.xreplace({n: v}))
                for v in sorted(values, key=sympy.default_sort_key)
            )
        return coefficient

    def clean_scalar(value):
        if value == 0:
            return NumberOrderedForm(basis.operators, {}, validate=False)
        return value.applyfunc(on_support)

    def clean(value):
        result = value.applyfunc(clean_scalar) if shape else clean_scalar(value)
        empty = (
            all(not any(x.terms.values()) for x in result.todok().values())
            if shape
            else not any(result.terms.values())
        )
        return zero if empty else result

    @cache
    def lifted(value):
        return _lift(value, basis, p, shape)

    def wrap(index, value):
        index = index[:2]
        if value is zero:
            return zero
        return _Block(index, basis._pullback(value) if index == (0, 0) else value)

    def multiply(a, b):
        index = (a.index[0], b.index[1])
        if a.index == b.index == (0, 0):
            return _Block(index, a.value * b.value)
        left = lifted(a.value) if a.index == (0, 0) else a.value
        right = lifted(b.value) if b.index == (0, 0) else b.value
        return wrap(index, clean(left * right))

    energies = []
    for i in range(shape[0] if shape else 1):
        entry = h0[i, i] if shape else h0
        if entry == 0:
            energies.append(sympy.S.Zero)
            continue
        if any(any(powers) and c != 0 for powers, c in entry.terms.items()):
            raise ValueError("Structured embeddings currently require diagonal H0")
        energies.append(sympy.expand(entry.terms.get((0,) * len(numbers), sympy.S.Zero)))
    if shape and any(i != j and x != 0 for (i, j), x in h0.todok().items()):
        raise ValueError("Structured embeddings currently require diagonal H0")

    def divide_scalar(value, row, col):
        terms = {}
        for powers, coefficient in value.terms.items():
            outgoing = {n: n + max(-int(power), 0) for n, power in zip(numbers, powers)}
            incoming = {n: n + max(int(power), 0) for n, power in zip(numbers, powers)}
            denominator = sympy.expand(
                energies[row].xreplace(outgoing) - energies[col].xreplace(incoming)
            )
            pinned = {
                n: sympy.S.Zero
                for n, power, size in zip(numbers, powers, dimensions)
                if power and size == 2
            }
            coefficient, denominator = (
                x.xreplace(pinned) for x in (coefficient, denominator)
            )
            binary = [
                n
                for n, size in zip(numbers, dimensions)
                if size == 2 and n in denominator.free_symbols
            ]
            result = sympy.S.Zero
            for values in product((0, 1), repeat=len(binary)):
                substitutions = dict(zip(binary, values))
                c = on_support(coefficient.xreplace(substitutions))
                d = denominator.xreplace(substitutions)
                if c == 0:
                    continue
                mask = sympy.prod(n if v else 1 - n for n, v in zip(binary, values))
                for term in sympy.Add.make_args(
                    sympy.expand(c) if c.has(sympy.Piecewise) else c
                ):
                    local = d
                    for delta in _projectors(term):
                        variables = set(numbers) & delta.free_symbols
                        if len(variables) == 1:
                            (n,) = variables
                            equation = sympy.expand(
                                delta.args[0].cond.lhs - delta.args[0].cond.rhs
                            )
                            slope = equation.coeff(n)
                            if slope.is_number and slope:
                                local = local.xreplace(
                                    {n: sympy.cancel(n - equation / slope)}
                                )
                    local = sympy.cancel(local)
                    if local == 0:
                        raise ZeroDivisionError(
                            "A virtual channel is degenerate with the retained space"
                        )
                    result += mask * term / local
            terms[powers] = result
        return NumberOrderedForm(basis.operators, terms, validate=False)

    def solve(value, index):
        if value is zero:
            return zero
        value = value.value
        if shape:
            return wrap(
                index,
                clean(
                    sympy.ImmutableSparseMatrix(
                        *shape,
                        {
                            key: divide_scalar(entry, *key)
                            for key, entry in value.todok().items()
                        },
                    )
                ),
            )
        return wrap(index, clean(divide_scalar(value, 0, 0)))

    def evaluate(i, j, *order):
        source = hamiltonian[tuple(order)]
        if source is zero or (i != j and tuple(order) == origin):
            return zero
        source = basis._source_form(source)
        if (source.shape if shape else None) != shape:
            raise ValueError(
                "All Hamiltonian coefficients must have the same source matrix shape"
            )
        return wrap((i, j), clean(blocks[i] * source * blocks[j]))

    options = dict(
        n_infinite=hamiltonian.n_infinite, dimension_names=hamiltonian.dimension_names
    )
    outputs, _ = series_computation(
        {"H": BlockSeries(eval=evaluate, shape=(2, 2), **options)},
        algorithm=main,
        scope={
            "solve_sylvester": solve,
            "use_linear_operator": np.zeros((2, 2), dtype=bool),
            "two_block_optimized": True,
            "commuting_blocks": [True, True],
        },
        operator=multiply,
    )

    def result(name):
        def evaluate(i, j, *order):
            value = outputs[name][i, j, *order]
            if value is zero or value is one:
                return value
            return value.value

        return BlockSeries(eval=evaluate, shape=(2, 2), **options)

    return tuple(result(name) for name in ("H_tilde", "U", "U†"))
