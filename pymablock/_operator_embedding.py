"""Prepare embedding blocks and their algebraic Sylvester solver."""

from functools import cache
from itertools import product

import sympy
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOrderedForm,
    _NOFTransition,
    _occupation_dimension,
)
from pymablock.operator_embedding import Embedding, _ReferenceBasis
from pymablock.series import BlockSeries, zero


def _occupation_projector(left, right):
    return sympy.Piecewise((1, sympy.Eq(left, right)), (0, True))


def _projectors(expression, numbers):
    """Yield point indicators and the occupation value they select."""
    for delta in expression.atoms(sympy.Piecewise):
        if (
            len(delta.args) != 2
            or delta.args[0].expr != 1
            or delta.args[1] != (0, sympy.true)
            or not isinstance(delta.args[0].cond, sympy.Equality)
        ):
            continue
        variables = set(numbers) & delta.free_symbols
        if len(variables) != 1:
            continue
        (n,) = variables
        equation = sympy.expand(delta.args[0].cond.lhs - delta.args[0].cond.rhs)
        slope = equation.coeff(n)
        if slope.is_number and slope:
            yield delta, n, sympy.cancel(n - equation / slope)


def _projector(basis):
    """Compile the occupation indicator of the retained subspace."""
    numbers = basis._source_placeholders

    def diagonal(expression):
        return NumberOrderedForm(basis.operators, {(0,) * len(numbers): expression}) * 1

    if isinstance(basis, _ReferenceBasis):
        return diagonal(
            sympy.prod(
                _occupation_projector(n, value)
                for n, value in zip(numbers, basis._references[0][1])
            )
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


def _support_reducer(basis):
    """Compile reduction of coefficients on point-projector support."""
    numbers = basis._source_placeholders

    @cache
    def on_support(coefficient):
        """Reduce point projectors without expanding unrelated coefficient factors."""
        replacements, points = {}, {}
        for delta, n, value in _projectors(coefficient, numbers):
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

    return on_support


def prepare(hamiltonian, embedding):
    """Return rectangular Hamiltonian blocks and their Sylvester solver."""
    if hamiltonian.shape:
        raise ValueError("Structured embeddings require an unseparated Hamiltonian.")
    basis = embedding._basis
    finite = isinstance(basis, _ReferenceBasis)
    origin = (0,) * hamiltonian.n_infinite
    source_h0 = basis._source_form(hamiltonian[origin])
    h0 = source_h0 if finite else sympy.ImmutableMatrix([[source_h0]])
    modes, numbers = basis.operators, basis._source_placeholders
    dimensions = tuple(map(_occupation_dimension, modes))
    if any(
        (i != j and x != 0)
        or (
            isinstance(x, NumberOrderedForm)
            and any(any(p) and c != 0 for p, c in x.terms.items())
        )
        for (i, j), x in h0.todok().items()
    ):
        raise ValueError("Structured embeddings currently require diagonal H0")
    vacuum = (0,) * len(modes)
    if finite:
        entry_embedding = Embedding(reference=[dict(zip(modes, vacuum))])
        w = sympy.zeros(h0.rows, len(basis._references))
        for col, (row, state) in enumerate(basis._references):
            monomial = NumberOrderedForm(modes, {tuple(-n for n in state): sympy.S.One})
            (transition,) = _NOFTransition.from_form(monomial)
            w[row, col] = entry_embedding._attach(
                monomial / transition.apply(vacuum).weight, 1
            )
        w = sympy.ImmutableMatrix(w)
    else:
        entry_embedding = embedding
        w = sympy.ImmutableMatrix([[embedding._attach(sympy.S.One, 1)]])
    frames = (w, sympy.eye(h0.rows) - w * w.adjoint())
    retained = w.adjoint() * h0 * w
    energies = [
        sympy.expand(h0[i, i].terms.get((0,) * len(modes), sympy.S.Zero))
        if h0[i, i] != 0
        else sympy.S.Zero
        for i in range(h0.rows)
    ]
    on_support = entry_embedding._on_support

    def divide_scalar(value, row, col):
        if value == 0 or value.is_zero:
            return sympy.S.Zero
        value = entry_embedding._clean(value.source * entry_embedding._projector)
        terms = {}
        for powers, coefficient in value.terms.items():
            outgoing = {n: n + max(-int(power), 0) for n, power in zip(numbers, powers)}
            incoming = {n: n + max(int(power), 0) for n, power in zip(numbers, powers)}
            denominator = sympy.expand(
                energies[row].xreplace(outgoing)
                - (retained[col, col] if finite else energies[col].xreplace(incoming))
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
                    for _, n, value in _projectors(term, numbers):
                        local = local.xreplace({n: value})
                    local = sympy.cancel(local)
                    if local == 0:
                        raise ZeroDivisionError(
                            "A virtual channel is degenerate with the retained space"
                        )
                    result += mask * term / local
            terms[powers] = result
        return NumberOrderedForm(
            basis.operators, terms, entry_embedding, 1, validate=False
        )

    def solve(value, index):
        reverse = index[:2] == (0, 1)
        source = value.adjoint() if reverse else value
        result = sympy.ImmutableMatrix(
            source.rows, source.cols, lambda i, j: divide_scalar(source[i, j], i, j)
        )
        return -result.adjoint() if reverse else result

    def evaluate(i, j, *order):
        source = hamiltonian[tuple(order)]
        if source is zero or (i != j and tuple(order) == origin):
            return zero
        source = basis._source_form(source)
        if not finite:
            source = sympy.ImmutableMatrix([[source]])
        if source.shape != h0.shape:
            raise ValueError(
                "All Hamiltonian coefficients must have the same source matrix shape"
            )
        result = frames[i].adjoint() * source * frames[j]
        return zero if result.is_zero_matrix else result if finite else result[0, 0]

    options = dict(
        n_infinite=hamiltonian.n_infinite, dimension_names=hamiltonian.dimension_names
    )
    blocks = BlockSeries(eval=evaluate, shape=(2, 2), **options)
    return blocks, solve
