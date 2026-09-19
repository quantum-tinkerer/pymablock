"""Prepare embedding blocks and their algebraic Sylvester solver."""

from itertools import product

import sympy
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOrderedForm,
    _NOFTransition,
    _occupation_dimension,
    _projectors,
    _reduce_projectors,
    _spectral_projector,
)
from pymablock.operator_embedding import Embedding, _ReferenceBasis
from pymablock.series import BlockSeries, zero


def _projector(basis):
    """Select the joint spectrum of retained number operators and constraints."""
    numbers = basis._source_placeholders
    if isinstance(basis, _ReferenceBasis):
        spectra = list(zip(numbers, ((n,) for n in basis._references[0][1])))
    else:
        offsets = sympy.Matrix(numbers) - sympy.Matrix(basis.reference)
        spectra = [
            (sympy.expand(normal.dot(offsets)), (0,))
            for normal in basis._occupation_matrix.T.nullspace()
        ]
        target = basis._occupation_left_inverse * offsets
        physical = {
            n: sympy.Dummy(integer=True, nonnegative=True)
            for source, n in zip(basis.operators, numbers)
            if not isinstance(source, LadderOp)
        }
        for q, op, size in zip(target, basis._target_operators, basis._target_dimensions):
            if (
                isinstance(op, BosonOp)
                and q.xreplace(physical).is_nonnegative is not True
            ):
                raise NotImplementedError(
                    "This bosonic embedding requires an occupation inequality"
                )
            spectra.append((q, range(size) if size is not None else sympy.S.Integers))
    indicator = sympy.prod(_spectral_projector(q, spectrum) for q, spectrum in spectra)
    return NumberOrderedForm(basis.operators, {(0,) * len(numbers): indicator}) * 1


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

    def divide_scalar(value, row, col):
        if value == 0 or value.is_zero:
            return sympy.S.Zero
        value = value.source * entry_embedding._projector
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
                c = _reduce_projectors(coefficient.xreplace(substitutions), modes)
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
