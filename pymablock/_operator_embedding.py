"""Prepare embedding blocks and their algebraic Sylvester solver."""

import sympy
from sympy.physics.quantum.boson import BosonOp

from pymablock.number_ordered_form import (
    LadderOp,
    NumberOrderedForm,
    _NOFTransition,
    _projectors,
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
    if any(i != j or any(any(p) for p in x.terms) for (i, j), x in h0.todok().items()):
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
        x.terms.get(vacuum, sympy.S.Zero) if x != 0 else sympy.S.Zero
        for x in h0.diagonal()
    ]

    coordinates = () if finite else basis.coordinate_symbols
    occupations = vacuum if finite else basis.source_occupations
    sizes = () if finite else basis._target_dimensions
    binary = [q for q, size in zip(coordinates, sizes) if size == 2]

    def divide_gap(coefficient, gap, weight):
        """Split gap-dependent binary sectors, checking zero gaps on active channels."""
        if weight == 0:
            return sympy.S.Zero
        gap = sympy.cancel(gap)
        variables = (gap if gap != 0 else weight).free_symbols
        q = next((q for q in binary if q in variables), None)
        if q is not None:
            return sum(
                (q if v else 1 - q)
                * divide_gap(*(x.xreplace({q: v}) for x in (coefficient, gap, weight)))
                for v in (sympy.S.Zero, sympy.S.One)
            )
        point = next(_projectors(coefficient, set(coordinates) & variables), None)
        if point is not None:
            delta, n, v = point
            arguments = coefficient, gap, weight
            inside = divide_gap(*(x.xreplace({n: v}) for x in arguments))
            outside = divide_gap(*(x.xreplace({delta: sympy.S.Zero}) for x in arguments))
            return sympy.Piecewise((inside, sympy.Eq(n, v)), (outside, True))
        if gap == 0:
            raise ZeroDivisionError(
                "A virtual channel is degenerate with the retained space"
            )
        return coefficient / gap

    def divide_scalar(value, row, col):
        if value == 0 or value.is_zero:
            return sympy.S.Zero
        value = basis._source_scalar(value.source)
        energy = (
            retained[col, col]
            if finite
            else basis._at_occupations(energies[col], occupations)
        )
        terms = {}
        for transition in _NOFTransition.from_form(value):
            powers = transition.powers
            action = transition.symbolic_action(occupations)
            if action.weight == 0:
                continue
            middle = [n - max(p, 0) for n, p in zip(occupations, powers)]
            coefficient = basis._at_occupations(value.terms[powers], middle)
            denominator = sympy.expand(
                basis._at_occupations(energies[row], action.output_state) - energy
            )
            result = divide_gap(coefficient, denominator, action.weight)
            if not finite:
                incoming = sympy.Matrix([n + max(p, 0) for n, p in zip(numbers, powers)])
                target = basis._occupation_left_inverse * (
                    incoming - sympy.Matrix(basis.reference)
                )
                result = result.xreplace(dict(zip(coordinates, target)))
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
