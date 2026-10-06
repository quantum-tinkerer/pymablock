"""Counterexamples from the independent embedding review."""

import gc
import importlib
import pickle
import weakref

import numpy as np
import pytest
import sympy as s
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus

from pymablock import block_diagonalize, operator_to_BlockSeries
from pymablock.number_ordered_form import LadderOp, NumberOrderedForm
from pymablock.number_ordered_form import NumberOperator as N
from pymablock.operator_embedding import Embedding
from pymablock.second_quantization import solve_scalar
from pymablock.series import cauchy_dot_product, zero
from pymablock.tests.second_quantization_helpers import nof_matrix


@pytest.mark.parametrize("selection", ["particle", "hole", "references"])
def test_vanishing_offdiagonal_order(selection):
    """A two-level avoided crossing has no third-order energy shift."""
    a, b, f = map(FermionOp, ("a", "b", "f"))
    if selection == "references":
        embedding = Embedding(reference=[{a: 1, b: 0}])
    else:
        image, filled = (a, 0) if selection == "particle" else (a.adjoint(), 1)
        embedding = Embedding({f: image}, reference={a: filled, b: 0})
    h, *_ = block_diagonalize(
        [2 * N(a) + 5 * N(b), a.adjoint() * b + b.adjoint() * a],
        subspace_eigenvectors=embedding,
    )
    assert h[0, 0, 3] is zero or h[0, 0, 3].is_zero
    fourth = h[0, 0, 4]
    expected = s.Rational(1, 27)
    if selection == "references":
        assert fourth == s.Matrix([[expected]])
    else:
        occupations = [range(2)]
        assert nof_matrix(fourth, occupations) == s.diag(
            *(expected if n == (0 if selection == "hole" else 1) else 0 for n in range(2))
        )


def test_binary_poles_preserve_nondegenerate_sectors():
    """Unresolved resonances may be poles but cannot turn valid sectors into nan."""
    f, g = FermionOp("f"), FermionOp("g")
    u = s.Symbol("U", positive=True)
    h, transform, *_ = block_diagonalize(
        [s.diag(u * N(f), u * N(g)), s.Matrix([[0, 1], [1, 0]])],
        subspace_indices=[0, 1],
    )
    for value in (transform[1, 0, 1][0, 0], h[0, 0, 2][0, 0]):
        assert not value.has(s.zoo, s.nan)
    for order, coefficient in ((2, 1 / u), (4, -1 / u**3), (6, 2 / u**5)):
        correction = h[0, 0, order][0, 0].as_expr()
        for nf, ng, sign in ((1, 0, 1), (0, 1, -1)):
            assert (
                s.cancel(correction.subs({N(f): nf, N(g): ng}) - sign * coefficient) == 0
            )


def test_embedding_binary_poles_against_two_level_spectrum():
    f, g, target_f, target_g = map(FermionOp, ("f", "g", "F", "G"))
    q = SigmaMinus("q")
    u = s.Symbol("U", positive=True)
    embedding = Embedding({target_f: f, target_g: g}, reference={q: 0, f: 0, g: 0})
    h, *_ = block_diagonalize(
        [u * N(f) * (1 - N(q)) + u * N(g) * N(q), q + q.adjoint()],
        subspace_eigenvectors=embedding,
    )
    for order, coefficient in ((2, 1 / u), (4, -1 / u**3), (6, 2 / u**5)):
        correction = h[0, 0, order].as_expr()
        for nf, ng, sign in ((1, 0, 1), (0, 1, -1)):
            value = correction.subs({N(target_f): nf, N(target_g): ng})
            assert s.cancel(value - sign * coefficient) == 0


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("selective", [False, True])
def test_diagonalize_retained_oscillator_levels(finite, selective):
    """Both retained levels obey the independent oscillator characteristic equation."""
    a, q = BosonOp("a"), SigmaMinus("q")
    embedding = (
        Embedding(reference=[{a: 0}, {a: 1}])
        if finite
        else Embedding({q: a}, reference={a: 0})
    )
    mask = s.Matrix([[0, 1], [1, 0]]) if finite else q + q.adjoint()
    h, *_ = block_diagonalize(
        [3 * N(a) + N(a) * (N(a) - 1), a + a.adjoint()],
        subspace_eigenvectors=embedding,
        fully_diagonalize={0: mask} if selective else (0,),
    )
    coefficients = []
    for order in range(5):
        value = h[0, 0, order]
        if value is zero or value == 0:
            matrix = s.zeros(2)
        elif isinstance(value, NumberOrderedForm):
            matrix = nof_matrix(value)
        else:
            matrix = value
        assert matrix.is_diagonal()
        coefficients.append(matrix)
    assert coefficients[2] == s.diag(-s.Rational(1, 3), -s.Rational(1, 15))

    # Four oscillator levels include every path of length four from levels 0,1.
    # Construct their Hamiltonian directly, without NOF or embedding conversion.
    g, energy = s.symbols("g energy")
    source = s.diag(0, 3, 8, 15)
    for n in range(3):
        source[n, n + 1] = source[n + 1, n] = g * s.sqrt(n + 1)
    characteristic = source.charpoly(energy).as_expr()
    for row in range(2):
        effective_energy = sum(
            c[row, row] * g**order for order, c in enumerate(coefficients)
        )
        residual = s.Poly(characteristic.subs(energy, effective_energy), g)
        assert all(residual.nth(order) == 0 for order in range(5))


def test_diagonalize_finite_matrix_embedding():
    """A retained matrix block includes internal and virtual energy corrections."""
    h, *_ = block_diagonalize(
        [s.diag(0, 2, 5), s.Matrix([[0, 1, 1], [1, 0, 2], [1, 2, 0]])],
        subspace_eigenvectors=Embedding(reference=[(0, {}), (1, {})]),
        fully_diagonalize=(0,),
    )
    assert h[0, 0, 1].is_zero_matrix
    # Ordinary second-order perturbation theory sums over both other levels.
    assert h[0, 0, 2] == s.diag(-s.Rational(7, 10), -s.Rational(5, 6))
    assert h[0, 0, 3] == s.diag(s.Rational(2, 5), -s.Rational(2, 3))


@pytest.mark.parametrize("selective", [False, True])
def test_diagonalize_embedded_infinite_oscillators(selective):
    """Displacing independent oscillators gives their exact energy corrections."""
    a, b, c, f, g = map(BosonOp, ("a", "b", "c", "f", "g"))
    h, *_ = block_diagonalize(
        [3 * N(a) + 5 * N(b) + 11 * N(c), sum(op + op.adjoint() for op in (a, b, c))],
        subspace_eigenvectors=Embedding({f: a, g: b}, reference={a: 0, b: 0, c: 0}),
        fully_diagonalize={0: f + f.adjoint()} if selective else (0, 1),
    )
    if selective:
        assert h[0, 0, 1] == NumberOrderedForm.from_expr(g + g.adjoint())
    else:
        assert h[0, 0, 1] is zero or h[0, 0, 1] == 0
    expected = -s.Rational(1, 3) - s.Rational(1, 11)
    if not selective:
        expected -= s.Rational(1, 5)
        assert h[1, 1, 1] is zero or h[1, 1, 1] == 0
        complement = nof_matrix(h[1, 1, 2], [range(2)] * 3)
        assert complement == s.diag(0, expected, 0, expected, 0, expected, 0, expected)
    assert h[0, 0, 2] == expected


@pytest.mark.parametrize("angle", [0, s.pi / 2, s.pi / 4])
@pytest.mark.parametrize("method", ["subs", "xreplace"])
def test_substitution_preserves_independent_rotation_groups(angle, method):
    a, b, c, d, f, g = map(FermionOp, ("a", "b", "c", "d", "f", "g"))
    theta, phi = s.symbols("theta phi", real=True)
    embedding = Embedding(
        {f: s.cos(theta) * a + s.sin(theta) * b, g: s.cos(phi) * c + s.sin(phi) * d},
        reference={a: 0, b: 0, c: 0, d: 0},
    )
    replacements = {theta: angle, phi: s.pi / 4}
    updated = getattr(embedding, method)(replacements)
    ket = NumberOrderedForm.from_expr(
        N(a) + 3 * N(b) + 2 * N(c)
    ) * NumberOrderedForm.from_expr(embedding)
    transformed = getattr(ket, method)(replacements)
    value = NumberOrderedForm.from_expr(updated).adjoint() * transformed
    # Independent local occupation probabilities, in the declared target basis.
    expected = (s.cos(angle) ** 2 + 3 * s.sin(angle) ** 2) * N(f) + N(g)
    assert nof_matrix(value) == nof_matrix(NumberOrderedForm.from_expr(expected))


def test_binary_equality_after_multiplication():
    """Canonical binary products retain structural equality."""
    a, b = FermionOp("a"), FermionOp("b")
    lhs = NumberOrderedForm.from_expr((1 - N(a)) * (1 - N(b))) * 1
    rhs = NumberOrderedForm.from_expr(1 - N(a) - N(b) + N(a) * N(b))
    assert lhs == rhs
    assert hash(lhs) == hash(rhs)


def test_bilateral_zero_rhs_convention():
    """The negative ladder site is physical and selects the zero solution."""
    ell = LadderOp("ell")
    result = solve_scalar(N(ell) + 1, N(ell) + 1, 0)
    assert nof_matrix(result, [range(-2, 2)]) == s.diag(1, 0, 1, 1)


@pytest.mark.parametrize("finite", [False, True])
def test_dressed_observable_conversion(finite):
    """Dressing N in a driven oscillator adds the vacuum population g²/omega²."""
    a, q = BosonOp("a"), SigmaMinus("q")
    embedding = (
        Embedding(reference=[{a: 0}]) if finite else Embedding({q: a}, reference={a: 0})
    )
    _, transform, inverse = block_diagonalize(
        [3 * N(a), a + a.adjoint()], subspace_eigenvectors=embedding
    )
    observable = operator_to_BlockSeries({(0,): N(a)}, subspace_eigenvectors=embedding)
    # The same reusable path accepts a non-diagonal zeroth-order observable.
    cross = operator_to_BlockSeries(
        {(0,): a + a.adjoint()}, subspace_eigenvectors=embedding
    )
    assert cross[0, 1, 0] is not zero
    dressed = cauchy_dot_product(
        inverse, observable, transform, operator=lambda a, b: a * b
    )
    value = dressed[0, 0, 2]
    if finite:
        assert value == s.Matrix([[s.Rational(1, 9)]])
    else:
        # Generator selection retains levels 0,1, so only level 1 is dressed by Q.
        assert nof_matrix(value) == s.diag(0, s.Rational(2, 9))


def test_mixed_mode_public_composition_and_reconstruction():
    """A rank-one occupied projector fixes the independently known matrix element."""
    a, b, f = map(FermionOp, ("a", "b", "f"))
    theta = s.Symbol("theta", real=True)
    embedding = Embedding(
        {f: s.cos(theta) * a + s.sin(theta) * b}, reference={a: 0, b: 0}
    )
    w = NumberOrderedForm.from_expr(embedding)
    x = NumberOrderedForm.from_expr(N(a))
    y = NumberOrderedForm.from_expr(N(b))
    product = (x * w) * (w.adjoint() * y)
    retained = embedding.restrict(product)
    expected = NumberOrderedForm.from_expr(s.cos(theta) ** 2 * s.sin(theta) ** 2 * N(f))
    assert (retained - expected).applyfunc(s.trigsimp).is_zero
    for restore in (lambda v: v.func(*v.args), lambda v: pickle.loads(pickle.dumps(v))):
        restored = restore(x * w)
        assert restored == x * w
        assert (
            (w.adjoint() * restored - embedding.restrict(N(a)))
            .applyfunc(s.trigsimp)
            .is_zero
        )
    for replace in (
        lambda v: v.subs(theta, s.pi / 4),
        lambda v: v.xreplace({theta: s.pi / 4}),
    ):
        ket = replace(x * w)
        bra = NumberOrderedForm.from_expr(replace(embedding)).adjoint()
        assert (bra * ket - NumberOrderedForm.from_expr(N(f) / 2)).is_zero


def test_compiled_bases_are_collectable():
    """Caching useful conversions cannot keep dropped embeddings alive."""
    a, q = BosonOp("a"), SigmaMinus("q")
    bases = []
    for _ in range(5):
        embedding = Embedding({q: a}, reference={a: 0})
        embedding.restrict(a)
        w = NumberOrderedForm.from_expr(embedding)
        w * NumberOrderedForm.from_expr(q)
        bases.append(weakref.ref(embedding._basis))
    del embedding, w
    gc.collect()
    assert all(basis() is None for basis in bases)


@pytest.mark.parametrize("preconverted", [False, True])
def test_float_fourth_order_and_finite_entry_types(preconverted):
    """Stored floating inputs agree with their exact values through fourth order."""
    a, b = BosonOp("a"), BosonOp("b")
    embedding = Embedding(reference=[{a: 0, b: 0}, {a: 1, b: 0}])
    h0 = s.Float(1.1) * N(a) + s.Float(2.3) * N(b)
    v = s.Float(0.2) * (a + a.adjoint()) + s.Float(0.3) * (
        a.adjoint() * b + b.adjoint() * a
    )

    def exact(expr):
        return expr.xreplace({v: s.Rational(v) for v in expr.atoms(s.Float)})

    coefficients = [h0, v]
    if preconverted:
        coefficients = [NumberOrderedForm.from_expr(x, (a, b)) for x in coefficients]
    floating, *_ = block_diagonalize(coefficients, subspace_eigenvectors=embedding)
    rational, *_ = block_diagonalize(
        [exact(h0), exact(v)], subspace_eigenvectors=embedding
    )
    for order in (2, 4):
        assert floating[0, 0, order] == rational[0, 0, order]
        np.asarray(floating[0, 0, order], dtype=complex)
        assert not any(
            isinstance(entry, NumberOrderedForm) for entry in floating[0, 0, order]
        )


@pytest.mark.parametrize("option", [{"direct_solver": False}, {"atol": 1e-8}])
def test_unsupported_solver_options_are_explicit(option):
    """Embedding solvers cannot silently ignore numeric options."""
    with pytest.raises(NotImplementedError):
        block_diagonalize(
            [s.diag(0, 1), s.Matrix([[0, 1], [1, 0]])],
            subspace_eigenvectors=Embedding(reference=[(0, {})]),
            **option,
        )


def test_source_target_shadow_and_missing_reference():
    """Source/target ambiguity and an omitted reference fail at construction."""
    a, b = BosonOp("a"), BosonOp("b")
    with pytest.raises(ValueError, match="shadow"):
        Embedding({a: b}, reference={a: 0, b: 0})
    with pytest.raises(TypeError, match="reference"):
        Embedding({a: b})


def test_violated_binary_identity_is_decidable():
    """An explicit nonzero binary polynomial is an invalid representation."""
    a, b, q, r = map(SigmaMinus, ("a", "b", "q", "r"))
    with pytest.raises(ValueError, match="normalized"):
        Embedding({q: (1 + N(b)) * a, r: b}, reference={a: 0, b: 0})


def test_sympy_workaround_is_idempotent_and_preserves_scalars():
    """Repeated imports preserve scalar assumptions and operator condition order."""
    from pymablock import _sympy_compat

    x = s.Symbol("x", real=True)
    scalar = s.Piecewise((1, x > 0), (0, True))
    before = scalar.is_commutative, scalar.is_real
    importlib.reload(_sympy_compat)
    assert (scalar.is_commutative, scalar.is_real) == before == (True, True)
    a = BosonOp("a")
    operator = s.Piecewise((1, s.Eq(N(a), 0)), (0, True))
    assert operator.is_real is not True
    assert operator.is_commutative is False
