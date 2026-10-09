"""Counterexamples from the independent embedding review."""

import gc
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


@pytest.mark.parametrize("embedded", [False, True])
def test_binary_poles_preserve_nondegenerate_sectors(embedded):
    f, g, source_f, source_g = map(FermionOp, ("f", "g", "F", "G"))
    q = SigmaMinus("q")
    u = s.Symbol("U", positive=True)
    if embedded:
        embedding = Embedding({source_f: f, source_g: g}, reference={q: 0, f: 0, g: 0})
        h, *_ = block_diagonalize(
            [u * N(f) * (1 - N(q)) + u * N(g) * N(q), q + q.adjoint()],
            subspace_eigenvectors=embedding,
        )
    else:
        source_f, source_g = f, g
        h, *_ = block_diagonalize(
            [s.diag(u * N(f), u * N(g)), s.Matrix([[0, 1], [1, 0]])],
            subspace_indices=[0, 1],
        )
    for order, coefficient in ((2, 1 / u), (4, -1 / u**3), (6, 2 / u**5)):
        value = h[0, 0, order] if embedded else h[0, 0, order][0, 0]
        for nf, ng, sign in ((1, 0, 1), (0, 1, -1)):
            actual = value.as_expr().subs({N(source_f): nf, N(source_g): ng})
            assert s.cancel(actual - sign * coefficient) == 0


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("selective", [False, True])
def test_diagonalize_retained_oscillator_levels(finite, selective):
    """Both retained levels obey the independent oscillator characteristic equation."""
    a, q = BosonOp("a"), SigmaMinus("q")
    embedding = (
        Embedding({}, reference=[{a: 0}, {a: 1}])
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
    target = s.diag(0, 3, 8, 15)
    for n in range(3):
        target[n, n + 1] = target[n + 1, n] = g * s.sqrt(n + 1)
    characteristic = target.charpoly(energy).as_expr()
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
        subspace_eigenvectors=Embedding({}, reference=[{}, {Embedding.row: 1}]),
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
        Embedding({}, reference=[{a: 0}])
        if finite
        else Embedding({q: a}, reference={a: 0})
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


@pytest.mark.parametrize("reference_list", [False, True])
def test_embeddings_are_collectable(reference_list):
    """Caching useful conversions cannot keep dropped embeddings alive."""
    a, q = BosonOp("a"), SigmaMinus("q")
    bases = []
    for _ in range(5):
        embedding = Embedding({q: a}, reference=[{a: 0}] if reference_list else {a: 0})
        embedding.restrict(a)
        w = embedding._retained_frame(1)[0, 0]
        w * NumberOrderedForm.from_expr(q)
        bases.append(weakref.ref(embedding))
        bases.extend(weakref.ref(lattice) for _, lattice in embedding._lattices)
    del embedding, w
    # SymPy's bounded expression cache may retain equal frame expressions.
    s.core.cache.clear_cache()
    gc.collect()
    assert all(basis() is None for basis in bases)


@pytest.mark.parametrize("preconverted", [False, True])
def test_float_fourth_order_matches_exact_values(preconverted):
    """Floating inputs agree numerically with their exact values through fourth order."""
    a, b = BosonOp("a"), BosonOp("b")
    embedding = Embedding({}, reference=[{a: 0, b: 0}, {a: 1, b: 0}])
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
        np.testing.assert_allclose(
            np.asarray(floating[0, 0, order], dtype=complex),
            np.asarray(rational[0, 0, order], dtype=complex),
            rtol=1e-12,
        )


@pytest.mark.parametrize("mode_type", [BosonOp, FermionOp])
def test_attachment_substitution_and_reconstruction(mode_type):
    a, b, z = map(mode_type, ("a", "b", "0z"))
    q = mode_type("q")
    g = s.Symbol("g", real=True)
    e = Embedding({q: a}, reference={a: 0, b: 1})
    renamed = Embedding({q: a}, reference={a: 0, z: 1})
    w = NumberOrderedForm.from_expr(e)
    x = NumberOrderedForm.from_expr(g * N(b) + a.adjoint() * b)
    expected = NumberOrderedForm.from_expr(2 * N(z) + a.adjoint() * z) * renamed
    for value in (x * w, (x * w).adjoint()):
        for restored in (
            value.func(*value.args),
            pickle.loads(pickle.dumps(value)),
            NumberOrderedForm.from_expr(value.as_expr()),
        ):
            assert restored == value
        assert value.adjoint().adjoint() == value
    for method in ("subs", "xreplace"):
        replaced = getattr(x * w, method)({g: 2})
        actual = replaced.xreplace({b: z, b.adjoint(): z.adjoint()})
        assert actual == expected
        assert replaced.adjoint().xreplace({b: z}) == expected.adjoint()
    assert (x * w) ** 1 == x * w
    with pytest.raises(ValueError):
        (x * w) ** 2
