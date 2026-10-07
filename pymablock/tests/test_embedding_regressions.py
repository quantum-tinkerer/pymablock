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
    f, g, source_f, source_g = map(FermionOp, ("f", "g", "F", "G"))
    q = SigmaMinus("q")
    u = s.Symbol("U", positive=True)
    embedding = Embedding({source_f: f, source_g: g}, reference={q: 0, f: 0, g: 0})
    h, *_ = block_diagonalize(
        [u * N(f) * (1 - N(q)) + u * N(g) * N(q), q + q.adjoint()],
        subspace_eigenvectors=embedding,
    )
    for order, coefficient in ((2, 1 / u), (4, -1 / u**3), (6, 2 / u**5)):
        correction = h[0, 0, order].as_expr()
        for nf, ng, sign in ((1, 0, 1), (0, 1, -1)):
            value = correction.subs({N(source_f): nf, N(source_g): ng})
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


def test_attached_reconstruction():
    """Attached NOFs survive reconstruction and pickling."""
    a, q = BosonOp("a"), SigmaMinus("q")
    embedding = Embedding({q: a}, reference={a: 0})
    w = NumberOrderedForm.from_expr(embedding)
    x = NumberOrderedForm.from_expr(N(a) + a)
    for restore in (lambda v: v.func(*v.args), lambda v: pickle.loads(pickle.dumps(v))):
        restored = restore(x * w)
        assert restored == x * w
        assert w.adjoint() * restored == embedding.restrict(N(a) + a)


@pytest.mark.parametrize("method", ["subs", "xreplace"])
def test_attached_blocks_substitute_parameters(method):
    """Number-operator conditions in rectangular blocks survive substitution."""
    a, q = BosonOp("a"), SigmaMinus("q")
    omega, alpha, g = s.symbols("omega alpha g", positive=True)
    h0 = omega * N(a) + alpha * N(a) * (N(a) - 1) / 2
    embedding = Embedding({q: a}, reference={a: 0})
    _, u, _ = block_diagonalize(
        [h0, g * (a + a.adjoint())], subspace_eigenvectors=embedding, symbols=[g]
    )
    values = {omega: 2, alpha: 3}
    _, expected, _ = block_diagonalize(
        [h0.subs(values), g * (a + a.adjoint())],
        subspace_eigenvectors=embedding,
        symbols=[g],
    )
    for index in ((1, 0, 1), (0, 1, 1)):
        value = getattr(u[index], method)(values)
        assert value.embedding == embedding
        assert value.side == expected[index].side
        assert nof_matrix(value.target, [range(4)]) == nof_matrix(
            expected[index].target, [range(4)]
        )


def test_attached_xreplace_renames_target_modes():
    """Renaming a target mode also renames its number operator."""
    a, b, q = BosonOp("a"), BosonOp("b"), SigmaMinus("q")
    renamed = Embedding({q: b}, reference={b: 0})
    attached = NumberOrderedForm.from_expr(N(a)) * NumberOrderedForm.from_expr(
        Embedding({q: a}, reference={a: 0})
    )
    value = attached.xreplace({a: b, a.adjoint(): b.adjoint()})
    assert value.embedding == renamed
    contracted = NumberOrderedForm.from_expr(renamed).adjoint() * value
    assert contracted == renamed.restrict(N(b))


def test_attached_xreplace_reorders_target_modes():
    """A rename that changes the mode order keeps fermion signs consistent."""
    c, d, q, z = map(FermionOp, ("c", "d", "q", "0z"))
    attached = Embedding({q: c}, reference={c: 0, d: 1})._attach(c.adjoint() * d, 1)
    renamed = Embedding({q: c}, reference={c: 0, z: 1})
    assert attached.xreplace({d: z}) == renamed._attach(c.adjoint() * z, 1)


@pytest.mark.parametrize("reference_list", [False, True])
def test_embeddings_are_collectable(reference_list):
    """Caching useful conversions cannot keep dropped embeddings alive."""
    a, q = BosonOp("a"), SigmaMinus("q")
    bases = []
    for _ in range(5):
        embedding = Embedding({q: a}, reference=[{a: 0}] if reference_list else {a: 0})
        embedding.restrict(a)
        w = embedding._frames(1)[0][0, 0]
        w * NumberOrderedForm.from_expr(q)
        bases.append(weakref.ref(embedding))
    del embedding, w
    # SymPy's bounded expression cache may retain equal frame expressions.
    s.core.cache.clear_cache()
    gc.collect()
    assert all(basis() is None for basis in bases)


@pytest.mark.parametrize("preconverted", [False, True])
def test_float_fourth_order_and_finite_entry_types(preconverted):
    """Floating inputs agree numerically with their exact values through fourth order."""
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
        np.testing.assert_allclose(
            np.asarray(floating[0, 0, order], dtype=complex),
            np.asarray(rational[0, 0, order], dtype=complex),
            rtol=1e-12,
        )
        assert not any(
            isinstance(entry, NumberOrderedForm) for entry in floating[0, 0, order]
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
    """Repeated installation preserves scalar assumptions and operator condition order."""
    from pymablock.number_ordered_form import _install_piecewise_patch

    x = s.Symbol("x", real=True)
    scalar = s.Piecewise((1, x > 0), (0, True))
    before = scalar.is_commutative, scalar.is_real
    _install_piecewise_patch()
    _install_piecewise_patch()
    assert (scalar.is_commutative, scalar.is_real) == before == (True, True)
    a = BosonOp("a")
    operator = s.Piecewise((1, s.Eq(N(a), 0)), (0, True))
    assert operator.is_real is not True
    assert operator.is_commutative is False
