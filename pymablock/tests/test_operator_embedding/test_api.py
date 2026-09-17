"""Public-API tests for algebraic second-quantized embeddings."""

from __future__ import annotations

import pytest
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.pauli import SigmaMinus
from sympy.physics.quantum.spin import JminusOp, JzOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm
from pymablock.second_quantization import Embedding


def test_fermion_embedding_returns_target_nof() -> None:
    source, virtual = FermionOp("source"), FermionOp("virtual")
    target = FermionOp("target")
    source_energy, virtual_energy, coupling = sympy.symbols(
        "source_energy virtual_energy coupling",
        nonzero=True,
        real=True,
    )
    h_0 = source_energy * NumberOperator(source) + virtual_energy * NumberOperator(
        virtual
    )
    perturbation = coupling * (Dagger(virtual) * source + Dagger(source) * virtual)
    embedding = Embedding(
        target=(target,), occupations={source: NumberOperator(target), virtual: 0}
    )

    effective, *_ = block_diagonalize(
        [h_0, perturbation],
        subspace_eigenvectors=embedding,
    )

    assert effective[0, 0, 2] == NumberOrderedForm.from_expr(
        coupling**2 * NumberOperator(target) / (source_energy - virtual_energy),
        operators=(target,),
    )


def test_frozen_fermion_phase_is_internal() -> None:
    fixed, source = FermionOp("a_fixed"), FermionOp("b_source")
    target, virtual = FermionOp("target"), FermionOp("virtual")
    source_energy, virtual_energy, coupling = sympy.symbols(
        "source_energy virtual_energy coupling",
        nonzero=True,
        real=True,
    )
    h_0 = source_energy * NumberOperator(source) + virtual_energy * NumberOperator(
        virtual
    )
    perturbation = coupling * (Dagger(virtual) * source + Dagger(source) * virtual)

    effective, *_ = block_diagonalize(
        [h_0, perturbation],
        subspace_eigenvectors=Embedding(
            target=(target,),
            occupations={fixed: 1, source: NumberOperator(target), virtual: 0},
        ),
    )

    assert effective[0, 0, 2].terms[(0,)] == (
        coupling**2
        * NumberOrderedForm.from_expr(NumberOperator(target), operators=(target,)).terms[
            (0,)
        ]
        / (source_energy - virtual_energy)
    )


def test_nonbinary_target_uses_package_matrix_interface() -> None:
    source = BosonOp("source")
    retained_spin = JminusOp("S")
    frequency, coupling = sympy.symbols(
        "frequency coupling",
        nonzero=True,
        real=True,
    )
    embedding = Embedding(target={retained_spin: 3}, occupations={source: JzOp("S") + 1})

    effective, *_ = block_diagonalize(
        [
            frequency * NumberOperator(source),
            coupling * (source + Dagger(source)),
        ],
        subspace_eigenvectors=embedding,
    )

    assert isinstance(effective[0, 0, 2], sympy.MatrixBase)
    assert effective[0, 0, 2] == sympy.diag(
        0,
        0,
        -3 * coupling**2 / frequency,
    )


def test_bosonic_excursion_is_not_a_product_of_compressions() -> None:
    """The spin target retains the virtual second boson level in products."""
    from pymablock._embedding.selection import _EmbeddingBackend

    a, s = BosonOp("a"), SigmaMinus("s")
    backend = _EmbeddingBackend(
        Embedding(target=(s,), occupations={a: NumberOperator(s)})
    )

    def pullback(expr):
        return backend.pullback(backend.source_form(expr))

    def expected(expr):
        return NumberOrderedForm.from_expr(expr, operators=(s,))

    assert pullback(a) == expected(s)
    assert pullback(a * Dagger(a)) == expected(1 + NumberOperator(s))
    assert pullback(a) * pullback(Dagger(a)) == expected(1 - NumberOperator(s))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("frozen", [0, 1])
def test_retained_fermions_preserve_car_and_mode_correspondence(reverse, frozen) -> None:
    """Frozen particles and permutations must not change the target CAR."""
    from pymablock._embedding.selection import _EmbeddingBackend

    a, fixed, b = (FermionOp(name) for name in ("a", "m", "z"))
    f, g = FermionOp("f"), FermionOp("g")
    left, right = (g, f) if reverse else (f, g)
    backend = _EmbeddingBackend(
        Embedding(
            target=(g, f),
            occupations={
                a: NumberOperator(left),
                fixed: frozen,
                b: NumberOperator(right),
            },
        )
    )
    assert backend.target_operators == (f, g)
    for source, target in (
        (a, left),
        (b, right),
        (Dagger(a), Dagger(left)),
        (Dagger(b), Dagger(right)),
        (Dagger(a) * b, Dagger(left) * right),
        (a * b, left * right),
        (b * Dagger(b), right * Dagger(right)),
    ):
        result = backend.pullback(backend.source_form(source))
        assert result == NumberOrderedForm.from_expr(target, operators=(f, g))


def test_spin_in_two_fermions_uses_single_occupancy() -> None:
    """The target is one spin, with charge-changing source actions projected out."""
    from pymablock._embedding.selection import _EmbeddingBackend

    up, down = FermionOp("up"), FermionOp("down")
    s = SigmaMinus("s")
    backend = _EmbeddingBackend(
        Embedding(
            target=(s,), occupations={up: NumberOperator(s), down: 1 - NumberOperator(s)}
        )
    )

    def expected(expr):
        return NumberOrderedForm.from_expr(expr, operators=(s,))

    assert backend.pullback(backend.source_form(NumberOperator(up))) == expected(
        NumberOperator(s)
    )
    assert backend.pullback(backend.source_form(NumberOperator(down))) == expected(
        1 - NumberOperator(s)
    )
    assert not backend.pullback(backend.source_form(up))
    assert not backend.pullback(
        backend.source_form(NumberOperator(up) * NumberOperator(down))
    )
    # In canonical source order (down, up), this bilinear takes down to up.
    assert backend.pullback(backend.source_form(Dagger(up) * down)) == expected(Dagger(s))


@pytest.mark.parametrize(
    "rule, error",
    [
        (lambda _s, _t: {BosonOp("a"): 0}, "injective"),
        (lambda _s, t: {BosonOp("a"): NumberOperator(t)}, "declared target"),
        (lambda s, _t: {BosonOp("a"): s}, "declared target"),
        (lambda s, _t: {BosonOp("a"): NumberOperator(s) / 2}, "integer affine"),
        (lambda s, _t: {BosonOp("a"): NumberOperator(s) - 1}, "nonnegative"),
        (lambda s, _t: {FermionOp("a"): 2 * NumberOperator(s)}, "zero or one"),
    ],
)
def test_invalid_occupation_rules_are_rejected(rule, error) -> None:
    s, t = SigmaMinus("s"), SigmaMinus("t")
    with pytest.raises(ValueError, match=error):
        Embedding(target=(s,), occupations=rule(s, t))


def test_invalid_target_declarations_are_rejected() -> None:
    s = SigmaMinus("s")
    a = BosonOp("a")
    with pytest.raises(ValueError, match="distinct"):
        Embedding(target=(s, s), occupations={a: NumberOperator(s)})
    with pytest.raises(ValueError, match="explicit dimension"):
        Embedding(target=(JminusOp("S"),), occupations={a: 0})
    with pytest.raises(ValueError, match="dimension two"):
        Embedding(target={s: 3}, occupations={a: NumberOperator(s)})


def test_fermion_targets_reject_non_direct_encodings() -> None:
    f = FermionOp("f")
    a, b = FermionOp("a"), FermionOp("b")
    with pytest.raises(ValueError, match="direct"):
        Embedding(target=(f,), occupations={a: 1 - NumberOperator(f)})
    with pytest.raises(ValueError, match="exactly one"):
        Embedding(target=(f,), occupations={a: NumberOperator(f), b: NumberOperator(f)})


def test_binary_validation_does_not_enumerate_target(monkeypatch) -> None:
    from pymablock.operator_embedding import _TargetSpace

    def forbidden(_self):
        raise AssertionError("Target enumeration is not needed")

    monkeypatch.setattr(_TargetSpace, "states", property(forbidden))
    spins = tuple(SigmaMinus(f"s{i}") for i in range(20))
    embedding = Embedding(
        target=spins,
        occupations={BosonOp(f"a{i}"): NumberOperator(s) for i, s in enumerate(spins)},
    )
    assert embedding.target.dimension == 2**20


def test_finite_virtual_resonance_still_raises() -> None:
    """Dropping transitions within P must not hide a resonant state in Q."""
    a = BosonOp("a")
    n = NumberOperator(a)
    embedding = Embedding(target={JminusOp("S"): 3}, occupations={a: JzOp("S") + 1})
    # Retained n=2 and excluded n=3 both have energy -6.
    effective, *_ = block_diagonalize(
        [n * (n - 5), a + Dagger(a)], subspace_eigenvectors=embedding
    )
    with pytest.raises(ZeroDivisionError, match="degenerate"):
        _ = effective[0, 0, 2]


@pytest.mark.parametrize("finite", [False, True])
def test_boson_annihilation_preserves_occupation_dependent_denominator(finite):
    """Intermediate energies are 10 and 11, versus retained energies 1 and 4."""
    a, b = BosonOp("a"), BosonOp("b")
    s = JminusOp("S") if finite else SigmaMinus("s")
    n = JzOp("S") + sympy.S.Half if finite else NumberOperator(s)
    embedding = Embedding(target={s: 2}, occupations={a: 1 + n, b: 0})
    effective, *_ = block_diagonalize(
        [NumberOperator(a) ** 2 + 10 * NumberOperator(b), Dagger(b) * a + Dagger(a) * b],
        subspace_eigenvectors=embedding,
    )
    expected = (
        sympy.diag(-sympy.Rational(1, 9), -sympy.Rational(2, 7))
        if finite
        else NumberOrderedForm.from_expr(-(1 - n) / 9 - 2 * n / 7, operators=(s,))
    )
    assert effective[0, 0, 2] == expected


@pytest.mark.parametrize("coupled", [False, True])
def test_sector_resonance_requires_nonzero_virtual_channel(coupled):
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    n = NumberOperator(a)
    effective, *_ = block_diagonalize(
        [n + (1 - n) * NumberOperator(b), (1 if coupled else 1 - n) * (b + Dagger(b))],
        subspace_eigenvectors=Embedding(
            target=(s,), occupations={a: NumberOperator(s), b: 0}
        ),
    )
    if coupled:
        with pytest.raises(ZeroDivisionError, match="degenerate"):
            _ = effective[0, 0, 2]
    else:
        assert effective[0, 0, 2] == NumberOrderedForm.from_expr(
            NumberOperator(s) - 1, operators=(s,)
        )


def test_empty_boson_channel_does_not_create_a_resonance():
    """The nominal zero gap at n=0 has zero annihilation amplitude."""
    a, b, s = BosonOp("a"), BosonOp("b"), SigmaMinus("s")
    effective, *_ = block_diagonalize(
        [NumberOperator(a) ** 2 - NumberOperator(b), Dagger(b) * a + Dagger(a) * b],
        subspace_eigenvectors=Embedding(
            target=(s,), occupations={a: NumberOperator(s), b: 0}
        ),
    )
    assert effective[0, 0, 2] == NumberOrderedForm.from_expr(
        NumberOperator(s) / 2, operators=(s,)
    )
