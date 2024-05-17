"""Algorithm definitions."""


class _Part:
    def __neg__(self):
        return _Negated(self)

    def __call__(self):
        return _Eval(self)


class _Negated(_Part):
    def __init__(self, child):
        self.child = _to_series(child)

    def __repr__(self):
        return f"(-{self.child})"


class _Series(_Part):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'series["{self.name}"][index]'


class _DaggerAntiHermitianProduct(_Series):
    def __repr__(self):
        return f'Dagger(series["{self.name}"][(index[1], index[0], *index[2:])])'


class _Factor(_Part):
    def __init__(self, child, factor):
        self.child = _to_series(child)
        self.factor = factor

    def __repr__(self):
        return f"({self.child} * {self.factor})"


class _Divide(_Factor):
    def __init__(self, child, factor):
        super().__init__(child, factor)

    def __repr__(self):
        return f"_safe_divide({self.child}, {self.factor})"


class _Sum(_Part):
    def __init__(self, *children):
        self.children = [_to_series(child) for child in children]

    def __repr__(self):
        return "_zero_sum(" + ", ".join(str(child) for child in self.children) + ")"


class _Zero(_Part):
    def __repr__(self):
        return "zero"


class _Dagger(_Part):
    def __init__(self, child):
        self.child = _to_series(child)

    def __repr__(self):
        return f"Dagger({self.child})"


class _Diag(_Part):
    def __init__(self, diag, offdiag=_Zero()):
        self.diag = _to_series(diag)
        self.offdiag = _to_series(offdiag)

    def __repr__(self):
        return f"({self.diag} if index[0] == index[1] else {self.offdiag})"


class _AntiHermitize(_Part):
    def __init__(self, child):
        self.child = _to_series(child)

    def __repr__(self):
        return f"(-Dagger(series[series_name][(0, 1, *index[2:])]) if index[:2] == (1, 0) else {self.child})"


class _SolveSylvester(_Part):
    def __init__(self, child):
        self.child = _to_series(child)

    def __repr__(self):
        return f"(solve_sylvester(Y) if (Y := {self.child}) is not zero else zero)"


class _Product:
    def __init__(self, *, hermitian=False):
        self.hermitian = hermitian


class _Eval:
    def __init__(self, child: _Part):
        self.child = child

    def __repr__(self):
        return f"""
def series_eval(*index):
    return {self.child}
"""


class _ZeroData(_Part):
    def __repr__(self):
        return "zero_data"


class _IdentityData(_Part):
    def __repr__(self):
        return "identity_data"


class _H0Data(_Part):
    def __repr__(self):
        return "H_0_data"


def _to_series(item):
    if isinstance(item, str):
        item = _Series(item)
    if isinstance(item, _Part):
        return item
    raise NotImplementedError


# The main algorithm closely follows the notation in the notes, and is hard
# to understand otherwise. Consult the docs/source/algorithms.md in order to
# understand the logic of what is happening.
main_algorithm = {
    "H'_diag": {
        "eval": _Diag("H"),
        "data": _ZeroData(),
    },
    "H'_offdiag": {
        "eval": _Diag(diag=_Zero(), offdiag="H"),
        "data": _ZeroData(),
    },
    "U'": {
        "eval": _Diag(
            _Divide("U'† @ U'", -2),
            offdiag=_AntiHermitize(
                -_SolveSylvester(
                    _Sum(
                        "X",
                        "H'_diag @ U'",
                        _DaggerAntiHermitianProduct("H'_diag @ U'"),
                    )
                ),
            ),
        ),
        "data": _ZeroData(),
    },
    "U": {
        "eval": _Series("U'"),
        "data": _IdentityData(),
    },
    "U'†": {"eval": _Diag("U'", offdiag=_Negated("U'"))},
    "U†": {
        "eval": _Series("U'†"),
        "data": _IdentityData(),
    },
    "X": {
        "eval": _Sum("B", "H'_offdiag", "H'_offdiag @ U'"),
        "data": _ZeroData(),
    },
    # Used as common subexpression to save products, see docs/source/algorithms.md
    "B": {
        "eval": _Diag(
            _Divide(
                _Sum(
                    "U'† @ B",
                    -_Dagger("U'† @ B"),
                    "H'_offdiag @ U'",
                    _Dagger("H'_offdiag @ U'"),
                ),
                -2,
            ),
            offdiag=_Negated("U'† @ B"),
        ),
        "data": _ZeroData(),
    },
    "H_tilde": {
        "eval": _Sum(
            "H'_diag",
            _Divide(_Sum("H'_offdiag @ U'", _Dagger("H'_offdiag @ U'")), 2),
            _Divide(_Sum("U'† @ B", _Dagger("U'† @ B")), -2),
        ),
        "data": _H0Data(),
    },
    "U'† @ U'": {
        "eval": _Product(hermitian=True),
    },
    "H'_diag @ U'": {
        "eval": _Product(),
    },
    "H'_offdiag @ U'": {
        "eval": _Product(),
    },
    "U'† @ B": {
        "eval": _Product(),
    },
}
