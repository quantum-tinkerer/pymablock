"""Algorithm definitions."""


class _Part:
    def __init__(self, **kwargs):
        self.extra = kwargs.get("extra", [])
        if not isinstance(self.extra, list):
            self.extra = [self.extra]

    def __neg__(self):
        return _Negated(self)


class _Negated(_Part):
    def __init__(self, child, **kwargs):
        self.child = _to_part(child)
        super().__init__(**kwargs)

    def __repr__(self):
        return f"(-{self.child})"


class _Series(_Part):
    def __init__(self, name, index_dag=False, **kwargs):
        self.key = f'"{name}"'
        self.index_dag = index_dag
        super().__init__(**kwargs)

    def _index(self):
        if self.index_dag:
            return "(index[1], index[0], *index[2:])"
        return "index"

    def __repr__(self):
        return f"which[{self.key}][{self._index()}]"


class _Self(_Series):
    def __init__(self, index_dag=False, **kwargs):
        super().__init__("", index_dag, **kwargs)
        self.key = "series_name"


class _Factor(_Part):
    def __init__(self, child, factor, **kwargs):
        self.child = _to_part(child)
        self.factor = factor
        super().__init__(**kwargs)

    def __repr__(self):
        return f"({self.child} * {self.factor})"


class _Divide(_Factor):
    def __init__(self, child, factor, **kwargs):
        super().__init__(child, factor, **kwargs)

    def __repr__(self):
        return f"_safe_divide({self.child}, {self.factor})"


class _Sum(_Part):
    def __init__(self, *children, **kwargs):
        self.children = [_to_part(child) for child in children]
        super().__init__(**kwargs)

    def __repr__(self):
        return "_zero_sum(" + ", ".join(str(child) for child in self.children) + ")"


class _Zero(_Part):
    def __repr__(self):
        return "zero"


class _Dagger(_Part):
    def __init__(self, child, index_dag=False, **kwargs):
        self.child = _to_part(child)
        if index_dag:
            if not isinstance(self.child, _Series):
                raise ValueError("Index dagger can only be used with _Term")
            self.child.index_dag = True
        super().__init__(**kwargs)

    def __repr__(self):
        return f"Dagger({self.child})"


class _SolveSylvester(_Part):
    def __init__(self, child):
        self.child = _to_part(child)

    def __repr__(self):
        return f"(solve_sylvester(Y) if (Y := {self.child}) is not zero else zero)"


class _Eval:
    def __init__(
        self,
        term=None,
        diag=None,
        offdiag=None,
        hermitian=False,
        antihermitian=False,
    ):
        if term is not None and (diag is not None or offdiag is not None):
            raise ValueError("Cannot have both term and diag/offdiag non-zero")
        self.term = _to_part(term or _Zero())
        self.diag = _to_part(diag or _Zero())
        self.offdiag = _to_part(offdiag or _Zero())
        if hermitian and antihermitian:
            raise ValueError("Cannot be both Hermitian and anti-Hermitian")
        self.hermitian = hermitian
        self.antihermitian = antihermitian

    @staticmethod
    def _return(expression: _Part, indent=""):
        return "".join(
            f"{indent}{line}\n" for line in [*expression.extra, f"return {expression}"]
        )

    def _function_body(self):
        if not isinstance(self.term, _Zero):
            result = self._return(self.term)
        else:
            result = "if index[0] == index[1]:\n"
            result += self._return(self.diag, indent="    ")
            if self.hermitian or self.antihermitian:
                result += "elif index[0] < index[1]:\n"
                result += self._return(self.offdiag, indent="    ")
                result += self._return(
                    _Dagger(_Self(index_dag=True))
                    if self.hermitian
                    else -_Dagger(_Self(index_dag=True))
                )
            else:
                result += self._return(self.offdiag)
        return result

    def __call__(self):
        result = "def series_eval(*index):\n"
        result += "    which = linear_operator_series if use_implicit and index[0] == index[1] == 1 else series\n"
        for line in str(self._function_body()).split("\n"):
            result += "    " + line + "\n"
        return result


class _Product:
    def __init__(self, *, hermitian=False):
        self.hermitian = hermitian


class _ZeroData(_Part):
    def __repr__(self):
        return "zero_data"


class _IdentityData(_Part):
    def __repr__(self):
        return "identity_data"


class _H0Data(_Part):
    def __repr__(self):
        return "H_0_data"


def _to_part(item):
    if isinstance(item, str):
        item = _Series(item)
    if isinstance(item, _Part):
        return item
    raise NotImplementedError


def _compile_evals(algorithm):
    for definition in algorithm.values():
        if not isinstance(definition["eval"], _Product):
            if not isinstance(definition["eval"], _Eval):
                definition["eval"] = _Eval(definition["eval"])
            definition["eval"] = compile(definition["eval"](), "<string>", "exec")


# The main algorithm closely follows the notation in the notes, and is hard
# to understand otherwise. Consult the docs/source/algorithms.md in order to
# understand the logic of what is happening.
main_algorithm = {
    "H'_diag": {
        "eval": _Eval(diag="H"),
        "data": _ZeroData(),
    },
    "H'_offdiag": {
        "eval": _Eval(offdiag="H"),
        "data": _ZeroData(),
    },
    "U'": {
        "eval": _Eval(
            diag=_Divide("U'† @ U'", -2),
            offdiag=-_SolveSylvester(
                _Sum(
                    "X",
                    "H'_diag @ U'",
                    _Dagger("H'_diag @ U'", index_dag=True),
                ),
            ),
            antihermitian=True,
        ),
        "data": _ZeroData(),
    },
    "U": {
        "eval": "U'",
        "data": _IdentityData(),
    },
    "U'†": {"eval": _Eval(diag="U'", offdiag=_Negated("U'"))},
    "U†": {
        "eval": "U'†",
        "data": _IdentityData(),
    },
    "X": {
        "eval": _Sum("B", "H'_offdiag", "H'_offdiag @ U'"),
        "data": _ZeroData(),
    },
    # Used as common subexpression to save products, see docs/source/algorithms.md
    "B": {
        "eval": _Eval(
            diag=_Divide(
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
        "eval": _Eval(
            diag=_Sum(
                "H'_diag",
                _Divide(_Sum("H'_offdiag @ U'", _Dagger("H'_offdiag @ U'")), 2),
                _Divide(_Sum("U'† @ B", _Dagger("U'† @ B")), -2),
            )
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

_compile_evals(main_algorithm)
