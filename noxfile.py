import nox


@nox.session(venv_backend="mamba")
@nox.parametrize(
    "python,numpy,scipy,sympy",
    [
        ("3.10", "=1.23", "=1.9", "=1.10"),
        ("3.11", "=1.24", "=1.10", "=1.11"),
        ("3.12", ">=1.6", ">=1.13", ">=1.12"),
    ],
    ids=["minimal", "mid", "latest"],
)
def tests(session, numpy, scipy, sympy):
    session.run(
        "mamba",
        "install",
        "-y",
        f"numpy{numpy}",
        f"scipy{scipy}",
        f"sympy{sympy}",
        "python-mumps>=0.0.1,<0.1",
        "packaging==22.0",
        "pytest-cov",
        "pytest-randomly",
        "pytest-repeat",
        "pytest-regressions",
        "-c",
        "conda-forge",
    )
    session.run(
        "pip", "install", "pytest-ruff"
    )
    session.run(
        "pytest",
        "-W",
        "error",
        "-W",
        "ignore:ast.NameConstant:DeprecationWarning",
    )
