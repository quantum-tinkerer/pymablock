import nox


@nox.session(venv_backend="mamba")
@nox.parametrize(
    "python,numpy,scipy,sympy",
    [
        ("3.9", "1.23", "1.8", "1.10"),
        ("3.10", "1.24", "1.10", "1.10"),
        ("3.12", "1.26", "1.12", "1.12"),
    ],
    ids=["minimal", "mid", "latest"],
)
def tests(session, numpy, scipy, sympy):
    session.run(
        "mamba",
        "install",
        "-y",
        f"numpy={numpy}",
        f"scipy={scipy}",
        f"sympy={sympy}",
        "python-mumps>=0.0.1,<0.1",
        "packaging==22.0",
        "pytest-cov",
        "pytest-randomly",
        "pytest-regressions",
        "-c",
        "conda-forge",
    )
    session.run("pip", "install", "ruff", "pytest-ruff", "pytest-regressions")
    session.run(
        "pytest",
        "--ruff",
        "-W",
        "error",
        "-W",
        "ignore:ast.NameConstant:DeprecationWarning",
    )
