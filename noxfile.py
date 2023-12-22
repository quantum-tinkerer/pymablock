import nox


@nox.session(venv_backend="mamba")
@nox.parametrize(
    "python,numpy,scipy,sympy",
    [
        ("3.9", "1.23.0", "1.8.0", "1.10.0"),
        ("3.10", "1.24", "1.10", "1.10"),
        ("3.11", "1.24", "1.10", "1.12"),
    ],
    ids=["minimal", "mid", "latest"],
)
def tests(session, numpy, scipy, sympy):
    session.run(
        "mamba",
        "install",
        "-y",
        f"numpy=={numpy}",
        f"scipy=={scipy}",
        f"sympy=={sympy}",
        "packaging==22.0",
        "kwant",
        "pytest-cov",
        "pytest-randomly",
        "-c",
        "conda-forge",
    )
    session.run("pip", "install", "ruff", "pytest-ruff")
    session.run("pytest", "--ruff")


@nox.session(venv_backend="mamba")
def tests_without_kwant(session):
    session.run(
        "mamba",
        "install",
        "-y",
        "python==3.11",
        "numpy==1.24.0",
        "scipy==1.10.0",
        "sympy==1.12.0",
        "packaging==22.0",
        "pytest-cov",
        "pytest-randomly",
        "-c",
        "conda-forge",
    )
    session.run("pip", "install", "ruff", "pytest-ruff")
    session.run("pytest", "--ruff")
