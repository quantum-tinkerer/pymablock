import nox


@nox.session(venv_backend="mamba")
@nox.parametrize(
    "python,numpy,scipy,sympy",
    [
        (python, numpy, scipy, sympy)
        for python, numpy, scipy, sympy in zip(
            ("3.9", "3.10", "3.11"),
            ("1.23.0", "1.24.0", "1.24.0"),
            ("1.8.0", "1.10.0", "1.10.0"),
            ("1.10.0", "1.10.0", "1.12.0"),
        )
    ],
)
def tests(session, numpy, scipy, sympy):
    session.run(
        "mamba",
        "install",
        "-y",
        f"numpy=={numpy}",
        f"scipy=={scipy}",
        f"sympy=={sympy}",
        "packaging==23.1.0",
        "kwant",
    )
    session.run("pip", "install", "-e", ".")
    session.run(
        "pip", "install", "pytest", "pytest-cov", "pytest-randomly", "pytest-ruff"
    )

    session.run("pytest", "--ruff")
