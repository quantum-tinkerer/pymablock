import nox


@nox.session(venv_backend="mamba")
@nox.parametrize(
    "python,numpy,sympy",
    [
        (python, numpy, sympy)
        for python in ("3.9", "3.10", "3.11")
        for numpy in ("1.23.0", "1.24.0")
        for sympy in ("1.10.0", "1.12.0")
        if (
            (python, numpy) != ("3.9", "1.24.0")
            and (python, numpy) != ("3.10", "1.23.0")
            and (python, numpy) != ("3.11", "1.23.0")
            and (python, sympy) != ("3.9", "1.12.0")
            and (python, sympy) != ("3.10", "1.12.0")
            and (python, sympy) != ("3.11", "1.10.0")
        )
    ],
)
def tests(session, numpy, sympy):
    session.run(
        "mamba",
        "install",
        "-y",
        f"numpy=={numpy}",
        "scipy==1.10.0",
        f"sympy=={sympy}",
        "packaging==23.1.0",
        "kwant",
    )
    session.run("pip", "install", "-e", ".")
    session.run(
        "pip", "install", "pytest", "pytest-cov", "pytest-randomly", "pytest-ruff"
    )

    session.run("pytest", "--ruff")
