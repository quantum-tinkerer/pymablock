This project uses pixi for managing the environment, stored in `pyproject.toml`. To learn about pixi configuration if necessary, see https://pixi.sh/latest/reference/pixi_manifest/

To run tests use `pixi run tests-latest <pytest args>`

Always run pre-commit to format the code using `pixi run pre-commit run --all-files`.

This project uses numpy style docstrings and aims to use type annotations for all functions and methods in the codebase.

This project does not use pytest test classes, and instead uses test functions.

When working on a task, interpret it narrowly and ask for next steps rather than writing a lot of code at once.

When working on code, always implement tests for new functionality and run tests.

If tests fail, always ask for advice on whether and how to fix them.

When implementing a plan from a design document, keep the document up to date and mark tasks as they are completed.
