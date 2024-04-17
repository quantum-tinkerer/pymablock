# Developer documentation

Pymablock is an open source project and we welcome contributions from the community.
To contribute code, please follow the guidelines below.

## Development environment

Pymablock uses [pre-commit](https://pre-commit.com/), run `pre-commit install` to enable it after cloning the repository.

We use `py.test` for testing, run it with `py.test`.
To test against multiple dependency versions, run `nox`.

## Dependency versions

Pymablock adopted [SPEC-0](https://scientific-python.org/specs/spec-0000/) for setting minimal requirements on Python, NumPy, and SciPy.

Before making a release check that the minimal versions specified in `pyproject.toml` and in `noxfile.py` adhere to SPEC-0.

## Release checklist

To make a release, do the following:

1. Confirm that the [changelog](CHANGELOG.md) contains all relevant user-visible changes, and update it if necessary.
2. Confirm that all contributors have been added to the [authors.md](authors.md) file.
3. Add a new level two header to the changelog with the title `[X.Y.Z] - YYYY-MM-DD`, but keep the `[Unreleased]` header.
4. Check that CI runs.
5. Tag the version with `git tag --sign vX.Y.Z --annotate -m 'release vX.Y.Z'` (skip `--sign` if you do not have git signing configured) and push the tag `git push origin vX.Y.Z`. This publishes the release to pypi.
6. @isidora.araya updates the Zenodo repository (as its owner).
7. Maintainers of the `pymablock-feedstock` review and merge the pull request created by the conda-forge bot.
