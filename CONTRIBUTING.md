# Developer documentation

Pymablock is an open source project and we welcome contributions from the community.
To contribute code, please follow the guidelines below.

## Development environment

We use [pixi](https://pixi.sh/latest/) for dependency management, run `pixi install` to install the default environment.

Pymablock uses [pre-commit](https://pre-commit.com/), run `pixi run pre-commit install` to enable it after cloning the repository.

We use `py.test` for testing, run it with `pixi run tests`.
To test against multiple dependency versions, run `pixi run tests-all`.

## Documentation

Pymablock uses markdown for documentation, run `pixi run docs-build` to build it.
When writing documentation, ensure that you write one sentence per line to make it easier to review changes.

## Dependency versions

Pymablock adopted [SPEC-0](https://scientific-python.org/specs/spec-0000/) for setting minimal requirements on Python, NumPy, and SciPy.

Before making a release check that the minimal versions specified in `pyproject.toml` adhere to SPEC-0.

## Release checklist

To make a release, do the following:

1. Confirm that the [changelog](CHANGELOG.md) contains all relevant user-visible changes, and update it if necessary.
2. Confirm that all contributors have been added to the [authors.md](authors.md) file by running `git shortlog -s $(git describe --tags --abbrev=0)..HEAD| sed -e "s/^ *[0-9\t ]*//"`.
3. Add a new level two header to the changelog with the title `[X.Y.Z] - YYYY-MM-DD`, but keep the `[Unreleased]` header. Commit with the message `release vX.Y.Z` and push.
4. Check that CI runs successfully.
5. Tag the version with `git tag --sign vX.Y.Z --annotate -m 'release vX.Y.Z'` (skip `--sign` if you do not have git signing configured) and push the tag `git push origin vX.Y.Z`. This publishes the release to pypi.
6. @isidora.araya updates the Zenodo repository (as its owner) or @anton-akhmerov as an administrator of the quantumtinkerer community.
  To do so, download the zip file from the [tags page](https://gitlab.kwant-project.org/qt/pymablock/-/tags), then click "create new version" on [Zenodo](https://doi.org/10.5281/zenodo.7995683), upload the zip file, and update the metadata.
7. Maintainers of the `pymablock-feedstock` review and merge the pull request created by the conda-forge bot.
