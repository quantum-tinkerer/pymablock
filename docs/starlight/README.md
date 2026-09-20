# Pymablock documentation

The documentation uses Astro and Starlight. All 16 authored pages come directly
from `docs/source/`, including the root changelog, author list, and contributor
guide through MyST includes. The original navigation and all executable cells
are retained. `starlight-pydocs` generates ten API module pages; the original
reference page renders the complete API inline through MyST `autodoc` directives
and Pydocs components, with full-width parameter descriptions.

```sh
pixi run docs-build
pixi run -e starlight starlight-dev --port 51000
```

The HTML output is `docs/starlight/dist/`. GitLab CI and Read the Docs use this
build. `READTHEDOCS_CANONICAL_URL` supplies the site's domain and version prefix;
`DOCS_BASE` and `DOCS_SITE` override the prefix and domain for other hosts.
Legacy in-page anchors are retained without generating `.html` redirect stubs.
The hosting layer owns extension and trailing-slash routing. API objects also
have generated module pages; `objects.inv` publishes their cross-project references.
MathJax's five expandable equations use ordinary MyST derivation
dropdowns, preserving their mathematical content.

## Execution and dependencies

Native execution uses fresh Jupyter kernels through `astro-myst-notebooks`.
Every build runs all cells, and errors fail the build. The native environment
uses the existing `mid` dependency baseline so the original examples execute
unchanged. Pixi locks Python, Node, Jupyter, scientific libraries, and build
tools; `package-lock.json` locks the npm graph.

The **Enable interactivity** button enables Thebe's editable cells with browser-local
JupyterLite **Xeus Python**, using Emscripten-forge packages, including Kwant.
`environment.yml` owns the browser environment. JupyterLite builds and bundles
that environment alongside a wheel from this checkout. Published pages serve
their own runtime assets; activation does not need Binder or a Python server.
The environment specification is not yet a complete transitive lockfile. Native
MUMPS is unavailable in the browser; Kwant uses its SciPy fallback.

The reusable integration is installed from the compiled GitHub release asset
[`preview-2026-09-20` (0.3.0)](https://github.com/akhmerov/astro-myst-notebooks/releases/tag/preview-2026-09-20).
`package.json` pins the asset URL; `package-lock.json` records its integrity and
dependency graph. Builds require no sibling checkout or vendored archive.
Pymablock owns content, navigation, styling, API configuration,
inventory snapshots, and scientific environments. `docs.config.mjs` configures
these choices. To upgrade, install a specific release's `.tgz` asset from this
directory, commit the npm manifest and lockfile, and run the checks below. Use
the compiled release asset, not a Git dependency or GitHub source archive.

## Validation

From `docs/starlight/`, in the Starlight Pixi environment:

```sh
npm run check
npm test
npm run preview -- --port 51000
DOCS_URL=http://localhost:51000 npm run test:browser
```

`npm test` performs successive real builds, proving that unchanged pages execute
again, execution failures fail the build, and a subsequent build recovers.
It temporarily adds and removes `docs/source/build-contract-check.md` and writes
its output to a temporary directory. Run it without another build or dev server
using the same source collection. A preview of the existing build can stay open.

Playwright checks all authored pages, cell counts, images, local links,
equation targets, search, mobile layout, both themes, and exact text selections
mapped to their source files. Browser execution checks use real Xeus workers,
including edits, ordered execution, reset, teardown, and failed-download recovery.
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` selects an installed Chromium. Integration unit
tests live in the upstream repository; the checks here exercise the installed
release in this consumer.

The parity audit is recorded in [PARITY.md](PARITY.md). The checked-in Sphinx
fixture compares original object coverage, prose, equation numbers, and typed
inventory entries; it is independent of the new renderer. Regenerate it with
`tests/capture-sphinx.py` against a Sphinx build of commit `f85fcd1`.

The Python API adapter uses Griffe's NumPy parser and `rst-to-myst` for RST
markup. Constructor docs, `__new__` signatures, and dynamic SymPy properties
are explicitly retained. Failed docstring rendering fails the build. MyST
page text keeps exact source mappings; generated docstring fragments use the
API source link and do not claim exact Markdown offsets.

Unsupported MyST constructs fail explicitly. Arbitrary Sphinx extensions,
Jupyter widgets, and persistent reader sessions are outside this integration.
