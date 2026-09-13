# Pymablock documentation

The documentation uses Astro and Starlight. All 16 authored pages come directly
from `docs/source/`, including the root changelog, author list, and contributor
guide through MyST includes. The original navigation and all executable cells
are retained. `starlight-pydocs` generates ten API module pages; the original
reference page links to the same documented objects, with full-width parameter
descriptions on their generated pages.

```sh
pixi run docs-build
pixi run -e starlight starlight-dev --port 51000
```

The HTML output is `docs/starlight/dist/`. GitLab CI and Read the Docs use this
build. `READTHEDOCS_CANONICAL_URL` supplies the site's domain and version prefix;
`DOCS_BASE` and `DOCS_SITE` override the prefix and domain for other hosts. Original `.html` page URLs
serve aliases of the corresponding Starlight pages. API objects now live on their
module pages; `objects.inv` publishes their cross-project references. MathJax's five expandable equations use ordinary MyST derivation
dropdowns, preserving their mathematical content.

## Execution and dependencies

Native execution uses fresh Jupyter kernels through `astro-myst-notebooks`.
Every build runs all cells, and errors fail the build. The native environment
uses the existing `mid` dependency baseline so the original examples execute
unchanged. Pixi locks Python, Node, Jupyter, scientific libraries, and build
tools; `package-lock.json` locks the npm graph.

The **Run interactively** button enables Thebe's editable cells with browser-local
JupyterLite **Xeus Python**, using Emscripten-forge packages, including Kwant.
`environment.yml` owns the browser environment. JupyterLite builds and bundles
that environment alongside a wheel from this checkout. Published pages serve
their own runtime assets; activation does not need Binder or a Python server.
The environment specification is not yet a complete transitive lockfile. Native
MUMPS is unavailable in the browser; Kwant uses its SciPy fallback.

The reusable integration lives in `~/src/astro-myst-notebooks`. This checkout
installs its compiled archive from `vendor/` and builds without the sibling
repository. Pymablock owns content, navigation, styling, API configuration,
inventory snapshots, and scientific environments. `docs.config.mjs` configures
these choices; see `vendor/README.md` for updating the integration.

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
tests live in the sibling repository and execute its compiled package.

Unsupported MyST constructs fail explicitly. Arbitrary Sphinx extensions,
Jupyter widgets, and persistent reader sessions are outside this integration.
