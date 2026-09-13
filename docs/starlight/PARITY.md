# Sphinx conversion audit

The comparison baseline is commit `f85fcd1`, immediately before the conversion.
Its original Sphinx/MyST/Napoleon/autodoc configuration was rebuilt in an isolated
directory, with notebook execution and remote tooltip fetching disabled for the
comparison. Native notebook execution is validated separately by the new build.

The earlier page-count and cell-count checks were insufficient: they missed
lost API prose, changed equation numbers, missing anchors, and inventory types.

## Repairs

| Area | What was missing | Current implementation |
| --- | --- | --- |
| Package reference | Full reference replaced with a link list | MyST `autodoc` directives render Pydocs components inline, including module introductions |
| API navigation | Object-level table of contents missing | Parsed MyST placements and the Pydocs collection supply the native Starlight table of contents |
| Constructors | `BlockSeries.__init__` and `NumberOperator.__new__` prose omitted | Griffe combines class and constructor sections, respecting the original class-only exceptions |
| Immutable classes | Signatures omitted `__new__` arguments | Public Pydocs signature component receives the appropriate constructor |
| Sentinel objects | `one` and `zero` had no descriptions | Griffe copies their defining classes' docstrings, as Sphinx autodata did |
| Dynamic properties | 25 SymPy assumption properties missing | Narrow runtime enrichment of the explicitly documented class |
| Docstring markup | Literal RST references, lost links and malformed examples | Griffe's NumPy parser plus RST-to-MyST; Pydocs resolves API references |
| Render failures | Pydocs could warn and omit prose | Missing rendered sections and unknown section kinds fail the build |
| Equation numbers | Every display equation was numbered | Labelled equations retain the original numbering; explicit MyST choices remain supported |
| Deep links | Old heading, equation and footnote anchors missing | 79 in-page aliases, recorded in `legacy-anchors.json`; no redirect files |
| Intersphinx export | Methods/properties/data mislabeled; document references omitted | Typed Python entries plus document and named-target entries |
| Page actions | Source download, edit, print and issue links omitted | Exact MyST downloads, native Starlight edit links, print action and issue link |
| Branding | Dark logo variant omitted | Original light and dark logo assets |
| Text source maps | Anonymous API fragments could claim a fictitious Markdown file | Authored MyST retains exact mappings; API fragments are marked generated and retain Python source links |

The assumption properties previously displayed Sphinx's generic `bool`
constructor documentation. They now describe SymPy's actual three-valued
assumption queries (`True`, `False`, or unknown `None`). This is an intentional
correction, rather than copying that boilerplate.

## Comparison coverage

`tests/fixtures/sphinx-parity.json` captures the original 54 documented API
objects, 189 API prose passages, 506 prose passages across the other 15 pages,
labelled equation numbers, and all 82 original inventory entries. Browser tests
compare the rendered site with this independent fixture. Prose comparisons
normalize whitespace and punctuation and exclude math and code; equation
numbers, signatures, links, code cells and execution have separate checks.
This is semantic coverage, not a claim of byte-identical HTML or screenshots.

The existing checks also cover all 16 authored pages, all 104 executable cells,
local links and assets, footnotes, search, mobile layouts, both themes, source
selections, and real Thebe/Xeus execution, including the Kwant tutorial.

`tests/capture-sphinx.py` regenerates the fixture from an original Sphinx HTML
build (requires BeautifulSoup). Keep the recorded baseline commit with any
regenerated fixture. Random UUID anchors on unlabelled equations are excluded
because they were never stable between Sphinx builds.

Validation on 2026-09-13: 39 static browser checks passed, including the
independent Sphinx comparison and every exported inventory fragment. All 19
browser execution/lifecycle checks passed in the preceding execution run,
covering all nine notebooks. The four-build contract test passed (reexecution,
intentional cell failure, and recovery), as did 19 integration unit tests,
TypeScript checks in both repositories, and the consumer's pre-commit hooks.
Desktop and mobile API screenshots were also inspected. These are local
validation results; this repair has not been deployed.

## Remaining differences

- Rich Sphinx-Tippy hover cards for cross-references, equations and DOI links
  are not implemented. Links and ordinary tooltips remain available.
- Sphinx's general/module index pages are replaced by Pydocs' searchable API
  index and module navigation. Search uses Starlight's search dialog. Exported
  inventory labels point to these replacements.
- Matomo tracking, its opt-out footer, and the Sphinx release string in the page
  title have not been ported. Tracking needs deployment-specific configuration;
  the personal preview should not silently count as production documentation.
- Read the Docs version switching is a hosting integration, not verified by a
  local or personal-site preview.
- The five MathJax equation expansion controls use MyST derivation dropdowns.
  Their mathematical content is retained; the interaction differs.
- Exact selections inside generated API docstrings are not mapped back to
  Python docstring character offsets. The Python source links are available.

The original scientific dependencies and examples are retained. Browser-local
execution is the requested extension: Xeus runs the Emscripten-forge build of
Kwant. Its browser environment is still a specification rather than a complete
transitive lockfile, and native MUMPS is not available there.
