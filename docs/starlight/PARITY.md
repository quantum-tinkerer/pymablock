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
Desktop and mobile API screenshots were also inspected. The validated build
was subsequently published to `https://antonakhmerov.org/misc/pymablock-starlight/`.
All 39 static checks passed against that public site, as did a real Thebe test
covering notebook execution, edits, output clearing, and kernel reset. The
published API HTML, inventory, and API MyST download match the local artifact
byte for byte.

## Remaining differences

- Sphinx-style MyST substitutions are not implemented; a parser probe leaves
  the expression literal. This feature is configured but unused by current pages.
- Rich Sphinx-Tippy hover cards for cross-references, equations and DOI links
  are not implemented. Links and ordinary tooltips remain available.
- Sphinx's general/module index pages are replaced by Pydocs' searchable API
  index and module pages. Search uses Starlight's search dialog. Exported
  inventory labels point to these replacements.
- Matomo tracking, its opt-out footer, and the Sphinx release string in the page
  title, plus the original author/copyright footer, have not been ported. Tracking needs deployment-specific configuration;
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


## Configuration audit (2026-09-13)

The original `docs/source/conf.py` at `f85fcd1` was checked setting by setting.
Content parity does not imply full Sphinx configuration compatibility.

| Original setting or extension | Result in this conversion |
| --- | --- |
| `napoleon`, `autodoc`, `autoclass_content = "both"` | Griffe NumPy sections plus the RST-to-MyST adapter; constructor prose and original explicit member coverage verified |
| `autosummary` | Pydocs module index and generated module pages; not the original Sphinx index URLs |
| `autodoc_typehints = "description"`, short type names | Types retained, but Pydocs also puts them in signatures; presentation is not identical |
| `viewcode` | GitLab source links replace embedded `_modules/` source pages; offline source browsing and old source-page URLs are missing |
| `intersphinx` | All five original inventory sources retained; typed outbound inventory and its destination anchors verified |
| `default_role = "autolink"` | Registered MyST role and Griffe docstring translation; ordinary unresolved variable names remain code |
| `nitpick_ignore` for `SigmaOpBase` | No direct configuration translation; Pydocs controls annotation resolution, and the current API builds successfully |
| `myst_nb`, timeout 480, raise on error | Fresh nbclient/Jupyter kernels, 480-second cell timeout, build rejection and recovery tested |
| `dollarmath`, `amsmath`, `colon_fence` | Existing MyST parser handles these; original equations build with KaTeX, not MathJax |
| `substitution` | **Missing.** A real parser probe leaves `{{ label }}` literal with both Sphinx frontmatter forms. No current authored page uses this feature, so the content comparison did not reveal it |
| `myst_heading_anchors = 3` | MyST heading anchors plus 79 explicit old-anchor aliases; original stable labels verified |
| `mathjax`, `sphinx_togglebutton` | KaTeX plus MyST dropdowns; five custom expansion controls differ in interaction |
| `sphinx_copybutton` | Expressive Code copy controls exist on static source blocks, including notebook input; signatures and output blocks are not copy-button equivalents |
| `sphinx_tippy` | **Missing:** internal reference/equation/footnote cards and external previews |
| Custom `tippy_doi_template` and Crossref/DataCite patch | **Missing:** no DOI metadata fetching, cache, or author/publisher/date cards |
| Repository, issue, edit, source download, home navigation | Present; source downloads contain the exact MyST files, and printing uses the browser |
| Light/dark logos and favicon | Original assets used |
| `show_toc_level` and book-theme layout | Native Starlight navigation and object-level API TOC; different theme behavior |
| Project/release/author/copyright metadata | Project branding retained; release in document titles and original author/copyright footer are missing |
| `extra_footer` and original tracking template | Matomo and opt-out not ported |
| `_templates`, `_static`, `local.css` | Required assets adapted; purple bold name styling is restored through parsed HTML text nodes; Tippy-only CSS is not ported, and Jinja templates are not executed |

### Hover-preview direction

[MyST's own web renderer supports rich hover references](https://mystmd.org/guide/cross-references).
Using its parser in Astro does not install that renderer's browser behavior.
The [Starlight plugin directory](https://starlight.astro.build/resources/plugins/)
currently lists no direct Sphinx-Tippy equivalent. A Starlight implementation
would need a reusable preview layer consuming resolved MyST/Pydocs targets,
plus a cached DOI metadata provider covering both Crossref and DataCite.
A generic tooltip widget alone would not reproduce the original functionality.

### URL compatibility boundary

Canonical directory routes and restored in-page anchors work together. Old
Sphinx `.html` paths still require hosting redirects before their fragments can
reach those anchors. On the published preview, all 15 non-homepage authored `.html` paths return
404, including `algorithms.html`, `tutorial/getting_started.html`, and
`documentation/pymablock.html`. Their directory routes work. `index.html` works.

The hosting migration should map authored `<page>.html` to `<page>/`, and map
`genindex.html` and `py-modindex.html` to `api/pymablock/`. Search needs a deliberate
replacement route, while `_modules/...html` requires source-page replacements
or explicit source-link redirects. A blanket `.html` rewrite cannot cover those
special routes. These server changes have not been made here, and no HTML
redirect stubs have been restored.


The later sidebar correction removes the additional "API by module" group.
Only the original authored navigation is shown; generated module routes remain
available for API links and inventory targets. The original purple bold
"Pymablock" typography is restored in page titles and prose, without rewriting
code, output blocks, URLs, or source-location metadata.
