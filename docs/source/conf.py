# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx_tippy

import pymablock

package_path = os.path.abspath("../pymablock")
# Suppress superfluous frozen modules warning.
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
sys.path.insert(0, package_path)


# -- Project information -----------------------------------------------------

project = "pymablock"
copyright = "2023, Pymablock developers"
author = "Pymablock developers"
gitlab_url = "https://gitlab.kwant-project.org/qt/pymablock"

# The full version, including alpha/beta/rc tags
release = pymablock.__version__
major, minor = pymablock.__version_tuple__[:2]
version = f"{major}.{minor}"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "sphinx_tippy",
]
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "substitution",
    "colon_fence",
]
nb_execution_timeout = 480
nb_execution_raise_on_error = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "kwant": ("https://kwant-project.org/doc/1", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    # TODO: Switch to latest when sympy 1.15 is released.
    "sympy": ("https://docs.sympy.org/dev/", None),
}

default_role = "autolink"

# This is an undocumented base class.
nitpick_ignore = [
    ("py:class", "sympy.physics.quantum.pauli.SigmaOpBase"),
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autoclass_content = "both"
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://gitlab.kwant-project.org/qt/pymablock",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source",
    "repository_branch": "main",
    "use_download_button": True,
    "home_page_in_toc": True,
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo_dark.svg",
    },
    "extra_footer": (
        '<hr><div id="matomo-opt-out"></div>'
        '<script src="https://piwik.kwant-project.org/index.php?'
        "module=CoreAdminHome&action=optOutJS&divId=matomo-opt-out"
        '&language=auto&showIntro=1"></script>'
    ),
    "show_toc_level": 2,
}

tippy_doi_template = """
{% if "message" in data %} {# Crossref #}
{% set data = data.message %}
{% set title = data.title[0] %}
{% set authors = data.author | map_join('given', 'family') | join(', ') %}
{% set publisher = data.publisher %}
{% set created = data.created['date-time'] %}
{% else %} {# Datacite #}
{% set data = data.data.attributes %}
{% set title = data.titles[0].title %}
{% set authors = data.creators | map_join('givenName', 'familyName') | join(', ') %}
{% set publisher = data.publisher %}
{% set created = data.created %}
{% endif %}
<div>
    <h3>{{ title }}</h3>
    <p><b>Authors:</b> {{ authors }}</p>
    <p><b>Publisher:</b> {{ publisher }}</p>
    <p><b>Published:</b> {{ created | iso8601_to_date }}</p>
</div>
"""

# Patch sphinx-tippy to resolve both crossref and datacite DOIs.
sphinx_tippy_patch = """
from datetime import datetime

def iso8601_to_date(iso8601: str) -> datetime:
    return datetime.fromisoformat(iso8601).strftime("%Y-%m-%d")

def fetch_doi_tips(app: Sphinx, data: dict[str, TippyPageData]) -> dict[str, str]:
    '''fetch the doi tooltips, caching them for rebuilds.'''
    config = get_tippy_config(app)
    doi_cache: dict[str, str]
    doi_cache_path = Path(app.outdir, "tippy_doi_cache.json")
    if doi_cache_path.exists():
        with doi_cache_path.open("r") as file:
            doi_cache = json.load(file)
    else:
        doi_cache = {}
    doi_fetch = {
        doi for page in data.values() for doi in page["dois"] if doi not in doi_cache
    }
    for doi in status_iterator(doi_fetch, "Fetching DOI tips", length=len(doi_fetch)):
        # Resolve the RA from doi.org
        url = (
            "https://doi.org/api/handles/"
            + doi.replace("https://doi.org/", "").split("/")[0]
        )
        api_url = config.doi_api
        try:
            authorities = requests.get(url).json()["values"]
            for authority in authorities:
                if "10." in authority["data"]["value"]:
                    authority = authority["data"]["value"]
                    break
            else:
                raise ValueError(f"No DOI authority found for {doi}")
            if authority == "10.SERV/DATACITE":
                api_url = "https://api.datacite.org/dois/"
            elif authority == "10.SERV/CROSSREF":
                api_url = "https://api.crossref.org/works/"
            else:
                raise ValueError(f"Unknown DOI authority: {authority}")
        except Exception as exc:
            LOGGER.warning(
                f"Could not fetch DOI authority for {doi}: {exc} [tippy.doi]",
                type="tippy",
                subtype="doi",
            )
            continue
        url = f"{api_url}{doi}"
        try:
            data = requests.get(url).json()
        except Exception as exc:
            LOGGER.warning(
                f"Could not fetch DOI data for {doi}: {exc} [tippy.doi]",
                type="tippy",
                subtype="doi",
            )
        try:
            env = Environment()
            env.filters["map_join"] = map_join
            env.filters["iso8601_to_date"] = iso8601_to_date
            template = env.from_string(config.doi_template)
            doi_cache[doi] = template.render(data=data)
        except Exception as exc:
            LOGGER.warning(
                f"Could not render DOI template for {doi}: {exc} [tippy.doi]",
                type="tippy",
                subtype="doi",
            )
    with doi_cache_path.open("w") as file:
        json.dump(doi_cache, file)
    return doi_cache
"""
exec(sphinx_tippy_patch, sphinx_tippy.__dict__)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["local.css"]
html_favicon = "favicon.png"
