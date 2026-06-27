# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Sphinx configuration for the qbalance documentation."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "qbalance"
copyright = "Soumyadip Sarkar"
author = "Soumyadip Sarkar"

try:
    release = version(project)
except PackageNotFoundError:  # pragma: no cover - used when building without installation
    release = "0.0.0"

version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

nitpicky = True
nitpick_ignore = [
    ("py:class", "qiskit.circuit.QuantumCircuit"),
]

html_theme = "furo"
html_title = "qbalance documentation"
html_static_path = ["_static"]
html_extra_path = [".nojekyll"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/neuralsorcerer/qbalance/",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_baseurl = "https://neuralsorcerer.github.io/qbalance/"
html_copy_source = False
html_show_sourcelink = True
html_show_sphinx = False
html_last_updated_fmt = "%Y-%m-%d"

pygments_style = "sphinx"
pygments_dark_style = "monokai"

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://pydantic.dev/docs/validation/latest/", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit/", None),
}

linkcheck_ignore = [
    r"https://doi\.org/.*",
]
