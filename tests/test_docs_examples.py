# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run the ``pycon`` examples of the documentation and check their output.

Each page's examples form one session, run in order in an empty working
directory, so the outputs printed in the docs are the outputs the code gives.
A ``...`` in an expected output matches anything (doctest's ELLIPSIS).
"""

from __future__ import annotations

import doctest
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
_PYCON_BLOCK = re.compile(r"^```pycon\n(.*?)^```", re.S | re.M)
# The examples exercise every optional integration.
_EXTRAS = (
    "qiskit_aer",
    "mthree",
    "qiskit_addon_cutting",
    "qiskit_ibm_runtime",
    "jinja2",
)


def _pages_with_examples() -> list[str]:
    return sorted(
        page.name
        for page in DOCS.glob("*.md")
        if _PYCON_BLOCK.search(page.read_text(encoding="utf-8"))
    )


def test_api_reference_has_examples():
    assert "api-references.md" in _pages_with_examples()


@pytest.mark.parametrize("page", _pages_with_examples())
def test_docs_examples(page, tmp_path, monkeypatch):
    for module in _EXTRAS:
        pytest.importorskip(module)
    import qbalance.builtin_data as builtin_data
    import qbalance.utils as utils

    # Keep the built-in dataset and the compile cache inside the sandbox.
    monkeypatch.setattr(
        builtin_data, "user_data_dir", lambda app: str(tmp_path / ".data" / app)
    )
    monkeypatch.setattr(
        utils, "user_cache_dir", lambda app: str(tmp_path / ".cache" / app)
    )
    monkeypatch.chdir(tmp_path)

    path = DOCS / page
    text = path.read_text(encoding="utf-8")
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    globs: dict = {"__name__": "__docs__"}
    report: list[str] = []
    for match in _PYCON_BLOCK.finditer(text):
        lineno = text.count("\n", 0, match.start(1))
        test = parser.get_doctest(match.group(1), globs, page, str(path), lineno)
        runner.run(test, out=report.append, clear_globs=False)
        # Later blocks build on the names earlier ones defined.
        globs.update(test.globs)
    results = runner.summarize(verbose=False)
    assert results.attempted > 0
    assert results.failed == 0, _version_note(text) + "".join(report)


# "... produced with Qiskit 2.5.2, qiskit-aer 0.17.2, ..." on the page.
_RECORDED_VERSION = re.compile(
    r"\b(Qiskit|qiskit-aer|mthree|qiskit-addon-cutting|qiskit-ibm-runtime) "
    r"(\d+(?:\.\d+)+)"
)


def _version_note(text: str) -> str:
    """Name the packages installed at other versions than the page records.

    Transpiler-derived outputs legitimately change between releases; the
    note tells a reader whether to refresh the page or install its versions.
    """
    from importlib.metadata import PackageNotFoundError, version

    differing = []
    for name, recorded in dict(_RECORDED_VERSION.findall(text)).items():
        try:
            installed = version(name.lower())
        except PackageNotFoundError:
            installed = "not installed"
        if installed != recorded:
            differing.append(f"{name} {installed} (page: {recorded})")
    if not differing:
        return ""
    return (
        "The page's outputs were recorded with other versions: "
        + ", ".join(differing)
        + ". Refresh the outputs (and the version list) or install those "
        "versions.\n"
    )
