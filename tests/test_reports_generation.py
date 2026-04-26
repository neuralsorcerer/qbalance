# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import sys
import types

from qbalance.reports import common as report_common
from qbalance.reports import html as report_html
from qbalance.reports import markdown as report_md


def test_reports_generation(tmp_path, monkeypatch):

    matrix = {
        "results": [
            {
                "backend": "b1",
                "strategy": {"optimization_level": 1},
                "metrics": {
                    "depth": 1,
                    "two_qubit_ops": 2,
                    "estimated_error": 0.1,
                    "compile_time_s": 0.2,
                },
            }
        ]
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(matrix), encoding="utf-8")
    data = report_common.load_matrix(mpath)
    assert report_common.strategy_key(data["results"][0]["strategy"]).startswith("opt")
    assert report_common.aggregate(data["results"])["depth"] == 1.0

    md = report_md.render_markdown(mpath, tmp_path)
    assert md.exists()

    jinja2 = types.ModuleType("jinja2")

    class Tpl:
        def __init__(self, text):

            self.text = text

        def render(self, **kwargs):

            return "html:" + kwargs["matrix_name"]

    jinja2.Template = Tpl
    monkeypatch.setitem(sys.modules, "jinja2", jinja2)
    html = report_html.render_html(mpath, tmp_path)
    assert html.read_text(encoding="utf-8").startswith("html:")


def test_aggregate_skips_non_finite_values():
    rows = [
        {
            "metrics": {
                "depth": 10,
                "two_qubit_ops": "4",
                "estimated_error": 0.2,
                "compile_time_s": 1.0,
            }
        },
        {
            "metrics": {
                "depth": float("nan"),
                "two_qubit_ops": float("inf"),
                "estimated_error": "-inf",
                "compile_time_s": "not-a-number",
            }
        },
    ]

    aggregated = report_common.aggregate(rows)

    assert aggregated["depth"] == 10.0
    assert aggregated["two_qubit_ops"] == 4.0
    assert aggregated["estimated_error"] == 0.2
    assert aggregated["compile_time_s"] == 1.0
