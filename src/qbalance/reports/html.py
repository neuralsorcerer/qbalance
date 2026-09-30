# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from qbalance.errors import OptionalDependencyError
from qbalance.reports.common import (
    aggregate,
    failed_trials,
    load_matrix,
    matrix_results,
    sort_value,
    strategy_key,
)


def render_html(matrix_json: Path, out_dir: Path) -> Path:
    """Write an HTML report of a matrix run to ``out_dir/report.html``.

    The tables are those of :func:`~qbalance.reports.markdown.render_markdown`.
    Every value taken from the matrix file is HTML-escaped.

    Args:
        matrix_json: Matrix results written by ``run_matrix``.
        out_dir: Directory to write into; created if missing.

    Returns:
        The path of the written report.

    Raises:
        OptionalDependencyError: If jinja2 is not installed.
        ValueError: If the matrix JSON cannot be read or has an unusable shape.
    """
    try:
        from jinja2 import Template
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "jinja2 is required (install qbalance[report])"
        ) from e

    # Read and validate before touching out_dir, so a bad matrix leaves no
    # empty report directory behind.
    results = matrix_results(load_matrix(matrix_json))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        b = r["backend"]
        sk = strategy_key(r["strategy"])
        grouped[b][sk].append(r)

    model: List[Dict[str, Any]] = []
    for backend, strat_map in grouped.items():
        rows: List[Dict[str, Any]] = []
        items: List[Tuple[str, Dict[str, float], str]] = []
        for sk, rs in strat_map.items():
            agg = aggregate(rs)
            items.append((sk, agg, f"{failed_trials(rs)}/{len(rs)}"))
        items.sort(
            key=lambda t: (
                sort_value(t[1].get("depth")),
                sort_value(t[1].get("two_qubit_ops")),
            )
        )
        for sk, agg, failed in items:
            rows.append(
                {
                    "strategy": sk,
                    "depth": agg["depth"],
                    "twoq": agg["two_qubit_ops"],
                    "err": agg["estimated_error"],
                    "compile": agg["compile_time_s"],
                    "failed": failed,
                }
            )
        model.append({"backend": backend, "rows": rows})

    # Backend names and strategy keys come from the matrix file, which is data
    # the report's reader did not necessarily write.  Without autoescape those
    # land in the document as live markup.
    tpl = Template(
        """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>qbalance report</title>
<style>
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; }
th, td { border: 1px solid #ddd; padding: 8px; }
th { background: #f7f7f7; text-align: left; }
code { background: #f3f3f3; padding: 1px 4px; border-radius: 4px; }
</style>
</head>
<body>
<h1>qbalance report</h1>
<p>Matrix: <code>{{ matrix_name }}</code></p>
{% for b in model %}
<h2>Backend: {{ b.backend }}</h2>
<table>
<thead>
<tr><th>Strategy</th><th>mean depth</th><th>mean 2q ops</th><th>mean est error</th><th>mean compile time (s)</th><th>failed trials</th></tr>
</thead>
<tbody>
{% for r in b.rows %}
<tr>
<td><code>{{ r.strategy }}</code></td>
<td>{{ "%.4g"|format(r.depth) }}</td>
<td>{{ "%.4g"|format(r.twoq) }}</td>
<td>{{ "%.4g"|format(r.err) }}</td>
<td>{{ "%.4g"|format(r.compile) }}</td>
<td>{{ r.failed }}</td>
</tr>
{% endfor %}
</tbody>
</table>
{% endfor %}
</body>
</html>""",
        autoescape=True,
    )
    html = tpl.render(model=model, matrix_name=Path(matrix_json).name)
    out = out_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out
