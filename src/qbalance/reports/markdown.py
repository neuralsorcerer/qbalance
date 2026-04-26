# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from qbalance.reports.common import aggregate, load_matrix, strategy_key


def render_markdown(matrix_json: Path, out_dir: Path) -> Path:
    """Render markdown used by the qbalance workflow.

    Args:
        matrix_json: Matrix json value consumed by this routine.
        out_dir: Out dir value consumed by this routine.

    Returns:
        Path with the computed result.

    Raises:
        None.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_matrix(matrix_json)
    results = data["results"]

    # group by backend -> strategy
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in results:
        b = r["backend"]
        sk = strategy_key(r["strategy"])
        grouped.setdefault(b, {}).setdefault(sk, []).append(r)

    lines = []
    lines.append("# qbalance report\n")
    lines.append(f"Matrix: `{Path(matrix_json).name}`\n")
    for backend, strat_map in grouped.items():
        lines.append(f"## Backend: {backend}\n")
        lines.append(
            "| Strategy | mean depth | mean 2q ops | mean est error | mean compile time (s) |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        # sort by depth then 2q
        items = []
        for sk, rows in strat_map.items():
            agg = aggregate(rows)
            items.append((sk, agg))
        items.sort(
            key=lambda t: (t[1].get("depth", 1e9), t[1].get("two_qubit_ops", 1e9))
        )
        for sk, agg in items:
            lines.append(
                f"| `{sk}` | {agg['depth']:.4g} | {agg['two_qubit_ops']:.4g} | {agg['estimated_error']:.4g} | {agg['compile_time_s']:.4g} |"
            )
        lines.append("")
    out = out_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
