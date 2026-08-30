# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, cast


def load_matrix(path: Path) -> Dict[str, Any]:
    """Load matrix from serialized data or persisted storage.

    Args:
        path: Path value consumed by this routine.

    Returns:
        Dict[str, Any] with the computed result.

    Raises:
        None.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return cast(Dict[str, Any], data)


_DEFAULT_ZNE_FACTORS = (1.0, 2.0, 3.0)


def _format_zne_factors(factors: Any) -> str:
    """Render ZNE noise factors compactly for a report row label."""
    try:
        return "|".join(f"{float(factor):g}" for factor in factors)
    except (TypeError, ValueError):
        return str(factors)


def strategy_key(spec: Dict[str, Any]) -> str:
    """Return the report row label for one strategy.

    Report rows are grouped by this key, so two strategies that compile or
    execute differently must never produce the same one.  Knobs that do not
    change behavior under the rest of the spec (for example ``zne_degree`` when
    ``zne`` is off) are deliberately left out so labels stay readable; those
    strategies really do describe the same experiment.

    Args:
        spec: Strategy/backend specification controlling compilation behavior.

    Returns:
        str with the computed result.

    Raises:
        None.
    """
    parts = []
    parts.append(f"opt{spec.get('optimization_level')}")
    lm = spec.get("layout_method")
    rm = spec.get("routing_method")
    tm = spec.get("translation_method")
    if lm:
        parts.append(f"layout={lm}")
    if rm:
        parts.append(f"route={rm}")
    if tm:
        parts.append(f"xlate={tm}")
    seed_transpiler = spec.get("seed_transpiler", 0)
    if seed_transpiler != 0:
        parts.append(f"seedt={seed_transpiler}")
    if spec.get("pauli_twirling"):
        parts.append(f"twirl{spec.get('num_twirls',1)}")
    if spec.get("dynamical_decoupling"):
        parts.append(f"dd={spec.get('dd_sequence','XY4')}")
    if spec.get("measurement_twirling"):
        parts.append("meas_twirl")
    if spec.get("pauli_twirling") or spec.get("measurement_twirling"):
        seed_suppression = spec.get("seed_suppression", 0)
        if seed_suppression != 0:
            parts.append(f"seeds={seed_suppression}")
    if spec.get("mthree"):
        parts.append("mthree")
    if spec.get("zne"):
        parts.append("zne")
        zne_degree = spec.get("zne_degree", 1)
        if zne_degree != 1:
            parts.append(f"zdeg={zne_degree}")
        zne_factors = spec.get("zne_factors", _DEFAULT_ZNE_FACTORS)
        if tuple(zne_factors or ()) != _DEFAULT_ZNE_FACTORS:
            parts.append(f"zf={_format_zne_factors(zne_factors)}")
    if spec.get("cutting"):
        parts.append(f"cut{spec.get('max_subcircuit_qubits')}")
    resilience_level = spec.get("resilience_level")
    if resilience_level is not None:
        parts.append(f"res={resilience_level}")
    return ",".join(parts) if parts else "default"


def sort_value(value: Any) -> float:
    """Return a total-order sort key: non-finite/missing values sort last.

    ``aggregate`` emits NaN for metrics with no finite samples; NaN compares
    False against everything, which makes ``list.sort`` ordering depend on
    input order. Mapping NaN (and any unparsable value) to +inf keeps report
    tables deterministic.
    """
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("inf")
    return number if math.isfinite(number) else float("inf")


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate used by the qbalance workflow.

    Args:
        rows: Rows value consumed by this routine.

    Returns:
        Dict[str, float] with the computed result.

    Raises:
        None.
    """
    keys = ("depth", "two_qubit_ops", "estimated_error", "compile_time_s")
    sums = {key: 0.0 for key in keys}
    counts = {key: 0 for key in keys}

    for row in rows:
        metrics = row.get("metrics", {})
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(number):
                continue
            sums[key] += number
            counts[key] += 1

    return {
        key: (sums[key] / counts[key]) if counts[key] else float("nan") for key in keys
    }
