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


def strategy_key(spec: Dict[str, Any]) -> str:
    """Strategy key used by the qbalance workflow.

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
    if lm:
        parts.append(f"layout={lm}")
    if rm:
        parts.append(f"route={rm}")
    if spec.get("pauli_twirling"):
        parts.append(f"twirl{spec.get('num_twirls',1)}")
    if spec.get("dynamical_decoupling"):
        parts.append(f"dd={spec.get('dd_sequence','XY4')}")
    if spec.get("measurement_twirling"):
        parts.append("meas_twirl")
    if spec.get("mthree"):
        parts.append("mthree")
    if spec.get("zne"):
        parts.append("zne")
    if spec.get("cutting"):
        parts.append(f"cut{spec.get('max_subcircuit_qubits')}")
    return ",".join(parts) if parts else "default"


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
