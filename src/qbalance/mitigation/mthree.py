# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List

from qbalance.errors import OptionalDependencyError
from qbalance.logging import get_logger

log = get_logger(__name__)


def apply_mthree_mitigation(
    backend: Any,
    raw_counts: Dict[str, int],
    measured_qubits: List[int],
    shots: int,
    calibration_shots: int = 10_000,
) -> Dict[str, float]:
    """Apply mthree mitigation used by the qbalance workflow.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        raw_counts: Raw counts value consumed by this routine.
        measured_qubits: Measured qubits value consumed by this routine.
        shots: Number of shots used when executing circuits on a backend.
        calibration_shots (default: 10000): Calibration shots value consumed by this routine.

    Returns:
        Dict[str, float] with the computed result.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    try:
        import mthree
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "mthree is required (install qbalance[mitigation])"
        ) from e

    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(measured_qubits, calibration_shots)
    quasi = mit.apply_correction(raw_counts, measured_qubits)
    probs = quasi.nearest_probability_distribution()  # best-effort true probs
    # Convert to python dict
    return {k: float(v) for k, v in probs.items()}
