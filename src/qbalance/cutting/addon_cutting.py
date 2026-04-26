# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Tuple

from qbalance.errors import OptionalDependencyError
from qbalance.logging import get_logger

log = get_logger(__name__)


def find_cuts_best_effort(
    circuit: Any,
    max_subcircuit_qubits: int,
    max_backjumps: int = 10_000,
    max_gamma: float = 1e6,
) -> Tuple[Any, Dict[str, Any]]:
    """Find cuts best effort used by the qbalance workflow.

    Args:
        circuit: QuantumCircuit instance to inspect, transform, or execute.
        max_subcircuit_qubits: Max subcircuit qubits value consumed by this routine.
        max_backjumps (default: 10000): Max backjumps value consumed by this routine.
        max_gamma (default: 1000000.0): Max gamma value consumed by this routine.

    Returns:
        Tuple[Any, Dict[str, Any]] with the computed result.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    try:
        from qiskit_addon_cutting.cutting import (
            DeviceConstraints,
            OptimizationParameters,
            find_cuts,
        )
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "qiskit-addon-cutting is required (install qbalance[cutting])"
        ) from e

    optimization = OptimizationParameters(
        max_backjumps=max_backjumps, max_gamma=max_gamma
    )
    constraints = DeviceConstraints(max_subcircuit_width=int(max_subcircuit_qubits))
    cut_circuit, meta = find_cuts(
        circuit, optimization=optimization, constraints=constraints
    )
    return cut_circuit, dict(meta)
