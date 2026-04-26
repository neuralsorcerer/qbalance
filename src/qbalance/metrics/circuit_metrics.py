# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict


def extract_circuit_metrics(circuit: Any) -> Dict[str, float]:
    """Extract circuit metrics used by the qbalance workflow.

    Args:
        circuit: QuantumCircuit instance to inspect, transform, or execute.

    Returns:
        Dict[str, float] with the computed result.

    Raises:
        None.
    """
    twoq = sum(1 for inst, qargs, _ in circuit.data if len(qargs) == 2)
    meas = sum(
        1 for inst, _, _ in circuit.data if getattr(inst, "name", "") == "measure"
    )
    t = sum(
        1 for inst, _, _ in circuit.data if getattr(inst, "name", "") in ("t", "tdg")
    )
    return {
        "depth": float(circuit.depth()),
        "size": float(circuit.size()),
        "width": float(circuit.num_qubits),
        "two_qubit_ops": float(twoq),
        "measures": float(meas),
        "t_count": float(t),
    }
