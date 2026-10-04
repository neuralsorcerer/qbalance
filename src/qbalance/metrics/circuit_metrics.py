# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict

from qbalance.utils import (
    executed_operation_count,
    operation_depth,
    operation_size,
    two_qubit_operation_count,
)


def extract_circuit_metrics(circuit: Any) -> Dict[str, float]:
    """Return structural metrics of ``circuit``, as floats.

    ``depth`` and ``size`` count operations only: barriers and delays are
    directives, not operations (see :func:`qbalance.utils.operation_depth`).
    ``width`` is the number of qubits, ``two_qubit_ops`` counts operations
    on exactly two qubits, ``measures`` the measurements, and ``t_count``
    the ``t`` and ``tdg`` gates.  Every count is of what one shot executes: a
    control-flow instruction contributes what its blocks run (see
    :func:`qbalance.utils.control_flow_cost`).

    Args:
        circuit: The circuit, typically a compiled one.

    Returns:
        ``depth``, ``size``, ``width``, ``two_qubit_ops``, ``measures`` and
        ``t_count``.
    """
    meas = executed_operation_count(
        circuit, lambda operation, qargs: getattr(operation, "name", "") == "measure"
    )
    t = executed_operation_count(
        circuit,
        lambda operation, qargs: getattr(operation, "name", "") in ("t", "tdg"),
    )
    return {
        "depth": float(operation_depth(circuit)),
        "size": float(operation_size(circuit)),
        "width": float(circuit.num_qubits),
        "two_qubit_ops": float(two_qubit_operation_count(circuit)),
        "measures": float(meas),
        "t_count": float(t),
    }
