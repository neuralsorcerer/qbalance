# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict

from qbalance.utils import (
    SCHEDULING_DIRECTIVES,
    instruction_parts,
    operation_depth,
    operation_size,
)


def extract_circuit_metrics(circuit: Any) -> Dict[str, float]:
    """Return structural metrics of ``circuit``, as floats.

    ``depth`` and ``size`` count operations only: barriers and delays are
    directives, not operations (see :func:`qbalance.utils.operation_depth`).
    ``width`` is the number of qubits, ``two_qubit_ops`` counts operations
    on exactly two qubits, ``measures`` the measurements, and ``t_count``
    the ``t`` and ``tdg`` gates.

    Args:
        circuit: The circuit, typically a compiled one.

    Returns:
        ``depth``, ``size``, ``width``, ``two_qubit_ops``, ``measures`` and
        ``t_count``.
    """
    instruction_data = [instruction_parts(entry) for entry in circuit.data]
    twoq = sum(
        1
        for inst, qargs, _ in instruction_data
        if len(qargs) == 2 and getattr(inst, "name", "") not in SCHEDULING_DIRECTIVES
    )
    meas = sum(
        1 for inst, _, _ in instruction_data if getattr(inst, "name", "") == "measure"
    )
    t = sum(
        1
        for inst, _, _ in instruction_data
        if getattr(inst, "name", "") in ("t", "tdg")
    )
    return {
        "depth": float(operation_depth(circuit)),
        "size": float(operation_size(circuit)),
        "width": float(circuit.num_qubits),
        "two_qubit_ops": float(twoq),
        "measures": float(meas),
        "t_count": float(t),
    }
