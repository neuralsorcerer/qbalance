# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from qbalance.logging import get_logger

log = get_logger(__name__)


def _safe_get_qubit_readout_error(backend: Any, q: int) -> Optional[float]:
    # Best-effort across backend versions

    """Safely read backend calibration data and return a conservative fallback when unavailable.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        q: Q value consumed by this routine.

    Returns:
        Optional[float] with the computed result.

    Raises:
        None.
    """
    try:
        props = backend.properties()
        if props is None:
            return None
        # Qiskit properties: readout_error in qubit properties
        qprops = props.qubits[q]
        for item in qprops:
            if getattr(item, "name", None) == "readout_error":
                return float(item.value)
    except Exception:
        return None
    return None


def _safe_get_t1(backend: Any, q: int) -> Optional[float]:
    """Safely read backend calibration data and return a conservative fallback when unavailable.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        q: Q value consumed by this routine.

    Returns:
        Optional[float] with the computed result.

    Raises:
        None.
    """
    try:
        props = backend.properties()
        if props is None:
            return None
        qprops = props.qubits[q]
        for item in qprops:
            if getattr(item, "name", None) == "T1":
                return float(item.value)
    except Exception:
        return None
    return None


def _safe_get_t2(backend: Any, q: int) -> Optional[float]:
    """Safely read backend calibration data and return a conservative fallback when unavailable.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        q: Q value consumed by this routine.

    Returns:
        Optional[float] with the computed result.

    Raises:
        None.
    """
    try:
        props = backend.properties()
        if props is None:
            return None
        qprops = props.qubits[q]
        for item in qprops:
            if getattr(item, "name", None) == "T2":
                return float(item.value)
    except Exception:
        return None
    return None


def _safe_get_2q_error(backend: Any, gate: str, q0: int, q1: int) -> Optional[float]:
    """Safely read backend calibration data and return a conservative fallback when unavailable.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        gate: Gate value consumed by this routine.
        q0: Q0 value consumed by this routine.
        q1: Q1 value consumed by this routine.

    Returns:
        Optional[float] with the computed result.

    Raises:
        None.
    """
    try:
        props = backend.properties()
        if props is None:
            return None
        g = props.gate_error(gate, [q0, q1])
        if g is None:
            return None
        return float(g)
    except Exception:
        return None


def estimate_circuit_error(backend: Any, circuit: Any) -> float:
    """Estimate circuit error used by the qbalance workflow.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        circuit: QuantumCircuit instance to inspect, transform, or execute.

    Returns:
        float with the computed result.

    Raises:
        None.
    """
    # 1 - Π(1-e_i) approximation
    total_survival = 1.0
    try:
        for inst, qargs, _ in circuit.data:
            name = getattr(inst, "name", "").lower()
            if len(qargs) == 2:
                q0 = qargs[0].index
                q1 = qargs[1].index
                e = _safe_get_2q_error(backend, name, q0, q1)
                if e is None:
                    e = 0.01
                total_survival *= 1.0 - float(e)
            elif name == "measure" and len(qargs) == 1:
                q0 = qargs[0].index
                e = _safe_get_qubit_readout_error(backend, q0)
                if e is None:
                    e = 0.02
                total_survival *= 1.0 - float(e)
            else:
                # 1q gate errors: best-effort use 0.001
                total_survival *= 1.0 - 0.001
    except Exception:
        return 1.0
    return float(max(0.0, 1.0 - total_survival))


def noise_aware_initial_layout(backend: Any, circuit: Any) -> Optional[Any]:
    """Noise aware initial layout used by the qbalance workflow.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        circuit: QuantumCircuit instance to inspect, transform, or execute.

    Returns:
        Optional[Any] with the computed result.

    Raises:
        None.
    """
    try:
        from qiskit.transpiler import Layout
    except Exception:  # pragma: no cover
        return None

    n = getattr(circuit, "num_qubits", None)
    if n is None:
        return None

    # logical activity: interaction graph degree
    deg = np.zeros(n, dtype=float)
    for inst, qargs, _ in circuit.data:
        if len(qargs) == 2:
            a = qargs[0].index
            b = qargs[1].index
            deg[a] += 1
            deg[b] += 1
    logical_order = list(np.argsort(-deg))  # most active first

    # physical quality: lower readout error; higher T1/T2
    phys_n = getattr(backend, "num_qubits", None)
    if phys_n is None:
        try:
            phys_n = backend.configuration().num_qubits
        except Exception:
            return None

    qualities = []
    for q in range(phys_n):
        ro = _safe_get_qubit_readout_error(backend, q)
        t1 = _safe_get_t1(backend, q)
        t2 = _safe_get_t2(backend, q)
        # normalize best-effort; missing => neutral
        ro = ro if ro is not None else 0.02
        t1 = t1 if t1 is not None else 50e3
        t2 = t2 if t2 is not None else 50e3
        # quality higher is better
        qscore = (1.0 - ro) + 0.00001 * (t1 + t2)
        qualities.append(qscore)
    physical_order = list(np.argsort(-np.array(qualities)))  # best first

    if len(physical_order) < n:
        return None

    layout = Layout()
    for lq, pq in zip(logical_order, physical_order[:n]):
        layout[circuit.qubits[lq]] = (
            backend.qubits[pq] if hasattr(backend, "qubits") else pq
        )
    return layout
