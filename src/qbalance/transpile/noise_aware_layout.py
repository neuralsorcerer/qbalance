# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from qbalance.logging import get_logger
from qbalance.utils import bit_index, instruction_parts

log = get_logger(__name__)


_DIRECTIVE_NAMES = {"barrier", "delay"}


def _backend_target(backend: Any) -> Any:
    """Return the BackendV2 transpiler target when one is exposed."""
    return getattr(backend, "target", None)


def _target_instruction_error(
    target: Any, name: str, qubits: tuple[int, ...]
) -> Optional[float]:
    """Read an instruction error rate from a BackendV2 target, best effort."""
    if target is None:
        return None
    try:
        properties_map = target[name]
        props = properties_map.get(qubits)
        if props is None:
            return None
        return _coerce_error_rate(getattr(props, "error", None))
    except Exception:
        return None


def _target_qubit_property(target: Any, q: int, attr: str) -> Optional[float]:
    """Read a per-qubit property (e.g. t1/t2) from a BackendV2 target."""
    if target is None:
        return None
    try:
        qubit_properties = getattr(target, "qubit_properties", None)
        if qubit_properties is None:
            return None
        return _coerce_finite_float(getattr(qubit_properties[q], attr, None))
    except Exception:
        return None


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
    # BackendV2: readout error lives on the target's measure instruction.
    error = _target_instruction_error(_backend_target(backend), "measure", (q,))
    if error is not None:
        return error

    try:
        props = backend.properties()
        if props is None:
            return None
        # Qiskit properties: readout_error in qubit properties
        qprops = props.qubits[q]
        for item in qprops:
            if getattr(item, "name", None) == "readout_error":
                return _coerce_error_rate(item.value)
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
    value = _target_qubit_property(_backend_target(backend), q, "t1")
    if value is not None:
        return value

    try:
        props = backend.properties()
        if props is None:
            return None
        qprops = props.qubits[q]
        for item in qprops:
            if getattr(item, "name", None) == "T1":
                return _coerce_finite_float(item.value)
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
    value = _target_qubit_property(_backend_target(backend), q, "t2")
    if value is not None:
        return value

    try:
        props = backend.properties()
        if props is None:
            return None
        qprops = props.qubits[q]
        for item in qprops:
            if getattr(item, "name", None) == "T2":
                return _coerce_finite_float(item.value)
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
    # BackendV2: gate errors live on the target, keyed by qubit tuple.  Try
    # the reversed direction too because some targets only list one direction.
    target = _backend_target(backend)
    error = _target_instruction_error(target, gate, (q0, q1))
    if error is None:
        error = _target_instruction_error(target, gate, (q1, q0))
    if error is not None:
        return error

    try:
        props = backend.properties()
        if props is None:
            return None
        g = props.gate_error(gate, [q0, q1])
        if g is None:
            return None
        return _coerce_error_rate(g)
    except Exception:
        return None


def _coerce_finite_float(value: Any) -> Optional[float]:
    """Return a finite float, or None when the value is not usable."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _coerce_error_rate(value: Any) -> Optional[float]:
    """Return a finite probability-like error rate clipped to [0, 1]."""
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(rate):
        return None
    return float(np.clip(rate, 0.0, 1.0))


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
        target = _backend_target(backend)
        for entry in circuit.data:
            inst, qargs, _ = instruction_parts(entry)
            name = getattr(inst, "name", "").lower()
            if name in _DIRECTIVE_NAMES:
                # Barriers/delays are scheduling directives, not error channels;
                # a two-qubit barrier must not be billed as a two-qubit gate.
                continue
            if len(qargs) == 2:
                q0 = bit_index(circuit, qargs[0])
                q1 = bit_index(circuit, qargs[1])
                e = _safe_get_2q_error(backend, name, q0, q1)
                if e is None:
                    e = 0.01
                total_survival *= 1.0 - e
            elif name == "measure" and len(qargs) == 1:
                q0 = bit_index(circuit, qargs[0])
                e = _safe_get_qubit_readout_error(backend, q0)
                if e is None:
                    e = 0.02
                total_survival *= 1.0 - e
            elif len(qargs) == 1:
                # 1q gate errors: prefer target calibration, else 0.001
                q0 = bit_index(circuit, qargs[0])
                e = _target_instruction_error(target, name, (q0,))
                if e is None:
                    e = 0.001
                total_survival *= 1.0 - e
            elif len(qargs) > 0:
                # multi-qubit (>2) operations: conservative default
                total_survival *= 1.0 - 0.001
    except Exception:
        return 1.0
    if not np.isfinite(total_survival):
        return 1.0
    return float(np.clip(1.0 - total_survival, 0.0, 1.0))


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

    # logical activity: interaction graph degree (directives are not gates)
    deg = np.zeros(n, dtype=float)
    for entry in circuit.data:
        inst, qargs, _ = instruction_parts(entry)
        if getattr(inst, "name", "").lower() in _DIRECTIVE_NAMES:
            continue
        if len(qargs) == 2:
            a = bit_index(circuit, qargs[0])
            b = bit_index(circuit, qargs[1])
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
