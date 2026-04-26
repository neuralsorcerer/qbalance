# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qbalance.errors import OptionalDependencyError
from qbalance.logging import get_logger

log = get_logger(__name__)


def apply_pauli_twirling(
    circuit: Any, num_twirls: int = 1, seed: Optional[int] = None, target: Any = None
) -> List[Any]:
    """Apply pauli twirling used by the qbalance workflow.

    Args:
        circuit: QuantumCircuit instance to inspect, transform, or execute.
        num_twirls (default: 1): Num twirls value consumed by this routine.
        seed (default: None): Seed used for deterministic randomization.
        target (default: None): Target value consumed by this routine.

    Returns:
        List[Any] with the computed result.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    try:
        from qiskit.circuit import pauli_twirl_2q_gates
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError("qiskit is required for pauli twirling") from e

    out = pauli_twirl_2q_gates(circuit, seed=seed, num_twirls=num_twirls, target=target)
    if isinstance(out, list):
        return out
    return [out]


def _dd_sequence(name: str):
    """Internal helper that dd sequence.

    Args:
        name: Name/identifier for a circuit, dataset, or lookup record.

    Returns:
        Computed value produced by this routine.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    try:
        from qiskit.circuit.library import XGate, YGate
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "qiskit is required for dynamical decoupling"
        ) from e

    name = name.upper()
    if name == "XY4":
        return [XGate(), YGate(), XGate(), YGate()]
    if name == "XX":
        return [XGate(), XGate()]
    if name == "YY":
        return [YGate(), YGate()]
    # default
    return [XGate(), YGate(), XGate(), YGate()]


def build_dd_pass_manager(backend: Any, sequence: str = "XY4") -> Any:
    """Build dd pass manager from the provided configuration parameters.

    Args:
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        sequence (default: 'XY4'): Sequence value consumed by this routine.

    Returns:
        Any with the computed result.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    try:
        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import (
            ALAPScheduleAnalysis,
            BasisTranslator,
            PadDynamicalDecoupling,
            Unroll3qOrMore,
        )
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "qiskit is required for dynamical decoupling"
        ) from e

    dd_seq = _dd_sequence(sequence)

    # Qiskit recommends using ALAPScheduleAnalysis + BasisTranslator + PadDynamicalDecoupling
    target = getattr(backend, "target", None)
    if target is None:
        # older backends
        basis = getattr(backend.configuration(), "basis_gates", None) or []
    else:
        basis = list(target.operation_names)

    pm = PassManager()
    pm.append(Unroll3qOrMore())
    pm.append(ALAPScheduleAnalysis(backend))
    pm.append(BasisTranslator(SessionEquivalenceLibrary, basis))
    pm.append(PadDynamicalDecoupling(backend, dd_seq))
    return pm


def apply_measurement_twirling(
    circuit: Any, seed: Optional[int] = None
) -> Tuple[Any, Dict[int, int]]:
    """Apply measurement twirling used by the qbalance workflow.

    Args:
        circuit: QuantumCircuit instance to inspect, transform, or execute.
        seed (default: None): Seed used for deterministic randomization.

    Returns:
        Tuple[Any, Dict[int, int]] with the computed result.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    try:
        import_module("qiskit")
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "qiskit is required for measurement twirling"
        ) from e

    rng = np.random.default_rng(seed)
    qc = circuit.copy()
    flip_map: Dict[int, int] = {}

    # scan measurements; if mid-circuit measurement, we flip before each measurement
    for inst, qargs, cargs in list(qc.data):
        if (
            getattr(inst, "name", "") == "measure"
            and len(qargs) == 1
            and len(cargs) == 1
        ):
            cb = cargs[0].index
            flip = int(rng.integers(0, 2))
            if flip == 1:
                qb = qargs[0].index
                qc.x(qb)
                flip_map[cb] = flip_map.get(cb, 0) ^ 1
    return qc, flip_map


def apply_measurement_untwirl_counts(
    counts: Dict[str, int], flip_map: Dict[int, int]
) -> Dict[str, int]:
    """Apply measurement untwirl counts used by the qbalance workflow.

    Args:
        counts: Counts value consumed by this routine.
        flip_map: Flip map value consumed by this routine.

    Returns:
        Dict[str, int] with the computed result.

    Raises:
        None.
    """
    if not flip_map:
        return counts

    out: Dict[str, int] = {}
    for bitstr, n in counts.items():
        b = list(bitstr)
        # qiskit bitstrings are little-endian in many contexts; here we interpret index from right
        for cb, flip in flip_map.items():
            if flip:
                pos = len(b) - 1 - cb
                if 0 <= pos < len(b):
                    b[pos] = "1" if b[pos] == "0" else "0"
        out["".join(b)] = out.get("".join(b), 0) + n
    return out
