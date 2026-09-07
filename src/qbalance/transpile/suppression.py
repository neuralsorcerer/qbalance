# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from qbalance.errors import OptionalDependencyError
from qbalance.logging import get_logger
from qbalance.utils import bit_index, instruction_parts, shares_bit

log = get_logger(__name__)

# Operations that cannot observe a measurement twirl inserted before them:
# they neither change the qubit state nor read the classical bit.
_TWIRL_TRANSPARENT_OPS = {"barrier", "delay"}


def normalize_measurement_flip_map(flip_map: Any) -> Dict[int, int]:
    """Return a sanitized clbit-index to XOR-flip map.

    Measurement twirling records flips as integer clbit indices, but the
    metadata can be serialized through JSON (which stringifies keys) or built by
    user code.  Keep only non-negative integer indices with odd/truthy flip
    values and normalize every retained flip to ``1`` because correction is an
    XOR operation.
    """
    if not isinstance(flip_map, Mapping):
        return {}

    normalized: Dict[int, int] = {}
    for raw_cb, raw_flip in flip_map.items():
        try:
            cb = int(raw_cb)
            flip = int(raw_flip)
        except (TypeError, ValueError, OverflowError):
            continue
        if cb >= 0 and flip % 2:
            normalized[cb] = 1
    return normalized


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


def _dd_sequence(name: str) -> List[Any]:
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


def _operation_names(target: Any) -> set[str]:
    """Return normalized operation names exposed by a transpiler target."""
    if target is None:
        return set()

    raw_names = getattr(target, "operation_names", ())
    if callable(raw_names):
        raw_names = raw_names()

    names: set[str] = set()
    for raw_name in raw_names or ():
        name = str(raw_name).strip().lower()
        if name:
            names.add(name)
    return names


def _backend_basis_gates(backend: Any) -> set[str]:
    """Return normalized basis-gate names from BackendV1-style objects."""
    configuration = getattr(backend, "configuration", None)
    if not callable(configuration):
        return set()

    try:
        raw_basis = getattr(configuration(), "basis_gates", ())
    except Exception:
        return set()

    names: set[str] = set()
    for raw_name in raw_basis or ():
        name = str(raw_name).strip().lower()
        if name:
            names.add(name)
    return names


def _gate_name(gate: Any) -> str:
    """Return a normalized gate name for Qiskit gates and lightweight stubs."""
    name = getattr(gate, "name", None)
    if name is None:
        name = gate.__class__.__name__
        if name.endswith("Gate"):
            name = name[:-4]
    return str(name).strip().lower()


def _compatible_dd_sequence(dd_seq: List[Any], supported_ops: set[str]) -> List[Any]:
    """Return a DD sequence that can be represented by the target basis."""
    if not supported_ops:
        return dd_seq

    if all(_gate_name(gate) in supported_ops for gate in dd_seq):
        return dd_seq

    # Many BackendV2 targets, including GenericBackendV2 and IBM-style bases,
    # expose X but not Y as a native scheduled instruction.  PadDynamicalDecoupling
    # validates the DD sequence directly against the target, so choose a fully
    # supported echo sequence when the requested sequence is unavailable.
    for fallback_name in ("XX", "YY"):
        fallback = _dd_sequence(fallback_name)
        if all(_gate_name(gate) in supported_ops for gate in fallback):
            return fallback

    return dd_seq


def _backend_instruction_durations(backend: Any) -> Any:
    """Return instruction durations from BackendV1-style objects when available."""
    durations = getattr(backend, "instruction_durations", None)
    if callable(durations):
        try:
            return durations()
        except TypeError:
            return durations
    return durations


def _make_pass(factory: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Instantiate a Qiskit pass, falling back for older positional APIs."""
    try:
        return factory(**kwargs)
    except TypeError:
        return factory(*args)


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

    # Qiskit recommends scheduling first, then padding idle intervals with the
    # requested DD sequence.  Modern Qiskit scheduler/padding passes accept a
    # Target (or InstructionDurations), not a Backend object.  Passing the
    # Backend positionally makes the scheduler treat it as an iterable duration
    # table and fails for BackendV2 implementations such as GenericBackendV2.
    target = getattr(backend, "target", None)
    if target is None:
        # Older backends may only expose a basis-gate list.  Keep translator
        # behavior when a basis is available; otherwise skip translation rather
        # than constructing a guaranteed-invalid empty-basis translator.
        supported_ops = _backend_basis_gates(backend)
        basis = sorted(supported_ops)
        dd_seq = _compatible_dd_sequence(dd_seq, supported_ops)
        durations = _backend_instruction_durations(backend)
        schedule_pass = _make_pass(ALAPScheduleAnalysis, durations, durations=durations)
        dd_pass = _make_pass(
            PadDynamicalDecoupling,
            durations,
            dd_seq,
            durations=durations,
            dd_sequence=dd_seq,
        )
    else:
        supported_ops = _operation_names(target)
        basis = sorted(supported_ops)
        dd_seq = _compatible_dd_sequence(dd_seq, supported_ops)
        schedule_pass = _make_pass(ALAPScheduleAnalysis, target, target=target)
        dd_pass = _make_pass(
            PadDynamicalDecoupling, target, dd_seq, target=target, dd_sequence=dd_seq
        )

    pm = PassManager()
    pm.append(Unroll3qOrMore())
    if basis:
        pm.append(BasisTranslator(SessionEquivalenceLibrary, basis))
    pm.append(schedule_pass)
    pm.append(dd_pass)
    return pm


def _is_terminal_measurement(
    data: list[Any],
    index: int,
    qargs: tuple[Any, ...] = (),
    cargs: tuple[Any, ...] = (),
) -> bool:
    """Return true when no later operation can observe the inserted flip.

    A measurement twirl inserts an ``X`` immediately before the measurement and
    undoes it by flipping the recorded classical bit.  That rewrite is only
    equivalent when nothing after the measurement can see either half of it:

    * no later instruction may act on the measured qubit -- a second
      measurement of the same qubit would read the flipped post-measurement
      state while the flip map only records one flip per classical bit, and any
      later gate would act on a conjugated state; and
    * no later instruction may write the classical bit that carries the
      correction.

    Barriers and delays neither change the state nor read the bit, so they stay
    transparent.
    """
    for later_entry in data[index + 1 :]:
        later_inst, later_qargs, later_cargs = instruction_parts(later_entry)
        if getattr(later_inst, "name", "") in _TWIRL_TRANSPARENT_OPS:
            continue
        if shares_bit(qargs, later_qargs) or shares_bit(cargs, later_cargs):
            return False
    return True


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
    flip_map: Dict[int, int] = {}

    # Measurement twirling is only semantics-preserving for the terminal
    # measurement block when correction is represented as a final count-key
    # permutation. Mid-circuit measurements may feed classical control flow or
    # define the post-measurement quantum state, so this lightweight transform
    # intentionally leaves them unchanged.
    copy_empty_like = getattr(circuit, "copy_empty_like", None)
    if callable(copy_empty_like):
        qc = copy_empty_like()
        data = list(circuit.data)
        for index, entry in enumerate(data):
            inst, qargs, cargs = instruction_parts(entry)
            should_twirl = (
                getattr(inst, "name", "") == "measure"
                and len(qargs) == 1
                and len(cargs) == 1
                and _is_terminal_measurement(data, index, qargs, cargs)
            )
            if should_twirl:
                cb = bit_index(circuit, cargs[0])
                flip = int(rng.integers(0, 2))
                if flip == 1:
                    qc.x(qargs[0])
                    flip_map[cb] = flip_map.get(cb, 0) ^ 1
            qc.append(inst, qargs, cargs)
        return qc, flip_map

    # Lightweight circuit stubs used by tests may not support reconstruction.
    # Preserve compatibility by falling back to in-place-style copy behavior.
    qc = circuit.copy()
    data = list(qc.data)
    for index, entry in enumerate(data):
        inst, qargs, cargs = instruction_parts(entry)
        if (
            getattr(inst, "name", "") == "measure"
            and len(qargs) == 1
            and len(cargs) == 1
            and _is_terminal_measurement(data, index, qargs, cargs)
        ):
            cb = bit_index(qc, cargs[0])
            flip = int(rng.integers(0, 2))
            if flip == 1:
                qb = bit_index(qc, qargs[0])
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

    normalized_flip_map = normalize_measurement_flip_map(flip_map)
    if not normalized_flip_map:
        return counts

    out: Dict[str, int] = {}
    for bitstr, n in counts.items():
        b = list(bitstr)
        # Qiskit renders multiple classical registers with spaces in count keys,
        # while classical-bit indices are over the flattened register.  Count only
        # actual binary digits when mapping flat little-endian clbit indices back
        # to display positions so separators are preserved and never flipped.
        bit_positions = [idx for idx, char in enumerate(b) if char in {"0", "1"}]
        for cb, flip in normalized_flip_map.items():
            if flip:
                bit_pos_idx = len(bit_positions) - 1 - cb
                if 0 <= bit_pos_idx < len(bit_positions):
                    pos = bit_positions[bit_pos_idx]
                    b[pos] = "1" if b[pos] == "0" else "0"
        corrected = "".join(b)
        out[corrected] = out.get(corrected, 0) + n
    return out
