# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, cast

from platformdirs import user_cache_dir


def validate_integral(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> int:
    """Validate an integer-like option and return it as a builtin ``int``.

    Args:
        name: User-facing option name used in error messages.
        value: Candidate value to validate.
        positive: Require the value to be greater than zero.
        non_negative: Require the value to be zero or greater.

    Returns:
        The validated value as a builtin ``int``.

    Raises:
        ValueError: If the value is bool, non-integral, or violates bounds.
    """
    if positive and non_negative:
        raise ValueError("positive and non_negative are mutually exclusive")
    if isinstance(value, bool) or not isinstance(value, Integral):
        if positive:
            raise ValueError(f"{name} must be a positive integer")
        if non_negative:
            raise ValueError(f"{name} must be a non-negative integer")
        raise ValueError(f"{name} must be an integer")

    value_int = int(value)
    if positive and value_int <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if non_negative and value_int < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value_int


def stable_hash_bytes(data: bytes) -> str:
    """Stable hash bytes used by the qbalance workflow.

    Args:
        data: Data value consumed by this routine.

    Returns:
        str with the computed result.

    Raises:
        None.
    """
    return hashlib.sha256(data).hexdigest()


def stable_hash_str(s: str) -> str:
    """Stable hash str used by the qbalance workflow.

    Args:
        s: S value consumed by this routine.

    Returns:
        str with the computed result.

    Raises:
        None.
    """
    return stable_hash_bytes(s.encode("utf-8"))


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    """Dump json used by the qbalance workflow.

    Args:
        path: Path value consumed by this routine.
        obj: Obj value consumed by this routine.

    Returns:
        None. This method updates state or performs side effects only.

    Raises:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    """Load json from serialized data or persisted storage.

    Args:
        path: Path value consumed by this routine.

    Returns:
        Dict[str, Any] with the computed result.

    Raises:
        None.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return cast(Dict[str, Any], data)


def instruction_parts(entry: Any) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    """Return instruction, qubits, and clbits for Qiskit and tuple-style entries.

    Args:
        entry: Circuit instruction entry from Qiskit or a lightweight tuple-style stub.

    Returns:
        Tuple containing the operation object, qubit tuple, and clbit tuple.

    Raises:
        ValueError: Raised when the entry cannot be interpreted as a circuit instruction.
    """
    if hasattr(entry, "operation"):
        return entry.operation, tuple(entry.qubits), tuple(entry.clbits)

    try:
        inst, qargs, cargs = entry
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "instruction entry must provide operation, qubits, and clbits"
        ) from exc
    return inst, tuple(qargs), tuple(cargs)


def bit_index(circuit: Any, bit: Any) -> int:
    """Return a stable bit index for Qiskit bit objects and lightweight stubs.

    Args:
        circuit: Circuit that owns ``bit`` when available.
        bit: Qubit or clbit object whose index should be resolved.

    Returns:
        Zero-based bit index.

    Raises:
        AttributeError: Raised when no supported index representation is present.
        ValueError: Raised when the resolved index is negative.
    """
    finder = getattr(circuit, "find_bit", None)
    if callable(finder):
        try:
            index = int(finder(bit).index)
            if index < 0:
                raise ValueError("bit index must be non-negative")
            return index
        except Exception:
            pass

    for attr in ("index", "_index"):
        raw_index = getattr(bit, attr, None)
        if raw_index is None:
            continue
        index_int = int(raw_index)
        if index_int < 0:
            raise ValueError("bit index must be non-negative")
        return index_int

    raise AttributeError("Unable to determine bit index")


def default_cache_dir(app: str = "qbalance") -> Path:
    """Return the default cache dir configuration used by qbalance.

    Args:
        app (default: 'qbalance'): App value consumed by this routine.

    Returns:
        Path with the computed result.

    Raises:
        None.
    """
    cache_path = Path(user_cache_dir(app))
    if cache_path.name != app:
        cache_path = cache_path / app
    return cache_path
