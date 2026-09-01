# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, cast

from platformdirs import user_cache_dir

# Concurrent replacements of the same destination can deny one another access
# on Windows.  A fixed set of locks avoids an unbounded path-to-lock registry;
# unrelated paths only serialize in the uncommon event of a hash collision.
_ATOMIC_WRITE_LOCKS = tuple(threading.Lock() for _ in range(64))


def _atomic_write_lock(path: Path) -> threading.Lock:
    normalized_path = os.path.normcase(os.path.abspath(path))
    return _ATOMIC_WRITE_LOCKS[hash(normalized_path) % len(_ATOMIC_WRITE_LOCKS)]


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


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so readers never observe a partial file.

    The payload goes to a temporary file in the destination directory and is
    then renamed into place, which is atomic on every supported platform.  A run
    interrupted mid-write therefore leaves either the previous file or none at
    all, never a truncated one for the next run to choke on.

    Args:
        path: Destination file path.
        data: Bytes to write.

    Returns:
        None. This method updates state or performs side effects only.

    Raises:
        OSError: If the temporary file cannot be written or renamed.
    """
    with _atomic_write_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
            # A reader or another process can still hold the destination open
            # briefly on Windows.  Retry the same completed temporary file so
            # the replacement remains atomic.
            for attempt in range(8):
                try:
                    os.replace(tmp_path, path)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(min(0.001 * (2**attempt), 0.05))
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    """Dump json used by the qbalance workflow.

    Args:
        path: Path value consumed by this routine.
        obj: Obj value consumed by this routine.

    Returns:
        None. This method updates state or performs side effects only.

    Raises:
        OSError: If the file cannot be written.
    """
    payload = json.dumps(obj, indent=2, sort_keys=True)
    atomic_write_bytes(path, payload.encode("utf-8"))


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


def shares_bit(bits: Any, other: Any) -> bool:
    """Return True when two qubit/clbit collections have a bit in common.

    Uses linear identity/equality comparison rather than set intersection:
    bit objects are not guaranteed to be hashable (lightweight stubs commonly
    are not), and instruction bit tuples are short enough that the scan is free.

    Args:
        bits: First collection of qubit or clbit objects.
        other: Second collection of qubit or clbit objects.

    Returns:
        True when at least one bit appears in both collections.

    Raises:
        None.
    """
    for bit in bits:
        for candidate in other:
            if bit is candidate or bit == candidate:
                return True
    return False


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
        # Only lookup/conversion failures fall through to the attribute probes
        # below; a successfully resolved but negative index is an error and must
        # not be masked by the fallback path.
        try:
            index: int | None = int(finder(bit).index)
        except Exception:
            index = None
        if index is not None:
            if index < 0:
                raise ValueError("bit index must be non-negative")
            return index

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
