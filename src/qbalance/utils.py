# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import errno
import hashlib
import json
import os
import secrets
import shutil
import threading
import time
from contextlib import contextmanager, suppress
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, cast

import numpy as np
from platformdirs import user_cache_dir

# Concurrent replacements of the same destination can deny one another access
# on Windows.  A fixed set of locks avoids an unbounded path-to-lock registry;
# unrelated paths only serialize in the uncommon event of a hash collision.
_ATOMIC_WRITE_LOCKS = tuple(threading.Lock() for _ in range(64))


def _atomic_write_lock(path: Path) -> threading.Lock:
    normalized_path = os.path.normcase(os.path.abspath(path))
    return _ATOMIC_WRITE_LOCKS[hash(normalized_path) % len(_ATOMIC_WRITE_LOCKS)]


# Temporary entries are staged under random names beside their destination and
# renamed into place.  tempfile.mkstemp and mkdtemp would pick the names too,
# but they create owner-only entries (0600 and 0700) meant to stay private; the
# rename keeps that mode, so every dataset, workload, report and cache entry
# came out unreadable to anyone else whatever the user's umask.  Creating them
# with the ordinary 0o666 and 0o777 lets the kernel apply the umask, exactly as
# a plain open() or mkdir() of the destination would.
_TEMPORARY_NAME_ATTEMPTS = 100
_TEMPORARY_FILE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    # Without O_BINARY, Windows writes through the descriptor in text mode.
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _temporary_sibling(directory: Path, prefix: str, suffix: str) -> Path:
    """Return a random absolute path in ``directory``.

    Absolute like tempfile's, so a caller changing the working directory while
    it writes the staged entry still commits the right one.
    """
    name = f"{prefix}{secrets.token_hex(8)}{suffix}"
    return Path(os.path.abspath(directory)) / name


def _create_temporary_file(
    directory: Path, prefix: str, suffix: str
) -> Tuple[int, Path]:
    """Create a new file in ``directory`` with umask-governed permissions.

    Returns:
        The open, write-only descriptor and the file's path.

    Raises:
        FileExistsError: If no unused name was found.
        OSError: If the file cannot be created.
    """
    for _ in range(_TEMPORARY_NAME_ATTEMPTS):
        candidate = _temporary_sibling(directory, prefix, suffix)
        try:
            return os.open(candidate, _TEMPORARY_FILE_FLAGS, 0o666), candidate
        except FileExistsError:
            continue
    raise FileExistsError(
        errno.EEXIST, "No unused temporary file name found", str(directory)
    )


def _create_temporary_directory(directory: Path, prefix: str) -> Path:
    """Create a new directory in ``directory`` with umask-governed permissions.

    Raises:
        FileExistsError: If no unused name was found.
        OSError: If the directory cannot be created.
    """
    for _ in range(_TEMPORARY_NAME_ATTEMPTS):
        candidate = _temporary_sibling(directory, prefix, "")
        try:
            os.mkdir(candidate, 0o777)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(
        errno.EEXIST, "No unused temporary directory name found", str(directory)
    )


def backend_display_name(backend: Any) -> str:
    """Return a readable name for a backend.

    ``BackendV2`` exposes ``name`` as a string attribute, while BackendV1-style
    objects expose it as a method.  A bare ``getattr`` therefore yields a bound
    method -- complete with a memory address -- for precisely the backends that
    most often need naming in a diagnostic.

    Args:
        backend: Backend object (or backend-like handle) to name.

    Returns:
        The backend's name, falling back to its class name.

    Raises:
        None.
    """
    name = getattr(backend, "name", None)
    if callable(name):
        name = name()
    return str(name or backend.__class__.__name__)


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


def validate_flag(name: str, value: Any) -> bool:
    """Validate a boolean option and return it as a builtin ``bool``.

    Truthiness is not enough: ``execute="no"`` would otherwise run every
    circuit.  NumPy booleans are accepted.

    Args:
        name: User-facing option name used in error messages.
        value: Candidate value to validate.

    Returns:
        The value as a builtin ``bool``.

    Raises:
        ValueError: If the value is not a boolean.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def stable_hash_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def stable_hash_str(s: str) -> str:
    """Return the SHA-256 hex digest of ``s`` encoded as UTF-8.

    Unlike ``hash(s)``, it is the same in every process.
    """
    return stable_hash_bytes(s.encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so readers never observe a partial file.

    The payload goes to a temporary file in the destination directory and is
    then renamed into place, which is atomic on every supported platform.  A run
    interrupted mid-write therefore leaves either the previous file or none at
    all, never a truncated one for the next run to choke on.  The file gets the
    permissions the umask gives any newly created file.

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
        handle, tmp_path = _create_temporary_file(
            path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
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


@contextmanager
def replacing_directory(target: Path) -> Iterator[Path]:
    """Yield an empty staging directory that replaces ``target`` on success.

    The staging directory is a sibling of ``target``, so the final rename
    stays on one filesystem.  When the ``with`` block raises, ``target`` is
    left exactly as it was and the staging directory is removed.  Otherwise
    whatever is at ``target`` is moved aside, the staging directory renamed
    into its place, and the old entry deleted; should that rename fail, the
    old entry is put back.  Callers decide beforehand whether ``target`` may
    be replaced at all.  The new directory gets the permissions the umask
    gives any newly created directory.

    Args:
        target: Directory to create or replace.

    Yields:
        The staging directory to write the new contents into.

    Raises:
        OSError: If the staging directory cannot be created or swapped in.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = _create_temporary_directory(target.parent, prefix=f".{target.name}.tmp-")
    committed = False
    try:
        yield staging
        backup = Path(f"{staging}.backup")
        backed_up = False
        try:
            if os.path.lexists(target):
                os.replace(target, backup)
                backed_up = True
            os.replace(staging, target)
            committed = True
        except BaseException:
            if backed_up and not os.path.lexists(target):
                os.replace(backup, target)
            raise
        # The new contents are in place: failing to delete the old ones must
        # not report the completed replacement as a failure.
        if backed_up:
            if backup.is_dir() and not backup.is_symlink():
                shutil.rmtree(backup, ignore_errors=True)
            else:
                with suppress(OSError):
                    backup.unlink()
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)


def strict_json_value(value: Any) -> Any:
    """Return ``value`` with every non-finite float replaced by ``None``.

    ``json.dumps`` writes NaN and infinity as the bare tokens ``NaN`` and
    ``Infinity``, which are not JSON: strict parsers (JavaScript's
    ``JSON.parse``, most other languages) reject the whole file, and ``jq``
    silently turns ``Infinity`` into ``1.8e308``.  ``null`` is what
    JavaScript's ``JSON.stringify`` writes for them.  Mappings, lists and
    tuples are converted recursively (tuples become lists, as in JSON);
    mapping keys are kept.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    return value


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write ``obj`` to ``path`` as indented JSON with sorted keys, atomically.

    Raises:
        TypeError: If ``obj`` is not JSON-serializable.
        OSError: If the file cannot be written.
    """
    payload = json.dumps(obj, indent=2, sort_keys=True)
    atomic_write_bytes(path, payload.encode("utf-8"))


def load_json(path: Path) -> Dict[str, Any]:
    """Read and decode the JSON file at ``path``.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If it is not valid JSON (``json.JSONDecodeError``).
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
        ValueError: If the entry cannot be interpreted as a circuit instruction.
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


# Instructions that neither change the quantum state nor read a classical bit.
SCHEDULING_DIRECTIVES = frozenset({"barrier", "delay"})


def is_terminal_measurement(
    data: list[tuple[Any, tuple[Any, ...], tuple[Any, ...]]], index: int
) -> bool:
    """Return True when nothing after ``data[index]`` can observe that measurement.

    A measurement is terminal only if no later instruction acts on the measured
    qubit or touches the classical bit it wrote; scheduling directives are
    transparent.

    Args:
        data: Instruction parts ``(operation, qubits, clbits)`` in circuit order.
        index: Position of the measurement in ``data``.

    Returns:
        True when the measurement is terminal.

    Raises:
        None.
    """
    _, qargs, cargs = data[index]
    for later_inst, later_qargs, later_cargs in data[index + 1 :]:
        if getattr(later_inst, "name", "") in SCHEDULING_DIRECTIVES:
            continue
        if shares_bit(qargs, later_qargs) or shares_bit(cargs, later_cargs):
            return False
    return True


def _counts_as_operation(instruction: Any) -> bool:
    """Whether ``instruction`` adds to depth and size (it is not a directive)."""
    operation = getattr(instruction, "operation", None)
    if getattr(operation, "_directive", False):
        return False
    return getattr(operation, "name", "") not in SCHEDULING_DIRECTIVES


def operation_depth(circuit: Any) -> int:
    """Return the depth of ``circuit`` counting operations, not directives.

    Qiskit's default depth already skips barriers but counts ``delay``.  A
    scheduled circuit (dynamical decoupling pads every idle window with
    delays) would then be several layers deeper than the same circuit left
    unscheduled, although it runs for exactly as long.  Directives still
    synchronize the wires they touch, as Qiskit's own barrier handling does.
    """
    return int(circuit.depth(filter_function=_counts_as_operation))


def operation_size(circuit: Any) -> int:
    """Return the number of operations in ``circuit``, excluding directives."""
    return int(circuit.size(filter_function=_counts_as_operation))


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


def measured_qubits_by_clbit(circuit: Any) -> list[int]:
    """Return the qubit measured into each clbit, ordered by clbit index.

    Measurement mitigation needs the qubit that feeds every classical bit of
    the observed bitstrings.  Compiled circuits are usually wider than the
    number of measured bits, so ``range(num_qubits)`` mismatches the counts
    keys on any backend wider than the logical circuit.  Falls back to
    ``range(num_qubits)`` when no per-clbit mapping can be recovered.

    Args:
        circuit: Compiled circuit whose measurements to inspect.

    Returns:
        Physical qubit index per measured classical bit.

    Raises:
        ValueError: If a control-flow block of ``circuit`` measures (see
            :func:`_reject_conditional_measurements`).
    """
    _reject_conditional_measurements(circuit)
    mapping = _measurement_map(circuit)
    if not mapping:
        return list(range(int(getattr(circuit, "num_qubits", 0) or 0)))
    return [qubit for _, qubit in sorted(mapping.items())]


def measured_clbits(circuit: Any) -> Optional[list[int]]:
    """Return the classical bits measurements write, in increasing order.

    The ``i``-th entry is the classical bit :func:`measured_qubits_by_clbit`'s
    ``i``-th qubit feeds.  A circuit can carry classical bits no measurement
    writes (``QuantumCircuit(3, 3)`` measuring two qubits); those stay ``0``
    in every count key, and measurement mitigation has to skip them.

    Returns:
        The measured classical bit indices, or ``None`` when no per-bit
        mapping can be recovered (matching the fallback of
        :func:`measured_qubits_by_clbit`).

    Raises:
        ValueError: If a control-flow block of ``circuit`` measures (see
            :func:`_reject_conditional_measurements`).
    """
    _reject_conditional_measurements(circuit)
    mapping = _measurement_map(circuit)
    return sorted(mapping) if mapping else None


def _block_measures(data: Any) -> bool:
    """Whether ``data`` holds a measurement, inside nested blocks too."""
    for entry in data:
        operation = instruction_parts(entry)[0]
        if getattr(operation, "name", "") == "measure":
            return True
        for block in getattr(operation, "blocks", None) or ():
            if _block_measures(getattr(block, "data", None) or ()):
                return True
    return False


def _reject_conditional_measurements(circuit: Any) -> None:
    """Refuse a circuit whose control-flow blocks measure.

    The per-bit map only sees top-level measurements.  A measurement inside a
    control-flow block (``if_else``, ``switch``, a loop, ``box``) acts on the
    block's own bits, may run on only some shots or several times per shot,
    and on the shots an ``if_else`` branch skips leaves its classical bit
    holding whatever it held -- no readout of any one qubit.  Mitigating with
    the top-level map projected such a bit out and restored it as ``0``: a
    circuit whose counts were ``00`` and ``11`` came back as ``00`` and
    ``01``.  Readout mitigation must not be attempted on such a circuit.

    Raises:
        ValueError: If a control-flow block of ``circuit`` measures.
    """
    try:
        conditional = any(
            _block_measures(getattr(block, "data", None) or ())
            for entry in list(getattr(circuit, "data", None) or [])
            for block in getattr(instruction_parts(entry)[0], "blocks", None) or ()
        )
    except ValueError:
        # Not a readable circuit: there is no measurement map to protect.
        return
    if conditional:
        raise ValueError(
            "Measurement mitigation needs every measurement at the top level of "
            "the circuit: a measurement inside a control-flow block may run on "
            "only some shots, or several times, so its classical bit cannot be "
            "attributed to the readout of one qubit."
        )


def _measurement_map(circuit: Any) -> Dict[int, int]:
    """Map each measured classical bit to the qubit last measured into it."""
    mapping: Dict[int, int] = {}
    try:
        for entry in list(getattr(circuit, "data", None) or []):
            inst, qargs, cargs = instruction_parts(entry)
            if getattr(inst, "name", "") != "measure":
                continue
            if len(qargs) != 1 or len(cargs) != 1:
                continue
            mapping[bit_index(circuit, cargs[0])] = bit_index(circuit, qargs[0])
    except Exception:
        return {}
    return mapping


def default_cache_dir(app: str = "qbalance") -> Path:
    """Return the platform's user cache directory for ``app``.

    The path always ends in ``app``: platforms whose cache directory ends
    in another component (``.../app/Cache`` on Windows) get ``app``
    appended.  Nothing is created on disk.
    """
    cache_path = Path(user_cache_dir(app))
    if cache_path.name != app:
        cache_path = cache_path / app
    return cache_path
