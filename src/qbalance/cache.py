# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qbalance.errors import OptionalDependencyError
from qbalance.transpile.suppression import normalize_measurement_flip_map
from qbalance.utils import (
    atomic_write_bytes,
    default_cache_dir,
    stable_hash_bytes,
)

# Only publication is serialized. Circuit serialization can run concurrently.
_CACHE_WRITE_LOCK = threading.Lock()


@dataclass
class CacheEntry:
    """Filesystem location for a cached compilation artifact.

    Args:
        key: Stable cache key.
        dir: Directory that stores cache artifacts for the key.
    """

    key: str
    dir: Path


def fingerprint_circuit(circuit: Any) -> str:
    """Return the SHA-256 hex digest of ``circuit``'s QPY serialization.

    Raises:
        OptionalDependencyError: If qiskit is not installed.
    """
    try:
        from qiskit import qpy
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError(
            "qiskit is required for circuit fingerprinting"
        ) from e

    buf = io.BytesIO()
    qpy.dump(circuit, buf)
    return stable_hash_bytes(buf.getvalue())


def cache_dir(root: Optional[Path] = None) -> Path:
    """Return the compile-cache directory: ``ROOT/cache``.

    Args:
        root (default: None): Cache root; the platform's user cache
            directory for qbalance when ``None``.
    """
    return (root or default_cache_dir("qbalance")) / "cache"


def get_entry(key: str, root: Optional[Path] = None) -> CacheEntry:
    """Return the cache entry for ``key``, stored at ``cache_dir(root)/key[:2]/key``.

    Nothing is created on disk. A key or shard that would escape the cache
    hierarchy raises ``ValueError``.
    """
    if (
        not isinstance(key, str)
        or key in ("", ".")
        or key.startswith("..")
        or any(char in key for char in "/\\:\0")
    ):
        raise ValueError(
            "cache key and its two-character prefix must be plain path components"
        )
    d = cache_dir(root) / key[:2] / key
    return CacheEntry(key=key, dir=d)


def _normalize_cached_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return cached metadata with JSON-loaded flip-map keys restored."""
    if "measurement_flip_map" not in meta and "measurement_flip_maps" not in meta:
        return meta

    normalized_meta = dict(meta)
    if "measurement_flip_map" in meta:
        normalized_meta["measurement_flip_map"] = normalize_measurement_flip_map(
            meta.get("measurement_flip_map")
        )
    if "measurement_flip_maps" in meta:
        raw_maps = meta.get("measurement_flip_maps")
        normalized_meta["measurement_flip_maps"] = [
            normalize_measurement_flip_map(flip_map)
            for flip_map in (raw_maps if isinstance(raw_maps, list) else [])
        ]
    return normalized_meta


def load_compiled_ensemble(entry: CacheEntry) -> Optional[Tuple[List[Any], Dict]]:
    """Load every cached circuit of an entry, with its metadata.

    Args:
        entry: CacheEntry describing where cached circuit artifacts are stored.

    Returns:
        ``(circuits, meta)``, or ``None`` when the entry is incomplete.

    Raises:
        OptionalDependencyError: If qiskit is unavailable.
        ValueError: If the entry holds no circuit.
    """
    meta = entry.dir / "meta.json"
    qpy_path = entry.dir / "compiled.qpy"
    if not (meta.exists() and qpy_path.exists()):
        return None
    try:
        from qiskit import qpy
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError("qiskit is required for cache load") from e
    # Writers publish metadata first. Read the circuit first so that even a
    # legacy entry (a plain metrics object without a digest) is consistent
    # during its first update to the new format.
    try:
        payload = qpy_path.read_bytes()
        stored = json.loads(meta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None

    if isinstance(stored, list):
        if len(stored) != 2 or not isinstance(stored[1], dict):
            raise ValueError(f"Invalid cache metadata in {meta}")
        digest, stored = stored
        if digest != stable_hash_bytes(payload):
            return None
    elif not isinstance(stored, dict):
        raise ValueError(f"Invalid cache metadata in {meta}")
    m = _normalize_cached_meta(stored)
    circuits = list(qpy.load(io.BytesIO(payload)))
    if not circuits:
        raise ValueError(f"Cache entry {entry.dir} holds no circuit")
    return circuits, m


def load_compiled(entry: CacheEntry) -> Optional[Tuple[Any, Dict]]:
    """Load the first cached circuit of an entry, with its metadata.

    Args:
        entry: CacheEntry describing where cached circuit artifacts are stored.

    Returns:
        ``(circuit, meta)``, or ``None`` when the entry is incomplete.

    Raises:
        OptionalDependencyError: If qiskit is unavailable.
    """
    hit = load_compiled_ensemble(entry)
    if hit is None:
        return None
    circuits, m = hit
    return circuits[0], m


def save_compiled(entry: CacheEntry, circuit: Any, meta: Dict) -> None:
    """Persist a compiled circuit, or a list of them, with its metadata.

    Both files are serialized before publication. The metadata records the
    digest of the circuit bytes, so a reader that observes an interrupted or
    overlapping update treats the entry as a miss.

    Args:
        entry: Where to store the circuits.
        circuit: Compiled QuantumCircuit, or a list of them (a twirl ensemble).
        meta: JSON-serializable compile metrics.

    Raises:
        OptionalDependencyError: If qiskit is not installed.
        OSError: If the entry cannot be written.
        Exception: Whatever QPY raises for a circuit it cannot serialize;
            nothing is written then.
    """
    try:
        from qiskit import qpy
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError("qiskit is required for cache save") from e

    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    payload = buffer.getvalue()
    metadata = json.dumps([stable_hash_bytes(payload), meta], indent=2, sort_keys=True)

    with _CACHE_WRITE_LOCK:
        atomic_write_bytes(entry.dir / "meta.json", metadata.encode("utf-8"))
        atomic_write_bytes(entry.dir / "compiled.qpy", payload)
