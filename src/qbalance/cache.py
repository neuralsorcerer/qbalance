# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qbalance.errors import OptionalDependencyError
from qbalance.transpile.suppression import normalize_measurement_flip_map
from qbalance.utils import (
    atomic_write_bytes,
    default_cache_dir,
    dump_json,
    load_json,
    stable_hash_bytes,
)


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

    Nothing is created on disk.
    """
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
    m = _normalize_cached_meta(load_json(meta))
    with qpy_path.open("rb") as f:
        circuits = list(qpy.load(f))
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

    ``compiled.qpy`` and then ``meta.json`` are each written atomically, so
    an interrupted save leaves an entry that reads as a miss rather than a
    corrupt one.

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

    # Serialize before touching disk: a circuit QPY cannot represent then fails
    # here, leaving no partial artifact behind.
    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)

    entry.dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(entry.dir / "compiled.qpy", buffer.getvalue())
    # meta.json is written last: load_compiled requires both files, so a run
    # interrupted between them leaves an entry that simply reads as a miss.
    dump_json(entry.dir / "meta.json", meta)
