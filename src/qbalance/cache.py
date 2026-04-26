# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from qbalance.errors import OptionalDependencyError
from qbalance.utils import default_cache_dir, dump_json, load_json, stable_hash_bytes


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
    """Fingerprint circuit used by the qbalance workflow.

    Args:
        circuit: QuantumCircuit instance to inspect, transform, or execute.

    Returns:
        str with the computed result.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
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
    """Cache dir used by the qbalance workflow.

    Args:
        root (default: None): Root directory used to resolve local cache/dataset files.

    Returns:
        Path with the computed result.

    Raises:
        None.
    """
    return (root or default_cache_dir("qbalance")) / "cache"


def get_entry(key: str, root: Optional[Path] = None) -> CacheEntry:
    """Return entry for the provided inputs.

    Args:
        key: Stable key used to identify a cache artifact.
        root (default: None): Root directory used to resolve local cache/dataset files.

    Returns:
        CacheEntry with the computed result.

    Raises:
        None.
    """
    d = cache_dir(root) / key[:2] / key
    return CacheEntry(key=key, dir=d)


def load_compiled(entry: CacheEntry) -> Optional[Tuple[Any, Dict]]:
    """Load compiled from serialized data or persisted storage.

    Args:
        entry: CacheEntry describing where cached circuit artifacts are stored.

    Returns:
        Optional[Tuple[Any, Dict]] with the computed result.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    meta = entry.dir / "meta.json"
    qpy_path = entry.dir / "compiled.qpy"
    if not (meta.exists() and qpy_path.exists()):
        return None
    try:
        from qiskit import qpy
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError("qiskit is required for cache load") from e
    m = load_json(meta)
    with qpy_path.open("rb") as f:
        c = qpy.load(f)[0]
    return c, m


def save_compiled(entry: CacheEntry, circuit: Any, meta: Dict) -> None:
    """Persist compiled and return a reference to the saved artifact.

    Args:
        entry: CacheEntry describing where cached circuit artifacts are stored.
        circuit: QuantumCircuit instance to inspect, transform, or execute.
        meta: Meta value consumed by this routine.

    Returns:
        None. This method updates state or performs side effects only.

    Raises:
        OptionalDependencyError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    entry.dir.mkdir(parents=True, exist_ok=True)
    try:
        from qiskit import qpy
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError("qiskit is required for cache save") from e
    with (entry.dir / "compiled.qpy").open("wb") as f:
        qpy.dump(circuit, f)
    dump_json(entry.dir / "meta.json", meta)
