# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, cast

from platformdirs import user_cache_dir


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
