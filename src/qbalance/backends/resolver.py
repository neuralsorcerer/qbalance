# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable, Dict, Union

from qbalance.errors import QBalanceError
from qbalance.logging import get_logger

log = get_logger(__name__)

BackendLike = Any


def _load_backend_plugins() -> Dict[str, Callable[[str], BackendLike]]:
    """Internal helper that load backend plugins.

    Args:
        None.

    Returns:
        Dict[str, Callable[[str], BackendLike]] with the computed result.

    Raises:
        None.
    """
    eps = entry_points()
    group = eps.select(group="qbalance.backends")
    out: Dict[str, Callable[[str], BackendLike]] = {}
    for ep in group:
        try:
            out[ep.name] = ep.load()
        except Exception as e:  # pragma: no cover
            log.warning("Failed to load backend plugin %s: %s", ep.name, e)
    return out


_PLUGINS: Dict[str, Callable[[str], BackendLike]] | None = None


def resolve_backend(spec_or_obj: Union[str, BackendLike]) -> BackendLike:
    """Resolve a backend/plugin implementation from user-provided configuration.

    Args:
        spec_or_obj: Backend spec string or already-instantiated backend object.

    Returns:
        BackendLike with the computed result.

    Raises:
        QBalanceError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    if not isinstance(spec_or_obj, str):
        return spec_or_obj

    global _PLUGINS
    if _PLUGINS is None:
        _PLUGINS = _load_backend_plugins()

    spec = spec_or_obj.strip()
    kind = spec.split(":")[0]
    plugin = _PLUGINS.get(kind)
    if plugin is None:
        raise QBalanceError(
            f"Unknown backend kind {kind!r}. Available: {sorted(_PLUGINS.keys())}"
        )
    return plugin(spec)


__all__ = ["resolve_backend", "BackendLike"]
