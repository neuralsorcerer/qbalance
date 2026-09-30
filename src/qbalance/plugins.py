# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Dict, List


def list_plugins() -> Dict[str, List[str]]:
    """Return the names registered in each qbalance entry-point group, sorted.

    The groups are ``qbalance.backends`` (resolved by
    :func:`qbalance.backends.resolve_backend`), ``qbalance.objectives`` and
    ``qbalance.reports``; the last two are listed only, nothing resolves
    them.
    """
    eps = entry_points()
    groups = ["qbalance.backends", "qbalance.objectives", "qbalance.reports"]
    out: Dict[str, List[str]] = {}
    for g in groups:
        out[g] = sorted([ep.name for ep in eps.select(group=g)])
    return out
