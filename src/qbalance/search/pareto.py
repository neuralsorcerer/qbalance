# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, List, Tuple


def _metric_value(metrics: Mapping[str, object] | None, key: str) -> float:
    """Internal helper that metric value.

    Args:
        metrics: Mapping of metric names to numeric values used for scoring.
        key: Stable key used to identify a cache artifact.

    Returns:
        float with the computed result.

    Raises:
        None.
    """
    if metrics is None:
        return float("inf")
    try:
        raw: Any = metrics.get(key, float("inf"))
        value = float(raw)
    except (AttributeError, TypeError, ValueError):
        return float("inf")
    return value if math.isfinite(value) else float("inf")


def pareto_front(
    items: Sequence[Tuple[object, Mapping[str, object] | None]], keys: Sequence[str]
) -> List[int]:
    """Pareto front used by the qbalance workflow.

    Args:
        items: Items value consumed by this routine.
        keys: Keys value consumed by this routine.

    Returns:
        List[int] with the computed result.

    Raises:
        None.
    """
    n = len(items)
    if n == 0:
        return []

    key_tuple = tuple(keys)
    if len(key_tuple) == 0:
        # With no optimization objectives, all candidates are equivalent and
        # therefore nondominated.
        return list(range(n))

    # Single-objective fast path: Pareto front equals all items that achieve
    # the minimum value on that key.
    if len(key_tuple) == 1:
        key = key_tuple[0]
        values = [_metric_value(metrics, key) for _, metrics in items]
        best = min(values)
        return [idx for idx, value in enumerate(values) if value == best]

    # Precompute normalized metric values once to avoid repeated float casts
    # and dictionary lookups inside the quadratic dominance loop.
    metric_vectors = [
        tuple(_metric_value(metrics, key) for key in key_tuple) for _, metrics in items
    ]

    # Group duplicate vectors so we compare unique objective tuples only.
    # ``dict.fromkeys`` preserves first-seen order and avoids redundant
    # membership checks in Python loops.
    unique_vectors = list(dict.fromkeys(metric_vectors))

    # Maintain an incremental front over unique vectors.
    # This avoids comparing every pair when the nondominated set is small.
    nondominated_vectors: list[tuple[float, ...]] = []
    for candidate in unique_vectors:
        dominated = False
        survivors: list[tuple[float, ...]] = []
        for incumbent in nondominated_vectors:
            # incumbent dominates candidate if incumbent <= candidate on all
            # keys and < on at least one.
            incumbent_le_all = True
            incumbent_lt_any = False
            # candidate dominates incumbent if candidate <= incumbent on all
            # keys and < on at least one.
            candidate_le_all = True
            candidate_lt_any = False
            for cv, iv in zip(candidate, incumbent):
                if iv > cv:
                    incumbent_le_all = False
                elif iv < cv:
                    incumbent_lt_any = True

                if cv > iv:
                    candidate_le_all = False
                elif cv < iv:
                    candidate_lt_any = True

                if not incumbent_le_all and not candidate_le_all:
                    # Neither can dominate the other anymore.
                    break

            if incumbent_le_all and incumbent_lt_any:
                dominated = True
                break

            if not (candidate_le_all and candidate_lt_any):
                survivors.append(incumbent)

        if not dominated:
            survivors.append(candidate)
            nondominated_vectors = survivors

    nondominated_set = set(nondominated_vectors)
    front: List[int] = []
    for idx, vector in enumerate(metric_vectors):
        if vector in nondominated_set:
            front.append(idx)
    return front
