# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class Objective:
    """Multi-objective scoring with weights. Lower is better."""

    weights: Mapping[str, float]
    _valid_weights: Tuple[Tuple[str, float], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate and normalize dataclass state immediately after initialization.

        Args:
            None.

        Returns:
            None. This method updates state or performs side effects only.

        Raises:
            None.
        """
        copied_weights = dict(self.weights)
        object.__setattr__(self, "weights", copied_weights)

        valid_weights: list[tuple[str, float]] = []
        for key, weight in copied_weights.items():
            try:
                weight_f = float(weight)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(weight_f):
                continue
            valid_weights.append((key, weight_f))
        object.__setattr__(self, "_valid_weights", tuple(valid_weights))

    def score(self, metrics: Mapping[str, Any]) -> float:
        """Score used by the qbalance workflow.

        Args:
            metrics: Mapping of metric names to numeric values used for scoring.

        Returns:
            float with the computed result.

        Raises:
            None.
        """
        score = 0.0
        for key, weight in self._valid_weights:
            value = metrics.get(key)
            if value is None:
                continue

            try:
                value_f = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(value_f):
                continue
            term = weight * value_f
            if not math.isfinite(term):
                continue
            score += term
        return score


def default_objective() -> Objective:
    # Reasonable default: depth + 2q gates + estimated error

    """Return the default objective configuration used by qbalance.

    Args:
        None.

    Returns:
        Objective with the computed result.

    Raises:
        None.
    """
    return Objective(
        weights={
            "depth": 1.0,
            "two_qubit_ops": 2.0,
            "estimated_error": 10.0,
            "compile_time_s": 0.1,
        }
    )
