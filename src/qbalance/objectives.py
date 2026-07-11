# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
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


def load_objective(path: Path | str) -> Objective:
    """Load objective weights from a JSON file.

    Accepted shapes are a direct mapping of metric names to weights, for
    example ``{"depth": 1.0, "two_qubit_ops": 2.0}``, an object containing a
    ``"weights"`` mapping, or a saved-results-style object containing an
    ``"objective"`` mapping.  The loader is intentionally stricter than
    :class:`Objective`: every loaded metric name must be a non-empty string and
    every loaded weight must be numeric, finite, and not a boolean.

    Args:
        path: JSON file containing objective weights.

    Returns:
        Objective initialized with validated weights.

    Raises:
        ValueError: If the file cannot be read, is not strict JSON, does not
            contain a supported JSON object shape, or contains invalid weights.
    """
    import json

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda constant: _raise_invalid_json_constant(constant),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid objective JSON in {source}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read objective JSON from {source}: {exc}") from exc

    weights = _objective_weights_from_payload(payload)
    return Objective(_normalize_objective_weights(weights))


def _raise_invalid_json_constant(constant: str) -> None:
    """Reject non-standard JSON numeric constants such as NaN and Infinity."""
    raise ValueError(f"Invalid objective JSON constant: {constant}")


def _objective_weights_from_payload(payload: Any) -> Mapping[str, Any]:
    """Extract objective weights from a supported decoded JSON payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("Objective JSON must be an object")

    if "weights" in payload:
        weights = payload["weights"]
        field = "weights"
    elif "objective" in payload:
        weights = payload["objective"]
        field = "objective"
    else:
        weights = payload
        field = "top-level objective"

    if not isinstance(weights, Mapping):
        raise ValueError(f"Objective JSON field '{field}' must be an object")
    return weights


def _normalize_objective_weights(weights: Mapping[str, Any]) -> dict[str, float]:
    """Validate and coerce loaded objective weights to finite floats."""
    if not weights:
        raise ValueError("Objective JSON must contain at least one weight")

    normalized: dict[str, float] = {}
    for key, value in weights.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Objective metric names must be non-empty strings")
        if isinstance(value, bool):
            raise ValueError(f"Objective weight for {key!r} must not be a boolean")
        try:
            value_f = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Objective weight for {key!r} must be numeric and finite"
            ) from exc
        if not math.isfinite(value_f):
            raise ValueError(f"Objective weight for {key!r} must be numeric and finite")
        normalized[key] = value_f

    return normalized


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
