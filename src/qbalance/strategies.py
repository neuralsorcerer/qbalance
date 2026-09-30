# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Dynamical-decoupling sequences qbalance knows how to build.
_DD_SEQUENCES = frozenset({"XY4", "XX", "YY"})


class StrategySpec(BaseModel):
    """A strategy = compilation knobs + suppression + mitigation knobs."""

    # Compilation knobs
    optimization_level: int = Field(default=1, ge=0, le=3)
    layout_method: Optional[str] = None  # e.g., "sabre", "trivial"
    routing_method: Optional[str] = None  # e.g., "sabre", "basic"
    translation_method: Optional[str] = None
    seed_transpiler: Optional[int] = 0

    # Suppression knobs
    pauli_twirling: bool = False
    num_twirls: int = 1
    dynamical_decoupling: bool = False
    dd_sequence: str = "XY4"  # "XY4" | "XX" | "YY"
    measurement_twirling: bool = False
    seed_suppression: Optional[int] = 0

    # Mitigation knobs
    mthree: bool = False
    zne: bool = False
    # Global folding realizes odd factors only, so the default asks for three
    # distinct noise levels; (1, 2, 3) would run at (1, 3, 3).
    zne_factors: tuple[float, ...] = (1.0, 3.0, 5.0)
    zne_degree: int = 1

    # Circuit cutting knobs (optional)
    cutting: bool = False
    max_subcircuit_qubits: Optional[int] = None

    # Runtime knobs (optional)
    resilience_level: Optional[int] = None  # IBM Runtime EstimatorV2 concept (0..2)

    @field_validator(
        "optimization_level",
        "num_twirls",
        "zne_degree",
        "max_subcircuit_qubits",
        "resilience_level",
        "seed_transpiler",
        "seed_suppression",
        mode="before",
    )
    @classmethod
    def _reject_bool_integer_fields(cls, value: Any, info: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be an integer, not a boolean")
        return value

    @field_validator("dd_sequence", mode="before")
    @classmethod
    def _normalize_dd_sequence(cls, value: Any) -> Any:
        # An unknown name used to fall back to XY4 silently, so "XY8" ran --
        # and was reported as a distinct strategy -- while compiling exactly
        # like XY4.  Reject it instead, and normalize case so "xx" and "XX"
        # are one strategy.
        if not isinstance(value, str):
            raise ValueError("dd_sequence must be a string")
        normalized = value.strip().upper()
        if normalized not in _DD_SEQUENCES:
            raise ValueError(
                f"dd_sequence must be one of {', '.join(sorted(_DD_SEQUENCES))}"
            )
        return normalized

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "StrategySpec":
        if isinstance(self.optimization_level, bool):
            raise ValueError("optimization_level must be an integer between 0 and 3")
        if isinstance(self.seed_transpiler, bool):
            raise ValueError("seed_transpiler must be an integer or None")
        if isinstance(self.seed_suppression, bool):
            raise ValueError("seed_suppression must be an integer or None")
        # Both seeds reach numpy/Qiskit RNGs that reject negative values, which
        # otherwise surfaces as an opaque error deep inside compilation.
        for seed_field in ("seed_transpiler", "seed_suppression"):
            seed_value = getattr(self, seed_field)
            if seed_value is not None and seed_value < 0:
                raise ValueError(f"{seed_field} must be a non-negative integer or None")

        if isinstance(self.num_twirls, bool) or self.num_twirls < 1:
            raise ValueError("num_twirls must be an integer >= 1")

        if isinstance(self.zne_degree, bool) or self.zne_degree < 0:
            raise ValueError("zne_degree must be a non-negative integer")

        if self.zne:
            if len(self.zne_factors) == 0:
                raise ValueError("zne_factors must be non-empty when zne=True")
            if any(not isfinite(f) for f in self.zne_factors):
                raise ValueError("zne_factors must all be finite when zne=True")
            if any(f < 1.0 for f in self.zne_factors):
                raise ValueError("zne_factors must all be >= 1.0 when zne=True")
            if tuple(self.zne_factors) != tuple(sorted(self.zne_factors)):
                raise ValueError("zne_factors must be sorted in non-decreasing order")
            if 1.0 not in self.zne_factors:
                raise ValueError("zne_factors must include 1.0 when zne=True")
            if self.zne_degree < 1:
                raise ValueError("zne_degree must be >= 1 when zne=True")
            if self.zne_degree >= len(self.zne_factors):
                raise ValueError(
                    "zne_degree must be less than len(zne_factors) when zne=True"
                )
            from qbalance.mitigation.zne import realized_fold_factor

            realized = {realized_fold_factor(f) for f in self.zne_factors}
            if self.zne_degree >= len(realized):
                raise ValueError(
                    "zne_degree must be less than the number of distinct fold "
                    f"factors zne_factors realize ({sorted(realized)}): folding "
                    "rounds each factor up to an odd integer"
                )

        if self.max_subcircuit_qubits is not None and (
            isinstance(self.max_subcircuit_qubits, bool)
            or self.max_subcircuit_qubits < 1
        ):
            raise ValueError("max_subcircuit_qubits must be an integer >= 1")
        if self.cutting and self.max_subcircuit_qubits is None:
            raise ValueError("max_subcircuit_qubits must be set when cutting=True")
        if self.cutting:
            # A cut circuit runs as sampled subexperiments whose results are
            # recombined into an expectation value; there is no single count
            # distribution for twirl flips, M3 or ZNE folding to act on.
            unsupported = [
                name
                for name in ("pauli_twirling", "measurement_twirling", "mthree", "zne")
                if getattr(self, name)
            ]
            if unsupported:
                raise ValueError(
                    "cutting=True cannot be combined with " + ", ".join(unsupported)
                )

        if isinstance(self.resilience_level, bool):
            raise ValueError("resilience_level must be one of 0, 1, 2")
        if self.resilience_level is not None and self.resilience_level not in (0, 1, 2):
            raise ValueError("resilience_level must be one of 0, 1, 2")

        return self

    # ``extra="forbid"`` matters for a comparison tool: pydantic would otherwise
    # drop an unknown key silently, so a typo such as "optimisation_level" or
    # "routing" would yield a default strategy and a run that reports several
    # visibly different entries as one identical configuration.
    model_config = dict(frozen=True, extra="forbid")


def load_strategy_specs(path: Path | str) -> list[StrategySpec]:
    """Load one or more strategy specs from a JSON file.

    Accepted shapes are:
    - a single StrategySpec object,
    - a list of StrategySpec objects,
    - ``{"strategies": [...]}``,
    - saved workload selections (``{"selections": {name: {"spec": ...}}}``), or
    - matrix results (``{"results": [{"strategy": ...}]}``).

    Duplicate strategies are removed while preserving first-seen order.
    """
    import json

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid strategy JSON in {source}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read strategy JSON from {source}: {exc}") from exc

    return coerce_strategy_specs(_strategy_items_from_payload(payload))


def _strategy_items_from_payload(payload: Any) -> Iterable[Any]:
    """Extract strategy-like items from supported JSON payload shapes."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Strategy JSON must be an object, a list, or a supported qbalance results file"
        )

    if "strategies" in payload:
        strategies = payload["strategies"]
        if not isinstance(strategies, list):
            raise ValueError("Strategy JSON field 'strategies' must be a list")
        return strategies

    if "selections" in payload:
        selections = payload["selections"]
        if not isinstance(selections, Mapping):
            raise ValueError("Strategy JSON field 'selections' must be an object")
        return [
            entry.get("spec") if isinstance(entry, Mapping) else entry
            for entry in selections.values()
        ]

    if "results" in payload:
        results = payload["results"]
        if not isinstance(results, list):
            raise ValueError("Strategy JSON field 'results' must be a list")
        return [
            entry.get("strategy") if isinstance(entry, Mapping) else entry
            for entry in results
        ]

    if "spec" in payload and isinstance(payload["spec"], Mapping):
        return [payload["spec"]]
    if "strategy" in payload and isinstance(payload["strategy"], Mapping):
        return [payload["strategy"]]
    return [payload]


def coerce_strategy_specs(
    strategies: Iterable[StrategySpec | Mapping[str, Any]],
    *,
    deduplicate: bool = True,
) -> list[StrategySpec]:
    """Validate and normalize an explicit strategy iterable.

    Args:
        strategies: ``StrategySpec`` objects or mappings of their fields.
        deduplicate (default: True): Drop repeats, keeping the first of each.

    Returns:
        The strategies as ``StrategySpec`` objects, in their given order.

    Raises:
        ValueError: If ``strategies`` is a single strategy or string rather
            than an iterable of them, is empty, or holds an entry that is
            neither a ``StrategySpec`` nor a mapping of valid fields.
    """
    if isinstance(strategies, (StrategySpec, Mapping)):
        # Both are iterable -- over field pairs or keys -- which would
        # otherwise surface as a baffling complaint about "entry 0".
        raise ValueError(
            "strategies must be an iterable of strategies, not a single "
            "strategy; wrap it in a list"
        )
    if isinstance(strategies, (str, bytes)) or not isinstance(strategies, Iterable):
        raise ValueError(
            "strategies must be an iterable of StrategySpec or mapping objects"
        )

    specs: list[StrategySpec] = []
    seen: set[StrategySpec] = set()
    count = 0
    for idx, spec in enumerate(strategies):
        count += 1
        if isinstance(spec, StrategySpec):
            normalized = spec
        elif isinstance(spec, Mapping):
            try:
                normalized = StrategySpec(**dict(spec))
            except Exception as exc:
                raise ValueError(f"Invalid strategy entry {idx}: {exc}") from exc
        else:
            raise ValueError(f"Strategy entry {idx} must be a StrategySpec or mapping")

        if not deduplicate or normalized not in seen:
            specs.append(normalized)
            seen.add(normalized)

    if count == 0:
        raise ValueError("strategies must contain at least one strategy")
    if not specs:
        raise ValueError("strategies must contain at least one unique strategy")
    return specs


@dataclass
class Strategy:
    """A strategy together with the metrics one evaluation of it produced."""

    spec: StrategySpec
    # Arbitrary metadata produced by execution/compile/analysis
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_dir: Optional[str] = None
