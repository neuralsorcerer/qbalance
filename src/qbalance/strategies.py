# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


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
    dd_sequence: str = "XY4"  # "XY4" | "XX" etc.
    measurement_twirling: bool = False
    seed_suppression: Optional[int] = 0

    # Mitigation knobs
    mthree: bool = False
    zne: bool = False
    zne_factors: tuple[float, ...] = (1.0, 2.0, 3.0)
    zne_degree: int = 1

    # Circuit cutting knobs (optional)
    cutting: bool = False
    max_subcircuit_qubits: Optional[int] = None

    # Runtime knobs (optional)
    resilience_level: Optional[int] = None  # IBM Runtime EstimatorV2 concept (0..2)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "StrategySpec":
        if isinstance(self.num_twirls, bool) or self.num_twirls < 1:
            raise ValueError("num_twirls must be an integer >= 1")

        if isinstance(self.zne_degree, bool) or self.zne_degree < 0:
            raise ValueError("zne_degree must be a non-negative integer")

        if self.zne:
            if len(self.zne_factors) == 0:
                raise ValueError("zne_factors must be non-empty when zne=True")
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

        if self.cutting:
            if self.max_subcircuit_qubits is None:
                raise ValueError("max_subcircuit_qubits must be set when cutting=True")
            if (
                isinstance(self.max_subcircuit_qubits, bool)
                or self.max_subcircuit_qubits < 1
            ):
                raise ValueError("max_subcircuit_qubits must be an integer >= 1")

        if self.resilience_level is not None and self.resilience_level not in (0, 1, 2):
            raise ValueError("resilience_level must be one of 0, 1, 2")

        return self

    model_config = dict(frozen=True)


@dataclass
class Strategy:
    spec: StrategySpec
    # Arbitrary metadata produced by execution/compile/analysis
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_dir: Optional[str] = None
