# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


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

    model_config = dict(frozen=True)


@dataclass
class Strategy:
    spec: StrategySpec
    # Arbitrary metadata produced by execution/compile/analysis
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_dir: Optional[str] = None
