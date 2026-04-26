# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Sequence

from qbalance.backends import resolve_backend
from qbalance.dataset import load_dataset
from qbalance.execution import run_counts
from qbalance.logging import get_logger
from qbalance.mitigation.zne import fold_global, zne_extrapolate_counts
from qbalance.strategies import StrategySpec
from qbalance.transpile.pipeline import compile_one
from qbalance.transpile.suppression import apply_measurement_untwirl_counts

log = get_logger(__name__)


@dataclass
class TrialResult:
    circuit: str
    backend: str
    strategy: Dict[str, Any]
    metrics: Dict[str, Any]


def _validate_seed(seed: int) -> None:
    """Internal helper that validate seed.

    Args:
        seed: Seed used for deterministic randomization.

    Returns:
        None. This method updates state or performs side effects only.

    Raises:
        ValueError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer.")


def run_matrix(
    dataset_dir: Path,
    backend_specs: Sequence[str],
    strategies: Sequence[StrategySpec],
    out_json: Path,
    execute: bool = False,
    shots: int = 1024,
    seed: int = 0,
    profile: bool = False,
) -> Path:
    """Execute matrix and return the collected results.

    Args:
        dataset_dir: Directory containing the dataset index and circuit artifacts.
        backend_specs: Backend specs value consumed by this routine.
        strategies: Strategies value consumed by this routine.
        out_json: Destination path for matrix JSON output.
        execute (default: False): Whether to run compiled circuits and collect counts.
        shots (default: 1024): Number of shots used when executing circuits on a backend.
        seed (default: 0): Seed used for deterministic randomization.
        profile (default: False): Whether pass-level transpiler profiling is enabled.

    Returns:
        Path with the computed result.

    Raises:
        ValueError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots <= 0:
        raise ValueError("shots must be a positive integer.")
    _validate_seed(seed)

    ds = load_dataset(dataset_dir)
    circuits = ds.load_circuits()
    n_records = len(ds.records)
    n_circuits = len(circuits)
    if n_circuits != n_records:
        raise ValueError(
            "Dataset records/circuits length mismatch: "
            f"{n_records} records but {n_circuits} loaded circuits."
        )

    # Strategy serialization is pure; compute once and reuse in inner loops.
    strategy_entries = tuple(
        (
            spec,
            spec.model_dump(),
            (
                tuple(getattr(spec, "zne_factors", ()))
                if getattr(spec, "zne", False)
                else ()
            ),
        )
        for spec in strategies
    )

    results: List[TrialResult] = []
    for bspec in backend_specs:
        backend = resolve_backend(bspec)
        for qc, rec in zip(circuits, ds.records):
            for spec, serialized_spec, zne_factors in strategy_entries:
                compiled, m = compile_one(
                    qc, backend=backend, spec=spec, profile=profile
                )
                if execute:
                    try:
                        counts = run_counts(
                            backend, compiled, shots=shots, seed_simulator=seed
                        )
                        flip_map = m.get("measurement_flip_map") or {}
                        counts = apply_measurement_untwirl_counts(counts, flip_map)
                        m["counts"] = counts
                        m["shots"] = int(sum(counts.values()))
                        if zne_factors:
                            counts_pf = []
                            for f in zne_factors:
                                c_fold = fold_global(compiled, f)
                                cts = run_counts(
                                    backend, c_fold, shots=shots, seed_simulator=seed
                                )
                                cts = apply_measurement_untwirl_counts(cts, flip_map)
                                counts_pf.append(cts)
                            m["zne_probs"] = zne_extrapolate_counts(
                                zne_factors, counts_pf, degree=spec.zne_degree
                            )
                    except Exception as e:
                        m["exec_error"] = str(e)

                results.append(
                    TrialResult(
                        circuit=rec.name,
                        backend=bspec,
                        strategy=serialized_spec,
                        metrics=m,
                    )
                )

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "results": [asdict(r) for r in results]}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_json
