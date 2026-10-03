# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from qbalance.backends import resolve_backend
from qbalance.cutting.addon_cutting import (
    CuttingExperiment,
    evaluate_cutting_experiment,
    prepare_cutting_experiment,
)
from qbalance.dataset import load_dataset
from qbalance.execution.ensemble import run_ensemble
from qbalance.logging import get_logger
from qbalance.strategies import StrategySpec, coerce_strategy_specs
from qbalance.transpile.pipeline import compile_ensemble
from qbalance.utils import (
    atomic_write_bytes,
    strict_json_value,
    validate_flag,
    validate_integral,
)

log = get_logger(__name__)


@dataclass
class TrialResult:
    """One row of a matrix run: a circuit on a backend under a strategy."""

    circuit: str
    backend: str
    strategy: Dict[str, Any]
    metrics: Dict[str, Any]


def run_matrix(
    dataset_dir: Path,
    backend_specs: Iterable[str],
    strategies: Iterable[StrategySpec | Mapping[str, Any]],
    out_json: Path,
    execute: bool = False,
    shots: int = 1024,
    seed: int = 0,
    profile: bool = False,
) -> Path:
    """Evaluate every backend, circuit and strategy combination into a JSON file.

    Each combination is compiled, a cutting strategy through its
    subexperiments, and with ``execute=True`` run and mitigated as its
    strategy asks.  A failing stage is recorded in that row
    (``compile_error``, ``cutting_error``, ``exec_error``, ``mthree_error``,
    ``zne_error``) instead of aborting the run.

    Args:
        dataset_dir: Directory containing the dataset index and circuit artifacts.
        backend_specs: Backend spec strings to evaluate on.
        strategies: Strategies to evaluate: ``StrategySpec`` objects or
            mappings of their fields, each evaluated even if repeated.
        out_json: Destination path for matrix JSON output; written atomically.
        execute (default: False): Whether to run compiled circuits and collect counts.
        shots (default: 1024): Shots per execution, split across an ensemble's
            instances.
        seed (default: 0): Simulator seed; instance ``i`` of an ensemble runs
            with ``seed + i``.
        profile (default: False): Whether pass-level transpiler profiling is enabled.

    Returns:
        ``out_json``.

    Raises:
        ValueError: If ``shots`` is not a positive integer, ``seed`` not a
            non-negative integer, ``execute`` or ``profile`` not a boolean,
            ``backend_specs`` or ``strategies`` empty or a single item, a
            strategy invalid, ``out_json`` a directory, or the dataset
            invalid.
        QBalanceError: If a backend spec cannot be resolved.
    """
    shots = validate_integral("shots", shots, positive=True)
    # Per-instance simulator seeds are seed + index, which must stay valid.
    seed = validate_integral("seed", seed, non_negative=True)
    execute = validate_flag("execute", execute)
    profile = validate_flag("profile", profile)
    if isinstance(backend_specs, (str, bytes)):
        raise ValueError("backend_specs must be a sequence of backend specs")
    backend_specs = list(backend_specs)
    if not backend_specs:
        raise ValueError("backend_specs must contain at least one backend spec")
    if not all(isinstance(spec, str) for spec in backend_specs):
        # Rows and metadata record the spec strings, which name the backends.
        raise ValueError("backend_specs must be backend spec strings")
    # Kept in order and with repeats: every requested strategy gets its rows.
    strategies = coerce_strategy_specs(strategies, deduplicate=False)
    out_json = Path(out_json)
    if out_json.is_dir():
        # Checked up front: the file is written only after every trial ran.
        raise ValueError(
            f"out_json {out_json} is a directory; pass the matrix JSON file path"
        )

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
                m = _evaluate_trial(
                    qc,
                    backend,
                    spec,
                    zne_factors=zne_factors,
                    execute=execute,
                    shots=shots,
                    seed=seed,
                    profile=profile,
                )
                results.append(
                    TrialResult(
                        circuit=rec.name,
                        backend=bspec,
                        strategy=serialized_spec,
                        metrics=m,
                    )
                )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "metadata": {
            "dataset_dir": str(Path(dataset_dir)),
            "backends": list(backend_specs),
            "execute": bool(execute),
            "shots": int(shots),
            "seed": int(seed),
            "profile": bool(profile),
        },
        "results": [asdict(r) for r in results],
    }
    # Read back by `qbalance report`, and written at the end of a run that
    # may have taken hours; a truncated file fails the next step outright.
    # Non-finite numbers are written as null, keeping the file strict JSON.
    atomic_write_bytes(
        out_json,
        json.dumps(strict_json_value(payload), indent=2, allow_nan=False).encode(
            "utf-8"
        ),
    )
    return out_json


def _evaluate_trial(
    qc: Any,
    backend: Any,
    spec: StrategySpec,
    *,
    zne_factors: Sequence[float],
    execute: bool,
    shots: int,
    seed: int,
    profile: bool,
) -> Dict[str, Any]:
    """Compile (and optionally execute) one circuit under one strategy."""
    max_subcircuit_qubits = getattr(spec, "max_subcircuit_qubits", None)
    if getattr(spec, "cutting", False) and max_subcircuit_qubits:
        try:
            experiment = prepare_cutting_experiment(qc, max_subcircuit_qubits)
        except Exception as e:
            return {"cutting_error": str(e)}
        if experiment is not None:
            return _evaluate_cut_trial(
                experiment,
                backend,
                spec,
                execute=execute,
                shots=shots,
                seed=seed,
                profile=profile,
            )

    try:
        instances, m = compile_ensemble(qc, backend=backend, spec=spec, profile=profile)
    except Exception as e:
        # One strategy the backend cannot compile (a circuit wider than the
        # device, say) must not abort the whole matrix.
        return {"compile_error": str(e)}
    if getattr(spec, "cutting", False):
        # The circuit already fits the subcircuit width, so it runs uncut.
        m["cut_count"] = 0
    if not execute:
        return m
    try:
        run = run_ensemble(
            backend,
            instances,
            m,
            shots=shots,
            seed=seed,
            mthree=getattr(spec, "mthree", False),
            zne_factors=zne_factors,
            zne_degree=getattr(spec, "zne_degree", 1),
        )
    except Exception as e:
        m["exec_error"] = str(e)
        return m
    m["counts"] = run.counts
    m["shots"] = int(sum(run.counts.values()))
    # A failed mitigation keeps the unmitigated counts and is reported as
    # mthree_error or zne_error, not as a failed execution.
    if run.mthree_error is not None:
        m["mthree_error"] = run.mthree_error
    elif run.mthree_probs is not None:
        m["mthree_probs"] = run.mthree_probs
    if run.zne_realized_factors is not None:
        m["zne_realized_factors"] = run.zne_realized_factors
    if run.zne_error is not None:
        m["zne_error"] = run.zne_error
    elif run.zne_probs is not None:
        m["zne_probs"] = run.zne_probs
    return m


def _evaluate_cut_trial(
    experiment: CuttingExperiment,
    backend: Any,
    spec: StrategySpec,
    *,
    execute: bool,
    shots: int,
    seed: int,
    profile: bool,
) -> Dict[str, Any]:
    """Compile every subexperiment of a cut circuit and optionally run them."""
    return evaluate_cutting_experiment(
        experiment,
        backend,
        lambda sub: compile_ensemble(sub, backend=backend, spec=spec, profile=profile),
        execute=execute,
        shots=shots,
        seed=seed,
    )
