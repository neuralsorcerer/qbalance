# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from qbalance.backends import resolve_backend
from qbalance.cache import (
    fingerprint_circuit,
    get_entry,
    load_compiled_ensemble,
    save_compiled,
)
from qbalance.cutting.addon_cutting import (
    CuttingExperiment,
    evaluate_cutting_experiment,
    prepare_cutting_experiment,
)
from qbalance.dataset import DATASET_INDEX, CircuitDataset
from qbalance.dataset import CircuitRecord as DatasetCircuitRecord
from qbalance.dataset import load_dataset
from qbalance.diagnostics.distribution import cvm_1d, emd_1d, ks_1d
from qbalance.errors import QBalanceError
from qbalance.execution.ensemble import run_ensemble
from qbalance.logging import get_logger
from qbalance.mitigation.zne import parity_expectation_value
from qbalance.objectives import Objective, default_objective
from qbalance.search import BanditSearcher, default_candidate_strategies, pareto_front
from qbalance.strategies import Strategy, StrategySpec, coerce_strategy_specs
from qbalance.transpile.pipeline import compile_ensemble
from qbalance.utils import (
    backend_display_name,
    dump_json,
    replacing_directory,
    stable_hash_str,
    strict_json_value,
    validate_flag,
    validate_integral,
)

log = get_logger(__name__)

CircuitRecord = DatasetCircuitRecord


@dataclass
class BalancedWorkload:
    """The strategy selected for every circuit of a workload, with its evidence.

    Returned by :meth:`Workload.adjust` and by :func:`load_balanced_workload`.

    Args:
        dataset: The circuits.
        backend_spec: The backend spec they were adjusted for.
        selections: Circuit name to the selected strategy and its metrics.
        baseline_metrics: Circuit name to the baseline strategy's metrics.
        objective: The objective the selection minimized.
        evaluation_history: Circuit name to every evaluated candidate, in
            evaluation order.
    """

    dataset: CircuitDataset
    backend_spec: str
    selections: Dict[str, Strategy]  # circuit_name -> Strategy
    baseline_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    objective: Objective = field(default_factory=default_objective)
    evaluation_history: Dict[str, List[Strategy]] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a plain-text comparison of the baseline and the selection.

        It lists the backend, the number of circuits, the candidate evaluations,
        the mean ``depth``, ``two_qubit_ops``, ``estimated_error`` and
        ``compile_time_s`` of baseline and selection (over finite values; NaN
        when there are none), the EMD, CvM and KS distances between their
        ``depth`` and ``two_qubit_ops`` distributions, and the mean objective
        delta with how many circuits strictly improved or stayed unchanged.
        """
        keys = ["depth", "two_qubit_ops", "estimated_error", "compile_time_s"]
        lines = []
        lines.append("qbalance summary")
        lines.append(f"  backend: {self.backend_spec}")
        lines.append(f"  circuits: {len(self.selections)}")
        if self.evaluation_history:
            counts = [len(v) for v in self.evaluation_history.values()]
            if counts:
                lines.append(
                    "  candidate evaluations: "
                    f"total={sum(counts)} mean_per_circuit={float(np.mean(counts)):.4g}"
                )

        def agg(ms: List[Dict[str, Any]]) -> Dict[str, float]:
            """Return the mean of each summary metric over its finite values."""
            out: Dict[str, float] = {}
            for k in keys:
                vals: List[float] = []
                for m in ms:
                    value = _finite_float_or_none(m.get(k))
                    if value is not None:
                        vals.append(value)
                out[k] = float(np.mean(vals)) if vals else float("nan")
            return out

        sel_ms = [s.metrics for s in self.selections.values()]
        base_ms = [self.baseline_metrics.get(n, {}) for n in self.selections.keys()]

        a_sel = agg(sel_ms)
        a_base = agg(base_ms)

        lines.append("  mean metrics (baseline -> balanced):")
        for k in keys:
            lines.append(f"    {k}: {a_base.get(k):.4g} -> {a_sel.get(k):.4g}")

        # Distribution diagnostics inspired by balance's EMD/CVMD/KS additions
        for k in ["depth", "two_qubit_ops"]:
            x1 = _finite_samples(base_ms, k)
            x2 = _finite_samples(sel_ms, k)
            if not x1 or not x2:
                # A workload with no selections (an empty dataset, or an empty
                # split) has nothing to compare; the distance helpers reject
                # empty samples, and reporting must not fail on an empty run.
                lines.append(f"  dist[{k}]: n/a (no comparable samples)")
                continue
            lines.append(
                f"  dist[{k}]: EMD={emd_1d(x1, x2):.4g}  CVM={cvm_1d(x1, x2):.4g}  KS={ks_1d(x1, x2):.4g}"
            )

        diagnostics = self.selection_diagnostics()
        objective_deltas = [
            float(item["objective_delta"])
            for item in diagnostics.values()
            if _is_finite_number(item.get("objective_delta"))
        ]
        comparable = [
            float(item["objective_delta"])
            for item in diagnostics.values()
            if item.get("objective_improved") is not None
            and _is_finite_number(item.get("objective_delta"))
        ]
        # objective_improved means "no worse" (ties included, as for the
        # regression guard); the headline separates real gains from ties, or
        # a run that kept every baseline would read as improving every circuit.
        improved = sum(1 for delta in comparable if delta < 0.0)
        unchanged = sum(1 for delta in comparable if delta == 0.0)
        if objective_deltas:
            lines.append(
                "  objective deltas: "
                f"mean={float(np.mean(objective_deltas)):.4g} "
                f"improved={improved}/{len(comparable)} "
                f"unchanged={unchanged}/{len(comparable)}"
            )
        return "\n".join(lines)

    def selection_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """Return per-circuit baseline-vs-selected diagnostic deltas.

        The adjustment workflow optimizes a weighted objective, but downstream
        reviews often need to know *why* a strategy was selected and whether it
        actually improves on the baseline for each circuit.  This method
        computes deterministic, JSON-serializable diagnostics without requiring
        recompilation: baseline score, selected score, absolute and relative
        deltas for common compile metrics, and candidate evaluation counts.
        Negative deltas indicate improvements for minimized metrics.

        ``objective_delta`` is computed over ``comparable_objective_terms``,
        the objective terms both the baseline and the selection report, so a
        metric only one side can have never masquerades as an improvement or
        a regression.
        """
        metric_keys = ("depth", "two_qubit_ops", "estimated_error", "compile_time_s")
        diagnostics: Dict[str, Dict[str, Any]] = {}

        for name, selected in self.selections.items():
            baseline = self.baseline_metrics.get(name, {})
            selected_metrics = selected.metrics or {}
            baseline_score, baseline_terms = _diagnostic_objective_score(
                self.objective, baseline
            )
            selected_score, selected_terms = _diagnostic_objective_score(
                self.objective, selected_metrics
            )

            metric_deltas: Dict[str, Dict[str, Optional[float]]] = {}
            for key in metric_keys:
                base_value = _finite_float_or_none(baseline.get(key))
                selected_value = _finite_float_or_none(selected_metrics.get(key))
                delta = (
                    selected_value - base_value
                    if base_value is not None and selected_value is not None
                    else None
                )
                if delta is not None and base_value is not None and base_value != 0.0:
                    relative_delta = delta / abs(base_value)
                else:
                    relative_delta = None
                metric_deltas[key] = {
                    "baseline": base_value,
                    "selected": selected_value,
                    "delta": delta,
                    "relative_delta": relative_delta,
                }

            # Compare only the objective terms both sides report -- the rule the
            # regression guard applies.  A term only one side has (a
            # mitigation metric, the cut sampling overhead of a legacy result)
            # says nothing about improvement.  With identical term sets this
            # is exactly selected_score - baseline_score.
            comparable_terms = [
                key
                for key, _ in self.objective._valid_weights
                if key in baseline_terms and key in selected_terms
            ]
            score_delta = (
                sum(selected_terms[key] for key in comparable_terms)
                - sum(baseline_terms[key] for key in comparable_terms)
                if comparable_terms
                and baseline_score is not None
                and selected_score is not None
                else None
            )
            diagnostics[name] = {
                "baseline_objective_score": baseline_score,
                "selected_objective_score": selected_score,
                "objective_delta": score_delta,
                "comparable_objective_terms": comparable_terms,
                "objective_improved": (
                    score_delta <= 0.0 if score_delta is not None else None
                ),
                "objective_terms": {
                    "baseline": baseline_terms,
                    "selected": selected_terms,
                },
                "evaluated_candidates": len(self.evaluation_history.get(name, [])),
                "metric_deltas": metric_deltas,
            }

        return diagnostics

    def candidate_rankings(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return objective-ranked candidate evaluations for every circuit.

        The raw ``evaluation_history`` preserves execution order, which is useful
        for reproducing grid or bandit searches but awkward for audit reports.
        This helper derives a stable, JSON-serializable leaderboard per circuit
        from the already-collected metrics.  Entries with finite objective
        scores sort ahead of incomparable entries; ties are resolved by original
        evaluation order so repeated calls are deterministic.
        """

        rankings: Dict[str, List[Dict[str, Any]]] = {}
        for name, strategies in self.evaluation_history.items():
            selected = self.selections.get(name)
            rows: List[Dict[str, Any]] = []
            for original_index, strategy in enumerate(strategies):
                rows.append(
                    _candidate_ranking_row(
                        strategy,
                        self.objective,
                        original_index=original_index,
                        selected=selected is not None
                        and strategy.spec == selected.spec
                        and strategy.metrics
                        == _without_guard_metadata(selected.metrics),
                    )
                )

            # A guarded no-regression selection can intentionally choose the
            # baseline strategy, which is not part of the evaluated candidate
            # history when users provide an explicit strategy list.  Include it
            # in the audit leaderboard so saved rankings always identify the
            # final selection exactly once.
            if selected is not None and not any(row["selected"] for row in rows):
                rows.append(
                    _candidate_ranking_row(
                        selected,
                        self.objective,
                        original_index=None,
                        selected=True,
                    )
                )

            rows.sort(
                key=lambda row: (
                    row["selection_score"] is None,
                    (
                        float("inf")
                        if row["selection_score"] is None
                        else float(row["selection_score"])
                    ),
                    # The selection is first among equal scores, as selection
                    # broke the tie in its favor; evaluation order breaks the
                    # rest.
                    not row["selected"],
                    int(row["_tie_index"]),
                )
            )
            for rank, row in enumerate(rows, start=1):
                row["rank"] = rank
                del row["_tie_index"]
            rankings[name] = rows

        return rankings

    def covars(self) -> Dict[str, Dict[str, float]]:
        """Return distances between the baseline and selected metric distributions.

        For ``depth``, ``two_qubit_ops`` and ``estimated_error``: the EMD, CvM
        and KS distances (see :mod:`qbalance.diagnostics`) between the
        baseline's and the selection's finite values; NaN when a side has none.

        Returns:
            ``{metric: {"emd": ..., "cvm": ..., "ks": ...}}``.
        """
        out: Dict[str, Dict[str, float]] = {}
        sel_ms = [s.metrics for s in self.selections.values()]
        base_ms = [self.baseline_metrics.get(n, {}) for n in self.selections.keys()]
        for k in ["depth", "two_qubit_ops", "estimated_error"]:
            x1 = _finite_samples(base_ms, k)
            x2 = _finite_samples(sel_ms, k)
            if not x1 or not x2:
                # Match ``agg``: an empty sample set reports NaN rather than
                # raising out of the distance helpers.
                nan = float("nan")
                out[k] = {"emd": nan, "cvm": nan, "ks": nan}
                continue
            out[k] = {"emd": emd_1d(x1, x2), "cvm": cvm_1d(x1, x2), "ks": ks_1d(x1, x2)}
        return out

    def save(self, out_dir: Path, overwrite: bool = False) -> None:
        """Save the workload to ``out_dir``.

        It writes ``dataset/`` (this workload's records, with an index for
        exactly them), ``results.json`` (selections, baseline metrics,
        diagnostics, rankings and the evaluation history), and ``summary.txt``.
        :func:`load_balanced_workload` reads it back.

        Args:
            out_dir: Directory to write.
            overwrite (default: False): Whether to replace an existing directory.

        Raises:
            FileExistsError: If ``out_dir`` exists and ``overwrite`` is false.
            NotADirectoryError: If ``out_dir`` exists but is not a directory.
            ValueError: If overwriting would delete this workload's own dataset.
        """
        out_dir = Path(out_dir)
        # is_symlink() catches a dangling link, which exists() reports as absent.
        if out_dir.exists() or out_dir.is_symlink():
            if not overwrite:
                raise FileExistsError(f"{out_dir} exists (use overwrite=True)")
            if not out_dir.is_dir():
                # overwrite=True means "replace this workload directory", never
                # "delete whatever file happens to sit at this path".
                raise NotADirectoryError(
                    f"Cannot overwrite {out_dir}: it exists and is not a directory."
                )
            dataset_root = Path(self.dataset.root).resolve()
            out_resolved = out_dir.resolve()
            if dataset_root == out_resolved or out_resolved in dataset_root.parents:
                raise ValueError(
                    f"Cannot overwrite {out_dir}: it contains this workload's "
                    "source dataset. Save to a different directory."
                )

        # Written beside out_dir and swapped in at the end, so a save that
        # fails midway leaves an existing one intact rather than deleted.
        with replacing_directory(out_dir) as staging:
            self._write_to(staging)

    def _write_to(self, out_dir: Path) -> None:
        """Write the workload's files into the empty directory ``out_dir``."""
        # Copy this workload's artifacts, plus selection metadata
        (out_dir / "dataset").mkdir(parents=True, exist_ok=True)
        for rec in self.dataset.records:
            shutil.copy2(
                self.dataset.root / rec.artifact, out_dir / "dataset" / rec.artifact
            )
        # Write an index for exactly these records.  Copying the source index
        # instead breaks every workload built on a split(): that index still
        # lists the other half's records, whose artifacts are not copied, so
        # load_balanced_workload rejects its own output.
        dump_json(
            out_dir / "dataset" / DATASET_INDEX,
            {"version": 1, "records": [asdict(rec) for rec in self.dataset.records]},
        )

        # Save results
        results = {
            "backend_spec": self.backend_spec,
            "objective": self.objective.weights,
            "selections": {
                name: {"spec": s.spec.model_dump(), "metrics": s.metrics}
                for name, s in self.selections.items()
            },
            "baseline_metrics": self.baseline_metrics,
            "selection_diagnostics": self.selection_diagnostics(),
            "candidate_rankings": self.candidate_rankings(),
            "evaluation_history": {
                name: [
                    {"spec": strategy.spec.model_dump(), "metrics": strategy.metrics}
                    for strategy in strategies
                ]
                for name, strategies in self.evaluation_history.items()
            },
        }
        # An infeasible candidate scores +inf in memory; written as null, so
        # results.json stays strict JSON (strategy_failed still marks it).
        (out_dir / "results.json").write_bytes(
            json.dumps(strict_json_value(results), indent=2, allow_nan=False).encode(
                "utf-8"
            )
        )
        (out_dir / "summary.txt").write_text(self.summary() + "\n", encoding="utf-8")

    def to_download(self, zip_path: Path, overwrite: bool = False) -> Path:
        """Save the workload (see :meth:`save`) into a zip archive.

        Args:
            zip_path: Archive to write.
            overwrite (default: False): Whether to replace an existing archive.

        Returns:
            ``zip_path``.

        Raises:
            FileExistsError: If ``zip_path`` exists and ``overwrite`` is false.
        """
        zip_path = Path(zip_path)
        if (zip_path.exists() or zip_path.is_symlink()) and not overwrite:
            raise FileExistsError(f"{zip_path} exists (use overwrite=True)")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        # A unique staging directory avoids clobbering unrelated user paths
        # that happen to match a predictable name next to the zip file.
        tmp = Path(tempfile.mkdtemp(prefix=f".{zip_path.stem}-", dir=zip_path.parent))
        try:
            content = tmp / "workload"
            self.save(content)
            archive = tmp / "archive.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for p in sorted(content.rglob("*")):
                    if p.is_file():
                        z.write(p, p.relative_to(content))
            # Moved into place only once complete, so a failed export leaves
            # an existing archive intact rather than truncated.
            archive.replace(zip_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return zip_path


def _candidate_ranking_row(
    strategy: Strategy,
    objective: Objective,
    *,
    original_index: Optional[int],
    selected: bool,
) -> Dict[str, Any]:
    """Build one JSON-serializable candidate-ranking row."""
    score, terms = _diagnostic_objective_score(objective, strategy.metrics or {})
    sort_score = _objective_score(strategy.metrics, objective)
    return {
        "original_index": original_index,
        "spec": strategy.spec.model_dump(),
        "objective_score": score,
        "selection_score": sort_score if math.isfinite(sort_score) else None,
        "objective_terms": terms,
        "selected": selected,
        "_tie_index": original_index if original_index is not None else 1_000_000_000,
    }


def _without_guard_metadata(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``metrics`` without the keys the regression guard adds.

    A candidate the guard selects is stored with that metadata, but its
    evaluation-history entry is not, so rankings compare without it.
    """
    return {k: v for k, v in metrics.items() if k not in _GUARD_METADATA_KEYS}


def _require_json_object(value: Any, field: str) -> Mapping[str, Any]:
    """Return *value* as a mapping or raise a precise loader error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Balanced workload {field} must be a JSON object.")
    return value


def _format_name_set(names: Iterable[str]) -> str:
    """Format circuit-name sets deterministically for error messages."""
    return ", ".join(sorted(names))


def _strategy_from_result_entry(entry: Any, field: str) -> Strategy:
    """Parse a saved Strategy payload containing spec and optional metrics."""
    entry_obj = _require_json_object(entry, field)
    spec_payload = entry_obj.get("spec")
    if not isinstance(spec_payload, Mapping):
        raise ValueError(f"{field.capitalize()} must include a spec object.")

    metrics_payload = entry_obj.get("metrics", {})
    if metrics_payload is None:
        metrics_payload = {}
    metrics = dict(_require_json_object(metrics_payload, f"{field} metrics"))

    try:
        spec = StrategySpec(**dict(spec_payload))
    except Exception as exc:
        raise ValueError(f"{field.capitalize()} has an invalid spec: {exc}") from exc
    return Strategy(spec=spec, metrics=metrics)


def _load_evaluation_history(
    payload: Mapping[str, Any], dataset_names: set[str]
) -> Dict[str, List[Strategy]]:
    """Load optional saved candidate evaluations, preserving legacy compatibility."""
    history_payload = payload.get("evaluation_history", {})
    if history_payload is None:
        history_payload = {}
    history_payload = _require_json_object(history_payload, "evaluation_history")

    evaluation_history: Dict[str, List[Strategy]] = {}
    for name, entries in history_payload.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Balanced workload evaluation history names must be non-empty strings."
            )
        if not isinstance(entries, list):
            raise ValueError(
                f"Balanced workload evaluation history for {name!r} must be a list."
            )
        evaluation_history[name] = [
            _strategy_from_result_entry(
                entry, f"evaluation history entry {idx} for {name!r}"
            )
            for idx, entry in enumerate(entries)
        ]

    unknown_history = set(evaluation_history) - dataset_names
    if unknown_history:
        raise ValueError(
            "Balanced workload evaluation history references circuits not present in dataset: "
            + _format_name_set(unknown_history)
        )
    return evaluation_history


def load_balanced_workload(out_dir: Path | str) -> BalancedWorkload:
    """Load a workload previously written by :meth:`BalancedWorkload.save`.

    Args:
        out_dir: Directory containing ``results.json`` and the copied ``dataset``
            subdirectory produced by :meth:`BalancedWorkload.save`.

    Returns:
        Reconstructed :class:`BalancedWorkload` with dataset, selections,
        baseline metrics, objective weights, and optional evaluation history.

    Raises:
        ValueError: If required artifacts are missing or malformed.
    """
    out_dir = Path(out_dir)
    results_path = out_dir / "results.json"
    dataset_dir = out_dir / "dataset"
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"Could not read balanced workload results from {results_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid balanced workload JSON in {results_path}: {exc}"
        ) from exc

    payload = _require_json_object(payload, "results")

    backend_spec = payload.get("backend_spec")
    if not isinstance(backend_spec, str) or not backend_spec:
        raise ValueError(
            "Balanced workload results must include a non-empty backend_spec."
        )

    objective_payload = payload.get("objective", {})
    if objective_payload is None:
        objective_payload = {}
    objective = Objective(dict(_require_json_object(objective_payload, "objective")))

    dataset = load_dataset(dataset_dir)
    dataset_names = set(dataset.names())

    raw_selections = _require_json_object(payload.get("selections"), "selections")
    selections: Dict[str, Strategy] = {}
    for name, entry in raw_selections.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Balanced workload selection names must be non-empty strings."
            )
        selections[name] = _strategy_from_result_entry(entry, f"selection for {name!r}")

    selection_names = set(selections)
    unknown_selections = selection_names - dataset_names
    if unknown_selections:
        raise ValueError(
            "Balanced workload selections reference circuits not present in dataset: "
            + _format_name_set(unknown_selections)
        )
    missing_selections = dataset_names - selection_names
    if missing_selections:
        raise ValueError(
            "Balanced workload selections are missing dataset circuits: "
            + _format_name_set(missing_selections)
        )

    baseline_payload = payload.get("baseline_metrics", {})
    if baseline_payload is None:
        baseline_payload = {}
    baseline_payload = _require_json_object(baseline_payload, "baseline_metrics")
    baseline_metrics: Dict[str, Dict[str, Any]] = {}
    for name, metrics in baseline_payload.items():
        if not isinstance(name, str):
            raise ValueError("Balanced workload baseline metric names must be strings.")
        baseline_metrics[name] = dict(
            _require_json_object(metrics, f"baseline metrics for {name!r}")
        )

    unknown_baselines = set(baseline_metrics) - dataset_names
    if unknown_baselines:
        raise ValueError(
            "Balanced workload baseline metrics reference circuits not present in dataset: "
            + _format_name_set(unknown_baselines)
        )

    evaluation_history = _load_evaluation_history(payload, dataset_names)

    return BalancedWorkload(
        dataset=dataset,
        backend_spec=backend_spec,
        selections=selections,
        baseline_metrics=baseline_metrics,
        objective=objective,
        evaluation_history=evaluation_history,
    )


@dataclass
class Workload:
    """A dataset of circuits to adjust, and the backend to adjust them for.

    Args:
        dataset: The circuits.
        backend_spec: Target backend spec; set it with :meth:`set_target`.
    """

    dataset: CircuitDataset
    backend_spec: Optional[str] = None

    @classmethod
    def from_dataset(cls, dataset: CircuitDataset) -> "Workload":
        """Return a workload over ``dataset``, without a target backend."""
        return cls(dataset=dataset)

    @classmethod
    def from_path(cls, dataset_dir: Path) -> "Workload":
        """Return a workload over the dataset saved in ``dataset_dir``.

        Raises:
            OSError: If the dataset index cannot be read.
            ValueError: If the dataset index is invalid (see
                :func:`~qbalance.dataset.load_dataset`).
        """
        return cls(dataset=load_dataset(dataset_dir))

    def set_target(self, backend_spec: str) -> "Workload":
        """Return a new workload over the same dataset, targeting ``backend_spec``.

        The spec is resolved when :meth:`adjust` runs (see
        :func:`~qbalance.backends.resolve_backend`).
        """
        return Workload(dataset=self.dataset, backend_spec=backend_spec)

    def adjust(
        self,
        objective: Optional[Objective | Mapping[str, float]] = None,
        search: str = "grid",
        pareto: bool = False,
        max_candidates: int = 24,
        warmup: int = 6,
        execute: bool = False,
        shots: int = 1024,
        profile: bool = False,
        cache_root: Optional[Path] = None,
        seed: int = 0,
        strategies: Optional[Iterable[StrategySpec | Mapping[str, Any]]] = None,
        allow_regression: bool = True,
        max_evaluations: Optional[int] = None,
    ) -> BalancedWorkload:
        """Select one strategy per circuit for the target backend.

        Every circuit is compiled with the baseline strategy,
        ``StrategySpec(optimization_level=1, routing_method="sabre")``, and with
        each candidate, scored with the objective (lower is better), and the
        best feasible candidate is selected; among equal scores, the one that
        comes first in the candidate order.  A candidate whose cutting,
        compilation, execution or requested mitigation fails stays in the
        evaluation history but cannot be selected.

        Args:
            objective (default: None): Objective to minimize;
                :func:`~qbalance.objectives.default_objective` when ``None``.
            search (default: 'grid'): ``"grid"`` evaluates candidates in order;
                ``"bandit"`` evaluates ``warmup`` random ones, then those a
                Thompson-sampling surrogate proposes.
            pareto (default: False): Select from the Pareto front over
                ``depth``, ``two_qubit_ops``, ``estimated_error`` and
                ``sampling_overhead``, ordered by objective score.
            max_candidates (default: 24): Size of the generated candidate pool
                when ``strategies`` is not given.
            warmup (default: 6): Random evaluations bandit search makes before
                its surrogate proposes.
            execute (default: False): Whether to run compiled circuits and
                record execution metrics.  Strategies with M3 or ZNE run
                regardless.
            shots (default: 1024): Shots per execution, split across an
                ensemble's instances.
            profile (default: False): Whether to record pass-level transpiler
                profiles.
            cache_root (default: None): Compile-cache directory; the platform
                cache when ``None``.
            seed (default: 0): Seeds the candidate shuffle, the bandit, and the
                simulator (instance ``i`` of an ensemble runs with ``seed + i``).
            strategies (default: None): Explicit candidate strategies. When
                provided, max_candidates is ignored and the supplied order is
                used for grid search.
            allow_regression (default: True): When False, keep the baseline strategy
                when the selected feasible candidate scores worse than the baseline
                on the objective terms both report for that circuit.
            max_evaluations (default: None): Evaluate at most this many candidates
                per circuit.  Grid search takes them in candidate order; bandit
                search spends ``warmup`` of them at random and the rest on the
                candidates its surrogate proposes.  ``None`` evaluates every
                candidate, in which case the search mode only changes the order.

        Returns:
            The selections, with the baseline metrics and the evaluation
            history of every circuit.

        Raises:
            ValueError: If the workload has no target, or an argument is
                invalid (``search`` not ``"grid"``/``"bandit"``, a count that is
                not a positive integer, a negative seed or warmup, a
                non-boolean ``allow_regression``, or no candidates).
            QBalanceError: If the backend spec cannot be resolved, or no
                candidate of some circuit is feasible.
        """
        if not self.backend_spec:
            raise ValueError(
                "Workload has no target backend; call set_target(...) first"
            )
        if search not in {"grid", "bandit"}:
            raise ValueError("search must be 'grid' or 'bandit'")
        shots = validate_integral("shots", shots, positive=True)
        # The seed feeds numpy.random.default_rng, which rejects negative seeds.
        seed = validate_integral("seed", seed, non_negative=True)
        warmup = validate_integral("warmup", warmup, non_negative=True)
        if strategies is None:
            max_candidates = validate_integral(
                "max_candidates", max_candidates, positive=True
            )

        allow_regression = validate_flag("allow_regression", allow_regression)
        pareto = validate_flag("pareto", pareto)
        execute = validate_flag("execute", execute)
        profile = validate_flag("profile", profile)
        if max_evaluations is not None:
            max_evaluations = validate_integral(
                "max_evaluations", max_evaluations, positive=True
            )

        if objective is None:
            obj = default_objective()
        elif isinstance(objective, Objective):
            obj = objective
        elif isinstance(objective, Mapping):
            obj = Objective(objective)
        else:
            raise ValueError(
                "objective must be an Objective, a metric-to-weight mapping, or None"
            )
        backend = resolve_backend(self.backend_spec)
        # Calibrations change under an unchanged name and spec (daily, on
        # hardware), and layouts and error estimates derive from them.
        backend_key = f"{self.backend_spec}|cal={_calibration_fingerprint(backend)}"
        rng = np.random.default_rng(seed)
        cache_root = Path(cache_root) if cache_root is not None else None

        candidates = (
            coerce_strategy_specs(strategies)
            if strategies is not None
            else default_candidate_strategies(max_candidates=max_candidates, seed=seed)
        )
        if not candidates:
            raise ValueError("at least one candidate strategy is required")
        candidate_position = {spec: index for index, spec in enumerate(candidates)}
        budget = (
            len(candidates)
            if max_evaluations is None
            else min(max_evaluations, len(candidates))
        )
        # The surrogate learns from scores relative to each circuit's baseline
        # (see _bandit_target), whose scale it cannot know in advance.
        bandit = BanditSearcher(sigma2=None)

        selections: Dict[str, Strategy] = {}
        baseline_metrics: Dict[str, Dict[str, Any]] = {}
        evaluation_history: Dict[str, List[Strategy]] = {}

        circuits = self.dataset.load_circuits()
        if len(circuits) != len(self.dataset.records):
            raise RuntimeError(
                "Dataset load_circuits() returned a circuit count that does not match "
                "the number of dataset records"
            )

        # Baseline (single default spec), evaluated through the same pipeline
        # as every candidate: with execute=True it runs too, so the regression
        # guard and the diagnostics compare executed metrics with executed
        # metrics rather than with compile-only ones.
        baseline_spec = StrategySpec(optimization_level=1, routing_method="sabre")
        for qc, rec in zip(circuits, self.dataset.records):
            baseline_m = _evaluate_candidate(
                qc,
                backend,
                baseline_spec,
                objective=obj,
                execute=execute,
                shots=shots,
                seed=seed,
                profile=profile,
                cache_root=cache_root,
                backend_key=backend_key,
            )
            baseline_metrics[rec.name] = baseline_m

        for qc, rec in zip(circuits, self.dataset.records):
            evals: List[Tuple[StrategySpec, Dict[str, Any]]] = []

            def _evaluate_and_record(
                spec: StrategySpec,
                *,
                circuit: Any = qc,
                record_evals: List[Any] = evals,
                baseline: Mapping[str, Any] = baseline_metrics[rec.name],
            ) -> Dict[str, Any]:
                if spec == baseline_spec:
                    # Evaluated already, with the same inputs and seeds;
                    # re-running it would only repeat its executions.
                    metrics = copy.deepcopy(dict(baseline))
                else:
                    metrics = _evaluate_candidate(
                        circuit,
                        backend,
                        spec,
                        objective=obj,
                        execute=execute,
                        shots=shots,
                        seed=seed,
                        profile=profile,
                        cache_root=cache_root,
                        backend_key=backend_key,
                    )
                record_evals.append((spec, metrics))
                return metrics

            if search == "grid":
                for spec in candidates[:budget]:
                    _evaluate_and_record(spec)
            else:
                baseline_score = _objective_score(baseline_metrics[rec.name], obj)

                def _observe(spec: StrategySpec, metrics: Dict[str, Any]) -> None:
                    target = _bandit_target(metrics["objective_score"], baseline_score)
                    if target is not None:
                        bandit.observe(spec, target)

                # Warmup evaluates a random subset first; warmup=0 intentionally
                # starts from the bandit's prior.  The remaining budget goes to
                # candidates proposed one at a time, observing each result
                # before the next proposal so the posterior stays current.
                remaining = list(candidates)
                rng.shuffle(remaining)
                warmup_specs = remaining[: min(warmup, budget)]
                remaining = remaining[len(warmup_specs) :]
                for spec in warmup_specs:
                    _observe(spec, _evaluate_and_record(spec))
                evaluated = len(warmup_specs)
                while remaining and evaluated < budget:
                    proposed = bandit.propose(list(remaining), rng=rng)
                    remaining.remove(proposed)
                    _observe(proposed, _evaluate_and_record(proposed))
                    evaluated += 1

            # Pareto selection if requested (otherwise min score)
            evaluation_history[rec.name] = [
                Strategy(spec=spec, metrics=dict(metrics)) for spec, metrics in evals
            ]
            try:
                # Equal scores go to the earlier candidate, not to whichever
                # the search happened to evaluate first: with every candidate
                # evaluated, grid and bandit search then select alike.
                ranked = _rank_candidates(
                    sorted(evals, key=lambda item: candidate_position[item[0]]),
                    pareto=pareto,
                    objective=obj,
                )
            except RuntimeError as exc:
                raise QBalanceError(f"Circuit {rec.name!r}: {exc}") from exc
            chosen_spec, chosen_m = ranked[0]
            if not allow_regression:
                chosen_spec, chosen_m = _guard_against_regression(
                    baseline_spec,
                    baseline_metrics[rec.name],
                    chosen_spec,
                    chosen_m,
                    objective=obj,
                    alternatives=ranked[1:],
                )
            selections[rec.name] = Strategy(spec=chosen_spec, metrics=chosen_m)

        _warn_about_unused_objective_terms(
            obj,
            [
                *baseline_metrics.values(),
                *(
                    s.metrics
                    for history in evaluation_history.values()
                    for s in history
                ),
            ],
        )

        return BalancedWorkload(
            dataset=self.dataset,
            backend_spec=self.backend_spec,
            selections=selections,
            baseline_metrics=baseline_metrics,
            objective=obj,
            evaluation_history=evaluation_history,
        )


def _warn_about_unused_objective_terms(
    objective: Objective, evaluated: Iterable[Mapping[str, Any]]
) -> None:
    """Log the objective terms no evaluated metrics report.

    ``Objective.score`` skips a term whose metric is absent, so a misspelled
    metric name, or an execution metric weighted in a run that executes
    nothing, drops out of the score without a trace -- and when every term
    does, every candidate scores 0 and the first one is selected.
    """
    reported: set[str] = set()
    for metrics in evaluated:
        reported.update(metrics)
    unused = [key for key, _ in objective._valid_weights if key not in reported]
    if unused:
        log.warning(
            "Objective term(s) %s appear in no evaluated metrics, so they do not "
            "affect selection (a misspelled metric name, or an execution metric "
            "without execute=True?)",
            ", ".join(repr(key) for key in unused),
        )


def _bandit_target(score: Any, baseline_score: float) -> Optional[float]:
    """Return the bandit's observation: a score relative to the circuit's baseline.

    One surrogate is shared by every circuit, and raw objective scores scale
    with circuit size: pooling them lets a large circuit's scores masquerade
    as the effect of whichever strategies happened to be tried on it.  The
    relative change from the circuit's own baseline removes that offset and
    the scale, so strategy effects learned on one circuit carry over to the
    next.  Candidates or baselines without a finite score are not observed.
    """
    if not _is_finite_number(score) or not math.isfinite(baseline_score):
        return None
    score_f = float(score)
    scale = abs(baseline_score)
    if scale == 0.0:
        return score_f - baseline_score
    return (score_f - baseline_score) / scale


# Metadata the regression guard adds to the metrics it selects.
_GUARD_METADATA_KEYS = (
    "selected_by_regression_guard",
    "rejected_candidate_spec",
    "rejected_candidate_objective_score",
)


def _guard_against_regression(
    baseline_spec: StrategySpec,
    baseline_metrics: Mapping[str, Any],
    chosen_spec: StrategySpec,
    chosen_metrics: Dict[str, Any],
    objective: Objective,
    alternatives: Sequence[Tuple[StrategySpec, Dict[str, Any]]] = (),
) -> Tuple[StrategySpec, Dict[str, Any]]:
    """Return the best candidate that does not regress, else the baseline.

    The normal optimizer is intentionally free to explore user-supplied
    candidates, but production workflows often need a safety rail: never ship a
    balanced selection whose objective is worse than the known baseline.  Scores
    are computed with the same finite-safe helper used by selection so malformed
    candidate metrics cannot bypass the guard.

    The baseline runs no mitigation, so a candidate can carry objective terms
    (``mitigated_top_prob``, ``zne_top_prob``, ``cut_parity_expval``, ...) the
    baseline cannot have.  Comparing the full sums would bill those extra
    terms as a regression (positive weights) or as an improvement (negative
    weights) whatever the circuits' shared metrics say.  The comparison
    therefore uses only the objective terms both sides report; with none in
    common the pair is incomparable and the candidate is kept.

    Because the chosen candidate can win on a term the comparison leaves out,
    it can regress while a lower-ranked candidate beats the baseline.  The
    ``alternatives`` -- the rest of the selection pool, best first -- are
    therefore tried in order, and the baseline is used only when every one of
    them regresses too.  A selection made here carries guard metadata naming
    the rejected top candidate.
    """
    baseline_score = _objective_score(baseline_metrics, objective)

    def regresses(metrics: Mapping[str, Any]) -> bool:
        if not math.isfinite(baseline_score):
            return False
        if not math.isfinite(_objective_score(metrics, objective)):
            return True
        shared = _shared_term_scores(objective, baseline_metrics, metrics)
        return shared is not None and shared[1] > shared[0]

    if not regresses(chosen_metrics):
        return chosen_spec, chosen_metrics
    chosen_score = _objective_score(chosen_metrics, objective)
    guard_metadata = {
        "selected_by_regression_guard": True,
        "rejected_candidate_spec": chosen_spec.model_dump(),
        "rejected_candidate_objective_score": (
            chosen_score if math.isfinite(chosen_score) else None
        ),
    }
    for spec, metrics in alternatives:
        if not regresses(metrics):
            kept = dict(metrics)
            kept.update(guard_metadata)
            return spec, kept
    guarded_metrics = dict(baseline_metrics)
    guarded_metrics["objective_score"] = baseline_score
    guarded_metrics.update(guard_metadata)
    return baseline_spec, guarded_metrics


def _shared_term_scores(
    objective: Objective, first: Mapping[str, Any], second: Mapping[str, Any]
) -> Optional[Tuple[float, float]]:
    """Score two metric sets on the objective terms both report finitely.

    Returns ``None`` when no weighted term is finite on both sides.
    """
    first_total = 0.0
    second_total = 0.0
    shared = False
    for key, weight in objective._valid_weights:
        first_value = _finite_float_or_none(first.get(key))
        second_value = _finite_float_or_none(second.get(key))
        if first_value is None or second_value is None:
            continue
        first_term = weight * first_value
        second_term = weight * second_value
        if not (math.isfinite(first_term) and math.isfinite(second_term)):
            continue
        first_total += first_term
        second_total += second_term
        shared = True
    if not shared or not (math.isfinite(first_total) and math.isfinite(second_total)):
        return None
    return first_total, second_total


def _evaluate_candidate(
    circuit: Any,
    backend: Any,
    spec: StrategySpec,
    *,
    objective: Objective,
    execute: bool,
    shots: int,
    seed: int,
    profile: bool,
    cache_root: Optional[Path],
    backend_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile and optionally execute one candidate strategy for one circuit.

    Returns the candidate's metrics.  Cutting, compilation, execution and
    mitigation failures are captured inside the metrics, so the candidate
    stays in the audit history but is marked infeasible.

    A twirled strategy compiles into several instances; execution splits the
    shot budget across all of them and untwirls each with its own flip map,
    because a twirl only suppresses noise averaged over its instances.
    """
    if spec.cutting and spec.max_subcircuit_qubits:
        try:
            experiment = prepare_cutting_experiment(circuit, spec.max_subcircuit_qubits)
        except Exception as e:
            # Recorded, not dropped: a silently skipped candidate is what let a
            # broken cutting integration look like "cutting simply was not
            # selected".
            log.warning(
                "Candidate is infeasible: circuit cutting failed for "
                "max_subcircuit_qubits=%s: %s",
                spec.max_subcircuit_qubits,
                e,
            )
            return _score_candidate(
                {"cutting_error": str(e)}, spec, objective=objective, execute=execute
            )
        if experiment is not None:
            return _evaluate_cut_experiment(
                experiment,
                backend,
                spec,
                objective=objective,
                execute=execute,
                shots=shots,
                seed=seed,
                profile=profile,
                cache_root=cache_root,
                backend_key=backend_key,
            )

    try:
        instances, m = _compile_cached(
            circuit,
            backend,
            spec,
            profile=profile,
            cache_root=cache_root,
            backend_key=backend_key,
        )
    except Exception as e:
        # A strategy the backend cannot compile (a circuit wider than the
        # device, an unknown layout method) is an infeasible candidate, not a
        # reason to abandon every other circuit and candidate of the run.
        log.warning("Candidate could not be compiled for this backend: %s", e)
        return _score_candidate(
            {"compile_error": str(e)}, spec, objective=objective, execute=execute
        )
    if spec.cutting:
        # The circuit already fits the subcircuit width, so it runs uncut.
        m["cut_count"] = 0

    # optional execution for mitigation or if execute=True
    if execute or spec.mthree or spec.zne:
        try:
            run = run_ensemble(
                backend,
                instances,
                m,
                shots=shots,
                seed=seed,
                mthree=spec.mthree,
                zne_factors=spec.zne_factors if spec.zne else (),
                zne_degree=spec.zne_degree,
            )
            m["raw_counts_entropy"] = _entropy_from_counts(run.counts)
            m["raw_top_prob"] = _top_prob(run.counts)
            m["raw_parity_expval"] = parity_expectation_value(run.counts)
        except Exception as e:
            m["exec_error"] = str(e)
        else:
            if run.mthree_error is not None:
                m["mthree_error"] = run.mthree_error
            elif run.mthree_probs is not None:
                _summarize_mitigated(m, run.mthree_probs, "mitigated", "mthree_error")
            if run.zne_realized_factors is not None:
                m["zne_realized_factors"] = run.zne_realized_factors
            if run.zne_error is not None:
                m["zne_error"] = run.zne_error
            elif run.zne_probs is not None:
                _summarize_mitigated(m, run.zne_probs, "zne", "zne_error")

    return _score_candidate(m, spec, objective=objective, execute=execute)


def _evaluate_cut_experiment(
    experiment: CuttingExperiment,
    backend: Any,
    spec: StrategySpec,
    *,
    objective: Objective,
    execute: bool,
    shots: int,
    seed: int,
    profile: bool,
    cache_root: Optional[Path],
    backend_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile (and optionally run) every subexperiment of a cut candidate.

    A subexperiment the backend cannot compile makes the candidate infeasible
    (``compile_error``) rather than aborting the run.
    """

    def compile_subexperiment(sub: Any) -> Tuple[List[Any], Dict[str, Any]]:
        return _compile_cached(
            sub,
            backend,
            spec,
            profile=profile,
            cache_root=cache_root,
            backend_key=backend_key,
        )

    m = evaluate_cutting_experiment(
        experiment,
        backend,
        compile_subexperiment,
        execute=execute,
        shots=shots,
        seed=seed,
    )
    if "compile_error" in m:
        log.warning(
            "Candidate is infeasible: the cut subexperiments for "
            "max_subcircuit_qubits=%s could not be compiled: %s",
            spec.max_subcircuit_qubits,
            m["compile_error"],
        )
    return _score_candidate(m, spec, objective=objective, execute=execute)


def _summarize_mitigated(
    m: Dict[str, Any], probs: Mapping[str, float], prefix: str, error_key: str
) -> None:
    """Record the top probability and parity of a mitigated distribution."""
    try:
        m[f"{prefix}_top_prob"] = float(max(probs.values()) if probs else 0.0)
        m[f"{prefix}_parity_expval"] = parity_expectation_value(probs)
    except Exception as e:
        m[error_key] = str(e)


def _score_candidate(
    m: Dict[str, Any], spec: StrategySpec, *, objective: Objective, execute: bool
) -> Dict[str, Any]:
    """Attach the objective score, or mark the candidate infeasible."""
    # Hard execution/mitigation failures make it infeasible for selection
    # rather than letting good compile-only metrics hide a failed runtime path.
    failure_reason = _strategy_failure_reason(m, spec, execute=execute)
    if failure_reason is not None:
        m["strategy_failed"] = True
        m["strategy_failure_reason"] = failure_reason
        m["objective_score"] = float("inf")
    else:
        m["objective_score"] = objective.score(m)
    return m


def _strategy_failure_reason(
    metrics: Mapping[str, Any], spec: StrategySpec, *, execute: bool
) -> Optional[str]:
    """Return why a candidate is infeasible for selection, if applicable.

    Compilation metrics alone can look attractive even when the requested
    runtime or mitigation stage failed.  Treat those candidates as evaluated
    but infeasible so audit history remains complete while selection and the
    bandit surrogate learn only from successful end-to-end strategies.
    """
    if metrics.get("cutting_error"):
        return "cutting_failed"
    if metrics.get("compile_error"):
        return "compile_failed"
    execution_requested = execute or spec.mthree or spec.zne
    if execution_requested and metrics.get("exec_error"):
        return "execution_failed"
    if spec.mthree and metrics.get("mthree_error"):
        return "mthree_failed"
    if spec.zne and metrics.get("zne_error"):
        return "zne_failed"
    return None


def _is_finite_number(value: Any) -> bool:
    """Return True when *value* can be safely used as a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_float_or_none(value: Any) -> Optional[float]:
    """Return a finite float for diagnostics, otherwise ``None``."""
    try:
        value_f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value_f if math.isfinite(value_f) else None


def _finite_samples(metrics: Iterable[Mapping[str, Any]], key: str) -> List[float]:
    """Return the finite values of ``key`` across metric mappings.

    A missing or malformed value is left out rather than counted as ``0.0``:
    an invented zero is a data point of its own, and it drags distribution
    distances toward whichever side is missing the metric.
    """
    samples: List[float] = []
    for m in metrics:
        value = _finite_float_or_none(m.get(key))
        if value is not None:
            samples.append(value)
    return samples


def _diagnostic_objective_score(
    objective: Objective, metrics: Mapping[str, Any]
) -> Tuple[Optional[float], Dict[str, float]]:
    """Compute a JSON-safe objective score and contributing terms.

    Unlike :meth:`Objective.score`, this helper returns ``None`` when no
    finite objective term contributes.  That distinction matters for review
    diagnostics: missing or malformed metrics should be reported as
    incomparable rather than as an accidental score of zero.
    """
    terms: Dict[str, float] = {}
    total = 0.0
    for key, weight in objective._valid_weights:
        value = _finite_float_or_none(metrics.get(key))
        if value is None:
            continue
        term = weight * value
        if not math.isfinite(term):
            continue
        terms[key] = term
        total += term

    if not terms or not math.isfinite(total):
        return None, terms
    return total, terms


def _derived_objective_score(metrics: Mapping[str, Any], objective: Objective) -> float:
    """Compute a finite-safe score from metrics and objective weights."""
    derived = 0.0
    contributed = False
    has_objective_key = False
    for key, weight in objective._valid_weights:
        if key not in metrics:
            continue
        has_objective_key = True
        raw_value = metrics.get(key)
        if raw_value is None:
            continue
        try:
            value_f = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(value_f):
            continue
        term = weight * value_f
        if not np.isfinite(term):
            continue
        derived += term
        contributed = True

    if not contributed and has_objective_key:
        return float("inf")
    if not contributed:
        return float("nan")
    return derived if np.isfinite(derived) else float("inf")


def _is_infeasible_metrics(metrics: Mapping[str, Any] | None) -> bool:
    """Return True when metrics explicitly mark a candidate as infeasible."""
    return not isinstance(metrics, Mapping) or bool(metrics.get("strategy_failed"))


def _objective_score(metrics: Mapping[str, Any] | None, objective: Objective) -> float:
    """Return the same finite-safe selection score used by workload choice."""
    if _is_infeasible_metrics(metrics):
        return float("inf")
    assert isinstance(metrics, Mapping)

    try:
        score = float(metrics.get("objective_score", float("inf")))
    except (TypeError, ValueError, OverflowError):
        score = float("inf")
    derived = _derived_objective_score(metrics, objective)
    if np.isfinite(score) and np.isfinite(derived):
        return score
    if np.isfinite(score) and np.isnan(derived):
        return score
    if np.isfinite(score):
        return float("inf")

    # Fallback: compute objective only if at least one objective-relevant
    # metric contributes a finite weighted term. This prevents malformed
    # metrics (e.g., {"depth": "bad"}) from receiving an accidental 0.0
    # score and being preferred over valid candidates.
    return derived if np.isfinite(derived) else float("inf")


# Bump whenever compile_ensemble's output for unchanged inputs changes, so
# entries written by the previous pipeline are no longer served.
_COMPILE_CACHE_VERSION = 8


def _calibration_fingerprint(backend: Any) -> str:
    """Return a stable digest of the calibration data compilation depends on.

    The noise-aware layout and ``estimated_error`` are derived from the
    backend's error rates and coherence times, which on hardware change with
    every calibration while the backend's name and spec stay the same.  Keying
    the compile cache on this digest stops yesterday's layouts and error
    estimates from being served as today's.  Returns ``"none"`` when the
    backend exposes no calibration data.
    """
    items: List[Any] = []
    try:
        target = getattr(backend, "target", None)
        if target is not None:
            items.append(("dt", getattr(target, "dt", None)))
            for name in sorted(str(n) for n in target.operation_names):
                for qargs, props in sorted(
                    target[name].items(), key=lambda item: repr(item[0])
                ):
                    items.append(
                        (
                            name,
                            qargs,
                            getattr(props, "error", None),
                            getattr(props, "duration", None),
                        )
                    )
            for qubit_props in getattr(target, "qubit_properties", None) or ():
                items.append(
                    tuple(
                        getattr(qubit_props, attr, None)
                        for attr in ("t1", "t2", "frequency")
                    )
                )
        properties = getattr(backend, "properties", None)
        if callable(properties):
            calibration = properties()
            if calibration is not None:
                items.append(
                    ("updated", str(getattr(calibration, "last_update_date", "")))
                )
    except Exception:
        # A backend that cannot describe itself is keyed by name and spec only.
        return "unknown"
    return stable_hash_str(repr(items)) if items else "none"


def _qiskit_version() -> str:
    """Return the installed qiskit version, or ``"unknown"`` when unavailable."""
    try:
        import qiskit

        return str(getattr(qiskit, "__version__", "unknown"))
    except Exception:  # pragma: no cover - qiskit is a hard dependency
        return "unknown"


def _compile_cached(
    circuit: Any,
    backend: Any,
    spec: StrategySpec,
    profile: bool,
    cache_root: Optional[Path],
    backend_key: Optional[str] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Compile ``circuit`` into its instance ensemble, reusing the compile cache.

    The cache key covers the circuit (its QPY fingerprint), the backend,
    the strategy, profiling, the qiskit version and the cache format
    version.  An unreadable entry is ignored and recompiled, and a failed
    save only logs a warning: the cache never fails a run.

    Args:
        circuit: Circuit to compile.
        backend: Backend to compile for.
        spec: Strategy to compile with.
        profile: Whether pass-level transpiler profiling is enabled.
        cache_root: Compile-cache directory; the platform cache when ``None``.
        backend_key (default: None): Backend spec string that resolved to
            ``backend`` (``adjust`` appends its calibration fingerprint).
            Backend display names are not unique -- every
            ``fake:generic:N:SEED`` shares one name while carrying different
            calibration data, and a device keeps its name across
            recalibrations -- so the key keeps calibration-derived metrics
            from leaking across backends and calibrations.

    Returns:
        ``(instances, metrics)`` as returned by :func:`compile_ensemble`.

    Raises:
        Exception: Whatever :func:`compile_ensemble` raises on a cache miss.
    """
    try:
        fpr = fingerprint_circuit(circuit)
    except Exception:
        # builtins.hash of a str is salted per process (PYTHONHASHSEED), so it
        # would give the same circuit a different key on every run and the
        # cache could never hit across processes.
        fpr = stable_hash_str(str(circuit))
    backend_name = backend_display_name(backend)
    backend_id = f"{backend_key if backend_key is not None else ''}|{backend_name}"
    # The toolchain is part of the key: a different qiskit transpiles the same
    # inputs differently, and a qbalance compile-pipeline change alters what
    # compile_ensemble returns.  Without it a stale entry keeps being served --
    # silently -- as if it had been produced by the running code.
    toolchain = f"cache=v{_COMPILE_CACHE_VERSION}|qiskit={_qiskit_version()}"
    key = f"{toolchain}:{backend_id}:{fpr}:{spec.model_dump_json()}:profile={profile}"
    import hashlib

    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    entry = get_entry(key_hash, root=cache_root)
    try:
        hit = load_compiled_ensemble(entry)
        if hit is not None and len(hit[0]) != hit[1].get("twirl_instances", 1):
            raise ValueError("circuit count does not match the recorded ensemble")
    except Exception as e:
        # An entry left corrupt by an interrupted run (or written by an
        # incompatible qiskit) must not abort this one; recompiling is always
        # correct, and the next save overwrites the bad entry.
        log.warning("Ignoring unreadable compile-cache entry %s: %s", entry.dir, e)
        hit = None
    if hit is not None:
        cached_instances, cached_m = hit
        return cached_instances, cached_m

    instances, m = compile_ensemble(
        circuit, backend=backend, spec=spec, profile=profile
    )
    try:
        save_compiled(entry, instances, m)
    except Exception as e:
        # The cache only saves work.  A read-only cache root, a full disk, or a
        # circuit QPY cannot serialize must not fail the adjustment run.
        log.warning("Could not write compile-cache entry %s: %s", entry.dir, e)
    return instances, m


def _choose(
    evals: List[Tuple[StrategySpec, Dict[str, Any]]],
    pareto: bool,
    objective: Objective,
) -> Tuple[StrategySpec, Dict[str, Any]]:
    """Return the selected candidate: the first of :func:`_rank_candidates`.

    Args:
        evals: Evaluated ``(spec, metrics)`` pairs, in evaluation order.
        pareto: Restrict the choice to the Pareto front.
        objective: Objective whose score orders the candidates.

    Returns:
        The best feasible ``(spec, metrics)``.

    Raises:
        RuntimeError: If no candidate was evaluated, or none is feasible.
    """
    return _rank_candidates(evals, pareto=pareto, objective=objective)[0]


def _rank_candidates(
    evals: List[Tuple[StrategySpec, Dict[str, Any]]],
    pareto: bool,
    objective: Objective,
) -> List[Tuple[StrategySpec, Dict[str, Any]]]:
    """Return the selection pool, best objective score first.

    The pool is every feasible candidate, or the Pareto front of them with
    ``pareto=True``; ties keep the order of ``evals``.

    Raises:
        RuntimeError: If no candidate was evaluated, or none is feasible.
    """
    if not evals:
        raise QBalanceError("No candidate strategies were successfully evaluated")

    feasible_evals = [
        (spec, metrics)
        for spec, metrics in evals
        if not _is_infeasible_metrics(metrics)
        and math.isfinite(_objective_score(metrics, objective))
    ]
    if not feasible_evals:
        raise QBalanceError(
            "No feasible candidate strategies were successfully evaluated"
            + _describe_failures(evals)
        )

    # sampling_overhead is an axis too: without it a cut candidate's shallower
    # subexperiments dominate -- and drop from the front -- every uncut
    # candidate, leaving the objective no uncut option to prefer even when its
    # overhead makes the cut the worse choice.  Uncut candidates report 1.0.
    pareto_keys = ("depth", "two_qubit_ops", "estimated_error", "sampling_overhead")

    pool = feasible_evals
    if pareto:
        # Pareto on key metrics, then tie-break by objective_score.
        # Failed or otherwise incomparable candidates are filtered before
        # Pareto construction so invalid low metric values cannot dominate
        # feasible work.
        front_idx = pareto_front(feasible_evals, keys=pareto_keys)
        pool = [feasible_evals[i] for i in front_idx]
    # sorted() is stable, so equal scores keep evaluation order -- the same
    # candidate min() picked.
    return sorted(pool, key=lambda t: _objective_score(t[1], objective))


def _describe_failures(evals: List[Tuple[StrategySpec, Dict[str, Any]]]) -> str:
    """Summarize why evaluated candidates were infeasible, for error messages.

    Without it, a run where every candidate failed the same way -- executing a
    circuit that has no measurements, say -- reports only that nothing was
    feasible, leaving the user to guess at the cause.
    """
    reasons: Dict[str, str] = {}
    for _, metrics in evals:
        if not isinstance(metrics, Mapping):
            continue
        reason = metrics.get("strategy_failure_reason")
        if not reason:
            reason = "no finite objective score"
        detail = next(
            (
                str(metrics[key])
                for key in (
                    "cutting_error",
                    "compile_error",
                    "exec_error",
                    "mthree_error",
                    "zne_error",
                )
                if metrics.get(key)
            ),
            "",
        )
        reasons.setdefault(str(reason), detail)
    if not reasons:
        return ""
    parts = [
        f"{reason} ({detail})" if detail else reason
        for reason, detail in sorted(reasons.items())
    ]
    return ": " + "; ".join(parts)


def _entropy_from_counts(counts: Dict[str, int]) -> float:
    """Return the Shannon entropy, in bits, of the outcome frequencies in ``counts``."""
    shots = sum(counts.values()) or 1
    ps = np.asarray([v / shots for v in counts.values()], dtype=float)
    ps = ps[ps > 0]
    return float(-(ps * np.log2(ps)).sum()) if len(ps) else 0.0


def _top_prob(counts: Dict[str, int]) -> float:
    """Return the frequency of the most common outcome in ``counts`` (0 when empty)."""
    shots = sum(counts.values()) or 1
    return float(max(counts.values()) / shots) if counts else 0.0
