# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from qbalance.backends import resolve_backend
from qbalance.cache import fingerprint_circuit, get_entry, load_compiled, save_compiled
from qbalance.cutting.addon_cutting import find_cuts_best_effort
from qbalance.dataset import CircuitDataset
from qbalance.dataset import CircuitRecord as DatasetCircuitRecord
from qbalance.dataset import load_dataset
from qbalance.diagnostics.distribution import cvm_1d, emd_1d, ks_1d
from qbalance.execution import run_counts
from qbalance.logging import get_logger
from qbalance.mitigation.mthree import apply_mthree_mitigation
from qbalance.mitigation.zne import fold_global, zne_extrapolate_counts
from qbalance.objectives import Objective, default_objective
from qbalance.search import BanditSearcher, default_candidate_strategies, pareto_front
from qbalance.strategies import Strategy, StrategySpec, coerce_strategy_specs
from qbalance.transpile.pipeline import compile_one
from qbalance.transpile.suppression import apply_measurement_untwirl_counts
from qbalance.utils import validate_integral

log = get_logger(__name__)

CircuitRecord = DatasetCircuitRecord


@dataclass
class BalancedWorkload:
    dataset: CircuitDataset
    backend_spec: str
    selections: Dict[str, Strategy]  # circuit_name -> Strategy
    baseline_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    objective: Objective = field(default_factory=default_objective)
    evaluation_history: Dict[str, List[Strategy]] = field(default_factory=dict)

    def summary(self) -> str:
        # Compare baseline vs selected over depth + two_qubit_ops + estimated_error

        """Summary used by the qbalance workflow.

        Args:
            None.

        Returns:
            str with the computed result.

        Raises:
            None.
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
            """Agg used by the qbalance workflow.

            Args:
                ms: Ms value consumed by this routine.

            Returns:
                Dict[str, float] with the computed result.

            Raises:
                None.
            """
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
            x1 = [_finite_float_or_default(m.get(k), 0.0) for m in base_ms]
            x2 = [_finite_float_or_default(m.get(k), 0.0) for m in sel_ms]
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
            item
            for item in diagnostics.values()
            if item.get("objective_improved") is not None
        ]
        improved = sum(1 for item in comparable if item["objective_improved"] is True)
        if objective_deltas:
            lines.append(
                "  objective deltas: "
                f"mean={float(np.mean(objective_deltas)):.4g} "
                f"improved={improved}/{len(comparable)}"
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

            score_delta = (
                selected_score - baseline_score
                if baseline_score is not None and selected_score is not None
                else None
            )
            diagnostics[name] = {
                "baseline_objective_score": baseline_score,
                "selected_objective_score": selected_score,
                "objective_delta": score_delta,
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
                score, terms = _diagnostic_objective_score(
                    self.objective, strategy.metrics or {}
                )
                sort_score = _objective_score(strategy.metrics, self.objective)
                rows.append(
                    {
                        "original_index": original_index,
                        "spec": strategy.spec.model_dump(),
                        "objective_score": score,
                        "selection_score": (
                            sort_score if math.isfinite(sort_score) else None
                        ),
                        "objective_terms": terms,
                        "selected": (
                            selected is not None
                            and strategy.spec == selected.spec
                            and strategy.metrics == selected.metrics
                        ),
                    }
                )

            rows.sort(
                key=lambda row: (
                    row["selection_score"] is None,
                    (
                        float("inf")
                        if row["selection_score"] is None
                        else float(row["selection_score"])
                    ),
                    int(row["original_index"]),
                )
            )
            for rank, row in enumerate(rows, start=1):
                row["rank"] = rank
            rankings[name] = rows

        return rankings

    def covars(self) -> Dict[str, Dict[str, float]]:
        # Return diagnostic distances for key metrics

        """Covars used by the qbalance workflow.

        Args:
            None.

        Returns:
            Dict[str, Dict[str, float]] with the computed result.

        Raises:
            None.
        """
        out: Dict[str, Dict[str, float]] = {}
        sel_ms = [s.metrics for s in self.selections.values()]
        base_ms = [self.baseline_metrics.get(n, {}) for n in self.selections.keys()]
        for k in ["depth", "two_qubit_ops", "estimated_error"]:
            x1 = [_finite_float_or_default(m.get(k), 0.0) for m in base_ms]
            x2 = [_finite_float_or_default(m.get(k), 0.0) for m in sel_ms]
            out[k] = {"emd": emd_1d(x1, x2), "cvm": cvm_1d(x1, x2), "ks": ks_1d(x1, x2)}
        return out

    def save(self, out_dir: Path, overwrite: bool = False) -> None:
        """Save used by the qbalance workflow.

        Args:
            out_dir: Out dir value consumed by this routine.
            overwrite (default: False): Whether existing files/directories may be replaced.

        Returns:
            None. This method updates state or performs side effects only.

        Raises:
            FileExistsError: Raised when input validation fails or a dependent operation cannot be completed.
        """
        out_dir = Path(out_dir)
        if out_dir.exists():
            if not overwrite:
                raise FileExistsError(f"{out_dir} exists (use overwrite=True)")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy dataset index and artifacts, plus selection metadata
        (out_dir / "dataset").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            self.dataset.root / "qbalance_dataset.json",
            out_dir / "dataset" / "qbalance_dataset.json",
        )
        for rec in self.dataset.records:
            shutil.copy2(
                self.dataset.root / rec.artifact, out_dir / "dataset" / rec.artifact
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
        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        (out_dir / "summary.txt").write_text(self.summary() + "\n", encoding="utf-8")

    def to_download(self, zip_path: Path, overwrite: bool = False) -> Path:
        """To download used by the qbalance workflow.

        Args:
            zip_path: Zip path value consumed by this routine.
            overwrite (default: False): Whether existing files/directories may be replaced.

        Returns:
            Path with the computed result.

        Raises:
            FileExistsError: Raised when input validation fails or a dependent operation cannot be completed.
        """
        zip_path = Path(zip_path)
        if zip_path.exists() and not overwrite:
            raise FileExistsError(f"{zip_path} exists (use overwrite=True)")
        tmp = zip_path.parent / (zip_path.stem + "_dir")
        if tmp.exists():
            shutil.rmtree(tmp)
        self.save(tmp, overwrite=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in tmp.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(tmp))
        shutil.rmtree(tmp, ignore_errors=True)
        return zip_path


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
    dataset: CircuitDataset
    backend_spec: Optional[str] = None

    @classmethod
    def from_dataset(cls, dataset: CircuitDataset) -> "Workload":
        """From dataset used by the qbalance workflow.

        Args:
            dataset: Dataset value consumed by this routine.

        Returns:
            'Workload' with the computed result.

        Raises:
            None.
        """
        return cls(dataset=dataset)

    @classmethod
    def from_path(cls, dataset_dir: Path) -> "Workload":
        """From path used by the qbalance workflow.

        Args:
            dataset_dir: Directory containing the dataset index and circuit artifacts.

        Returns:
            'Workload' with the computed result.

        Raises:
            None.
        """
        return cls(dataset=load_dataset(dataset_dir))

    def set_target(self, backend_spec: str) -> "Workload":
        """Set target used by the qbalance workflow.

        Args:
            backend_spec: Backend spec value consumed by this routine.

        Returns:
            'Workload' with the computed result.

        Raises:
            None.
        """
        return Workload(dataset=self.dataset, backend_spec=backend_spec)

    def adjust(
        self,
        objective: Optional[Objective] = None,
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
    ) -> BalancedWorkload:
        """Adjust used by the qbalance workflow.

        Args:
            objective (default: None): Objective value consumed by this routine.
            search (default: 'grid'): Search value consumed by this routine.
            pareto (default: False): Pareto value consumed by this routine.
            max_candidates (default: 24): Max candidates value consumed by this routine.
            warmup (default: 6): Warmup value consumed by this routine.
            execute (default: False): Whether to run compiled circuits and collect counts.
            shots (default: 1024): Number of shots used when executing circuits on a backend.
            profile (default: False): Whether pass-level transpiler profiling is enabled.
            cache_root (default: None): Cache root value consumed by this routine.
            seed (default: 0): Seed used for deterministic randomization.
            strategies (default: None): Explicit candidate strategies. When provided,
                max_candidates is ignored and the supplied order is used for grid search.

        Returns:
            BalancedWorkload with the computed result.

        Raises:
            ValueError: Raised when input validation fails or a dependent operation cannot be completed.
        """
        if not self.backend_spec:
            raise ValueError(
                "Workload has no target backend; call set_target(...) first"
            )
        if search not in {"grid", "bandit"}:
            raise ValueError("search must be 'grid' or 'bandit'")
        shots = validate_integral("shots", shots, positive=True)
        seed = validate_integral("seed", seed)
        warmup = validate_integral("warmup", warmup, non_negative=True)
        if strategies is None:
            max_candidates = validate_integral(
                "max_candidates", max_candidates, positive=True
            )

        obj = objective or default_objective()
        backend = resolve_backend(self.backend_spec)
        rng = np.random.default_rng(seed)
        cache_root = Path(cache_root) if cache_root is not None else None

        candidates = (
            coerce_strategy_specs(strategies)
            if strategies is not None
            else default_candidate_strategies(max_candidates=max_candidates, seed=seed)
        )
        if not candidates:
            raise ValueError("at least one candidate strategy is required")
        bandit = BanditSearcher()

        selections: Dict[str, Strategy] = {}
        baseline_metrics: Dict[str, Dict[str, Any]] = {}
        evaluation_history: Dict[str, List[Strategy]] = {}

        circuits = self.dataset.load_circuits()
        if len(circuits) != len(self.dataset.records):
            raise RuntimeError(
                "Dataset load_circuits() returned a circuit count that does not match "
                "the number of dataset records"
            )

        # Baseline compile (single default spec)
        baseline_spec = StrategySpec(optimization_level=1, routing_method="sabre")
        for qc, rec in zip(circuits, self.dataset.records):
            compiled, m = _compile_cached(
                qc, backend, baseline_spec, profile=profile, cache_root=cache_root
            )
            baseline_metrics[rec.name] = m

        for qc, rec in zip(circuits, self.dataset.records):
            # choose candidate evaluation order
            order: List[StrategySpec] = []
            if search == "grid":
                order = list(candidates)
            else:
                # warmup random subset; warmup=0 intentionally starts from the
                # bandit's prior and proposes every candidate adaptively.
                order = []
                perm = list(candidates)
                rng.shuffle(perm)
                order.extend(perm[: min(warmup, len(perm))])
                # then propose until exhaustion or budget
                while len(order) < len(candidates):
                    proposed = bandit.propose(
                        [c for c in candidates if c not in order], rng=rng
                    )
                    order.append(proposed)

            evals: List[Tuple[StrategySpec, Dict[str, Any]]] = []
            for spec in order:
                # apply circuit cutting before compile if requested
                working = qc
                cut_meta = None
                if spec.cutting and spec.max_subcircuit_qubits:
                    try:
                        working, cut_meta = find_cuts_best_effort(
                            working, spec.max_subcircuit_qubits
                        )
                    except Exception:
                        # if cutting fails, skip this candidate
                        continue

                compiled, m = _compile_cached(
                    working, backend, spec, profile=profile, cache_root=cache_root
                )

                # optional execution for mitigation or if execute=True
                if execute or spec.mthree or spec.zne:
                    try:
                        counts = run_counts(
                            backend, compiled, shots=shots, seed_simulator=seed
                        )
                        # undo measurement twirling flips if present
                        flip_map = m.get("measurement_flip_map") or {}
                        counts = apply_measurement_untwirl_counts(counts, flip_map)
                        m["raw_counts_entropy"] = _entropy_from_counts(counts)
                        m["raw_top_prob"] = _top_prob(counts)
                        if spec.mthree:
                            try:
                                measured = list(range(compiled.num_qubits))
                                probs = apply_mthree_mitigation(
                                    backend,
                                    counts,
                                    measured_qubits=measured,
                                    shots=shots,
                                )
                                m["mitigated_top_prob"] = float(
                                    max(probs.values()) if probs else 0.0
                                )
                            except Exception as e:
                                m["mthree_error"] = str(e)
                        if spec.zne:
                            try:
                                factors = list(spec.zne_factors)
                                counts_pf = []
                                for f in factors:
                                    c_fold = fold_global(compiled, f)
                                    cts = run_counts(
                                        backend,
                                        c_fold,
                                        shots=shots,
                                        seed_simulator=seed,
                                    )
                                    cts = apply_measurement_untwirl_counts(
                                        cts, flip_map
                                    )
                                    counts_pf.append(cts)
                                probs = zne_extrapolate_counts(
                                    factors, counts_pf, degree=spec.zne_degree
                                )
                                m["zne_top_prob"] = float(
                                    max(probs.values()) if probs else 0.0
                                )
                            except Exception as e:
                                m["zne_error"] = str(e)
                    except Exception as e:
                        m["exec_error"] = str(e)

                # score and observe for bandit
                m["objective_score"] = obj.score(m)
                evals.append((spec, m))
                if search == "bandit" and _is_finite_number(m["objective_score"]):
                    bandit.observe(spec, m["objective_score"])

            # Pareto selection if requested (otherwise min score)
            evaluation_history[rec.name] = [
                Strategy(spec=spec, metrics=dict(metrics)) for spec, metrics in evals
            ]
            chosen_spec, chosen_m = _choose(evals, pareto=pareto, objective=obj)
            selections[rec.name] = Strategy(spec=chosen_spec, metrics=chosen_m)

        return BalancedWorkload(
            dataset=self.dataset,
            backend_spec=self.backend_spec,
            selections=selections,
            baseline_metrics=baseline_metrics,
            objective=obj,
            evaluation_history=evaluation_history,
        )


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


def _finite_float_or_default(value: Any, default: float) -> float:
    """Return a finite float or a deterministic fallback value."""
    value_f = _finite_float_or_none(value)
    return default if value_f is None else value_f


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


def _objective_score(metrics: Mapping[str, Any] | None, objective: Objective) -> float:
    """Return the same finite-safe selection score used by workload choice."""
    if not isinstance(metrics, Mapping):
        return float("inf")

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


def _compile_cached(
    circuit: Any,
    backend: Any,
    spec: StrategySpec,
    profile: bool,
    cache_root: Optional[Path],
) -> Tuple[Any, Dict[str, Any]]:
    # Cache key depends on circuit fingerprint + backend name + spec

    """Internal helper that compile cached.

    Args:
        circuit: QuantumCircuit instance to inspect, transform, or execute.
        backend: Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        spec: Strategy/backend specification controlling compilation behavior.
        profile: Whether pass-level transpiler profiling is enabled.
        cache_root: Cache root value consumed by this routine.

    Returns:
        Tuple[Any, Dict[str, Any]] with the computed result.

    Raises:
        None.
    """
    try:
        fpr = fingerprint_circuit(circuit)
    except Exception:
        fpr = str(hash(str(circuit)))
    backend_name = getattr(backend, "name", None)
    if callable(backend_name):
        backend_name = backend.name()
    backend_name = str(backend_name or backend.__class__.__name__)
    key = f"{backend_name}:{fpr}:{spec.model_dump_json()}:profile={profile}"
    import hashlib

    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    entry = get_entry(key_hash, root=cache_root)
    hit = load_compiled(entry)
    if hit is not None:
        c, m = hit
        return c, m

    compiled, m = compile_one(circuit, backend=backend, spec=spec, profile=profile)
    save_compiled(entry, compiled, m)
    return compiled, m


def _choose(
    evals: List[Tuple[StrategySpec, Dict[str, Any]]],
    pareto: bool,
    objective: Objective,
) -> Tuple[StrategySpec, Dict[str, Any]]:
    """Internal helper that choose.

    Args:
        evals: Evals value consumed by this routine.
        pareto: Pareto value consumed by this routine.
        objective: Objective value consumed by this routine.

    Returns:
        Tuple[StrategySpec, Dict[str, Any]] with the computed result.

    Raises:
        RuntimeError: Raised when input validation fails or a dependent operation cannot be completed.
    """
    if not evals:
        raise RuntimeError("No candidate strategies were successfully evaluated")

    pareto_keys = ("depth", "two_qubit_ops", "estimated_error")

    if not pareto:
        best = min(evals, key=lambda t: _objective_score(t[1], objective))
        return best[0], best[1]

    # Pareto on key metrics, then tie-break by objective_score.
    # We intentionally pass raw metric mappings to pareto_front,
    # which already performs robust finite-safe normalization.
    front_idx = pareto_front(evals, keys=pareto_keys)
    front = [evals[i] for i in front_idx]
    best = min(front, key=lambda t: _objective_score(t[1], objective))
    return best[0], best[1]


def _entropy_from_counts(counts: Dict[str, int]) -> float:
    """Internal helper that entropy from counts.

    Args:
        counts: Counts value consumed by this routine.

    Returns:
        float with the computed result.

    Raises:
        None.
    """
    shots = sum(counts.values()) or 1
    ps = np.asarray([v / shots for v in counts.values()], dtype=float)
    ps = ps[ps > 0]
    return float(-(ps * np.log2(ps)).sum()) if len(ps) else 0.0


def _top_prob(counts: Dict[str, int]) -> float:
    """Internal helper that top prob.

    Args:
        counts: Counts value consumed by this routine.

    Returns:
        float with the computed result.

    Raises:
        None.
    """
    shots = sum(counts.values()) or 1
    return float(max(counts.values()) / shots) if counts else 0.0
