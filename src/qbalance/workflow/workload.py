# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
from qbalance.strategies import Strategy, StrategySpec
from qbalance.transpile.pipeline import compile_one
from qbalance.transpile.suppression import apply_measurement_untwirl_counts

log = get_logger(__name__)

CircuitRecord = DatasetCircuitRecord


@dataclass
class BalancedWorkload:
    dataset: CircuitDataset
    backend_spec: str
    selections: Dict[str, Strategy]  # circuit_name -> Strategy
    baseline_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    objective: Objective = field(default_factory=default_objective)

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
                    value = m.get(k)
                    if isinstance(value, (int, float)):
                        vals.append(float(value))
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
            x1 = [float(m.get(k, 0)) for m in base_ms]
            x2 = [float(m.get(k, 0)) for m in sel_ms]
            lines.append(
                f"  dist[{k}]: EMD={emd_1d(x1,x2):.4g}  CVM={cvm_1d(x1,x2):.4g}  KS={ks_1d(x1,x2):.4g}"
            )
        return "\n".join(lines)

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
            x1 = [float(m.get(k, 0) or 0) for m in base_ms]
            x2 = [float(m.get(k, 0) or 0) for m in sel_ms]
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

        Returns:
            BalancedWorkload with the computed result.

        Raises:
            ValueError: Raised when input validation fails or a dependent operation cannot be completed.
        """
        if not self.backend_spec:
            raise ValueError(
                "Workload has no target backend; call set_target(...) first"
            )
        obj = objective or default_objective()
        backend = resolve_backend(self.backend_spec)
        rng = np.random.default_rng(seed)

        candidates = default_candidate_strategies(
            max_candidates=max_candidates, seed=seed
        )
        bandit = BanditSearcher()

        selections: Dict[str, Strategy] = {}
        baseline_metrics: Dict[str, Dict[str, Any]] = {}

        circuits = self.dataset.load_circuits()

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
            elif search == "bandit":
                # warmup random subset
                order = []
                perm = list(candidates)
                rng.shuffle(perm)
                order.extend(perm[: max(1, min(warmup, len(perm)))])
                # then propose until exhaustion or budget
                while len(order) < len(candidates):
                    proposed = bandit.propose(
                        [c for c in candidates if c not in order], rng=rng
                    )
                    order.append(proposed)
            else:
                raise ValueError("search must be 'grid' or 'bandit'")

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
                if search == "bandit":
                    bandit.observe(spec, m["objective_score"])

            # Pareto selection if requested (otherwise min score)
            chosen_spec, chosen_m = _choose(evals, pareto=pareto, objective=obj)
            selections[rec.name] = Strategy(spec=chosen_spec, metrics=chosen_m)

        return BalancedWorkload(
            dataset=self.dataset,
            backend_spec=self.backend_spec,
            selections=selections,
            baseline_metrics=baseline_metrics,
            objective=obj,
        )


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
    key = f"{backend_name}:{fpr}:{spec.model_dump_json()}"
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
    valid_objective_terms = objective._valid_weights

    def _derived_objective_score(metrics: Mapping[str, Any]) -> float:
        """Compute a finite-safe objective score from raw metrics."""
        derived = 0.0
        contributed = False
        has_objective_key = False
        for key, weight in valid_objective_terms:
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

    def _objective_score(metrics: Mapping[str, Any] | None) -> float:
        """Internal helper that objective score.

        Args:
            metrics: Mapping of metric names to numeric values used for scoring.

        Returns:
            float with the computed result.

        Raises:
            None.
        """
        if not isinstance(metrics, Mapping):
            return float("inf")

        try:
            score = float(metrics.get("objective_score", float("inf")))
        except (TypeError, ValueError, OverflowError):
            score = float("inf")
        derived = _derived_objective_score(metrics)
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

    if not pareto:
        best = min(evals, key=lambda t: _objective_score(t[1]))
        return best[0], best[1]

    # Pareto on key metrics, then tie-break by objective_score.
    # We intentionally pass raw metric mappings to pareto_front,
    # which already performs robust finite-safe normalization.
    front_idx = pareto_front(evals, keys=pareto_keys)
    front = [evals[i] for i in front_idx]
    best = min(front, key=lambda t: _objective_score(t[1]))
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
