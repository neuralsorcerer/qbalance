# API reference

This page summarizes the public qbalance Python API and the lower-level modules most users extend or call directly.

## Package root: `qbalance`

The package root exports the primary workflow and dataset APIs:

| Symbol | Description |
| --- | --- |
| `CircuitDataset` | Dataset container with root path and records. |
| `load_dataset(path)` | Load a dataset directory containing `qbalance_dataset.json`. |
| `save_dataset(path, circuits, metadata=None, overwrite=False)` | Persist circuits and metadata as a qbalance dataset. |
| `load_data(name)` | Load built-in datasets such as `"tiny"`. |
| `Workload` | Fluent workflow entry point. |
| `BalancedWorkload` | Result object returned by `Workload.adjust(...)`. |
| `load_balanced_workload(out_dir)` | Reconstruct a saved balanced workload directory produced by `BalancedWorkload.save(...)`. |
| `Strategy` | Runtime selection object containing a `StrategySpec` plus metrics. |
| `StrategySpec` | Immutable strategy model. |
| `load_strategy_specs(path)` | Load, validate, normalize, and de-duplicate strategy specs from JSON. |
| `coerce_strategy_specs(strategies)` | Validate, normalize, and de-duplicate in-memory strategies. |

Example:

```python
from qbalance import StrategySpec, Workload, load_balanced_workload, load_data

balanced = (
    Workload.from_dataset(load_data("tiny"))
    .set_target("fake:generic:5")
    .adjust(strategies=[StrategySpec(optimization_level=1, routing_method="sabre")])
)
balanced.save("./balanced", overwrite=True)
reloaded = load_balanced_workload("./balanced")
assert reloaded.backend_spec == balanced.backend_spec
```

## Strategies: `qbalance.strategies`

### `StrategySpec`

`StrategySpec` is frozen and hashable, so it can be used for caching, de-duplication, and reproducible serialization.

Important validation rules:

- `optimization_level` must be `0..3`.
- Boolean values are rejected for integer-like fields: `optimization_level`, `num_twirls`, `zne_degree`, `max_subcircuit_qubits`, `resilience_level`, `seed_transpiler`, and `seed_suppression`.
- `num_twirls >= 1`.
- `zne_degree >= 0`; when `zne=True`, it must also be at least `1` and smaller than `len(zne_factors)`.
- When `zne=True`, `zne_factors` must be non-empty, finite, sorted, all `>= 1.0`, and include `1.0`.
- When `cutting=True`, `max_subcircuit_qubits` is required.
- If `max_subcircuit_qubits` is provided, it must be an integer `>= 1`.
- `resilience_level` must be `None`, `0`, `1`, or `2`.
- Unknown fields are rejected. A misspelled key such as `optimisation_level` would otherwise be dropped silently, leaving a default strategy and a run that reports several visibly different entries as one identical configuration.

### `load_strategy_specs(path: Path | str) -> list[StrategySpec]`

Loads JSON strategy definitions from these supported shapes:

- a single strategy object,
- a list of strategy objects,
- `{"strategies": [...]}`,
- saved balanced workload results with `{"selections": {name: {"spec": ...}}}`,
- matrix results with `{"results": [{"strategy": ...}]}`.

The function rejects malformed JSON, unreadable files, unsupported container shapes, non-object entries, and invalid strategy field combinations. Duplicate strategies are removed while preserving first-seen order.

### `coerce_strategy_specs(strategies: Iterable[StrategySpec | Mapping[str, Any]]) -> list[StrategySpec]`

Normalizes an in-memory iterable of `StrategySpec` instances or mapping objects. Strings/bytes and non-iterables are rejected. Empty iterables are rejected. Duplicate normalized strategies are removed.

## Objectives: `qbalance.objectives`

- `Objective(weights)`: immutable weighted minimization objective. Non-finite or non-numeric weights/metric values are ignored during scoring.
- `default_objective()`: returns the built-in depth/two-qubit/error/compile-time weighting used by `Workload.adjust(...)` when no custom objective is supplied.
- `load_objective(path)`: loads objective weights from JSON. The file may be a direct metric-to-weight object such as `{"depth": 1.0}`, an object with a `weights` mapping, or a saved-results-style object with an `objective` mapping; malformed files, empty mappings, blank metric names, booleans, non-numeric weights, and non-finite weights raise `ValueError`.

## Workflow: `qbalance.workflow.workload`

### `Workload`

Constructors:

- `Workload.from_dataset(dataset)`
- `Workload.from_path(dataset_dir)`

Methods:

- `set_target(backend_spec) -> Workload`
- `adjust(...) -> BalancedWorkload`

`adjust` parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `objective` | `None` | Objective instance; defaults to `default_objective()`. |
| `search` | `"grid"` | Candidate ordering mode: `"grid"` or `"bandit"`. |
| `pareto` | `False` | If true, select from the Pareto front before objective tie-break. |
| `max_candidates` | `24` | Number of generated candidates when `strategies` is not supplied. |
| `warmup` | `6` | Number of randomly ordered candidates evaluated before bandit proposals. `0` starts from the bandit prior. |
| `execute` | `False` | Execute compiled circuits and collect counts/mitigation metrics. Mitigation strategies such as M3/ZNE also trigger execution. |
| `shots` | `1024` | Positive integer execution-shot count. Boolean values are rejected. |
| `profile` | `False` | Enable pass-level transpiler profiling. |
| `cache_root` | `None` | Optional compiled-circuit cache root. `str` and `Path` values are accepted and normalized to `Path`. |
| `seed` | `0` | Integer seed for deterministic candidate ordering and execution helpers. Boolean values are rejected. |
| `strategies` | `None` | Explicit iterable of strategies. If provided, `max_candidates` is ignored and the supplied order is used for grid search. |
| `allow_regression` | `True` | Boolean safety rail. When `False`, final selection falls back to the baseline strategy if the best feasible candidate has a finite-safe objective score worse than the finite-safe baseline score. Equal scores and incomparable baselines do not trigger fallback. |

Validation and edge-case behavior:

- `search` must be exactly `"grid"` or `"bandit"`.
- `shots` must be a positive integer; `seed` must be an integer; `warmup` must be a non-negative integer. Integral scalar types are accepted, but booleans are rejected.
- `max_candidates` must be a positive integer when generated candidates are used. It is not consulted when explicit `strategies` are supplied.
- At least one candidate strategy must be available after generated or explicit strategy normalization.
- `allow_regression` must be a real boolean (`True` or `False`).
- `search="bandit"` evaluates up to `warmup` shuffled candidates first, then asks the bandit model to propose the remaining candidates one at a time, observing each evaluation before the next proposal. Non-finite objective scores are not observed by the bandit model, but they remain in the evaluated candidates and are handled by final selection.
- With `allow_regression=False`, qbalance still records the full candidate evaluation history. Only the final selected strategy changes: if the selected candidate score is worse than the baseline score, the saved selection uses the baseline spec and baseline metrics plus guard metadata (`selected_by_regression_guard`, `rejected_candidate_spec`, and `rejected_candidate_objective_score`).

### `BalancedWorkload`

Properties:

- `dataset`
- `backend_spec`
- `selections`: mapping from circuit name to the selected `Strategy`.
- `baseline_metrics`: mapping from circuit name to baseline compile metrics.
- `objective`: the objective used to score candidates.
- `evaluation_history`: optional mapping from circuit name to every evaluated candidate `Strategy`, in evaluation order. This is empty for legacy artifacts that predate candidate-history persistence.

Methods:

- `summary() -> str`: human-readable baseline-vs-balanced aggregate summary. When `evaluation_history` is present, the summary includes total and mean candidate-evaluation counts. When objective scores are comparable, it also reports the mean selected-minus-baseline objective delta and the number of comparable circuits whose objective improved.
- `selection_diagnostics() -> dict`: per-circuit baseline-vs-selected diagnostics computed from already stored metrics. Each entry includes `baseline_objective_score`, `selected_objective_score`, `objective_delta`, `objective_improved`, per-objective `objective_terms`, `evaluated_candidates`, and `metric_deltas` for `depth`, `two_qubit_ops`, `estimated_error`, and `compile_time_s`. Missing, non-numeric, NaN, and infinite metric values are normalized to `None`; objective scores are `None` when no finite weighted term contributes, so invalid metrics are not mistaken for zero-cost candidates.
- `candidate_rankings() -> dict`: per-circuit candidate leaderboards derived from `evaluation_history`, sorted by the same finite-safe selection score used by final strategy selection and then by original evaluation order. Each row contains `rank`, `original_index`, serialized `spec`, diagnostic `objective_score`, JSON-safe `selection_score`, finite weighted `objective_terms`, and a `selected` marker. `objective_score` is recomputed from finite objective terms for review; `selection_score` mirrors selection semantics, including stored finite `metrics["objective_score"]` when it is valid. Candidates with malformed, missing, or non-finite objective information get `selection_score=None` in the exported leaderboard and sort after comparable candidates. If `allow_regression=False` falls back to a baseline that was not part of the candidate list, the baseline is included as the selected ranking row with `original_index: null` so saved audit artifacts still identify the final selection exactly once.
- `covars() -> dict`: EMD/CVM/KS diagnostics for depth, two-qubit ops, and estimated error. Missing, non-numeric, NaN, and infinite metric values are treated as `0.0` for these distribution-distance summaries.
- `save(out_dir, overwrite=False) -> None`: save dataset copy, `results.json`, and `summary.txt`. The results file contains selections, baselines, objective weights, `selection_diagnostics`, `candidate_rankings`, and `evaluation_history`.
- `to_download(zip_path, overwrite=False) -> Path`: save a ZIP bundle.

### `load_balanced_workload(out_dir: Path | str) -> BalancedWorkload`

Reloads a directory written by `BalancedWorkload.save(...)`. The loader reconstructs the copied dataset, selected `Strategy` objects, baseline metrics, full candidate evaluation history, and `Objective` weights without re-running compilation or execution. Per-circuit selection diagnostics and candidate rankings are recomputed from the loaded metrics/history, so older artifacts that omit `selection_diagnostics` or `candidate_rankings` remain loadable. Artifacts written before `evaluation_history` existed also remain loadable; missing or `null` history is treated as an empty mapping. It validates the saved payload before returning:

- `results.json` must be a JSON object with a non-empty string `backend_spec`;
- `objective`, `selections`, `metrics`, `baseline_metrics`, and `evaluation_history` entries must have the expected JSON shapes where present;
- saved `selection_diagnostics` and `candidate_rankings`, if present, are treated as derived metadata and are not required for loading;
- every saved selection and evaluation-history entry must contain a strategy `spec` object and optional metrics object;
- every saved strategy spec must pass `StrategySpec` validation;
- selection names must exactly match the bundled dataset records;
- baseline metric names and evaluation-history names, if present, must refer to bundled dataset records.

Use it for offline inspection or follow-up processing of saved adjustment results:

```python
from qbalance import load_balanced_workload

balanced = load_balanced_workload("./balanced")
print(balanced.summary())
```

## Benchmarking: `qbalance.benchmarking`

### `run_matrix(dataset_dir, backend_specs, strategies, out_json, execute=False, shots=1024, seed=0, profile=False) -> Path`

Evaluates every `(backend, circuit, strategy)` combination and writes matrix JSON. It validates:

- `shots` is a positive integer and not a boolean,
- `seed` is an integer and not a boolean,
- `backend_specs` is a non-empty sequence of backend specs, not a single string/bytes value,
- `strategies` is a non-empty sequence of strategy objects, not a single string/bytes value,
- loaded circuit count matches dataset record count.

The JSON payload contains `version`, `metadata`, and `results`. `metadata` records `dataset_dir`, `backends`, `execute`, `shots`, `seed`, and `profile` so matrix artifacts are self-describing. When `execute=True`, each result's metrics can include counts and shot totals. If a strategy has `zne=True`, qbalance also runs folded circuits for `zne_factors` and stores extrapolated probabilities.

## Search: `qbalance.search`

- `default_candidate_strategies(max_candidates=24, seed=0)`: deterministic default candidate pool.
- `pareto_front(evals, keys)`: finite-safe Pareto filtering over metric keys.
- `BanditSearcher`: Thompson-sampling-style adaptive candidate ordering used by `search="bandit"`.

## Transpilation: `qbalance.transpile`

- `compile_one(circuit, backend, spec, profile=False)`: compile a circuit under one strategy and return `(compiled_circuit, metrics)`.
- `build_dd_pass_manager(backend, sequence="XY4")`: create a dynamical-decoupling pass manager where Qiskit support is available.
- `apply_pauli_twirling(circuit, num_twirls=1, seed=None)`: best-effort Pauli twirling transform.
- `qbalance.transpile.suppression.apply_measurement_untwirl_counts(counts, flip_map)`: reverse measurement-twirling bit flips on counts.

Common compile metrics include depth, size, two-qubit operation count, estimated error, compile time, optional measurement flip maps, and optional pass profiles.

## Execution and mitigation

- `qbalance.execution.run_counts(backend, circuit, shots=1024, seed_simulator=None, seed_transpiler=None)`: execute a circuit on a backend-like object and return counts. Seeds are forwarded only to `run` implementations that accept them.
- `qbalance.mitigation.apply_mthree_mitigation(backend, raw_counts, measured_qubits, shots, calibration_shots=10000)`: optional M3 mitigation integration. `measured_qubits` must list the physical qubit feeding each classical bit, ordered by classical bit index; `qbalance.workflow.workload` derives it from the compiled circuit.
- `qbalance.mitigation.runtime_options.build_runtime_estimator_options(...)`: builds an IBM Runtime EstimatorV2 options mapping. This is a standalone helper: qbalance's own compile/execute path does not call it, and `StrategySpec.resilience_level` is likewise carried through to artifacts as metadata rather than applied locally.
- `qbalance.mitigation.fold_global(circuit, factor)`: global folding helper for ZNE.
- `qbalance.mitigation.fold_global_for_backend(circuit, backend, factor)`: fold a compiled circuit and re-express it in the backend's native basis. Folding appends `circuit.inverse()`, which introduces adjoint gates (for example `sxdg`) that are outside the backend basis, so the ZNE workflow uses this helper to keep folded circuits runnable. The qubit layout and the measurement clbit mapping are preserved.
- `qbalance.mitigation.zne_extrapolate_counts(factors, counts_per_factor, degree=1)`: extrapolate probability distributions back to zero noise.

## Backend plugins

Backends are resolved by backend spec strings and entry-point plugins.

Built-in entry points include:

- `fake`: `fake:generic:<num_qubits>` (optionally `fake:generic:<num_qubits>:<calibration_seed>`; calibration data is seeded deterministically, seed 0 by default) and `fake:ibm:<name>` device snapshots via `qiskit-ibm-runtime` where available.
- `aer`: qiskit-aer backends when the `aer` extra is installed.

Third-party packages can register `qbalance.backends` entry points. `qbalance plugins list` shows discovered plugin groups.

## Reports

- `qbalance.reports.markdown.render_markdown(matrix_json, out_dir) -> Path`
- `qbalance.reports.html.render_html(matrix_json, out_dir) -> Path`

Markdown reports require only base dependencies. HTML reports require the optional report dependencies.
