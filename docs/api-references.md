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
| `warmup` | `6` | Bandit warmup count. |
| `execute` | `False` | Execute compiled circuits and collect counts/mitigation metrics. |
| `shots` | `1024` | Execution shots. |
| `profile` | `False` | Enable pass-level transpiler profiling. |
| `cache_root` | `None` | Optional compiled-circuit cache root. |
| `seed` | `0` | Deterministic seed for search/execution helpers. |
| `strategies` | `None` | Explicit iterable of strategies. If provided, `max_candidates` is ignored. |

### `BalancedWorkload`

Properties:

- `dataset`
- `backend_spec`
- `selections`: mapping from circuit name to selected `Strategy`
- `baseline_metrics`
- `objective`

Methods:

- `summary() -> str`: human-readable baseline-vs-balanced aggregate summary.
- `covars() -> dict`: EMD/CVM/KS diagnostics for depth, two-qubit ops, and estimated error.
- `save(out_dir, overwrite=False) -> None`: save dataset copy, `results.json`, and `summary.txt`.
- `to_download(zip_path, overwrite=False) -> Path`: save a ZIP bundle.

### `load_balanced_workload(out_dir: Path | str) -> BalancedWorkload`

Reloads a directory written by `BalancedWorkload.save(...)`. The loader reconstructs the copied dataset, selected `Strategy` objects, baseline metrics, and `Objective` weights without re-running compilation or execution. It validates the saved payload before returning:

- `results.json` must be a JSON object with a non-empty string `backend_spec`;
- `objective`, `selections`, `metrics`, and `baseline_metrics` entries must be JSON objects where present;
- every saved strategy spec must pass `StrategySpec` validation;
- selection names must exactly match the bundled dataset records;
- baseline metric names, if present, must refer to bundled dataset records.

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
- loaded circuit count matches dataset record count.

When `execute=True`, it records counts and shot totals. If a strategy has `zne=True`, it also runs folded circuits for `zne_factors` and stores extrapolated probabilities.

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

- `qbalance.execution.run_counts(backend, circuit, shots=1024, seed_simulator=None)`: execute a circuit on a backend-like object and return counts.
- `qbalance.mitigation.apply_mthree_mitigation(...)`: optional M3 mitigation integration.
- `qbalance.mitigation.fold_global(circuit, factor)`: global folding helper for ZNE.
- `qbalance.mitigation.zne_extrapolate_counts(factors, counts_per_factor, degree=1)`: extrapolate probability distributions back to zero noise.

## Backend plugins

Backends are resolved by backend spec strings and entry-point plugins.

Built-in entry points include:

- `fake`: `fake:generic:<num_qubits>` and fake-provider style backends where available.
- `aer`: qiskit-aer backends when the `aer` extra is installed.

Third-party packages can register `qbalance.backends` entry points. `qbalance plugins list` shows discovered plugin groups.

## Reports

- `qbalance.reports.markdown.render_markdown(matrix_json, out_dir) -> Path`
- `qbalance.reports.html.render_html(matrix_json, out_dir) -> Path`

Markdown reports require only base dependencies. HTML reports require the optional report dependencies.
