# API reference

This page documents every public name in qbalance, module by module: what it does, its parameters and validation rules, the formulas it computes, and an example with the output it prints.

## Reading the examples

The examples form one Python session: they run top to bottom, later ones reuse names defined earlier, and they start in an empty working directory. Every output shown is the real output, and `tests/test_docs_examples.py` re-runs the page to keep it that way. They need the optional extras (`pip install "qbalance[all]"`). Numbers that come out of Qiskit's transpiler were produced with Qiskit 2.5.2, qiskit-aer 0.17.2, mthree 3.0.0, qiskit-addon-cutting 0.10.0 and qiskit-ibm-runtime 0.50.0, and can differ under other versions. `...` in an output stands for text that changes from run to run, such as wall-clock timings.

The session starts with two helpers:

```pycon
>>> import json
>>> from pathlib import Path
>>> from pprint import pprint
>>> from qiskit import QuantumCircuit
>>> def ghz(n, name=None):
...     """An n-qubit GHZ circuit that measures every qubit."""
...     qc = QuantumCircuit(n, n, name=name or f"ghz{n}")
...     qc.h(0)
...     for q in range(n - 1):
...         qc.cx(q, q + 1)
...     qc.measure(range(n), range(n))
...     return qc
>>> def write_json(path, payload):
...     Path(path).write_text(json.dumps(payload), encoding="utf-8")
```

## Package root (`qbalance`)

The package root re-exports the workflow, dataset, strategy, and objective APIs:

| Names | Section |
| --- | --- |
| `CircuitDataset`, `load_dataset`, `save_dataset`, `load_data` | [Datasets](#datasets-qbalancedataset) |
| `StrategySpec`, `Strategy`, `load_strategy_specs`, `coerce_strategy_specs` | [Strategies](#strategies-qbalancestrategies) |
| `Objective`, `default_objective`, `load_objective` | [Objectives](#objectives-qbalanceobjectives) |
| `Workload`, `BalancedWorkload`, `load_balanced_workload` | [Workflow](#workflow-qbalanceworkflow) |

### `__version__`

`qbalance.__version__: str`

The installed distribution's version, read from its metadata; the in-tree `qbalance/_version.py` is used only when qbalance runs from a source checkout that is not installed.

```pycon
>>> import qbalance
>>> qbalance.__version__
'0.2.0'
>>> qbalance.__all__
['__version__', 'CircuitDataset', 'load_dataset', 'save_dataset', 'load_data', 'Objective', 'default_objective', 'load_objective', 'Workload', 'BalancedWorkload', 'load_balanced_workload', 'Strategy', 'StrategySpec', 'coerce_strategy_specs', 'load_strategy_specs']
```

## Datasets (`qbalance.dataset`)

A dataset is a directory holding one serialized circuit per record and an index, `qbalance_dataset.json` (see [artifacts](artifacts.md#dataset-layout)).

### `DATASET_INDEX`

`qbalance.dataset.DATASET_INDEX: str`

The index file name.

```pycon
>>> from qbalance.dataset import DATASET_INDEX
>>> DATASET_INDEX
'qbalance_dataset.json'
```

### `save_dataset`

`qbalance.save_dataset(dataset_dir, circuits, metadata=None, overwrite=False) -> CircuitDataset`

Serializes `circuits` to QPY and writes the index. Record names come from each circuit's `name` (`circuit_<i>` when it has none), and a repeated name gets a `_1`, `_2`, ... suffix. Artifact file names derive from the record names: characters outside `A-Z a-z 0-9 . _ -` become `_`, a stem longer than 120 characters is shortened and suffixed with a digest of the full name, Windows device names (`CON`, `NUL`, `COM1`, ...) get a `_` prefix, and names differing only in case get distinct files, so every record keeps its own artifact on case-insensitive filesystems. `metadata`, when given, holds one entry per circuit: `None`, or a JSON-serializable mapping with string keys (stored as a JSON round-trip copy). The dataset is written to a temporary sibling directory and then moved into place (see `replacing_directory` under [utilities](#utilities-qbalanceutils)), so an interrupted save never leaves a partial dataset, and a failed overwrite keeps the existing one.

Raises `FileExistsError` when `dataset_dir` exists (a dangling symbolic link included) and `overwrite` is false, and `ValueError` for metadata of the wrong length or shape.

```pycon
>>> from qbalance import save_dataset
>>> ds = save_dataset(
...     "circuits",
...     [ghz(2, "bell"), ghz(3), ghz(3)],
...     metadata=[{"family": "ghz"}, None, None],
... )
>>> ds.names()
['bell', 'ghz3', 'ghz3_1']
>>> [record.artifact for record in ds.records]
['bell.qpy', 'ghz3.qpy', 'ghz3_1.qpy']
>>> sorted(path.name for path in Path("circuits").iterdir())
['bell.qpy', 'ghz3.qpy', 'ghz3_1.qpy', 'qbalance_dataset.json']
>>> save_dataset("circuits", [ghz(2)])
Traceback (most recent call last):
    ...
FileExistsError: circuits exists (use overwrite=True)
```

### `CircuitRecord`

`qbalance.dataset.CircuitRecord(name, artifact, format, metadata={})`

One index entry: the circuit's name, its artifact file name relative to the dataset root, the artifact format (`"qpy"` or `"qasm"`), and free-form JSON metadata.

```pycon
>>> ds.records[0]
CircuitRecord(name='bell', artifact='bell.qpy', format='qpy', metadata={'family': 'ghz'})
```

### `CircuitDataset`

`qbalance.CircuitDataset(root, records)`

A dataset root directory and its ordered records.

- `len(ds)`: the number of records.
- `names() -> list[str]`: the record names, in order.
- `iter_records()`: iterates over the records.
- `load_circuits() -> list[QuantumCircuit]`: deserializes every artifact (the first circuit of each QPY file). Raises `ValueError`, naming the record or file, for an unknown format or an unreadable, truncated, or empty artifact.
- `split(seed=0, frac_train=0.8) -> (train, test)`: shuffles the records with `numpy.random.default_rng(seed)` and puts the first $\operatorname{round}(f\,n)$ of them in `train` and the rest in `test`, for `frac_train` $f$ and $n$ records (Python's `round`, which rounds halves to even). `frac_train` must be a finite, non-boolean number in $[0, 1]$; `0` and `1` put every record, unshuffled, in one half. Both halves share the dataset root.

```pycon
>>> len(ds)
3
>>> [record.name for record in ds.iter_records()]
['bell', 'ghz3', 'ghz3_1']
>>> [circuit.num_qubits for circuit in ds.load_circuits()]
[2, 3, 3]
>>> train, test = ds.split(seed=0, frac_train=0.67)
>>> train.names(), test.names()
(['ghz3_1', 'bell'], ['ghz3'])
>>> ds.split(frac_train=1.5)
Traceback (most recent call last):
    ...
ValueError: frac_train must be in the inclusive range [0, 1].
```

### `load_dataset`

`qbalance.load_dataset(dataset_dir) -> CircuitDataset`

Reads and validates the index: it must be a JSON object with a `records` list, and every record needs a non-empty `name`, an `artifact` that is a single relative file name (no directory part, no `..`), a `format` of `"qpy"` or `"qasm"`, and optional object `metadata`. Names and artifacts must be unique, and every artifact file must exist. A violation raises `ValueError` naming the record.

```pycon
>>> from qbalance import load_dataset
>>> load_dataset("circuits").names()
['bell', 'ghz3', 'ghz3_1']
```

### `load_data`

`qbalance.load_data(name) -> CircuitDataset`

Loads a built-in dataset. `"tiny"` holds `bell` (2 qubits), `ghz3` (3 qubits), and `qft4` (a 4-qubit QFT built from controlled-phase gates); it is written to the platform's user data directory on first use and reused afterwards. An unknown name raises `KeyError`.

```pycon
>>> from qbalance import load_data
>>> tiny = load_data("tiny")
>>> tiny.names()
['bell', 'ghz3', 'qft4']
>>> [circuit.num_qubits for circuit in tiny.load_circuits()]
[2, 3, 4]
>>> load_data("huge")
Traceback (most recent call last):
    ...
KeyError: 'Unknown built-in dataset: huge'
```

### `get_builtin_dataset_dir`

`qbalance.builtin_data.get_builtin_dataset_dir(name) -> Path`

The directory `load_data` reads, `<user data dir>/datasets/<name>`, written on first use. The name is checked before it reaches the filesystem, so it cannot point outside the data directory.

```pycon
>>> from qbalance.builtin_data import get_builtin_dataset_dir
>>> tiny_dir = get_builtin_dataset_dir("tiny")
>>> tiny_dir.name, (tiny_dir / DATASET_INDEX).is_file()
('tiny', True)
```

## Strategies (`qbalance.strategies`)

### `StrategySpec`

`qbalance.StrategySpec(**fields)`

An immutable, hashable pydantic model of one strategy: its compilation, suppression, mitigation, cutting, and runtime knobs. The fields, their defaults, and what each does are listed in [strategy configuration](strategies.md#fields). Specs with equal field values are equal and hash alike, which caching and de-duplication rely on.

Validation (a violation raises pydantic's `ValidationError`, a subclass of `ValueError`):

- `optimization_level` is one of 0, 1, 2, 3.
- Integer fields reject booleans: `optimization_level`, `num_twirls`, `zne_degree`, `max_subcircuit_qubits`, `resilience_level`, `seed_transpiler`, `seed_suppression`.
- `num_twirls` $\ge 1$; `seed_transpiler` and `seed_suppression` are non-negative or `None`.
- `dd_sequence` is `"XY4"`, `"XX"`, or `"YY"`, case-insensitive and stored upper-case.
- `zne_degree` $\ge 0$. With `zne=True`, `zne_factors` is non-empty, finite, sorted, includes `1.0`, and has every factor $\ge 1$; `zne_degree` is at least 1, below `len(zne_factors)`, and below the number of distinct *realized* fold factors (folding rounds a factor up to an odd integer, so `(1, 2, 3)` realizes only 1 and 3; see [`realized_fold_factor`](#realized_fold_factor)).
- `max_subcircuit_qubits` is `None` or $\ge 1$. `cutting=True` requires it and excludes `pauli_twirling`, `measurement_twirling`, `mthree`, and `zne`: a cut circuit yields a reconstructed expectation value, not the count distribution those act on.
- `resilience_level` is `None`, 0, 1, or 2.
- Unknown fields are rejected, so a misspelled key fails instead of silently producing a default strategy.

```pycon
>>> from qbalance import StrategySpec
>>> spec = StrategySpec(
...     optimization_level=2,
...     layout_method="qbalance_noise_aware",
...     routing_method="sabre",
...     dynamical_decoupling=True,
...     dd_sequence="xx",
... )
>>> pprint(spec.model_dump(), sort_dicts=False)
{'optimization_level': 2,
 'layout_method': 'qbalance_noise_aware',
 'routing_method': 'sabre',
 'translation_method': None,
 'seed_transpiler': 0,
 'pauli_twirling': False,
 'num_twirls': 1,
 'dynamical_decoupling': True,
 'dd_sequence': 'XX',
 'measurement_twirling': False,
 'seed_suppression': 0,
 'mthree': False,
 'zne': False,
 'zne_factors': (1.0, 3.0, 5.0),
 'zne_degree': 1,
 'cutting': False,
 'max_subcircuit_qubits': None,
 'resilience_level': None}
>>> StrategySpec() == StrategySpec(optimization_level=1)
True
>>> len({StrategySpec(), StrategySpec(optimization_level=1), spec})
2
>>> def rejection(**fields):
...     try:
...         StrategySpec(**fields)
...     except ValueError as exc:
...         return exc.errors()[0]["msg"]
>>> rejection(optimization_level=4)
'Input should be less than or equal to 3'
>>> rejection(num_twirls=True)
'Value error, num_twirls must be an integer, not a boolean'
>>> rejection(dd_sequence="XY8")
'Value error, dd_sequence must be one of XX, XY4, YY'
>>> rejection(zne=True, zne_factors=(1.0, 2.0, 3.0), zne_degree=2)
'Value error, zne_degree must be less than the number of distinct fold factors zne_factors realize ([1, 3]): folding rounds each factor up to an odd integer'
>>> rejection(cutting=True)
'Value error, max_subcircuit_qubits must be set when cutting=True'
>>> rejection(cutting=True, max_subcircuit_qubits=4, mthree=True)
'Value error, cutting=True cannot be combined with mthree'
>>> rejection(optimisation_level=2)
'Extra inputs are not permitted'
```

### `Strategy`

`qbalance.Strategy(spec, metrics={}, artifact_dir=None)`

A `StrategySpec` paired with the metrics measured for it. `BalancedWorkload` stores its selections and evaluation history as `Strategy` objects.

```pycon
>>> from qbalance import Strategy
>>> chosen = Strategy(spec=StrategySpec(), metrics={"depth": 4})
>>> chosen.spec.optimization_level, chosen.metrics, chosen.artifact_dir
(1, {'depth': 4}, None)
```

### `load_strategy_specs`

`qbalance.load_strategy_specs(path) -> list[StrategySpec]`

Loads strategies from a JSON file holding a single strategy object, a list of them, `{"strategies": [...]}`, a saved workload's `results.json` (`{"selections": {name: {"spec": ...}}}`), or a matrix file (`{"results": [{"strategy": ...}]}`). Every entry is validated as a `StrategySpec`, and duplicates are dropped, keeping the first. Malformed JSON, an unreadable file, an unsupported shape, a non-object entry, or an invalid strategy raises `ValueError`.

```pycon
>>> from qbalance import load_strategy_specs
>>> write_json("strategies.json", {"strategies": [
...     {"optimization_level": 1, "routing_method": "sabre"},
...     {"optimization_level": 1, "routing_method": "sabre"},
...     {"optimization_level": 2, "measurement_twirling": True, "num_twirls": 8},
... ]})
>>> specs = load_strategy_specs("strategies.json")
>>> [(s.optimization_level, s.routing_method, s.measurement_twirling) for s in specs]
[(1, 'sabre', False), (2, None, True)]
```

### `coerce_strategy_specs`

`qbalance.coerce_strategy_specs(strategies, *, deduplicate=True) -> list[StrategySpec]`

The in-memory counterpart: validates an iterable of `StrategySpec` objects and mappings and, with `deduplicate`, drops repeats, keeping the first. A single strategy (a `StrategySpec` or mapping, which are iterable themselves), a string, bytes, or non-iterable input, an empty iterable, or an invalid entry raises `ValueError`.

```pycon
>>> from qbalance import coerce_strategy_specs
>>> unique = coerce_strategy_specs([{"optimization_level": 3}, StrategySpec(optimization_level=3)])
>>> len(unique), unique[0].optimization_level
(1, 3)
>>> coerce_strategy_specs("opt3")
Traceback (most recent call last):
    ...
ValueError: strategies must be an iterable of StrategySpec or mapping objects
>>> coerce_strategy_specs([])
Traceback (most recent call last):
    ...
ValueError: strategies must contain at least one strategy
>>> coerce_strategy_specs(StrategySpec())
Traceback (most recent call last):
    ...
ValueError: strategies must be an iterable of strategies, not a single strategy; wrap it in a list
>>> len(coerce_strategy_specs([StrategySpec(), StrategySpec()], deduplicate=False))
2
>>> try:
...     coerce_strategy_specs([{"optimization_level": 7}])
... except ValueError as exc:
...     print(str(exc).splitlines()[0])
Invalid strategy entry 0: 1 validation error for StrategySpec
```

## Objectives (`qbalance.objectives`)

### `Objective`

`qbalance.Objective(weights)`

A weighted objective to minimize. For metrics $m$ and weights $w$ it scores

$$
S(m) = \sum_{k} w_k \, m_k ,
$$

summing only the terms whose weight and metric value both convert to finite floats (anything `float()` accepts); other terms are skipped, and a metric set with no usable term scores `0.0`. Lower is better, so a metric to maximize (such as `mitigated_top_prob`) takes a negative weight. `weights` keeps the mapping as given, including entries that scoring skips.

```pycon
>>> from qbalance import Objective
>>> objective = Objective({"depth": 1.0, "two_qubit_ops": 2.0, "note": "skipped"})
>>> objective.score({"depth": 10, "two_qubit_ops": 3, "estimated_error": 0.2})
16.0
>>> objective.score({"depth": float("nan"), "two_qubit_ops": "7"})
14.0
>>> objective.weights
{'depth': 1.0, 'two_qubit_ops': 2.0, 'note': 'skipped'}
```

### `default_objective`

`qbalance.default_objective() -> Objective`

The objective `Workload.adjust` uses when none is given:

$$
S = \mathrm{depth} + 2\,\mathrm{two\_qubit\_ops} + 10\,\mathrm{estimated\_error} + 0.1\,\mathrm{compile\_time\_s} + \mathrm{sampling\_overhead} .
$$

`sampling_overhead` is the factor by which a strategy multiplies the shots a given precision needs: 1 for a circuit run once, $n \sum_i c_i^2$ for ZNE (4.375 for the default linear fit at factors 1, 3, 5; see [`zne_sampling_overhead`](#zne_sampling_overhead)), and the quasi-probability overhead of a cut circuit (9 per cut CNOT, multiplied over the cuts). `compile_time_s` is wall-clock time, so among candidates that compile to identical depth, two-qubit count, error, and overhead, which one wins can change from run to run; use an objective without `compile_time_s` when selections must repeat exactly.

```pycon
>>> from qbalance import default_objective
>>> default_objective().weights
{'depth': 1.0, 'two_qubit_ops': 2.0, 'estimated_error': 10.0, 'compile_time_s': 0.1, 'sampling_overhead': 1.0}
```

### `load_objective`

`qbalance.load_objective(path) -> Objective`

Loads weights from a JSON file holding a direct mapping (`{"depth": 1.0}`), `{"weights": {...}}`, or a saved `results.json` (its `objective` mapping). It is stricter than `Objective`: the mapping must be non-empty, names non-empty strings, and weights finite numbers that are not booleans, with `NaN` and `Infinity` literals rejected. A violation raises `ValueError`.

```pycon
>>> from qbalance import load_objective
>>> write_json("objective.json", {"weights": {"depth": 1, "estimated_error": 20}})
>>> load_objective("objective.json").weights
{'depth': 1.0, 'estimated_error': 20.0}
>>> write_json("bad_objective.json", {"depth": True})
>>> load_objective("bad_objective.json")
Traceback (most recent call last):
    ...
ValueError: Objective weight for 'depth' must not be a boolean
```

## Workflow (`qbalance.workflow`)

### `Workload`

`qbalance.Workload(dataset, backend_spec=None)`

The fluent entry point:

- `Workload.from_dataset(dataset)` and `Workload.from_path(dataset_dir)` build a workload without a target.
- `set_target(backend_spec) -> Workload` returns a new workload targeting a backend spec (see [backends](#backends-qbalancebackends)).
- `adjust(...) -> BalancedWorkload` selects one strategy per circuit.

`adjust` compiles every circuit with the baseline strategy, `StrategySpec(optimization_level=1, routing_method="sabre")`, and with each candidate, scores every result with the objective, and selects the best feasible candidate per circuit. A candidate equal to the baseline strategy reuses the baseline's evaluation, which ran with the same inputs and seeds, rather than executing again. Its parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `objective` | `None` | An `Objective` or a metric-to-weight mapping; `None` uses `default_objective()`. |
| `search` | `"grid"` | `"grid"` evaluates candidates in order; `"bandit"` evaluates `warmup` randomly ordered candidates, then those a Thompson-sampling surrogate proposes (see [`BanditSearcher`](#banditsearcher)). The mode changes the selection only when `max_evaluations` is below the number of candidates. |
| `pareto` | `False` | Select from the Pareto front over `depth`, `two_qubit_ops`, `estimated_error`, and `sampling_overhead` (see [`pareto_front`](#pareto_front)), ordered by objective score. |
| `max_candidates` | `24` | Size of the generated candidate pool when `strategies` is not given (see [`default_candidate_strategies`](#default_candidate_strategies)). |
| `warmup` | `6` | Randomly ordered candidates bandit search evaluates before its surrogate proposes; `0` starts from the prior. A `max_evaluations` budget at or below `warmup` is spent entirely at random. |
| `execute` | `False` | Execute every compiled circuit and record execution metrics. Strategies with `mthree=True` or `zne=True` execute regardless. |
| `shots` | `1024` | Shots per execution, a positive integer. |
| `profile` | `False` | Record pass-level transpiler profiles (`pass_profile`). |
| `cache_root` | `None` | Compile-cache directory (`str` or `Path`); `None` uses the platform cache (see [compile cache](#compile-cache-qbalancecache)). |
| `seed` | `0` | Non-negative integer seeding the candidate shuffle, the bandit, the simulator (instance $i$ of a twirled ensemble runs with seed $\mathrm{seed} + i$), and M3's calibration on simulators. |
| `strategies` | `None` | Explicit candidates, any iterable of `StrategySpec` objects or mappings (validated and de-duplicated); `max_candidates` is then ignored, and grid search keeps their order. |
| `allow_regression` | `True` | With `False`, never select a candidate that scores worse than the baseline (see below). |
| `max_evaluations` | `None` | Evaluate at most this many candidates per circuit; `None` evaluates all of them. |

Validation: the workload needs a target; `search` must be `"grid"` or `"bandit"`; `shots`, `max_candidates` (when used), and `max_evaluations` (when given) must be positive integers, `seed` and `warmup` non-negative integers, and `pareto`, `execute`, `profile`, and `allow_regression` booleans (NumPy booleans included; truthy values such as `"no"` are rejected); integer arguments reject booleans, and `objective` must be an `Objective`, a mapping, or `None`. A violation raises `ValueError`.

Scoring and feasibility: every evaluated candidate gets `objective_score` $= S(m)$. For selection, a candidate that reports an objective metric only with an invalid value (`None`, non-numeric, or non-finite) ranks last, while a metric it does not report at all just contributes nothing; `adjust` logs a warning naming every objective term that no evaluated metrics report (a misspelled name, or an execution metric without execution), since such a term cannot influence the selection. A candidate whose cutting, compilation, execution, or requested mitigation fails stays in the history but is infeasible: its metrics carry the error (`cutting_error`, `compile_error`, `exec_error`, `mthree_error`, `zne_error`), `strategy_failed: true`, a `strategy_failure_reason` (`cutting_failed`, `compile_failed`, `execution_failed`, `mthree_failed`, `zne_failed`), and an infinite `objective_score`. A circuit wider than the device therefore fails its uncut candidates while a cutting candidate can still win; when no candidate of a circuit is feasible, `adjust` raises `QBalanceError` listing the failures. With execution, the metrics include `raw_counts_entropy` (the Shannon entropy $-\sum_b p_b \log_2 p_b$ of the outcome frequencies $p_b$, in bits), `raw_top_prob` ($\max_b p_b$), and `raw_parity_expval` (see [`parity_expectation_value`](#parity_expectation_value)); M3 adds `mitigated_top_prob` and `mitigated_parity_expval`, ZNE adds `zne_top_prob`, `zne_parity_expval`, and `zne_realized_factors`, and an executed cut adds `cut_parity_expval` and `cut_shots_total` (shots times `num_subexperiments`).

Bandit search observes each score relative to its circuit's baseline score $S_0$, as $(S - S_0)/|S_0|$ (or $S - S_0$ when $S_0 = 0$): one surrogate serves every circuit, and raw scores grow with circuit size. Candidates or baselines without a finite score are not observed.

The regression guard (`allow_regression=False`) compares a candidate with the baseline on the objective terms both report finitely; the baseline runs no mitigation, so terms such as `mitigated_top_prob` or `cut_parity_expval` count neither for nor against a candidate. A candidate regresses when its score over those shared terms is higher than the baseline's, or when it has no finite score at all. A baseline without a finite score, or with no term in common, never triggers the guard, and an equal score is not a regression. When the top candidate regresses, the next candidate of the selection pool (the Pareto front with `pareto=True`) that does not regress is selected, and the baseline strategy, with its metrics, only when all of them regress. Such a selection carries `selected_by_regression_guard: true`, `rejected_candidate_spec`, and `rejected_candidate_objective_score`.

```pycon
>>> from qbalance import Workload
>>> fixed = Objective({"depth": 1.0, "two_qubit_ops": 2.0, "estimated_error": 10.0, "sampling_overhead": 1.0})
>>> workload = Workload.from_dataset(tiny).set_target("fake:generic:5")
>>> workload.backend_spec
'fake:generic:5'
>>> candidates = [
...     StrategySpec(optimization_level=0),
...     StrategySpec(optimization_level=3),
...     StrategySpec(optimization_level=1, layout_method="qbalance_noise_aware", routing_method="sabre"),
... ]
>>> balanced = workload.adjust(objective=fixed, strategies=candidates, cache_root="cache")
>>> {name: selection.spec.optimization_level for name, selection in balanced.selections.items()}
{'bell': 3, 'ghz3': 3, 'qft4': 3}
>>> pprint({
...     name: {key: round(selection.metrics[key], 4) for key in ("depth", "two_qubit_ops", "estimated_error", "objective_score")}
...     for name, selection in balanced.selections.items()
... })
{'bell': {'depth': 5,
          'estimated_error': 0.0032,
          'objective_score': 8.0319,
          'two_qubit_ops': 1},
 'ghz3': {'depth': 6,
          'estimated_error': 0.008,
          'objective_score': 11.0803,
          'two_qubit_ops': 2},
 'qft4': {'depth': 23,
          'estimated_error': 0.0269,
          'objective_score': 48.2691,
          'two_qubit_ops': 12}}
>>> Workload.from_path("circuits").adjust()
Traceback (most recent call last):
    ...
ValueError: Workload has no target backend; call set_target(...) first
```

Bandit search with a budget evaluates exactly `max_evaluations` candidates per circuit:

```pycon
>>> searched = workload.adjust(objective=fixed, search="bandit", max_evaluations=8, warmup=3, seed=5, cache_root="cache")
>>> {name: len(history) for name, history in searched.evaluation_history.items()}
{'bell': 8, 'ghz3': 8, 'qft4': 8}
```

Execution adds counts-derived metrics; here with a measurement twirl over four instances and M3:

```pycon
>>> executed = workload.adjust(
...     objective=Objective({"raw_top_prob": -1.0}),
...     strategies=[
...         StrategySpec(optimization_level=1, routing_method="sabre"),
...         StrategySpec(optimization_level=1, routing_method="sabre", measurement_twirling=True, num_twirls=4, mthree=True),
...     ],
...     execute=True,
...     shots=2000,
...     seed=11,
...     cache_root="cache",
... )
>>> twirled = executed.evaluation_history["ghz3"][1].metrics
>>> twirled["twirl_instances"], len(twirled["measurement_flip_maps"])
(4, 4)
>>> {key: round(twirled[key], 4) for key in ("raw_top_prob", "raw_parity_expval", "mitigated_top_prob", "mitigated_parity_expval")}
{'raw_top_prob': 0.4965, 'raw_parity_expval': 0.018, 'mitigated_top_prob': 0.4989, 'mitigated_parity_expval': 0.0143}
```

The regression guard at work: dynamical decoupling pads idle windows with X pulses, which adds depth, so the only candidate here scores worse than the baseline, and the baseline strategy is kept:

```pycon
>>> guarded = workload.adjust(
...     objective=fixed,
...     strategies=[StrategySpec(optimization_level=1, routing_method="sabre", dynamical_decoupling=True, dd_sequence="XX")],
...     allow_regression=False,
...     cache_root="cache",
... )
>>> kept = guarded.selections["ghz3"]
>>> kept.spec == StrategySpec(optimization_level=1, routing_method="sabre")
True
>>> kept.metrics["selected_by_regression_guard"], kept.metrics["rejected_candidate_spec"]["dynamical_decoupling"]
(True, True)
>>> round(kept.metrics["rejected_candidate_objective_score"], 4), round(kept.metrics["objective_score"], 4)
(13.1419, 11.138)
```

### `BalancedWorkload`

`qbalance.BalancedWorkload(dataset, backend_spec, selections, baseline_metrics={}, objective=default_objective(), evaluation_history={})`

The result of `adjust`. Attributes:

- `dataset`: the `CircuitDataset`.
- `backend_spec`: the target spec.
- `selections`: circuit name to the selected `Strategy`.
- `baseline_metrics`: circuit name to the baseline's metrics, produced by the same pipeline as the candidates (with execution, they include the execution metrics).
- `objective`: the `Objective` used.
- `evaluation_history`: circuit name to every evaluated candidate `Strategy`, in evaluation order (empty for artifacts saved before the history existed).

Methods:

- `summary() -> str`: the baseline against the selection: the backend, the number of circuits, the total and mean candidate evaluations, the means of `depth`, `two_qubit_ops`, `estimated_error`, and `compile_time_s` over their finite values (`nan` when there are none), the EMD, CvM, and KS distances between the baseline and selected distributions of `depth` and `two_qubit_ops` (see [diagnostics](#diagnostics-qbalancediagnostics)), and, when any circuit's delta is comparable, the mean objective delta with the number of circuits that strictly improved and that stayed unchanged.
- `selection_diagnostics() -> dict`: per circuit, `baseline_objective_score` and `selected_objective_score` (sums of the finite weighted terms; `None` when there is none), `comparable_objective_terms` (the terms both sides report), `objective_delta` (the selected minus the baseline sum over the comparable terms; `None` when there is none), `objective_improved` (`objective_delta <= 0`, so a tie counts as no worse, as for the guard; `None` when not comparable), `objective_terms` (each side's weighted terms), `evaluated_candidates`, and `metric_deltas` for `depth`, `two_qubit_ops`, `estimated_error`, and `compile_time_s`, each with `baseline`, `selected`, `delta` (selected minus baseline), and `relative_delta` (delta over the baseline's absolute value; `None` when the baseline is 0). Missing or non-finite values become `None`.
- `candidate_rankings() -> dict`: per circuit, the evaluated candidates sorted by the score selection uses, then by evaluation order. Each row holds `rank`, `original_index`, the serialized `spec`, `objective_score` (recomputed from the finite weighted terms), `selection_score` (the stored `objective_score` when valid; `None` for an infeasible or incomparable candidate, which sorts last), `objective_terms`, and `selected`. When the guard selected a baseline that was not among the candidates, it appears as one extra selected row with `original_index: None`, so exactly one row per circuit is selected.
- `covars() -> dict`: the EMD, CvM, and KS distances between the baseline and selected distributions of `depth`, `two_qubit_ops`, and `estimated_error` over their finite values; `nan` when a side has none.
- `save(out_dir, overwrite=False)`: writes `dataset/` (the workload's own records, with an index for exactly them), `results.json`, and `summary.txt` (see [artifacts](artifacts.md#balanced-workload-output)). An existing `out_dir` needs `overwrite=True`; a path that is a file, or a directory containing the workload's own dataset, is refused. The files are written to a sibling directory that then replaces `out_dir`, so a save that fails midway leaves an existing one intact.
- `to_download(zip_path, overwrite=False) -> Path`: saves into a temporary directory and zips it; the archive is renamed into place only once complete, so a failed export leaves an existing one intact.

```pycon
>>> print(balanced.summary())
qbalance summary
  backend: fake:generic:5
  circuits: 3
  candidate evaluations: total=9 mean_per_circuit=3
  mean metrics (baseline -> balanced):
    depth: 12.33 -> 11.33
    two_qubit_ops: 5 -> 5
    estimated_error: 0.0224 -> 0.01271
    compile_time_s: ... -> ...
  dist[depth]: EMD=1  CVM=0.3333  KS=0.3333
  dist[two_qubit_ops]: EMD=0  CVM=0  KS=0
  objective deltas: mean=-1.097 improved=3/3 unchanged=0/3
>>> diagnostics = balanced.selection_diagnostics()["qft4"]
>>> {key: round(diagnostics[key], 4) for key in ("baseline_objective_score", "selected_objective_score", "objective_delta")}
{'baseline_objective_score': 51.4488, 'selected_objective_score': 48.2691, 'objective_delta': -3.1797}
>>> diagnostics["objective_improved"], diagnostics["comparable_objective_terms"], diagnostics["evaluated_candidates"]
(True, ['depth', 'two_qubit_ops', 'estimated_error', 'sampling_overhead'], 3)
>>> diagnostics["metric_deltas"]["depth"]
{'baseline': 26.0, 'selected': 23.0, 'delta': -3.0, 'relative_delta': -0.11538461538461539}
>>> [(row["rank"], row["original_index"], round(row["selection_score"], 4), row["selected"]) for row in balanced.candidate_rankings()["qft4"]]
[(1, 1, 48.2691, True), (2, 2, 51.3773, False), (3, 0, 51.4488, False)]
>>> {metric: {name: round(value, 4) for name, value in distances.items()} for metric, distances in balanced.covars().items()}
{'depth': {'emd': 1.0, 'cvm': 0.3333, 'ks': 0.3333}, 'two_qubit_ops': {'emd': 0.0, 'cvm': 0.0, 'ks': 0.0}, 'estimated_error': {'emd': 0.0097, 'cvm': 0.0033, 'ks': 0.6667}}
>>> balanced.save("balanced")
>>> sorted(path.name for path in Path("balanced").iterdir())
['dataset', 'results.json', 'summary.txt']
>>> balanced.to_download("balanced.zip").name
'balanced.zip'
```

### `load_balanced_workload`

`qbalance.load_balanced_workload(out_dir) -> BalancedWorkload`

Reloads a directory written by `BalancedWorkload.save` (extract a ZIP bundle first) without recompiling. It checks that `results.json` is a JSON object with a non-empty string `backend_spec`; that `objective`, `selections`, `baseline_metrics`, and `evaluation_history` have the expected shapes; that every selection and history entry holds a valid strategy `spec` and an optional `metrics` object; that the selections name exactly the bundled dataset's circuits; and that baseline and history names refer to them. `selection_diagnostics` and `candidate_rankings` are recomputed from the loaded data, and a missing or `null` `evaluation_history` loads as empty, so artifacts from earlier releases still load. A problem raises `ValueError`.

```pycon
>>> from qbalance import load_balanced_workload
>>> reloaded = load_balanced_workload("balanced")
>>> reloaded.backend_spec, reloaded.dataset.names()
('fake:generic:5', ['bell', 'ghz3', 'qft4'])
>>> reloaded.selections["qft4"].spec == balanced.selections["qft4"].spec
True
>>> reloaded.selection_diagnostics() == balanced.selection_diagnostics()
True
```

## Benchmarking (`qbalance.benchmarking`)

### `run_matrix`

`qbalance.benchmarking.run_matrix(dataset_dir, backend_specs, strategies, out_json, execute=False, shots=1024, seed=0, profile=False) -> Path`

Evaluates every backend, circuit, and strategy combination and writes the matrix JSON (see [artifacts](artifacts.md#matrix-json)), returning `out_json`. `shots` must be a positive integer and `seed` a non-negative integer, neither a boolean; `backend_specs` must be a non-empty iterable of spec strings, and `strategies` a non-empty iterable of `StrategySpec` objects or mappings (validated as by `coerce_strategy_specs`, but a repeated strategy keeps its rows); `execute` and `profile` must be booleans, and `out_json` must not be a directory. All of this is checked before any trial runs. Every combination is compiled, a cutting strategy through its subexperiments; a strategy the backend cannot compile records `compile_error` instead of metrics (for a cut, together with the cut's description), and a circuit that cannot be cut records `cutting_error`. With `execute=True`, each row also records `counts` (pooled and untwirled over a twirled ensemble's instances, which split the shots) and `shots`; an `mthree` strategy adds `mthree_probs` (or `mthree_error`), a `zne` strategy adds `zne_probs` and `zne_realized_factors` (or `zne_error`, keeping the unmitigated counts), and an executed cut adds `cut_parity_expval` and `cut_shots_total`. `exec_error` records a failure to run the circuit itself. The file is written atomically.

```pycon
>>> from qbalance.benchmarking import run_matrix
>>> matrix_path = run_matrix(
...     "circuits",
...     ["fake:generic:5"],
...     [
...         StrategySpec(optimization_level=1, routing_method="sabre"),
...         StrategySpec(optimization_level=1, routing_method="sabre", measurement_twirling=True, num_twirls=4, mthree=True),
...     ],
...     "matrix.json",
...     execute=True,
...     shots=2000,
...     seed=7,
... )
>>> matrix = json.loads(Path(matrix_path).read_text())
>>> matrix["version"], matrix["metadata"]
(1, {'dataset_dir': 'circuits', 'backends': ['fake:generic:5'], 'execute': True, 'shots': 2000, 'seed': 7, 'profile': False})
>>> len(matrix["results"])
6
>>> row = matrix["results"][3]
>>> row["circuit"], row["backend"], row["strategy"]["mthree"]
('ghz3', 'fake:generic:5', True)
>>> row["metrics"]["shots"], sorted(row["metrics"]["counts"].items(), key=lambda item: -item[1])[:2]
(2000, [('111', 985), ('000', 973)])
>>> {key: round(value, 4) for key, value in sorted(row["metrics"]["mthree_probs"].items()) if value > 0.01}
{'000': 0.4904, '111': 0.4958}
```

### `TrialResult`

`qbalance.benchmarking.matrix.TrialResult(circuit, backend, strategy, metrics)`

One matrix row before serialization: the circuit name, the backend spec, the strategy as a dict, and its metrics.

```pycon
>>> from dataclasses import asdict
>>> from qbalance.benchmarking.matrix import TrialResult
>>> asdict(TrialResult(circuit="bell", backend="fake:generic:5", strategy={"optimization_level": 1}, metrics={"depth": 3}))
{'circuit': 'bell', 'backend': 'fake:generic:5', 'strategy': {'optimization_level': 1}, 'metrics': {'depth': 3}}
```

## Search (`qbalance.search`)

### `default_candidate_strategies`

`qbalance.search.default_candidate_strategies(max_candidates=24, seed=0) -> list[StrategySpec]`

The generated candidate pool, 23 distinct strategies:

- optimization levels 0 to 3, each with default routing, with SABRE routing, with SABRE layout and routing, and with the noise-aware layout and SABRE routing (16);
- at level 2 with SABRE routing: Pauli twirling (8 instances), dynamical decoupling (`XY4`), measurement twirling (8 instances), and Pauli twirling combined with dynamical decoupling and measurement twirling (8 instances);
- M3 with measurement twirling and ZNE with measurement twirling, both at level 2 with SABRE routing and 8 instances;
- a cutting candidate: level 1, noise-aware layout, SABRE routing, 4-qubit subcircuits (at level 1 Qiskit would otherwise keep each small subexperiment on qubits 0, 1, ..., whatever their error rates).

The first candidate, `StrategySpec(optimization_level=0)`, stays first; the rest are shuffled with `random.Random(seed)`, and the first `max_candidates` are returned (all 23 for the default 24; `[]` for `max_candidates <= 0`). A non-integer or boolean argument raises `ValueError`.

```pycon
>>> from qbalance.search import default_candidate_strategies
>>> pool = default_candidate_strategies()
>>> len(pool), pool[0] == StrategySpec(optimization_level=0)
(23, True)
>>> sum(s.cutting for s in pool), sum(s.mthree for s in pool), sum(s.zne for s in pool)
(1, 1, 1)
>>> len(default_candidate_strategies(max_candidates=5))
5
>>> set(default_candidate_strategies(seed=1)) == set(pool), default_candidate_strategies(seed=1) == pool
(True, False)
```

### `pareto_front`

`qbalance.search.pareto_front(items, keys) -> list[int]`

The indices, in input order, of the non-dominated `(label, metrics)` items, every key minimized. Item $a$ dominates item $b$ when $a_k \le b_k$ for every key $k$ and $a_k < b_k$ for at least one. A missing, non-numeric, or non-finite value (or a `None` metrics mapping) counts as $+\infty$. Items with identical metric vectors are kept or dropped together, and with no keys every item is on the front.

```pycon
>>> from qbalance.search import pareto_front
>>> evaluations = [
...     ("a", {"depth": 10, "two_qubit_ops": 4}),
...     ("b", {"depth": 8, "two_qubit_ops": 6}),
...     ("c", {"depth": 12, "two_qubit_ops": 6}),
...     ("d", {"depth": 8, "two_qubit_ops": 6}),
...     ("e", {"depth": float("nan"), "two_qubit_ops": 1}),
... ]
>>> pareto_front(evaluations, keys=("depth", "two_qubit_ops"))
[0, 1, 3, 4]
>>> pareto_front(evaluations, keys=("depth",))
[1, 3]
```

### `BanditSearcher`

`qbalance.search.BanditSearcher(alpha=1.0, sigma2=1.0)`

The Thompson-sampling surrogate behind `search="bandit"`: a Bayesian linear regression of a candidate's score on its features $\phi(s)$, which are a constant, the optimization level, indicators for SABRE routing, SABRE layout, the noise-aware layout, Pauli twirling, dynamical decoupling, measurement twirling, M3, ZNE, and cutting, and `num_twirls` when the strategy twirls. With the prior $w \sim \mathcal{N}(0, \alpha^{-1} I)$ and observations $y \sim \mathcal{N}(X w, \sigma^2 I)$, for the feature matrix $X$ and scores $y$ observed so far, the posterior is

$$
w \mid X, y \sim \mathcal{N}\!\left(\mu, \Lambda^{-1}\right), \qquad \Lambda = \alpha I + \frac{X^\top X}{\sigma^2}, \qquad \mu = \Lambda^{-1} \frac{X^\top y}{\sigma^2} .
$$

- `observe(spec, score)` records a finite score (lower is better); a non-finite score raises `ValueError`.
- `propose(candidates, rng)` draws $\tilde w \sim \mathcal{N}(\mu, \Lambda^{-1})$ with the NumPy generator `rng` and returns the candidate minimizing $\phi(s)^\top \tilde w$.

`alpha` must be finite and positive. `sigma2` is the noise variance $\sigma^2$: a fixed positive value, or `None` for the sample variance of the observed scores (floored at $10^{-6}$, and 1 until two scores are observed). `adjust` uses `sigma2=None`.

```pycon
>>> import numpy as np
>>> from qbalance.search import BanditSearcher
>>> searcher = BanditSearcher(sigma2=None)
>>> searcher.observe(StrategySpec(optimization_level=0), 0.40)
>>> searcher.observe(StrategySpec(optimization_level=3), -0.20)
>>> searcher.observe(StrategySpec(optimization_level=1), 0.10)
>>> searcher.propose([StrategySpec(optimization_level=level) for level in (0, 2, 3)], rng=np.random.default_rng(0)).optimization_level
3
>>> BanditSearcher(alpha=0.0)
Traceback (most recent call last):
    ...
ValueError: alpha must be a finite positive value
```

## Transpilation (`qbalance.transpile`)

### `compile_ensemble`

`qbalance.transpile.compile_ensemble(circuit, backend, spec, profile=False) -> (instances, metrics)`

Compiles `circuit` once with Qiskit's preset pass manager, which honors `optimization_level`, `layout_method`, `routing_method`, `translation_method`, and `seed_transpiler` (the noise-aware layout is passed as `initial_layout`), and expands the result into its twirl instances: Pauli twirling of the compiled, target-native circuit into `num_twirls` instances; then, per instance, a fresh measurement-twirl pattern (instance $i$ seeded with `seed_suppression` $+\,i$) and, last, dynamical decoupling, so the padding is scheduled around every inserted gate. Without twirling there is one instance. A backend without a transpiler `Target` falls back to translation-only compilation, which ignores the optimization level, layout, and routing. A noise-aware layout, measurement twirl, or DD pass that fails is logged and skipped (the default layout is used, and `dd_applied` reports whether DD was applied), but a Pauli twirl that cannot be applied raises `QBalanceError`: continuing untwirled would report a twirled strategy that twirled nothing, so `adjust` and `run_matrix` record the strategy as a compile failure instead.

The metrics describe the whole ensemble:

- `depth`, `size`, `width`, `two_qubit_ops`, `estimated_error`: the maxima over the instances. `depth` and `size` count operations only (see [utilities](#utilities-qbalanceutils)); `two_qubit_ops` counts instructions on exactly two qubits other than `barrier` and `delay`; `estimated_error` is [`estimate_circuit_error`](#estimate_circuit_error).
- `compile_time_s`: the wall-clock seconds the transpilation took.
- `twirl_instances`; `measurement_flip_maps`, one classical-bit flip map per instance, and `measurement_flip_map`, the first of them.
- `dd_applied` (every instance was padded) and, when it was, `dd_sequence_applied`.
- `sampling_overhead`: [`zne_sampling_overhead`](#zne_sampling_overhead) of the strategy's factors and degree for a ZNE strategy, otherwise 1.0.
- `pass_profile`, with `profile=True` (see [profiling](#profilereport-passprofile-make_callback)).

```pycon
>>> from qbalance.backends import resolve_backend
>>> from qbalance.transpile import compile_ensemble
>>> backend = resolve_backend("fake:generic:5")
>>> instances, metrics = compile_ensemble(
...     ghz(3),
...     backend,
...     StrategySpec(optimization_level=2, routing_method="sabre", pauli_twirling=True, measurement_twirling=True, num_twirls=4),
... )
>>> len(instances), metrics["twirl_instances"]
(4, 4)
>>> metrics["measurement_flip_maps"]
[{0: 1, 1: 1, 2: 1}, {1: 1, 2: 1}, {0: 1}, {0: 1}]
>>> {key: metrics[key] for key in ("depth", "size", "width", "two_qubit_ops", "dd_applied", "sampling_overhead")}
{'depth': 10, 'size': 17, 'width': 5, 'two_qubit_ops': 2, 'dd_applied': False, 'sampling_overhead': 1.0}
>>> round(metrics["estimated_error"], 6)
0.011867
>>> _, zne_metrics = compile_ensemble(ghz(3), backend, StrategySpec(zne=True))
>>> round(zne_metrics["sampling_overhead"], 6)
4.375
```

### `compile_one`

`qbalance.transpile.compile_one(circuit, backend, spec, profile=False) -> (compiled, metrics)`

`compile_ensemble` returning only the first instance. The metrics still describe the whole ensemble; execute every instance from `compile_ensemble` to obtain a twirled result.

```pycon
>>> from qbalance.transpile import compile_one
>>> compiled, one_metrics = compile_one(ghz(3), backend, StrategySpec(optimization_level=1, routing_method="sabre", dynamical_decoupling=True, dd_sequence="XY4"))
>>> one_metrics["dd_applied"], one_metrics["dd_sequence_applied"]
(True, 'XX')
>>> compiled.count_ops()["measure"]
3
```

### `NOISE_AWARE_LAYOUT`

`qbalance.transpile.pipeline.NOISE_AWARE_LAYOUT: str`

The `layout_method` value that selects qbalance's noise-aware layout.

```pycon
>>> from qbalance.transpile.pipeline import NOISE_AWARE_LAYOUT
>>> NOISE_AWARE_LAYOUT
'qbalance_noise_aware'
```

### `estimate_circuit_error`

`qbalance.transpile.estimate_circuit_error(backend, circuit) -> float`

The probability that at least one instruction fails, treating failures as independent:

$$
\varepsilon = 1 - \prod_i \left(1 - e_i\right),
$$

over the circuit's instructions $i$, with $e_i$ the backend's calibrated error for that instruction on those qubits, clipped to $[0, 1]$. Two-qubit gates are looked up in both qubit orders, measurements use the `measure` error (or a legacy `readout_error` property), and one-qubit gates the target's error. An instruction without calibration data counts as $e_i = 10^{-3}$ (one-qubit gates), $10^{-2}$ (two-qubit gates), or $2 \times 10^{-2}$ (measurements), and a gate on three or more qubits always counts $10^{-3}$, so on a backend that reports no errors (the plain Aer simulator) $\varepsilon$ is a gate-count proxy rather than a physical estimate. `barrier` and `delay` are not billed, and idle decoherence is not modelled, so the benefit of dynamical decoupling shows only in executed metrics. A circuit the estimate cannot read gives 1.0.

```pycon
>>> import math
>>> from qbalance.transpile import estimate_circuit_error
>>> bell_compiled, _ = compile_one(ghz(2, "bell"), backend, StrategySpec(optimization_level=1, routing_method="sabre"))
>>> [(inst.operation.name, [bell_compiled.find_bit(q).index for q in inst.qubits]) for inst in bell_compiled.data]
[('rz', [0]), ('sx', [0]), ('rz', [0]), ('cx', [0, 1]), ('measure', [0]), ('measure', [1])]
>>> errors = [backend.target[inst.operation.name][tuple(bell_compiled.find_bit(q).index for q in inst.qubits)].error for inst in bell_compiled.data]
>>> [round(error, 6) for error in errors]
[0.0, 9.2e-05, 0.0, 0.004317, 0.003559, 0.000584]
>>> round(1 - math.prod(1 - error for error in errors), 6), round(estimate_circuit_error(backend, bell_compiled), 6)
(0.008531, 0.008531)
```

### `noise_aware_initial_layout`

`qbalance.transpile.noise_aware_initial_layout(backend, circuit) -> Layout | None`

The layout behind `layout_method="qbalance_noise_aware"`.

- Physical qubit $p$ gets the error score $r_p - 10^{-3}\left(T_{1,p}/\tilde T_1 + T_{2,p}/\tilde T_2\right)$: its readout error $r_p$, with coherence relative to the backend medians $\tilde T_1$ and $\tilde T_2$ as a tie-breaker (a missing value counts as the median).
- Two logical qubits interact when a gate acts on both; a gate on three or more qubits (still whole before transpilation) makes every pair of its qubits interact.
- On a backend with a coupling map, couplers calibrated with error 1.0 are treated as absent, and logical qubits are placed one at a time, the busiest first and then always the one most connected to those already placed. Each goes to the free physical qubit needing the fewest routing hops to its placed partners, so it lands next to them whenever a free neighbour exists, and among those to the one with the lowest cost in error units: its error score plus, per interaction, the pair's calibrated two-qubit gate error, or, for a pair $h$ hops apart, one typical (median) gate error, three more per extra hop (a SWAP), and the readout error routing adds by moving one of the pair onto a qubit of the path. The placement is grown from the 24 most promising start qubits, and the one with the lowest total cost wins.
- Without a coupling map, the best-scoring qubits go to the busiest logical qubits.

Returns `None` when the sizes are unknown or the circuit is wider than the backend.

```pycon
>>> from qbalance.transpile import noise_aware_initial_layout
>>> torino = resolve_backend("fake:ibm:torino")
>>> chain = ghz(8, "chain8")
>>> layout = noise_aware_initial_layout(torino, chain)
>>> physical = [layout[qubit] for qubit in chain.qubits]
>>> physical
[48, 36, 29, 28, 27, 17, 8, 7]
>>> all(torino.coupling_map.distance(a, b) == 1 for a, b in zip(physical, physical[1:]))
True
>>> _, placed = compile_one(chain, torino, StrategySpec(optimization_level=1, layout_method=NOISE_AWARE_LAYOUT, routing_method="sabre"))
>>> placed["two_qubit_ops"], round(placed["estimated_error"], 4)
(7, 0.1702)
```

### `apply_pauli_twirling`

`qbalance.transpile.apply_pauli_twirling(circuit, num_twirls=1, seed=None, target=None) -> list[QuantumCircuit]`

Wraps Qiskit's `pauli_twirl_2q_gates`: returns `num_twirls` circuits in which every two-qubit gate is conjugated by a random Pauli pair that leaves the circuit's unitary unchanged; with `target`, the inserted Paulis are synthesized into its native gates.

```pycon
>>> from qiskit.quantum_info import Operator
>>> from qbalance.transpile import apply_pauli_twirling
>>> bell_unitary = QuantumCircuit(2)
>>> _ = bell_unitary.h(0)
>>> _ = bell_unitary.cx(0, 1)
>>> twirls = apply_pauli_twirling(bell_unitary, num_twirls=3, seed=1)
>>> len(twirls), all(Operator(t).equiv(Operator(bell_unitary)) for t in twirls)
(3, True)
```

### `apply_measurement_twirling`

`qbalance.transpile.suppression.apply_measurement_twirling(circuit, seed=None) -> (circuit, flip_map)`

Flips each terminal measurement at random: with probability 1/2, an `X` is inserted just before it and its classical bit is recorded in the flip map. A measurement is terminal when no later instruction other than `barrier` or `delay` acts on its qubit or classical bit; other measurements are left alone.

```pycon
>>> from qbalance.transpile.suppression import apply_measurement_twirling
>>> twirled_ghz, flip_map = apply_measurement_twirling(ghz(3), seed=3)
>>> flip_map, twirled_ghz.count_ops().get("x", 0)
({0: 1}, 1)
```

### `apply_measurement_untwirl_counts`

`qbalance.transpile.suppression.apply_measurement_untwirl_counts(counts, flip_map) -> dict`

Undoes the flips on a counts or probability mapping by relabelling its keys: classical bit $c$ is the $c$-th binary digit from the right, register separators are skipped and kept, and outcomes that become equal are summed.

```pycon
>>> from qbalance.transpile.suppression import apply_measurement_untwirl_counts
>>> apply_measurement_untwirl_counts({"101": 7, "000": 3}, {0: 1, 2: 1})
{'000': 7, '101': 3}
>>> apply_measurement_untwirl_counts({"1 01": 5}, {2: 1})
{'0 01': 5}
```

### `normalize_measurement_flip_map`

`qbalance.transpile.suppression.normalize_measurement_flip_map(flip_map) -> dict[int, int]`

Sanitizes a flip map read back from JSON or built by hand: keeps the non-negative integer classical-bit keys (digit strings included) whose flip value is odd, each normalized to 1. Anything that is not a mapping gives `{}`.

```pycon
>>> from qbalance.transpile.suppression import normalize_measurement_flip_map
>>> normalize_measurement_flip_map({"0": 1, "1": 0, "2": 3, "-1": 1, "x": 1})
{0: 1, 2: 1}
```

### `resolve_dd_sequence` and `dd_sequence_label`

`qbalance.transpile.suppression.resolve_dd_sequence(backend, sequence="XY4") -> list[Gate]`, `qbalance.transpile.suppression.dd_sequence_label(gates) -> str`

`resolve_dd_sequence` returns the gates dynamical decoupling pads idle time with: `XY4` is X Y X Y, `XX` is X X, and `YY` is Y Y. When the backend cannot run every gate of the request, the first fully supported of `XX` and `YY` is used instead (the request itself when neither is). `dd_sequence_label` names a gate sequence.

```pycon
>>> from qbalance.transpile.suppression import dd_sequence_label, resolve_dd_sequence
>>> gates = resolve_dd_sequence(backend, "XY4")
>>> [gate.name for gate in gates], dd_sequence_label(gates)
(['x', 'x'], 'XX')
```

### `build_dd_pass_manager`

`qbalance.transpile.build_dd_pass_manager(backend, sequence="XY4") -> PassManager`

A pass manager that unrolls gates on three or more qubits, translates to the backend basis, schedules every instruction as late as possible (`ALAPScheduleAnalysis`), and pads the idle windows with the resolved sequence (`PadDynamicalDecoupling`).

```pycon
>>> from qbalance.transpile import build_dd_pass_manager
>>> dd_manager = build_dd_pass_manager(backend, "XX")
>>> [type(task).__name__ for task in dd_manager.to_flow_controller().tasks]
['Unroll3qOrMore', 'BasisTranslator', 'ALAPScheduleAnalysis', 'PadDynamicalDecoupling']
```

### `ProfileReport`, `PassProfile`, `make_callback`

`qbalance.transpile.profiling.ProfileReport(passes=[])`, `qbalance.transpile.profiling.PassProfile(name, time_s, index)`, `qbalance.transpile.profiling.make_callback(report)`

`ProfileReport` collects one `PassProfile` per pass run; `total_time_s()` sums their times, and `to_json()` serializes them. `make_callback(report)` returns a callback for `PassManager.run(callback=...)` that appends an entry for each pass Qiskit reports. `compile_ensemble(profile=True)` stores `report.to_json()` as `pass_profile`.

```pycon
>>> from qbalance.transpile.profiling import PassProfile, ProfileReport, make_callback
>>> report = ProfileReport()
>>> callback = make_callback(report)
>>> callback(pass_=object(), time=0.25, count=0)
>>> callback(pass_=None, time=0.5, count=1)
>>> report.passes
[PassProfile(name='object', time_s=0.25, index=0), PassProfile(name='UnknownPass', time_s=0.5, index=1)]
>>> report.total_time_s(), report.to_json()["passes"][1]
(0.75, {'name': 'UnknownPass', 'time_s': 0.5, 'index': 1})
>>> _, profiled = compile_one(ghz(3), backend, StrategySpec(), profile=True)
>>> sorted(profiled["pass_profile"]), len(profiled["pass_profile"]["passes"]) > 0
(['passes', 'total_time_s'], True)
```

## Metrics (`qbalance.metrics`)

### `extract_circuit_metrics`

`qbalance.metrics.extract_circuit_metrics(circuit) -> dict[str, float]`

Structural metrics of any circuit, with the definitions of the compile metrics: `depth` and `size` over operations (directives excluded), `width` (qubits), `two_qubit_ops` (instructions on exactly two qubits other than `barrier` and `delay`), `measures`, and `t_count` (`t` and `tdg` gates).

```pycon
>>> from qbalance.metrics import extract_circuit_metrics
>>> circuit = ghz(3)
>>> _ = circuit.t(0)
>>> extract_circuit_metrics(circuit)
{'depth': 4.0, 'size': 7.0, 'width': 3.0, 'two_qubit_ops': 2.0, 'measures': 3.0, 't_count': 1.0}
```

## Execution (`qbalance.execution`)

### `run_counts`

`qbalance.execution.run_counts(backend, circuit, shots=1024, seed_simulator=None, seed_transpiler=None) -> dict[str, int]`

Runs a compiled circuit and returns its counts. A backend without `run` is wrapped in `AerSimulator.from_backend`, and the seeds are passed only to `run` methods that accept them. `shots` must be a positive integer and the seeds integers or `None`, none of them booleans; a circuit without a measurement (measurements inside control-flow blocks count) raises `ValueError` before anything is submitted.

```pycon
>>> from qbalance.execution import run_counts
>>> counts = run_counts(backend, compiled, shots=1000, seed_simulator=3)
>>> sorted(counts.items(), key=lambda item: -item[1])[:3]
[('111', 498), ('000', 489), ('001', 4)]
>>> run_counts(backend, QuantumCircuit(1, name="idle"))
Traceback (most recent call last):
    ...
ValueError: Circuit 'idle' has no measurements, so executing it yields no counts.
```

### `split_shots`, `instance_flip_maps`, `merge_counts`

`qbalance.execution.ensemble.split_shots(shots, instances) -> list[int]`, `qbalance.execution.ensemble.instance_flip_maps(metrics, instances) -> list[dict]`, `qbalance.execution.ensemble.merge_counts(distributions) -> dict`

Bookkeeping for a twirled ensemble. `split_shots` divides `shots` as evenly as possible, the first `shots % instances` instances taking one extra shot (instances beyond the shot count get 0 and are not run). `instance_flip_maps` returns every instance's normalized flip map from `measurement_flip_maps` (or `measurement_flip_map` for a single instance), raising `ValueError` when the count does not match. `merge_counts` sums counts, or weights, key by key.

```pycon
>>> from qbalance.execution.ensemble import instance_flip_maps, merge_counts, split_shots
>>> split_shots(10, 4), split_shots(2, 3)
([3, 3, 2, 2], [1, 1, 0])
>>> instance_flip_maps({"measurement_flip_maps": [{"0": 1}, {}]}, 2)
[{0: 1}, {}]
>>> merge_counts([{"00": 3, "11": 1}, {"11": 2}])
{'00': 3, '11': 3}
```

### `run_ensemble` and `EnsembleRun`

`qbalance.execution.ensemble.run_ensemble(backend, instances, metrics, *, shots, seed, mthree=False, zne_factors=(), zne_degree=1) -> EnsembleRun`

Runs a compiled ensemble the way `Workload.adjust` and `run_matrix` do. The shots are split across the instances with `split_shots`, instance $i$ runs with `seed_simulator=seed + i`, and each instance's counts are untwirled with its own flip map from `instance_flip_maps(metrics, len(instances))` before they are merged. With `mthree=True`, the raw counts go to `mitigate_twirled_counts` together with the calibration seed `seed`. With non-empty `zne_factors`, every instance is folded at each factor and rerun with the same share and seed; a factor that realizes 1 reuses the unfolded runs, and a factor realizing the same fold as an earlier one (`(1, 2, 3)` folds 2 and 3 alike) is an independent second sample of that noise level, its $r$-th repeat running instance $i$ with seed $\mathrm{seed} + r \cdot n + i$ for $n$ instances, so a simulator does not just copy the first. The fit, by `zne_extrapolate_counts` of degree `zne_degree`, uses the realized factors. An error in the unmitigated runs is raised. A mitigation failure is recorded instead, and the counts are kept.

`EnsembleRun` holds `counts` (merged, untwirled). It also holds `mthree_probs` or `mthree_error`, and `zne_probs` or `zne_error`, each `None` when not requested. `zne_realized_factors` is recorded once every folded run has succeeded.

```pycon
>>> from qbalance.execution.ensemble import run_ensemble
>>> members, members_metrics = compile_ensemble(ghz(3), backend, StrategySpec(measurement_twirling=True, num_twirls=4, zne=True))
>>> ensemble_run = run_ensemble(backend, members, members_metrics, shots=1000, seed=7, mthree=True, zne_factors=(1.0, 3.0, 5.0))
>>> sum(ensemble_run.counts.values()), sorted(ensemble_run.counts.items(), key=lambda item: -item[1])[:2]
(1000, [('000', 503), ('111', 472)])
>>> {key: round(value, 4) for key, value in sorted(ensemble_run.mthree_probs.items()) if value > 0.005}
{'000': 0.5052, '001': 0.0054, '101': 0.0066, '111': 0.4757}
>>> ensemble_run.zne_realized_factors, {key: round(value, 4) for key, value in sorted(ensemble_run.zne_probs.items()) if value > 0.005}
([1.0, 3.0, 5.0], {'000': 0.499, '001': 0.0071, '101': 0.0069, '111': 0.4761})
>>> ensemble_run.mthree_error, ensemble_run.zne_error
(None, None)
>>> run_ensemble(backend, members, members_metrics, shots=1000, seed=7, zne_factors=(1.0, 3.0), zne_degree=2).zne_error
'not enough points for requested polynomial degree'
```

## Mitigation (`qbalance.mitigation`)

### `apply_mthree_mitigation`

`qbalance.mitigation.apply_mthree_mitigation(backend, raw_counts, measured_qubits, shots=None, calibration_shots=10000, clbits=None) -> dict[str, float]`

M3 readout mitigation of one counts mapping: calibrates the readout of `measured_qubits` on the backend with `calibration_shots` shots, corrects the counts, and returns the nearest probability distribution. `measured_qubits` lists the physical qubit feeding each measured classical bit, ordered by classical bit, and `clbits` the classical bit each of them writes; `measured_qubits_by_clbit` and `measured_clbits` (see [utilities](#utilities-qbalanceutils)) derive both from a compiled circuit. Count keys are projected onto the measured bits for the correction and expanded back afterwards, so register separators are kept and a classical bit no measurement writes stays `0`; with `clbits=None`, every bit of the keys must be measured. Pass counts exactly as measured, before any untwirling. `shots` is accepted for call-site symmetry and unused. mthree needs `backend.configuration()`, which `GenericBackendV2` (`fake:generic`) lacks, so for those backends the calibration runs on `AerSimulator.from_backend(backend)`, the noisy simulator `GenericBackendV2` itself executes on; every other backend is calibrated on itself. Requires the `mitigation` extra.

```pycon
>>> from qbalance.mitigation import apply_mthree_mitigation
>>> from qbalance.utils import measured_clbits, measured_qubits_by_clbit
>>> raw = run_counts(backend, compiled, shots=4000, seed_simulator=5)
>>> mitigated = apply_mthree_mitigation(backend, raw, measured_qubits_by_clbit(compiled), clbits=measured_clbits(compiled))
>>> round(sum(mitigated.values()), 6), max(mitigated, key=mitigated.get) in ("000", "111")
(1.0, True)
>>> mitigated["000"] + mitigated["111"] > (raw["000"] + raw["111"]) / 4000
True
```

### `mitigate_twirled_counts`

`qbalance.mitigation.mitigate_twirled_counts(backend, raw_counts, flip_maps, measured_qubits, calibration_shots=10000, seed=None, clbits=None) -> dict[str, float]`

M3 for a measurement-twirled ensemble: calibrates once, corrects each instance's raw counts (the calibration describes the physical readout, which only matches counts as measured), untwirls each with its own flip map, combines the instances weighted by their shots, and projects the result onto the nearest probability distribution. `seed` seeds the calibration on a local simulator -- the `AerSimulator` created for `GenericBackendV2`, or the backend itself for Aer and `fake:ibm` snapshots, whose calibration jobs all run with `seed_simulator=seed` while the backend object is left unmodified -- which makes the result reproducible there; hardware calibrates as it runs. `raw_counts` and `flip_maps` must have the same length and hold at least one shot.

```pycon
>>> from qbalance.mitigation import mitigate_twirled_counts
>>> ensemble, ensemble_metrics = compile_ensemble(ghz(3), backend, StrategySpec(measurement_twirling=True, num_twirls=4))
>>> ensemble_flips = ensemble_metrics["measurement_flip_maps"]
>>> ensemble_raw = [run_counts(backend, member, shots=1000, seed_simulator=20 + i) for i, member in enumerate(ensemble)]
>>> probs = mitigate_twirled_counts(backend, ensemble_raw, ensemble_flips, measured_qubits_by_clbit(ensemble[0]), seed=0, clbits=measured_clbits(ensemble[0]))
>>> {key: round(value, 4) for key, value in sorted(probs.items()) if value > 0.005}
{'000': 0.5111, '111': 0.4828}
```

### `realized_fold_factor`

`qbalance.mitigation.realized_fold_factor(scale) -> int`

Global folding realizes only odd integer noise factors, so a requested factor $\lambda \ge 1$ runs at

$$
k(\lambda) = \begin{cases} 1 & \text{if } \lambda = 1, \\ \lceil \lambda \rceil & \text{if } \lceil \lambda \rceil \text{ is odd}, \\ \lceil \lambda \rceil + 1 & \text{otherwise}. \end{cases}
$$

Extrapolate against these realized factors, as the workflow does. A non-finite, boolean, or sub-1 factor raises `ValueError`.

```pycon
>>> from qbalance.mitigation import realized_fold_factor
>>> [realized_fold_factor(factor) for factor in (1.0, 1.2, 2.0, 3.0, 4.0, 5.5)]
[1, 3, 3, 3, 5, 7]
>>> realized_fold_factor(0.5)
Traceback (most recent call last):
    ...
ValueError: scale must be a finite real value >= 1.0
```

### `zne_sampling_overhead`

`qbalance.mitigation.zne_sampling_overhead(factors, degree=1) -> float`

The factor by which ZNE multiplies the shots a given precision needs. For the realized factors $\lambda_1, \dots, \lambda_n$ and the Vandermonde matrix $V_{ij} = \lambda_i^{\,j}$ ($j = 0, \dots, d$ for degree $d$), the least-squares intercept is the linear combination $\hat y_0 = \sum_i c_i y_i$ of the measured values, with $c$ the first row of the pseudo-inverse $V^{+}$. Every factor runs on the full shot budget (the unfolded factor 1 is the primary execution, whose counts the workflow reuses), so matching the variance of a single unmitigated run takes

$$
\gamma_{\mathrm{ZNE}} = n \sum_{i=1}^{n} c_i^2
$$

times the shots: $4.375$ for the default linear fit at factors $(1, 3, 5)$, where $c = (\tfrac{13}{12}, \tfrac{1}{3}, -\tfrac{5}{12})$; $15.65625$ for a quadratic fit at the same factors; and $7.125$ for a linear fit at $(1, 2, 3)$, which realizes $(1, 3, 3)$. Factors that realize 1 all reuse that one unfolded execution, so they count as a single run whose coefficient is the sum of theirs: $(1, 1, 3)$ costs $5$, exactly what $(1, 3)$ does. An invalid factor or degree, or fewer than $d + 1$ distinct realized factors, raises `ValueError`. Compile metrics record the value as `sampling_overhead`.

```pycon
>>> from qbalance.mitigation import zne_sampling_overhead
>>> [round(zne_sampling_overhead(factors, degree), 6) for factors, degree in [((1, 3, 5), 1), ((1, 3, 5), 2), ((1, 2, 3), 1)]]
[4.375, 15.65625, 7.125]
>>> zne_sampling_overhead((1.0, 2.0, 3.0), degree=2)
Traceback (most recent call last):
    ...
ValueError: not enough distinct realized factors for the degree
```

### `fold_global`

`qbalance.mitigation.fold_global(circuit, scale) -> QuantumCircuit`

Global unitary folding. With $U$ the circuit without its terminal measurements and $k$ the [realized factor](#realized_fold_factor), it builds

$$
U \left(U^\dagger U\right)^{(k-1)/2}
$$

and re-appends the measurements, returning `circuit` itself when $k = 1$. The split is made per qubit, so gates listed after another qubit's measurement are folded too. Every measurement must be terminal (nothing later touches its qubit or classical bit), or `ValueError` is raised. A reset that acts on its qubit before anything else does only re-prepares $|0\rangle$, so it runs once, ahead of the folded unitary (Qiskit keeps a `reset(range(n))` preamble at every optimization level); any other reset is part of the unitary and cannot be inverted, so folding raises `CircuitError`. Folding appends `circuit.inverse()`, whose adjoint gates (for example `sxdg`) may lie outside a backend's basis; fold compiled circuits with `fold_global_for_backend`.

```pycon
>>> from qbalance.mitigation import fold_global
>>> folded = fold_global(ghz(3), 3)
>>> folded.name, dict(folded.count_ops())
('ghz3_fold3', {'cx': 6, 'h': 3, 'measure': 3})
>>> original = ghz(3)
>>> fold_global(original, 1.0) is original
True
```

### `fold_global_for_backend`

`qbalance.mitigation.fold_global_for_backend(circuit, backend, scale) -> QuantumCircuit`

Folds a compiled circuit and translates it back into the backend's native basis (a level-0 preset pass manager with the identity layout), keeping the physical qubits, the measurement mapping, and the compiled circuit's `TranspileLayout`. A backend the preset pass manager cannot target, or a circuit not sized for the backend, gets the folded circuit untranslated.

```pycon
>>> from qbalance.mitigation import fold_global_for_backend
>>> folded_compiled = fold_global_for_backend(compiled, backend, 3.0)
>>> set(folded_compiled.count_ops()) <= set(backend.target.operation_names)
True
>>> folded_compiled.layout.final_index_layout() == compiled.layout.final_index_layout()
True
```

### `parity_expectation_value`

`qbalance.mitigation.zne.parity_expectation_value(distribution) -> float`

The $Z$-parity expectation over every measured bit of a counts or (quasi-)probability mapping,

$$
\langle Z^{\otimes m} \rangle = \frac{\sum_b (-1)^{|b|}\, p_b}{\sum_b p_b},
$$

where $|b|$ is the number of ones in outcome $b$ (register separators ignored). It is the observable ZNE extrapolates and that uncut, mitigated, and cut executions report. A mapping without positive total weight raises `ValueError`.

```pycon
>>> from qbalance.mitigation.zne import parity_expectation_value
>>> parity_expectation_value({"000": 480, "111": 470, "001": 50})
-0.04
```

### `zne_extrapolate_counts`

`qbalance.mitigation.zne_extrapolate_counts(factors, counts_per_factor, degree=1) -> dict[str, float]`

Extrapolates the parity to zero noise. For each factor $\lambda_i$ it computes $y_i = \langle Z^{\otimes m}\rangle$ from the counts, fits a degree-$d$ polynomial $p$ by least squares, and takes $\hat y_0 = p(0)$. It returns the distribution measured at the factor closest to 1, with its even- and odd-parity outcomes rescaled to the masses

$$
P_{\mathrm{even}} = \min\!\left(1, \max\!\left(0, \frac{1 + \hat y_0}{2}\right)\right), \qquad P_{\mathrm{odd}} = 1 - P_{\mathrm{even}},
$$

keeping the shape within each class (a class that was never sampled gets one synthetic key, and a class without mass is spread evenly), so the result's parity is $\hat y_0$ clipped to $[-1, 1]$. Pass the realized factors. `factors` must be finite, at least 1, as many as the count mappings, and hold at least $d + 1$ distinct values; counts must be non-negative integers over binary keys.

```pycon
>>> from qbalance.mitigation import zne_extrapolate_counts
>>> extrapolated = zne_extrapolate_counts(
...     [1, 3, 5],
...     [{"00": 850, "11": 50, "01": 100}, {"00": 700, "11": 100, "01": 200}, {"00": 600, "11": 100, "01": 300}],
... )
>>> {key: round(value, 4) for key, value in extrapolated.items()}
{'00': 0.8972, '11': 0.0528, '01': 0.05}
>>> round(parity_expectation_value(extrapolated), 4)
0.9
```

### `build_runtime_estimator_options`

`qbalance.mitigation.runtime_options.build_runtime_estimator_options(resilience_level=1, enable_gate_twirling=None, enable_measurement_mitigation=None, enable_zne=None, layer_noise_model=None) -> dict`

Builds an IBM Runtime `EstimatorV2` options mapping, omitting options left as `None`. It is a standalone helper: qbalance's own compile and execute path does not call it, and `StrategySpec.resilience_level` is carried into artifacts as metadata only.

```pycon
>>> from qbalance.mitigation.runtime_options import build_runtime_estimator_options
>>> build_runtime_estimator_options(resilience_level=2, enable_gate_twirling=True, enable_zne=True)
{'resilience_level': 2, 'twirling': {'enable_gates': True}, 'resilience': {'zne_mitigation': True}}
```

## Circuit cutting (`qbalance.cutting`)

Requires the `cutting` extra (`qiskit-addon-cutting`).

### `prepare_cutting_experiment`

`qbalance.cutting.prepare_cutting_experiment(circuit, max_subcircuit_qubits, max_backjumps=10000, max_gamma=1e6, max_subexperiments=1024) -> CuttingExperiment | None`

Splits off the terminal measurements, which define the observable ($Z$ on every measured qubit), drops `barrier` and `delay`, unrolls gates on three or more qubits, and removes the qubits no gate acts on (a measured idle qubit reads $|0\rangle$ and contributes exactly $+1$ to the parity). It then cuts the rest with `find_cuts` and expands it into every quasi-probability term exactly (`num_samples=inf`, so results are reproducible). Returns `None` when the circuit is no wider than `max_subcircuit_qubits`, or no gate acts on any qubit; a wider circuit whose connected parts each fit is split into them without a cut. Raises `ValueError` for non-terminal measurements, classically controlled operations, or a cut needing more than `max_subexperiments` subexperiments.

### `CuttingExperiment`

`qbalance.cutting.CuttingExperiment(subexperiments, coefficients, subobservables, measured_qubits, metadata={})`

The subexperiment circuits per partition label, the quasi-probability coefficients, the observable per label, the measured qubits, and `metadata`: `cut_count`, `cut_types`, `sampling_overhead` (the cut's quasi-probability overhead $\prod_c \gamma_c^2$, 9 per cut CNOT), `num_subcircuits`, `subcircuit_widths`, and `num_subexperiments`. `qbalance.cutting.addon_cutting.MAX_SUBEXPERIMENTS` is the default limit, 1024.

```pycon
>>> from qbalance.cutting import prepare_cutting_experiment
>>> from qbalance.cutting.addon_cutting import MAX_SUBEXPERIMENTS
>>> experiment = prepare_cutting_experiment(ghz(6), max_subcircuit_qubits=4)
>>> experiment.metadata
{'cut_count': 1, 'cut_types': ['Gate Cut'], 'sampling_overhead': 9.0, 'num_subcircuits': 2, 'subcircuit_widths': [4, 2], 'num_subexperiments': 12}
>>> {label: len(circuits) for label, circuits in experiment.subexperiments.items()}, experiment.measured_qubits
({0: 6, 1: 6}, [0, 1, 2, 3, 4, 5])
>>> prepare_cutting_experiment(ghz(3), max_subcircuit_qubits=4) is None, MAX_SUBEXPERIMENTS
(True, 1024)
```

### `combine_subexperiment_metrics`

`qbalance.cutting.combine_subexperiment_metrics(metrics, experiment) -> dict`

Candidate metrics for a cut strategy. Every subexperiment runs as a circuit of its own, so `depth`, `size`, `width`, `two_qubit_ops`, and `estimated_error` are the maxima over the subexperiments' compile metrics, `compile_time_s` is their total, `dd_applied` holds when every subexperiment was padded, and the experiment's metadata is merged in. An empty `metrics` raises `ValueError`.

```pycon
>>> from qbalance.cutting import combine_subexperiment_metrics
>>> compiled_subexperiments, sub_metrics = {}, []
>>> for label, circuits in experiment.subexperiments.items():
...     compiled_subexperiments[label] = []
...     for sub in circuits:
...         sub_compiled, sub_m = compile_one(sub, backend, StrategySpec(optimization_level=1, routing_method="sabre"))
...         compiled_subexperiments[label].append(sub_compiled)
...         sub_metrics.append(sub_m)
>>> cut_metrics = combine_subexperiment_metrics(sub_metrics, experiment)
>>> {key: cut_metrics[key] for key in ("depth", "width", "two_qubit_ops", "cut_count", "sampling_overhead", "num_subexperiments")}
{'depth': 9, 'width': 5, 'two_qubit_ops': 3, 'cut_count': 1, 'sampling_overhead': 9.0, 'num_subexperiments': 12}
```

### `run_cutting_experiment`

`qbalance.cutting.run_cutting_experiment(experiment, compiled, backend, shots, seed=None) -> float`

Runs the compiled subexperiments (per label, in the experiment's order) through `BackendSamplerV2` with `shots` shots each and reconstructs the original circuit's $\langle Z^{\otimes m}\rangle$. `seed` goes to backends whose `run` accepts `seed_simulator`. A circuit that measured nothing raises `ValueError`. A GHZ state on an even number of qubits has parity exactly $+1$; the noisy backend here gives:

```pycon
>>> from qbalance.cutting import run_cutting_experiment
>>> round(run_cutting_experiment(experiment, compiled_subexperiments, backend, shots=4000, seed=1), 4)
0.9561
```

### `evaluate_cutting_experiment`

`qbalance.cutting.evaluate_cutting_experiment(experiment, backend, compile_subexperiment, *, execute, shots, seed=None) -> dict`

How `Workload.adjust` and `run_matrix` evaluate a cut candidate. `compile_subexperiment(circuit)` returns `(instances, metrics)` for one subexperiment. A cut strategy cannot twirl, so the first instance is the one that runs. The result is `combine_subexperiment_metrics` of the compiled subexperiments. With `execute=True` it adds `cut_parity_expval` from `run_cutting_experiment` and `cut_shots_total` ($\text{shots}\times$`num_subexperiments`), or `exec_error` when the run fails. A subexperiment that cannot be compiled yields `compile_error` together with the cut's metadata.

```pycon
>>> from qbalance.cutting import evaluate_cutting_experiment
>>> sabre = StrategySpec(optimization_level=1, routing_method="sabre")
>>> cut_run = evaluate_cutting_experiment(experiment, backend, lambda sub: compile_ensemble(sub, backend, sabre), execute=True, shots=4000, seed=1)
>>> round(cut_run["cut_parity_expval"], 4), cut_run["cut_shots_total"], cut_run["depth"]
(0.9561, 48000, 9)
>>> def too_wide(circuit):
...     raise ValueError("subcircuit is wider than the device")
>>> evaluate_cutting_experiment(experiment, backend, too_wide, execute=True, shots=4000)
{'compile_error': 'subcircuit is wider than the device', 'cut_count': 1, 'cut_types': ['Gate Cut'], 'sampling_overhead': 9.0, 'num_subcircuits': 2, 'subcircuit_widths': [4, 2], 'num_subexperiments': 12}
```

### `find_cuts_best_effort`

`qbalance.cutting.find_cuts_best_effort(circuit, max_subcircuit_qubits, max_backjumps=10000, max_gamma=1e6) -> (cut_circuit, metadata)`

The underlying cut finder: `qiskit_addon_cutting.find_cuts` with these optimization parameters and subcircuit width. The circuit must carry no classical bits. Returns the circuit with its cut instructions and the finder's metadata (`cuts`, `sampling_overhead`, ...).

```pycon
>>> from qbalance.cutting import find_cuts_best_effort
>>> unitary_chain = QuantumCircuit(4)
>>> _ = unitary_chain.h(0)
>>> for q in range(3):
...     _ = unitary_chain.cx(q, q + 1)
>>> _, cut_meta = find_cuts_best_effort(unitary_chain, max_subcircuit_qubits=2)
>>> cut_meta["cuts"], float(cut_meta["sampling_overhead"])
([('Gate Cut', 2)], 9.0)
```

## Diagnostics (`qbalance.diagnostics`)

Distances between two one-dimensional samples with optional weights. Negative weights are clipped to 0 and the rest normalized to sum to 1 (uniform when none are given or all are 0), and each sample defines the weighted empirical CDF

$$
F(t) = \sum_{i \,:\, x_i \le t} \tilde w_i, \qquad \tilde w_i = \frac{\max(w_i, 0)}{\sum_j \max(w_j, 0)} .
$$

Samples must be non-empty, one-dimensional, and finite, and weights finite and of the same length; a violation raises `ValueError`. `BalancedWorkload.summary()` and `covars()` apply these to the baseline and selected metric values.

- `weighted_cdf(x, w=None) -> (support, cdf)`: the sorted distinct values and $F$ at each of them (the last is exactly 1).
- `ks_1d(x1, x2, w1=None, w2=None) -> float`: the Kolmogorov-Smirnov statistic $\sup_t |F_1(t) - F_2(t)|$.
- `cvm_1d(x1, x2, w1=None, w2=None) -> float`: the Cramér-von Mises distance $\int_{-\infty}^{\infty} \left(F_1(t) - F_2(t)\right)^2 dt$, integrated over $t$, so it carries the metric's units.
- `emd_1d(x1, x2, w1=None, w2=None) -> float`: the earth mover's (Wasserstein-1) distance $\int_{-\infty}^{\infty} |F_1(t) - F_2(t)|\, dt$.

```pycon
>>> from qbalance.diagnostics import cvm_1d, emd_1d, ks_1d, weighted_cdf
>>> weighted_cdf([3, 1, 3, 2])
(array([1., 2., 3.]), array([0.25, 0.5 , 1.  ]))
>>> [round(distance([1, 2, 3], [2, 3, 4]), 6) for distance in (ks_1d, emd_1d, cvm_1d)]
[0.333333, 1.0, 0.333333]
>>> emd_1d([0, 10], [0, 10], w1=[1, 0], w2=[0, 1])
10.0
>>> ks_1d([], [1])
Traceback (most recent call last):
    ...
ValueError: Input samples must be non-empty.
```

## Backends (`qbalance.backends`)

### `resolve_backend`

`qbalance.backends.resolve_backend(spec_or_obj) -> BackendLike`

Resolves a backend spec through the `qbalance.backends` entry points: the part before the first `:` selects the plugin, which receives the whole spec. Anything that is not a string is returned unchanged. An empty spec or an unknown kind raises `QBalanceError` listing the available kinds. `qbalance.backends.resolver.BackendLike` is the type alias (`Any`) of what a plugin returns. The built-in specs:

- `fake:generic:<n>` or `fake:generic:<n>:<seed>`: a `GenericBackendV2` with $n \ge 2$ qubits whose calibration data is drawn from `seed` (0 by default), so a spec always yields the same backend.
- `fake:ibm:<name>`: an IBM device snapshot from `qiskit_ibm_runtime.fake_provider` (`sherbrooke` resolves to `FakeSherbrooke`); needs the `runtime` extra.
- `aer:simulator`: an ideal `AerSimulator`, and `aer:from_backend:<spec>`: `AerSimulator.from_backend` of any other spec (for example `aer:from_backend:fake:ibm:kyiv`); both need the `aer` extra.

Whitespace around the parts is ignored. A malformed spec raises `QBalanceError`, and a missing optional package `OptionalDependencyError`.

```pycon
>>> from qbalance.backends import resolve_backend
>>> generic = resolve_backend("fake:generic:5:3")
>>> type(generic).__name__, generic.num_qubits
('GenericBackendV2', 5)
>>> resolve_backend("fake : ibm : sherbrooke").name
'fake_sherbrooke'
>>> type(resolve_backend("aer:from_backend:fake:generic:5")).__name__
'AerSimulator'
>>> resolve_backend(generic) is generic
True
>>> resolve_backend("mock:5")
Traceback (most recent call last):
    ...
qbalance.errors.QBalanceError: Unknown backend kind 'mock'. Available: ['aer', 'fake']
```

### `fake.resolve` and `aer.resolve`

`qbalance.backends.fake.resolve(spec)`, `qbalance.backends.aer.resolve(spec)`

The built-in plugin functions registered as the `fake` and `aer` entry points; each parses the full spec as described above.

```pycon
>>> from qbalance.backends import aer, fake
>>> fake.resolve("fake:generic:3").num_qubits
3
>>> type(aer.resolve("aer:simulator")).__name__
'AerSimulator'
>>> fake.resolve("fake:generic:1")
Traceback (most recent call last):
    ...
qbalance.errors.QBalanceError: fake:generic requires at least 2 qubits
```

### `list_plugins`

`qbalance.plugins.list_plugins() -> dict[str, list[str]]`

The names registered in the `qbalance.backends`, `qbalance.objectives`, and `qbalance.reports` entry-point groups, sorted. Registering a `qbalance.backends` entry point (a function taking the spec string and returning a backend) makes its kind resolvable wherever a backend spec is accepted. The other two groups are only listed: nothing resolves them yet.

```pycon
>>> from qbalance.plugins import list_plugins
>>> list_plugins()
{'qbalance.backends': ['aer', 'fake'], 'qbalance.objectives': ['default'], 'qbalance.reports': ['html', 'markdown']}
```

## Reports (`qbalance.reports`)

### `render_markdown` and `render_html`

`qbalance.reports.render_markdown(matrix_json, out_dir) -> Path`, `qbalance.reports.render_html(matrix_json, out_dir) -> Path`

Render `report.md` or `report.html` into `out_dir` (created when missing) from a matrix JSON file. Rows are grouped per backend and per strategy label (`strategy_key`); each shows the means over its trials of `depth`, `two_qubit_ops`, `estimated_error`, and `compile_time_s` (over finite values; `nan` when there are none) and `failed trials` as `k/n`, and the rows are sorted by mean depth, then mean two-qubit count. The HTML report escapes every value and needs the `report` extra (`jinja2`). An unreadable or malformed matrix raises `ValueError`.

```pycon
>>> from qbalance.reports import render_html, render_markdown
>>> write_json("tiny_matrix.json", {"version": 1, "results": [
...     {"circuit": "bell", "backend": "fake:generic:5", "strategy": {"optimization_level": 1},
...      "metrics": {"depth": 4, "two_qubit_ops": 1, "estimated_error": 0.02, "compile_time_s": 0.01}},
...     {"circuit": "ghz3", "backend": "fake:generic:5", "strategy": {"optimization_level": 1},
...      "metrics": {"depth": 6, "two_qubit_ops": 2, "estimated_error": 0.04, "compile_time_s": 0.03}},
...     {"circuit": "bell", "backend": "fake:generic:5", "strategy": {"optimization_level": 3, "mthree": True},
...      "metrics": {"depth": 3, "two_qubit_ops": 1, "estimated_error": 0.01, "compile_time_s": 0.05, "mthree_error": "calibration failed"}},
... ]})
>>> print(render_markdown("tiny_matrix.json", "report").read_text())
# qbalance report
<BLANKLINE>
Matrix: `tiny_matrix.json`
<BLANKLINE>
## Backend: fake:generic:5
<BLANKLINE>
| Strategy | mean depth | mean 2q ops | mean est error | mean compile time (s) | failed trials |
|---|---:|---:|---:|---:|---:|
| `opt3,mthree` | 3 | 1 | 0.01 | 0.05 | 1/1 |
| `opt1` | 5 | 1.5 | 0.03 | 0.02 | 0/2 |
>>> render_html("tiny_matrix.json", "report").name
'report.html'
```

### `qbalance.reports.common`

- `load_matrix(path) -> dict`: reads a matrix JSON object (`ValueError` otherwise).
- `matrix_results(data) -> list[dict]`: checks that every result row has a non-empty string `backend`, a `strategy` object, and an object (or missing or `null`) `metrics`, and returns the normalized rows.
- `strategy_key(spec) -> str`: the row label. It names every knob that changes behavior (`opt<level>`, `layout=`, `route=`, `xlate=`, a non-zero transpiler seed `seedt=`, `twirl<n>`, `dd=<sequence>`, `meas_twirl` or `meas_twirl<n>`, a non-zero suppression seed `seeds=`, `mthree`, `zne` with a non-default degree `zdeg=` and factors `zf=`, `cut<n>`, `res=`), so strategies that run differently never share a label.
- `aggregate(rows) -> dict`: the per-metric means shown in the tables.
- `failed_trials(rows) -> int`: how many rows record one of `FAILURE_KEYS`.
- `sort_value(value) -> float`: a sort key that maps missing, non-numeric, and non-finite values to $+\infty$.

```pycon
>>> from qbalance.reports.common import FAILURE_KEYS, aggregate, failed_trials, load_matrix, matrix_results, sort_value, strategy_key
>>> rows = matrix_results(load_matrix("tiny_matrix.json"))
>>> [strategy_key(row["strategy"]) for row in rows]
['opt1', 'opt1', 'opt3,mthree']
>>> strategy_key({"optimization_level": 2, "routing_method": "sabre", "measurement_twirling": True, "num_twirls": 8, "zne": True, "zne_factors": [1, 3, 7]})
'opt2,route=sabre,meas_twirl8,zne,zf=1|3|7'
>>> aggregate(rows[:2])
{'depth': 5.0, 'two_qubit_ops': 1.5, 'estimated_error': 0.03, 'compile_time_s': 0.02}
>>> failed_trials(rows), FAILURE_KEYS
(1, ('compile_error', 'cutting_error', 'exec_error', 'mthree_error', 'zne_error'))
>>> sort_value(float("nan")), sort_value("3.5")
(inf, 3.5)
```

## Compile cache (`qbalance.cache`)

`Workload.adjust` caches every compiled ensemble on disk. The key is a SHA-256 digest of the cache format version, the Qiskit version, the backend spec with a digest of its calibration data, the backend name, a digest of the circuit's QPY serialization, the strategy's JSON, and the profile flag, so a changed input, toolchain, or calibration never serves a stale entry. An unreadable entry is ignored and recompiled, and a failed write never fails a run.

- `CacheEntry(key, dir)`: where one entry lives.
- `cache_dir(root=None) -> Path`: `<root>/cache`, with `root` defaulting to the platform cache directory (`qbalance.utils.default_cache_dir()`).
- `get_entry(key, root=None) -> CacheEntry`: the entry at `<cache_dir>/<key[:2]>/<key>`.
- `fingerprint_circuit(circuit) -> str`: the SHA-256 hex digest of the circuit's QPY serialization.
- `save_compiled(entry, circuit, meta)`: writes a circuit, or a list of circuits, to `compiled.qpy` (atomically), then `meta.json`.
- `load_compiled_ensemble(entry) -> (circuits, meta) | None`: every cached circuit and the metadata (flip-map keys restored to integers); `None` when the entry is incomplete.
- `load_compiled(entry) -> (circuit, meta) | None`: the first circuit only.

```pycon
>>> from qbalance.cache import CacheEntry, cache_dir, fingerprint_circuit, get_entry, load_compiled, load_compiled_ensemble, save_compiled
>>> entry = get_entry("ab" * 32, root=Path("my-cache"))
>>> isinstance(entry, CacheEntry), entry.dir.relative_to(cache_dir(Path("my-cache"))).as_posix()
(True, 'ab/abababababababababababababababababababababababababababababababab')
>>> load_compiled(entry) is None
True
>>> save_compiled(entry, [ghz(2), ghz(3)], {"measurement_flip_maps": [{"0": 1}, {}]})
>>> circuits, meta = load_compiled_ensemble(entry)
>>> [c.num_qubits for c in circuits], meta
([2, 3], {'measurement_flip_maps': [{0: 1}, {}]})
>>> load_compiled(entry)[0].num_qubits
2
>>> fingerprint_circuit(ghz(2)) == fingerprint_circuit(ghz(2)), len(fingerprint_circuit(ghz(2)))
(True, 64)
```

## Utilities (`qbalance.utils`)

- `measured_qubits_by_clbit(circuit) -> list[int]`: the qubit last measured into each measured classical bit, ordered by classical bit (`range(num_qubits)` when no measurement can be read).
- `measured_clbits(circuit) -> list[int] | None`: those classical bits, increasing (`None` when no measurement can be read).
- `operation_depth(circuit) -> int` and `operation_size(circuit) -> int`: depth and size counting operations only. `barrier`, `delay`, and other directives add nothing, although they still synchronize the wires they span, so the idle delays dynamical decoupling inserts are not billed as depth.
- `is_terminal_measurement(data, index) -> bool`: whether nothing after `data[index]` other than `barrier` or `delay` acts on the measured qubit or its classical bit; `data` holds `(operation, qubits, clbits)` triples. `SCHEDULING_DIRECTIVES` is `frozenset({"barrier", "delay"})`.
- `instruction_parts(entry) -> (operation, qubits, clbits)`, `shares_bit(bits, other) -> bool`, and `bit_index(circuit, bit) -> int`: accessors that also accept tuple-style instructions and bit objects without `find_bit`.
- `validate_integral(name, value, *, positive=False, non_negative=False) -> int`: validates an integer option, rejecting booleans, and returns it as an `int`; `validate_flag(name, value) -> bool` accepts only booleans (NumPy's included) and returns a `bool`.
- `backend_display_name(backend) -> str`: a backend's name, whether `name` is an attribute or a method, falling back to its class name.
- `stable_hash_bytes(data) -> str` and `stable_hash_str(s) -> str`: SHA-256 hex digests, stable across processes (unlike `hash()`).
- `atomic_write_bytes(path, data)`: writes through a temporary file in the same directory that is renamed into place, so readers never see a partial file; `dump_json(path, obj)` writes sorted, indented JSON that way, and `load_json(path)` reads JSON.
- `replacing_directory(target)`: a context manager yielding an empty staging directory beside `target`. When the block completes, whatever is at `target` is moved aside, the staging directory renamed into its place, and the old entry deleted (it is restored should that rename fail); when the block raises, `target` is left untouched and the staging directory removed. `save_dataset`, `BalancedWorkload.save`, and `qbalance compile` write through it.
- `default_cache_dir(app="qbalance") -> Path`: the platform user cache directory for `app`.

```pycon
>>> from qbalance import utils
>>> partial = QuantumCircuit(3, 3)
>>> _ = partial.h(0)
>>> _ = partial.measure(2, 0)
>>> _ = partial.measure(0, 2)
>>> utils.measured_qubits_by_clbit(partial), utils.measured_clbits(partial)
([2, 0], [0, 2])
>>> padded = QuantumCircuit(2)
>>> _ = padded.x(0)
>>> _ = padded.delay(100, 1)
>>> _ = padded.barrier()
>>> _ = padded.x(1)
>>> padded.depth(), utils.operation_depth(padded), padded.size(), utils.operation_size(padded)
(2, 2, 3, 2)
>>> parts = [utils.instruction_parts(inst) for inst in ghz(2).data]
>>> [utils.is_terminal_measurement(parts, i) for i, (op, _, _) in enumerate(parts) if op.name == "measure"]
[True, True]
>>> utils.SCHEDULING_DIRECTIVES == {"barrier", "delay"}, utils.shares_bit(parts[1][1], parts[2][1])
(True, True)
>>> utils.validate_integral("shots", 5, positive=True), utils.validate_integral("seed", 0, non_negative=True)
(5, 0)
>>> utils.validate_integral("shots", True, positive=True)
Traceback (most recent call last):
    ...
ValueError: shots must be a positive integer
>>> utils.validate_flag("execute", "no")
Traceback (most recent call last):
    ...
ValueError: execute must be a boolean
>>> utils.backend_display_name(backend), utils.stable_hash_str("qbalance")[:16]
('generic_backend_5q', '4d17798e357ac5e8')
>>> utils.dump_json(Path("out/data.json"), {"b": 1, "a": [1, 2]})
>>> print(Path("out/data.json").read_text())
{
  "a": [
    1,
    2
  ],
  "b": 1
}
>>> utils.load_json(Path("out/data.json"))
{'a': [1, 2], 'b': 1}
>>> with utils.replacing_directory(Path("out/run")) as staging:
...     _ = (staging / "result.txt").write_text("done")
>>> try:
...     with utils.replacing_directory(Path("out/run")) as staging:
...         _ = (staging / "partial.txt").write_text("half")
...         raise RuntimeError("interrupted")
... except RuntimeError:
...     pass
>>> sorted(p.name for p in Path("out/run").iterdir()), sorted(p.name for p in Path("out").iterdir())
(['result.txt'], ['data.json', 'run'])
>>> utils.default_cache_dir().name
'qbalance'
```

## Logging (`qbalance.logging`)

`qbalance.logging.get_logger(name="qbalance") -> logging.Logger` returns a standard logger. The first call installs one stream handler on the `qbalance` package logger (format `time | level | name | message`), but only when neither that logger nor the root logger has a handler. Records always propagate, and the handler prints only those emitted while the root logger has no handler, so an application that configures logging keeps full control whether it does so before or after importing qbalance, and no record is printed twice. The package logger's level is left unset, so it follows the root logger's (`WARNING` by default). `LOGGER_NAME` is `"qbalance"`.

```pycon
>>> from qbalance.logging import LOGGER_NAME, get_logger
>>> LOGGER_NAME, get_logger("qbalance.transpile").name
('qbalance', 'qbalance.transpile')
```

## Errors (`qbalance.errors`)

`QBalanceError`, a `RuntimeError`, is raised for failures qbalance detects itself: an unknown or malformed backend spec, a circuit with no feasible candidate, a cut finder the installed addon lacks. `OptionalDependencyError`, a `QBalanceError`, names the extra to install when an optional package is missing.

```pycon
>>> from qbalance.errors import OptionalDependencyError, QBalanceError
>>> issubclass(QBalanceError, RuntimeError), issubclass(OptionalDependencyError, QBalanceError)
(True, True)
```

## Command line (`qbalance.cli`)

`qbalance.cli.main()` is the console entry point behind `qbalance` and `python -m qbalance`; the commands are documented in the [CLI guide](cli.md). It runs the Typer application and reports a `QBalanceError`, `ValueError`, or `OSError` as one `Error: ...` line on standard error with exit status 1. Invalid command-line values exit with status 2 through Typer, and any other exception keeps its traceback.

```pycon
>>> import os, subprocess, sys
>>> finished = subprocess.run(
...     [sys.executable, "-m", "qbalance", "adjust", "no-such-dir", "--backend", "fake:generic:5", "--out", "out-dir"],
...     capture_output=True, text=True, env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
... )
>>> finished.returncode, finished.stderr.strip()
(1, "Error: [Errno 2] No such file or directory: 'no-such-dir/qbalance_dataset.json'")
```
