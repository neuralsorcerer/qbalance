# CLI guide

All commands are available through either the installed `qbalance` console script or `python -m qbalance`.

Errors in the input or environment (a missing dataset or strategy file, an unknown backend, an invalid option or strategy, a circuit with no feasible candidate, a circuit `compile` cannot compile for the backend) are reported as a single `Error: ...` line with exit status 1; invalid command-line values exit with status 2. `adjust`, `compile`, and `dataset` check `--out` before compiling or writing anything: an existing output needs `--overwrite`, an output path that is a file is refused, and so is an output that contains the source dataset.

## `dataset`

Create a built-in example dataset.

```bash
python -m qbalance dataset examples --out ./circuits --overwrite
```

Options:

- `--out, -o PATH`: output dataset directory.
- `--overwrite`: replace an existing output directory.

Currently, `examples` is the supported dataset action.

## `adjust`

Select one strategy per circuit for a target backend. The example shows every option; `--strategies` and `--objective` are optional files, described in [getting started](getting-started.md) and [strategies](strategies.md) (an objective file is a mapping such as `{"depth": 1.0, "two_qubit_ops": 2.0}`).

```bash
python -m qbalance adjust ./circuits \
  --backend fake:generic:5 \
  --out ./balanced \
  --search grid \
  --pareto \
  --max-candidates 24 \
  --execute \
  --shots 1024 \
  --seed 7 \
  --cache-root ./.qbalance-cache \
  --profile \
  --strategies ./strategies.json \
  --objective ./objective.json \
  --no-regression \
  --overwrite
```

Options:

- `--backend, -b TEXT`: backend spec, for example `fake:generic:5`.
- `--out, -o PATH`: output directory for the balanced workload.
- `--search TEXT`: `grid` or `bandit`.
- `--max-evaluations INTEGER`: evaluate at most this many candidates per circuit (bandit search spends the budget adaptively); omit to evaluate every candidate.
- `--warmup INTEGER`: randomly ordered candidates bandit search evaluates before its surrogate starts proposing (default `6`). Keep it below `--max-evaluations`, or the whole budget is spent at random.
- `--pareto`: restrict final selection to the Pareto front before objective tie-break.
- `--max-candidates INTEGER`: number of generated candidates when `--strategies` is not supplied.
- `--strategies PATH`: strategy JSON file. When supplied, this file defines the complete candidate set.
- `--objective PATH`: objective-weight JSON file: a direct metric-to-weight mapping, `{ "weights": { ... } }`, or a saved-results object with an `objective` mapping. All metric names must be non-empty strings and all weights must be finite non-boolean numbers.
- `--execute`: execute compiled circuits and collect counts.
- `--shots INTEGER`: execution shots; must be a positive integer.
- `--seed INTEGER`: non-negative seed for the candidate shuffle, bandit search, and execution (simulator and M3 calibration). With the default objective, candidates that compile identically are ordered by wall-clock compile time, so pass an `--objective` without `compile_time_s` when selections must repeat exactly across cold-cache runs.
- `--cache-root PATH`: compiled-circuit cache directory; omit to use the platform cache.
- `--profile`: record pass-level transpiler profiling where supported.
- `--no-regression`: never select a candidate that scores worse than the baseline on the objective terms both report (the baseline runs no mitigation, so mitigation-derived terms are not compared). A regressing top candidate is replaced by the best-ranked one that does not regress, and the baseline strategy is kept only when every candidate regresses; equal scores and incomparable baselines do not trigger fallback.
- `--overwrite`: replace the output directory.

## `matrix`

Evaluate a fixed strategy list across one or more backends. `--strategies` is optional (built-in defaults are used without it); the example assumes a `strategies.json` like the one in [getting started](getting-started.md), and `report` below reads the `matrix.json` it writes.

```bash
python -m qbalance matrix ./circuits \
  --backend fake:generic:5 \
  --backend fake:generic:10 \
  --out ./matrix.json \
  --execute \
  --shots 1024 \
  --seed 7 \
  --profile \
  --strategies ./strategies.json
```

Options:

- `--backend, -b TEXT`: repeatable backend spec.
- `--out, -o PATH`: output matrix JSON path.
- `--strategies PATH`: strategy JSON file. If omitted, the matrix runs five built-in strategies: optimization level 1 with SABRE routing, and at optimization level 2 with SABRE routing the noise-aware layout, Pauli twirling (8 instances), dynamical decoupling (`XY4`), and measurement twirling (8 instances).
- `--execute`: execute compiled circuits and include counts/shot data.
- `--shots INTEGER`: execution shots; must be a positive integer.
- `--seed INTEGER`: non-negative simulator seed; twirl instance `i` runs with `seed + i`.
- `--profile`: record pass-level transpiler profiling where supported.

## `report`

Render a matrix report.

```bash
python -m qbalance report ./matrix.json --out ./report --html
```

Each backend gets a table with one row per strategy: mean depth, two-qubit operations, estimated error, and compile time over the circuits, plus `failed trials` (`k/n`), the number of trials that recorded `compile_error`, `cutting_error`, `exec_error`, `mthree_error`, or `zne_error`. A failed execution or mitigation keeps its compile metrics, so that column is what tells a successful row from a failed one.

Options:

- `--out, -o PATH`: report output directory.
- `--html`: also render HTML output. This requires the report extra dependencies.

## `plugins`

List discovered entry-point plugins.

```bash
python -m qbalance plugins list
```

## `compile`

Compile every circuit in a dataset using one explicit strategy supplied as CLI flags.

```bash
python -m qbalance compile ./circuits \
  --backend fake:generic:5 \
  --out ./compiled \
  --optimization-level 1 \
  --routing-method sabre \
  --layout-method qbalance_noise_aware \
  --pauli-twirling \
  --num-twirls 8 \
  --dd \
  --meas-twirl \
  --overwrite
```

Options:

- `--backend, -b TEXT`: backend spec.
- `--out, -o PATH`: output directory.
- `--optimization-level INTEGER`: Qiskit optimization level, `0`-`3` (default `1`).
- `--routing-method TEXT`: routing method (default `sabre`).
- `--layout-method TEXT`: layout method, for example `sabre` or `qbalance_noise_aware` (default: Qiskit's choice).
- `--pauli-twirling`: twirl the compiled two-qubit gates.
- `--num-twirls INTEGER`: number of twirl instances (default `1`).
- `--dd`: pad idle windows with dynamical decoupling (`XY4`, or `XX` when the target has no `y` gate).
- `--meas-twirl`: twirl the terminal measurements.
- `--overwrite`: replace a previous output directory.

The command writes compiled QPY artifacts when Qiskit QPY support is available and always writes `meta.json` with compile metrics per record and, under `artifacts`, the compiled file each record was written to. With `--pauli-twirling` or `--meas-twirl`, each artifact holds all `--num-twirls` instances (load them with `qpy.load`, which returns a list), and `meta.json` records one entry of `measurement_flip_maps` per instance, in the same order. Each compiled artifact is named after its source artifact's stem (`bell.qpy` -> `compiled/bell.qpy`); stems that collide, including case-only differences, get a `_N` suffix. `--overwrite` replaces a previous output directory but refuses an `--out` that is, or contains, the source dataset. The output is written beside `--out` and moved into place once every circuit has compiled, so a failing circuit leaves a previous output intact.
