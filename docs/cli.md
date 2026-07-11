# CLI guide

All commands are available through either the installed `qbalance` console script or `python -m qbalance`.

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

Select one strategy per circuit for a target backend.

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
- `--pareto`: restrict final selection to the Pareto front before objective tie-break.
- `--max-candidates INTEGER`: number of generated candidates when `--strategies` is not supplied.
- `--strategies PATH`: strategy JSON file. When supplied, this file defines the complete candidate set.
- `--objective PATH`: objective-weight JSON file: a direct metric-to-weight mapping, `{ "weights": { ... } }`, or a saved-results object with an `objective` mapping. All metric names must be non-empty strings and all weights must be finite non-boolean numbers.
- `--execute`: execute compiled circuits and collect counts.
- `--shots INTEGER`: execution shots; must be a positive integer.
- `--seed INTEGER`: random seed used for deterministic bandit ordering and execution.
- `--cache-root PATH`: compiled-circuit cache directory; omit to use the platform cache.
- `--profile`: record pass-level transpiler profiling where supported.
- `--no-regression`: keep the baseline strategy when the best feasible candidate has a worse objective score than the finite-safe baseline score; equal scores and incomparable baselines do not trigger fallback.
- `--overwrite`: replace the output directory.

## `matrix`

Evaluate a fixed strategy list across one or more backends.

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
- `--strategies PATH`: strategy JSON file. If omitted, qbalance uses the built-in matrix defaults.
- `--execute`: execute compiled circuits and include counts/shot data.
- `--shots INTEGER`: execution shots; must be a positive integer.
- `--seed INTEGER`: random seed used for deterministic execution.
- `--profile`: record pass-level transpiler profiling where supported.

## `report`

Render a matrix report.

```bash
python -m qbalance report ./matrix.json --out ./report --html
```

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

The command writes compiled QPY artifacts when Qiskit QPY support is available and always writes `meta.json` with compile metrics.
