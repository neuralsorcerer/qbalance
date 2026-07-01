# Artifacts and reports

## Dataset layout

A qbalance dataset directory contains an index and circuit artifacts:

```text
<dataset_root>/
  qbalance_dataset.json
  <circuit>.qpy
  <other-circuit>.qpy
```

Each index record contains:

- `name`: circuit identifier,
- `artifact`: relative artifact path,
- `format`: usually `"qpy"` or `"qasm"`,
- `metadata`: optional JSON metadata.

Create datasets programmatically with `qbalance.save_dataset(...)` and load them with `qbalance.load_dataset(...)` or `qbalance.load_data("tiny")`.

## Balanced workload output

`BalancedWorkload.save(out_dir, overwrite=False)` writes:

```text
<out_dir>/
  dataset/
    qbalance_dataset.json
    <copied artifacts>
  results.json
  summary.txt
```

`results.json` includes:

- `backend_spec`,
- `objective`: objective weights,
- `selections`: selected strategy specs and metrics per circuit,
- `baseline_metrics`: baseline compile metrics per circuit,
- `evaluation_history`: every evaluated candidate strategy and metrics per circuit, in evaluation order.

A minimal shape is:

```json
{
  "backend_spec": "fake:generic:5",
  "objective": {"depth": 1.0},
  "selections": {
    "bell": {"spec": {"optimization_level": 1}, "metrics": {"depth": 3}}
  },
  "baseline_metrics": {"bell": {"depth": 4}},
  "evaluation_history": {
    "bell": [
      {"spec": {"optimization_level": 0}, "metrics": {"depth": 5}},
      {"spec": {"optimization_level": 1}, "metrics": {"depth": 3}}
    ]
  }
}
```

`BalancedWorkload.to_download(zip_path, overwrite=False)` creates a ZIP bundle containing the same saved workload layout.

Reload a saved workload directory with `qbalance.load_balanced_workload(out_dir)`. The loader expects the directory layout above (not the ZIP file itself), reconstructs the `BalancedWorkload`, and validates that selections, baseline metrics, and evaluation-history entries refer only to circuits in the copied dataset. Older artifacts that omit `evaluation_history` or set it to `null` still load, with an empty history mapping. Extract a ZIP bundle first if you need to reload a download archive.

```python
from qbalance import load_balanced_workload

balanced = load_balanced_workload("./balanced")
print(balanced.summary())
```

## Matrix JSON

`run_matrix(...)` and `qbalance matrix` write:

```json
{
  "version": 1,
  "metadata": {
    "dataset_dir": "./circuits",
    "backends": ["fake:generic:5"],
    "execute": false,
    "shots": 1024,
    "seed": 0,
    "profile": false
  },
  "results": [
    {
      "circuit": "bell",
      "backend": "fake:generic:5",
      "strategy": {"optimization_level": 1},
      "metrics": {"depth": 3}
    }
  ]
}
```

The top-level `metadata` block records the run context used to produce the artifact: dataset path, backend specs, execution/profile flags, shot count, and seed. `results` contains one row per backend/circuit/strategy combination. When `execute=True`, metrics can include counts, shot totals, execution errors, and ZNE probabilities depending on the strategy and backend.

## Reports

Markdown reports are always available:

```bash
python -m qbalance report ./matrix.json --out ./report
```

HTML reports require report extras:

```bash
pip install "qbalance[report]"
python -m qbalance report ./matrix.json --out ./report --html
```

The report layer groups matrix rows by serialized strategy settings and aggregates numeric metrics such as depth, two-qubit operation count, estimated error, and compile time.
