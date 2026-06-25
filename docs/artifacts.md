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
- objective weights,
- selected strategy specs and metrics per circuit,
- baseline metrics per circuit.

`BalancedWorkload.to_download(zip_path, overwrite=False)` creates a ZIP bundle containing the same saved workload layout.

## Matrix JSON

`run_matrix(...)` and `qbalance matrix` write:

```json
{
  "version": 1,
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

When `execute=True`, metrics can include counts, shot totals, execution errors, and ZNE probabilities depending on the strategy and backend.

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
