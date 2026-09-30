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
- `format`: `"qpy"` (what `save_dataset` writes) or `"qasm"`,
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

The copied `dataset/` holds exactly the workload's own records: its index is written from them rather than copied from the source, so a workload adjusted on one half of a `split()` saves and reloads cleanly.

`results.json` includes:

- `backend_spec`,
- `objective`: objective weights,
- `selections`: selected strategy specs and metrics per circuit. When `allow_regression=False` rejects the top candidate (falling back to the best non-regressing candidate, or to the baseline when every candidate regresses), the selected metrics include guard metadata such as `selected_by_regression_guard`, `rejected_candidate_spec`, and `rejected_candidate_objective_score`;
- `baseline_metrics`: baseline strategy metrics per circuit (compile metrics, plus execution metrics when `execute=True`),
- `selection_diagnostics`: derived per-circuit baseline-vs-selected diagnostics,
- `candidate_rankings`: derived per-circuit objective-ranked candidate leaderboards,
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
  "selection_diagnostics": {
    "bell": {
      "baseline_objective_score": 4.0,
      "selected_objective_score": 3.0,
      "objective_delta": -1.0,
      "comparable_objective_terms": ["depth"],
      "objective_improved": true,
      "objective_terms": {
        "baseline": {"depth": 4.0},
        "selected": {"depth": 3.0}
      },
      "evaluated_candidates": 2,
      "metric_deltas": {
        "depth": {
          "baseline": 4.0,
          "selected": 3.0,
          "delta": -1.0,
          "relative_delta": -0.25
        }
      }
    }
  },
  "candidate_rankings": {
    "bell": [
      {
        "original_index": 1,
        "spec": {"optimization_level": 1},
        "objective_score": 3.0,
        "selection_score": 3.0,
        "objective_terms": {"depth": 3.0},
        "selected": true,
        "rank": 1
      },
      {
        "original_index": 0,
        "spec": {"optimization_level": 0},
        "objective_score": 5.0,
        "selection_score": 5.0,
        "objective_terms": {"depth": 5.0},
        "selected": false,
        "rank": 2
      }
    ]
  },
  "evaluation_history": {
    "bell": [
      {"spec": {"optimization_level": 0}, "metrics": {"depth": 5}},
      {"spec": {"optimization_level": 1}, "metrics": {"depth": 3}}
    ]
  }
}
```

`BalancedWorkload.to_download(zip_path, overwrite=False)` creates a ZIP bundle containing the same saved workload layout.

Both write their output under a temporary name beside the destination and move it into place only once it is complete, so a failed save or export leaves an existing one intact; `save_dataset` and `qbalance compile` do the same.


Selection diagnostics are JSON-safe and finite-aware:

- `baseline_objective_score`, `selected_objective_score`, and `objective_delta` are numeric only when at least one finite weighted objective term contributes on both sides; otherwise they are `null`. `objective_delta` is computed over `comparable_objective_terms`, the terms both sides report.
- `objective_improved` is `true` when the selection is no worse than the baseline (`objective_delta <= 0`, so a tie counts, as it does for the regression guard), `false` when it is worse, and `null` when the baseline and selected scores cannot be compared.
- `objective_terms` records each finite weighted objective contribution used in the score.
- `metric_deltas` contains `baseline`, `selected`, `delta`, and `relative_delta` values for common compile metrics; invalid or non-finite inputs become `null`.

Candidate rankings are derived from `evaluation_history` and are also JSON-safe:

- entries are sorted by the same finite-safe selection score used for final strategy selection, then by `original_index` for deterministic ties;
- `objective_score` is the diagnostic score recomputed from finite weighted objective terms;
- `selection_score` mirrors selection semantics, including a valid stored `metrics["objective_score"]` when available, and becomes `null` for incomparable candidates;
- `objective_terms` records the finite weighted terms used for the diagnostic score;
- `selected` marks the row matching the saved selected strategy and metrics;
- when `allow_regression=False` chooses the baseline and that baseline was not in `evaluation_history`, `candidate_rankings` includes one synthetic selected baseline row with `original_index: null` so consumers can still identify the final selection exactly once.

Reload a saved workload directory with `qbalance.load_balanced_workload(out_dir)`. The loader expects the directory layout above (not the ZIP file itself), reconstructs the `BalancedWorkload`, and validates that selections, baseline metrics, and evaluation-history entries refer only to circuits in the copied dataset. `selection_diagnostics` and `candidate_rankings` are derived metadata and are recomputed by `BalancedWorkload.selection_diagnostics()` and `BalancedWorkload.candidate_rankings()` after loading, so older artifacts that omit them still load. Older artifacts that omit `evaluation_history` or set it to `null` also still load, with an empty history mapping. Extract a ZIP bundle first if you need to reload a download archive.

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

The top-level `metadata` block records the run context used to produce the artifact: dataset path, backend specs, execution/profile flags, shot count, and seed. `results` contains one row per backend/circuit/strategy combination. When `execute=True`, metrics can include counts, shot totals, M3-mitigated probabilities (`mthree_probs`), ZNE probabilities (`zne_probs`) with their realized fold factors, the reconstructed parity of a cut circuit (`cut_parity_expval`), and the reason a stage failed (`exec_error`, `mthree_error`, `zne_error`; `compile_error` and `cutting_error` replace the compile metrics of a strategy the backend could not compile or a circuit that could not be cut), depending on the strategy and backend.

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

The report layer groups matrix rows per backend and per strategy label and shows, for each, the means of depth, two-qubit operation count, estimated error, and compile time over its trials, sorted by mean depth and then mean two-qubit count. Each row also shows `failed trials` (`k/n`): how many of its trials recorded `compile_error`, `cutting_error`, `exec_error`, `mthree_error`, or `zne_error`. A failed execution or mitigation keeps its compile metrics, so this column is what distinguishes it from a successful row.
