<h1 align="center">
qBalance
</h1>
<h3 align="center">
A workflow toolkit for balancing quantum compilation, suppression, and mitigation choices over a dataset of quantum circuits.
</h3>

---

<div align="center">

[![Qiskit Ecosystem](https://qisk.it/e-88affdcb)](https://qisk.it/e)
[![Current Release](https://img.shields.io/github/release/neuralsorcerer/qbalance.svg)](https://github.com/neuralsorcerer/qbalance/releases)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.0%2B-purple?logo=qiskit&logoColor=white)](https://www.ibm.com/quantum/qiskit/)
[![Test Linux](https://github.com/neuralsorcerer/qbalance/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/neuralsorcerer/qbalance/actions/workflows/ubuntu.yml?query=branch%3Amain)
[![Test Windows](https://github.com/neuralsorcerer/qbalance/actions/workflows/windows.yml/badge.svg)](https://github.com/neuralsorcerer/qbalance/actions/workflows/windows.yml?query=branch%3Amain)
[![Test MacOS](https://github.com/neuralsorcerer/qbalance/actions/workflows/macos.yml/badge.svg)](https://github.com/neuralsorcerer/qbalance/actions/workflows/macos.yml?query=branch%3Amain)
[![Lints](https://github.com/neuralsorcerer/qbalance/actions/workflows/lint.yml/badge.svg)](https://github.com/neuralsorcerer/qbalance/actions/workflows/lint.yml?query=branch%3Amain)
[![CodeQL](https://github.com/neuralsorcerer/qbalance/actions/workflows/codeql.yml/badge.svg)](https://github.com/neuralsorcerer/qbalance/actions/workflows/codeql.yml?query=branch%3Amain)
[![Documentation](https://github.com/neuralsorcerer/qbalance/actions/workflows/docs.yml/badge.svg)](https://github.com/neuralsorcerer/qbalance/actions/workflows/docs.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/License-MIT-3c60b1.svg?logo=opensourceinitiative&logoColor=white)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.02966-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2605.02966)
[![DOI:48550/arXiv.2605.02966](https://img.shields.io/badge/DOI-10.48550/arXiv.2605.02966-blue.svg)](https://doi.org/10.48550/arXiv.2605.02966)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qbalance?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/qbalance)

</div>

At a high level, it provides a reproducible pipeline to:

1. load a dataset,
2. resolve one or more backend targets,
3. evaluate candidate strategies,
4. select one strategy per circuit,
5. export artifacts for downstream analysis/reporting.

It supports both a Python API and a CLI.

Full documentation lives in [`docs/`](docs/index.md), including the [API reference](docs/api-references.md), [CLI guide](docs/cli.md), [strategy configuration guide](docs/strategies.md), and [artifact/report reference](docs/artifacts.md).

---

## Why qbalance

Quantum workload tuning is naturally multi-objective. Typical goals conflict:

- lower depth vs. lower compile time,
- fewer two-qubit operations vs. hardware routing constraints,
- suppression/mitigation quality vs. runtime overhead.

qbalance provides a single workflow that keeps these trade-offs explicit and reproducible.

---

## Core capabilities

- **Dataset IO**: save/load datasets using a JSON index and QPY artifacts with record-level metadata.
- **Backend resolution**: plugin-based backend specification (`fake:*`, `aer:*`, custom).
- **Strategy model**: immutable `StrategySpec` with compile/suppression/mitigation controls.
- **Search modes**:
  - `grid`: evaluate candidates in order (all of them, or the first `max_evaluations`).
  - `bandit`: warmup + adaptive proposals via a Bayesian linear surrogate, spending a `max_evaluations` budget per circuit on the most promising candidates.
- **Selection modes**:
  - direct objective minimization,
  - optional Pareto filtering before tie-break by objective.
- **Mitigation/Suppression knobs**: Pauli twirling, dynamical decoupling, measurement twirling, M3, ZNE, optional cutting.
- **Diagnostics**: baseline-vs-selected distribution distances (EMD/CVM/KS).
- **Reporting**: markdown + optional HTML from matrix JSON.
- **Caching**: compiled-circuit cache keyed by backend (spec, name, and calibration digest) + circuit fingerprint + strategy payload + profile flag + toolchain version.

---

## Installation

### Base

```bash
pip install qbalance
```

### Optional extras

```bash
pip install "qbalance[aer]"         # qiskit-aer integration
pip install "qbalance[runtime]"     # IBM runtime integration helpers
pip install "qbalance[mitigation]"  # mthree integration
pip install "qbalance[cutting]"     # qiskit-addon-cutting integration
pip install "qbalance[report]"      # reporting stack (jinja2/matplotlib/pandas)
pip install "qbalance[all]"         # all optional dependencies + dev tools
```

### Development install

```bash
pip install -e ".[all]"
```

**Python requirement**: `>=3.10`.

---

## Quickstart

### Python API

```python
from qbalance import Workload, load_balanced_workload, load_data

# 1) built-in tiny dataset: bell, ghz3, qft4
ds = load_data("tiny")

# 2) choose backend target
wl = Workload.from_dataset(ds).set_target("fake:generic:5")

# 3) run strategy search/selection
balanced = wl.adjust(
    search="bandit",      # or "grid"
    pareto=True,           # optional Pareto pre-filtering
    max_candidates=24,
    execute=False,         # compile-only mode
    seed=7,                # deterministic candidate ordering
    cache_root="./.qbalance-cache",
    profile=False,
    allow_regression=False,  # optional: never select a candidate worse than the baseline
)

# 4) inspect/persist artifacts
print(balanced.summary())
print(balanced.candidate_rankings()["bell"][0])
balanced.save("./balanced", overwrite=True)
balanced.to_download("./balanced_bundle.zip", overwrite=True)

# Reload saved results later without recompilation.
reloaded = load_balanced_workload("./balanced")
print(reloaded.summary())
```

### CLI

```bash
# Create built-in dataset
python -m qbalance dataset examples --out ./circuits --overwrite

# Objective weights (lower score is better): the defaults without the
# wall-clock compile-time term, so scores are reproducible
cat > objective.json <<'JSON'
{"depth": 1.0, "two_qubit_ops": 2.0, "estimated_error": 10.0, "sampling_overhead": 1.0}
JSON

# Run per-circuit adjustment over the default candidate pool; bandit search
# spends a budget of 8 evaluations per circuit
python -m qbalance adjust ./circuits \
  --backend fake:generic:5 \
  --out ./balanced \
  --search bandit \
  --max-evaluations 8 \
  --pareto \
  --seed 7 \
  --cache-root ./.qbalance-cache \
  --objective ./objective.json \
  --no-regression \
  --overwrite

# A fixed strategy list for the matrix
cat > strategies.json <<'JSON'
{"strategies": [
  {"optimization_level": 1, "routing_method": "sabre"},
  {"optimization_level": 2, "layout_method": "qbalance_noise_aware", "routing_method": "sabre"},
  {"optimization_level": 2, "routing_method": "sabre", "measurement_twirling": true, "num_twirls": 8}
]}
JSON

# Evaluate fixed strategy matrix across backends
python -m qbalance matrix ./circuits \
  --backend fake:generic:5 \
  --backend fake:generic:10 \
  --out ./matrix.json \
  --execute \
  --seed 7 \
  --shots 1024 \
  --strategies ./strategies.json

# Render markdown/html reports
python -m qbalance report ./matrix.json --out ./report --html
```

---

## Architecture (execution flow)

```text
CircuitDataset
   │
   ├── Workload.from_dataset(...) / Workload.from_path(...)
   │
   ├── set_target("fake:generic:5" | "aer:..." | custom plugin)
   │
   └── adjust(...)
         ├── baseline (compiled, and executed when execute=True)
         ├── candidate search (grid | bandit, optional max_evaluations budget)
         ├── compile once (+ twirl instances, DD, or cut subexperiments)
         ├── optional execution of every instance (+ mitigation)
         ├── objective scoring
         └── choose best (optional Pareto pre-filter)
               ↓
         BalancedWorkload
             ├── summary()
             ├── save(out_dir)
             ├── to_download(zip)
             └── reload via load_balanced_workload(out_dir)
```

---

## Data model

Dataset directory layout:

```text
<dataset_root>/
  qbalance_dataset.json
  <name>.qpy
  <name_1>.qpy
  ...
```

`qbalance_dataset.json` indexes records with:

- `name`: circuit identifier,
- `artifact`: relative artifact path,
- `format`: `"qpy"` or `"qasm"`,
- `metadata`: optional JSON payload.

Programmatic dataset creation:

```python
from pathlib import Path
from qiskit import QuantumCircuit
from qbalance import save_dataset, load_dataset

qc = QuantumCircuit(2, 2, name="bell")
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

save_dataset(Path("./my_dataset"), [qc], metadata=[{"family": "bell"}], overwrite=True)
ds = load_dataset(Path("./my_dataset"))
print(ds.names())
```

---

## Strategy space

A strategy is an immutable `StrategySpec` with these groups:

1. **Compilation knobs**: `optimization_level`, layout/routing/translation methods, transpiler seed.
2. **Suppression knobs**: Pauli twirling, twirl count, dynamical decoupling, measurement twirling, suppression seed.
3. **Mitigation knobs**: M3 toggle, ZNE toggle + factors + polynomial degree.
4. **Cutting knobs**: cutting toggle + max subcircuit qubits.
5. **Runtime knob**: optional resilience level metadata.

Example:

```python
from qbalance import StrategySpec

spec = StrategySpec(
    optimization_level=2,
    layout_method="qbalance_noise_aware",
    routing_method="sabre",
    seed_transpiler=0,
    pauli_twirling=True,
    num_twirls=8,
    dynamical_decoupling=False,
    measurement_twirling=False,
    mthree=False,
    zne=False,
)
```

---

## Mathematical model (objective, Pareto, diagnostics)

### 1) Weighted objective

Given metric vector $\mathbf{m}$, qbalance minimizes:

```math
J(\mathbf{m}) = \sum_{k \in K} w_k\,m_k
```

Default terms:

```math
J = 1.0\cdot\text{depth}
  + 2.0\cdot\text{two\_qubit\_ops}
  + 10.0\cdot\text{estimated\_error}
  + 0.1\cdot\text{compile\_time\_s}
  + 1.0\cdot\text{sampling\_overhead}
```

Interpretation:

- higher `estimated_error` is penalized strongly (weight 10),
- two-qubit operation count has moderate penalty (weight 2),
- compile time contributes but with a small coefficient (0.1),
- `sampling_overhead` is 1 for a circuit that runs once (a constant that never reorders those candidates) and charges ZNE and cut candidates for their extra shots (see below),
- `depth` counts operation layers; scheduling directives (`barrier`, `delay`) add none, so the idle delays dynamical decoupling inserts are not billed.

### 2) Finite-safe scoring behavior

For each objective term, qbalance ignores a term when the metric is missing, non-numeric, or non-finite. Selection then distinguishes two cases: a candidate that reports an objective metric only with invalid values (`None`, non-numeric, or non-finite) is treated as worst-case (score $+\infty$), while a metric no candidate reports at all simply contributes nothing, so an objective made only of such terms scores every candidate 0 and keeps the first. `adjust` logs a warning naming every objective term that no evaluated metrics report (a misspelled name, or an execution metric without `execute=True`).

### 2b) Reproducibility of `objective_score`

Compilation itself is deterministic: with a fixed `seed`, two cold-cache runs produce identical `depth`, `two_qubit_ops` and `estimated_error` values for every candidate. So does execution on simulators: twirl instances, M3's calibration and ZNE's folded circuits all run with seeds derived from `seed`, so executed and mitigated metrics repeat exactly, whether the compiled circuits come fresh or from the cache. The default objective, however, weights `compile_time_s` at `0.1`, and compile time is wall-clock, so `objective_score` varies between runs. Candidates that compile to the same depth, two-qubit count, error and overhead (optimization levels 2 and 3 often do) are then ordered by timing noise, so which of them is selected can change from one cold-cache run to the next; a rerun on a warm cache replays the recorded compile times and repeats its selections. Drop the `compile_time_s` term to make scores and selection bit-reproducible:

```python
from qbalance import Objective

objective = Objective(
    weights={
        "depth": 1.0,
        "two_qubit_ops": 2.0,
        "estimated_error": 10.0,
        "sampling_overhead": 1.0,
    }
)
```

The default objective's `sampling_overhead` term is the factor by which a strategy multiplies the shots a given precision needs. It is `1.0` for a circuit that runs once, the quasi-probability sampling overhead (9 per cut CNOT, multiplied over the cuts) for circuit cutting, and $n\sum_i w_i^2$ for ZNE, where $w_i$ are the least-squares weights that give the extrapolated value from the $n$ noise factors (4.375 for the default linear fit at factors 1, 3, 5). So a cut candidate's shallower subexperiments, or a ZNE candidate's mitigation, are weighed against the extra shots they cost; without it a ZNE strategy scored like the same compile without ZNE.

### 3) Pareto pre-filtering

With `pareto=True`, qbalance first computes a non-dominated set on:

- `depth`,
- `two_qubit_ops`,
- `estimated_error`,
- `sampling_overhead` (1 for a circuit that runs once; without it a cut candidate's shallower subexperiments would dominate every uncut candidate regardless of their shot cost).

Candidate $a$ dominates candidate $b$ iff:

```math
\big(\forall i,\ a_i \le b_i\big)\ \land\ \big(\exists j,\ a_j < b_j\big)
```

Then qbalance chooses the minimum-objective strategy within that Pareto front.

### 4) Bandit proposal model

In `bandit` mode, candidates after warmup are proposed by a Thompson-sampling style linear surrogate over each circuit's score relative to its baseline, $y=(s-s_{\text{base}})/|s_{\text{base}}|$, so one model can learn from circuits of any size:

```math
\mathbf{y} \approx X\mathbf{w},\qquad
\Lambda = \alpha I + \frac{1}{\sigma^2}X^T X,
\qquad
\mu = \Lambda^{-1}\frac{1}{\sigma^2}X^T\mathbf{y}
```

A sample $\tilde{\mathbf{w}}\sim\mathcal{N}(\mu,\Lambda^{-1})$ is drawn implicitly from the Cholesky factor of $\Lambda$, and candidates are ranked by linear score $\phi(s)^T\tilde{\mathbf{w}}$ (lower is preferred). The noise variance $\sigma^2$ is the sample variance of the observed $y$. Proposals stop once `max_evaluations` candidates have been evaluated; without a budget every candidate is evaluated and the proposals only reorder the work.

### 5) Distribution diagnostics

For baseline and selected metric samples, qbalance computes:

- **KS distance**:
  ```math
  D_{KS}=\sup_x |F_1(x)-F_2(x)|
  ```
- **EMD/Wasserstein-1 (1D CDF form)**:
  ```math
  W_1=\int |F_1(x)-F_2(x)|\,dx
  ```
- **CVM-type distance used here**:
  ```math
  \mathrm{CVM}=\int (F_1(x)-F_2(x))^2\,dx
  ```

where $F_1, F_2$ are empirical weighted CDFs on aligned support grids.

### 6) Entropy and top-probability metrics (execution path)

When execution is available, helper metrics may include:

- Shannon entropy over observed counts:
  ```math
  H=-\sum_i p_i\log_2 p_i
  ```
- top observed probability:
  ```math
  p_{\max}=\max_i p_i
  ```

with $p_i = c_i / \sum_j c_j$.

- Z-parity expectation over the measured bits, $\langle Z\otimes\cdots\otimes Z\rangle=\sum_i (-1)^{|b_i|}p_i$, reported for every execution path so they can be compared: `raw_parity_expval` (observed counts), `mitigated_parity_expval` (M3), `zne_parity_expval` (zero-noise extrapolated) and `cut_parity_expval` (reconstructed from cut subexperiments).

Twirled strategies run every twirl instance on an equal share of the shots and pool the untwirled counts, since a twirl only suppresses noise averaged over its instances.

---

## Caching and computational performance

Compiled circuits may be reused via cache key:

```math
\text{key}=\text{SHA256}(\text{toolchain}:\text{backend\_spec}\,|\,\text{calibration}\,|\,\text{backend\_name}:\text{circuit\_fingerprint}:\text{strategy\_json}:\text{profile})
```

This avoids redundant transpilation for repeated `(circuit, backend, strategy)` tuples. The toolchain part (cache format and Qiskit version) and the calibration digest (the backend's gate errors and durations, qubit coherence times and frequencies, `dt`, and the calibration date when the backend reports one) make a new Qiskit, a changed pipeline, or a recalibrated device miss the cache instead of reusing layouts and `estimated_error` computed from stale data.

Additional performance-relevant behavior:

- matrix execution pre-serializes strategy payloads once before inner loops,
- distribution metrics aggregate and align sorted supports efficiently,
- optional `profile=False` avoids transpiler pass profiling overhead.

---

## Artifacts and output contracts

### `adjust` output directory

- `dataset/` copied dataset index + artifacts,
- `results.json` selected strategy specs/metrics + baseline metrics + objective weights + per-circuit selection diagnostics + per-circuit candidate rankings + per-circuit candidate evaluation history,
- `summary.txt` text summary.

Saved adjustment results also include per-circuit `candidate_rankings` derived from the full evaluation history. Each leaderboard row records the original evaluation index, serialized strategy spec, diagnostic objective score, finite-safe selection score, objective-term contributions, selected-candidate marker, and rank. Ranking uses the same objective-score semantics as final strategy selection and then original evaluation order for deterministic ties; incomparable candidates use `null` selection scores and sort after comparable candidates. If `--no-regression`/`allow_regression=False` falls back to a baseline that was not in the candidate history, the selected baseline is emitted as a synthetic ranking row with `original_index: null`.

### `matrix` output JSON

```json
{
  "version": 1,
  "metadata": {"dataset_dir": "...", "backends": ["..."], "execute": false, "shots": 1024, "seed": 0, "profile": false},
  "results": [
    {
      "circuit": "...",
      "backend": "...",
      "strategy": {"...": "..."},
      "metrics": {"...": "..."}
    }
  ]
}
```

### `report` output directory

- `report.md`,
- `report.html` when `--html` is used (requires the `report` extra).

---

## CLI reference

### `dataset`

```bash
python -m qbalance dataset examples --out ./circuits --overwrite
```

### `adjust`

The strategy and objective files are optional inputs; their formats are described below.

```bash
python -m qbalance adjust ./circuits \
  --backend fake:generic:5 \
  --out ./balanced \
  --search grid \
  --pareto \
  --max-candidates 24 \
  --strategies ./strategies.json \
  --objective ./objective.json \
  --execute \
  --shots 1024 \
  --profile \
  --no-regression \
  --overwrite
```

Use `--objective` to tune the metric trade-off used for candidate ranking and no-regression checks. The file may be a direct mapping such as `{"depth": 1.0, "two_qubit_ops": 2.0}`, an object with a `weights` field, or a saved-results object with an `objective` field. The loader rejects empty mappings, blank metric names, booleans, non-numeric values, and non-finite values.

Use `--no-regression` when the final selection must never score worse than the baseline strategy on the objective terms both report (the baseline runs no mitigation, so mitigation-derived terms are not compared): a regressing top candidate is replaced by the best-ranked one that does not regress, and the baseline is kept only when every candidate regresses; equal scores and incomparable baselines do not trigger fallback.

A strategy JSON file may be one strategy object, a list of strategy objects, an object with a `strategies` list, or a saved `results.json` / matrix JSON whose strategies are reused:

```json
{
  "strategies": [
    {"optimization_level": 1, "routing_method": "sabre"},
    {"optimization_level": 2, "layout_method": "qbalance_noise_aware", "routing_method": "sabre"}
  ]
}
```

### `matrix`

```bash
python -m qbalance matrix ./circuits \
  --backend fake:generic:5 \
  --backend fake:generic:10 \
  --out ./matrix.json \
  --execute \
  --shots 1024 \
  --profile \
  --strategies ./strategies.json
```

### `report`

```bash
python -m qbalance report ./matrix.json --out ./report --html
```

### `plugins`

```bash
python -m qbalance plugins list
```

### `compile`

```bash
python -m qbalance compile ./circuits \
  --backend fake:generic:5 \
  --out ./compiled_bundle \
  --optimization-level 2 \
  --routing-method sabre \
  --layout-method qbalance_noise_aware \
  --pauli-twirling \
  --num-twirls 8 \
  --dd \
  --meas-twirl \
  --overwrite
```

---

## Plugin system

Entry-point groups:

- `qbalance.backends` — **resolved at runtime.** A third-party package that registers
  here extends backend spec strings: register `mock` and `-b mock:5` resolves through
  your loader everywhere a backend spec is accepted.
- `qbalance.objectives` — inventory only.
- `qbalance.reports` — inventory only.

Only the backend group is loaded. Nothing resolves an objective or a report by plugin
name today: objectives come from `default_objective()` or `load_objective(path)`, and
`qbalance report` calls the markdown and HTML renderers directly. Registrations in
those two groups appear in `plugins list` but cannot yet be invoked.

Inspect active registrations with:

```bash
python -m qbalance plugins list
```

---

## Edge cases and failure semantics

- `adjust()` without `.set_target(...)` raises `ValueError`.
- Invalid search mode raises `ValueError`.
- If no candidate of a circuit is feasible, `adjust()` raises `QBalanceError` (a `RuntimeError`) listing the failures.
- `matrix` validates `shots` as a positive integer and `seed` as a non-negative integer.
- `matrix` raises `ValueError` if dataset record count and loaded circuit count mismatch.
- Existing output paths require explicit overwrite flags.
- Optional dependency features require installed extras.
- Cutting, compilation, execution, and mitigation failures are captured in metrics (`cutting_error`, `compile_error`, `exec_error`, `mthree_error`, `zne_error`) so runs can continue; in `adjust` such a candidate is kept in the history but is infeasible for selection.

---

## Correctness and complexity notes

For a dataset with $N$ circuits and $S$ strategy candidates per circuit:

- **Grid search** evaluates up to $N\times S$ candidates.
- **Bandit search** evaluates $N\times\min(B, S)$ candidates for a `max_evaluations` budget $B$, spending the budget on the candidates its surrogate expects to score best (grid search with the same budget takes the first $B$ in order). Without a budget both modes evaluate all $N\times S$.
- **Pareto filtering** in practice is close to linear in small fronts, with a quadratic worst-case bound on the number of unique objective vectors.

If a compile cache hit occurs, effective transpilation cost is reduced from repeated compile cost $T_{compile}$ to key lookup + deserialization overhead.


## Development

```bash
# Install editable package with all extras, plus the hook runner
# (pre-commit is not part of any extra)
pip install -e ".[all]" pre-commit

# Install hooks once
pre-commit install

# Run tests with warnings promoted to errors
pytest -q -W error

# Run formatting/lint/type hooks
pre-commit run --all-files
```

---

## Citation

If you use qbalance in your work and wish to refer to it, please use the following BibTeX entry.

```bibtex
@misc{sarkar2026qbalancereproduciblemultiobjectiveworkflow,
      title={QBalance: A Reproducible Multi-Objective Workflow for Quantum Compilation, Noise Suppression, and Error-Mitigation Strategy Selection},
      author={Soumyadip Sarkar},
      year={2026},
      eprint={2605.02966},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2605.02966},
}
```

## License

MIT License. See [LICENSE](./LICENSE).
