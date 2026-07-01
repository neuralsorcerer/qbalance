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
[![Test Linux](https://github.com/neuralsorcerer/qbalance/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/neuralsorcerer/qbalance/actions/workflows/ubuntu.yml?query=branch%3Amain)
[![Test Windows](https://github.com/neuralsorcerer/qbalance/actions/workflows/windows.yml/badge.svg)](https://github.com/neuralsorcerer/qbalance/actions/workflows/windows.yml?query=branch%3Amain)
[![Test MacOS](https://github.com/neuralsorcerer/qbalance/actions/workflows/macos.yml/badge.svg)](https://github.com/neuralsorcerer/qbalance/actions/workflows/macos.yml?query=branch%3Amain)
[![Lints](https://github.com/neuralsorcerer/qbalance/actions/workflows/lint.yml/badge.svg)](https://github.com/neuralsorcerer/qbalance/actions/workflows/lint.yml?query=branch%3Amain)
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
  - `grid`: evaluate all candidates in sequence.
  - `bandit`: warmup + adaptive ordering via a Bayesian linear surrogate.
- **Selection modes**:
  - direct objective minimization,
  - optional Pareto filtering before tie-break by objective.
- **Mitigation/Suppression knobs**: Pauli twirling, dynamical decoupling, measurement twirling, M3, ZNE, optional cutting.
- **Diagnostics**: baseline-vs-selected distribution distances (EMD/CVM/KS).
- **Reporting**: markdown + optional HTML from matrix JSON.
- **Caching**: compiled-circuit cache keyed by backend + circuit fingerprint + strategy payload.

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
)

# 4) inspect/persist artifacts
print(balanced.summary())
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

# Run per-circuit adjustment
python -m qbalance adjust ./circuits \
  --backend fake:generic:5 \
  --out ./balanced \
  --search bandit \
  --pareto \
  --max-candidates 24 \
  --seed 7 \
  --cache-root ./.qbalance-cache \
  --strategies ./strategies.json \
  --overwrite

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
         ├── baseline compile
         ├── candidate ordering (grid | bandit)
         ├── compile (+ optional suppression/cutting)
         ├── optional execution (+ mitigation)
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
```

Interpretation:

- higher `estimated_error` is penalized strongly (weight 10),
- two-qubit operation count has moderate penalty (weight 2),
- compile time contributes but with a small coefficient (0.1).

### 2) Finite-safe scoring behavior

For each objective term, qbalance ignores a term when the metric is missing, non-numeric, or non-finite. In selection fallback logic, if no finite objective term contributes, the candidate is treated as effectively worst-case (score $+\infty$).

### 3) Pareto pre-filtering

With `pareto=True`, qbalance first computes a non-dominated set on:

- `depth`,
- `two_qubit_ops`,
- `estimated_error`.

Candidate $a$ dominates candidate $b$ iff:

```math
\big(\forall i,\ a_i \le b_i\big)\ \land\ \big(\exists j,\ a_j < b_j\big)
```

Then qbalance chooses the minimum-objective strategy within that Pareto front.

### 4) Bandit proposal model

In `bandit` mode, candidate ordering after warmup uses a Thompson-sampling style linear surrogate:

```math
\mathbf{y} \approx X\mathbf{w},\qquad
\Lambda = \alpha I + \frac{1}{\sigma^2}X^T X,
\qquad
\mu = \Lambda^{-1}\frac{1}{\sigma^2}X^T\mathbf{y}
```

A sample $\tilde{\mathbf{w}}\sim\mathcal{N}(\mu,\Lambda^{-1})$ is drawn implicitly from the Cholesky factor of $\Lambda$, and candidates are ranked by linear score $\phi(s)^T\tilde{\mathbf{w}}$ (lower is preferred).

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

---

## Caching and computational performance

Compiled circuits may be reused via cache key:

```math
\text{key}=\text{SHA256}(\text{backend\_name}:\text{circuit\_fingerprint}:\text{strategy\_json})
```

This avoids redundant transpilation for repeated `(circuit, backend, strategy)` tuples.

Additional performance-relevant behavior:

- matrix execution pre-serializes strategy payloads once before inner loops,
- distribution metrics aggregate and align sorted supports efficiently,
- optional `profile=False` avoids transpiler pass profiling overhead.

---

## Artifacts and output contracts

### `adjust` output directory

- `dataset/` copied dataset index + artifacts,
- `results.json` selected strategy specs/metrics + baseline metrics + objective weights + per-circuit candidate evaluation history,
- `summary.txt` text summary.

### `matrix` output JSON

```json
{
  "version": 1,
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
- optional HTML files when `--html` is used and report dependencies are installed.

---

## CLI reference

### `dataset`

```bash
python -m qbalance dataset examples --out ./circuits --overwrite
```

### `adjust`

```bash
python -m qbalance adjust ./circuits \
  --backend fake:generic:5 \
  --out ./balanced \
  --search grid \
  --pareto \
  --max-candidates 24 \
  --strategies ./strategies.json \
  --execute \
  --shots 1024 \
  --profile \
  --overwrite
```

A strategy JSON file may be either one strategy object, a list of strategy objects, or an object with a `strategies` list:

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

- `qbalance.backends`
- `qbalance.objectives`
- `qbalance.reports`

Inspect active registrations with:

```bash
python -m qbalance plugins list
```

---

## Edge cases and failure semantics

- `adjust()` without `.set_target(...)` raises `ValueError`.
- Invalid search mode raises `ValueError`.
- If no candidate can be evaluated, selection raises `RuntimeError`.
- `matrix` validates `shots` as positive integer and validates integer seed.
- `matrix` raises `ValueError` if dataset record count and loaded circuit count mismatch.
- Existing output paths require explicit overwrite flags.
- Optional dependency features require installed extras.
- Execution/mitigation exceptions are captured in metrics (e.g., `exec_error`, `mthree_error`, `zne_error`) so runs can continue.

---

## Correctness and complexity notes

For a dataset with $N$ circuits and $S$ strategy candidates per circuit:

- **Grid search** evaluates up to $N\times S$ candidates.
- **Bandit search** also compiles/evaluates up to $N\times S$ candidates in the worst case, but may discover strong candidates earlier due to adaptive ordering.
- **Pareto filtering** in practice is close to linear in small fronts, with a quadratic worst-case bound on the number of unique objective vectors.

If a compile cache hit occurs, effective transpilation cost is reduced from repeated compile cost $T_{compile}$ to key lookup + deserialization overhead.


## Development

```bash
# Install editable package with all extras
pip install -e ".[all]"

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
