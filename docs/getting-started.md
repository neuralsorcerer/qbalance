# Getting started

## Install

Base install:

```bash
pip install qbalance
```

Development install with all optional integrations and test tools:

```bash
pip install -e ".[all]"
```

Optional extras:

```bash
pip install "qbalance[aer]"         # qiskit-aer execution backends
pip install "qbalance[runtime]"     # IBM Runtime helper integrations
pip install "qbalance[mitigation]"  # mthree mitigation
pip install "qbalance[cutting]"     # qiskit-addon-cutting
pip install "qbalance[report]"      # HTML/report stack
```

## Python quickstart

```python
from qbalance import Workload, load_balanced_workload, load_data

# Built-in tiny dataset: bell, ghz3, qft4.
ds = load_data("tiny")

balanced = (
    Workload.from_dataset(ds)
    .set_target("fake:generic:5")
    .adjust(
        search="bandit",
        pareto=True,
        max_candidates=24,
        execute=False,
        seed=7,
        cache_root="./.qbalance-cache",
        profile=False,
    )
)

print(balanced.summary())
print(balanced.candidate_rankings()["bell"][0])
balanced.save("./balanced", overwrite=True)
balanced.to_download("./balanced_bundle.zip", overwrite=True)

# Later, reload the saved directory without recompiling.
reloaded = load_balanced_workload("./balanced")
print(reloaded.summary())
```

## CLI quickstart

```bash
python -m qbalance dataset examples --out ./circuits --overwrite

python -m qbalance adjust ./circuits \
  --backend fake:generic:5 \
  --out ./balanced \
  --search bandit \
  --pareto \
  --max-candidates 24 \
  --seed 7 \
  --cache-root ./.qbalance-cache \
  --overwrite

python -m qbalance matrix ./circuits \
  --backend fake:generic:5 \
  --backend fake:generic:10 \
  --out ./matrix.json \
  --execute \
  --shots 1024 \
  --seed 7

python -m qbalance report ./matrix.json --out ./report --html
```

## Use a custom strategy file

Create `strategies.json`:

```json
{
  "strategies": [
    {"optimization_level": 1, "routing_method": "sabre"},
    {
      "optimization_level": 2,
      "layout_method": "qbalance_noise_aware",
      "routing_method": "sabre",
      "measurement_twirling": true,
      "num_twirls": 8
    }
  ]
}
```

Then pass it to either strategy-selection or matrix workflows:

```bash
python -m qbalance adjust ./circuits \
  --backend fake:generic:5 \
  --out ./balanced-custom \
  --strategies ./strategies.json \
  --overwrite

python -m qbalance matrix ./circuits \
  --backend fake:generic:5 \
  --out ./matrix-custom.json \
  --strategies ./strategies.json
```

When explicit strategies are supplied to `adjust`, `max_candidates` is ignored because the JSON file is the candidate set. Use `seed`/`--seed` to make randomized candidate ordering and execution helpers reproducible (the default objective also weights wall-clock `compile_time_s`, which decides between candidates that compile identically; drop that term, as in the [README](https://github.com/neuralsorcerer/qbalance#2b-reproducibility-of-objective_score), when selections must repeat exactly across cold-cache runs), and `cache_root`/`--cache-root` to choose where compiled-circuit cache artifacts are stored. Add `--no-regression` (or `allow_regression=False` in Python) when production runs must never select a candidate with a worse objective score than the baseline: a regressing top candidate is replaced by the best one that does not regress, or by the baseline strategy when every candidate regresses; equal objective scores and baselines with no finite objective score are left unchanged.
