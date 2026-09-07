# Strategy configuration

A strategy is represented by `qbalance.StrategySpec`. It is immutable and combines compilation, suppression, mitigation, cutting, and runtime knobs.

## Fields

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `optimization_level` | `int` | `1` | Qiskit preset optimization level, `0..3`. |
| `layout_method` | `str | None` | `None` | Qiskit layout method or `"qbalance_noise_aware"`. See the note below before using the noise-aware layout on sparse hardware. |
| `routing_method` | `str | None` | `None` | Qiskit routing method, commonly `"sabre"`. |
| `translation_method` | `str | None` | `None` | Qiskit translation method. |
| `seed_transpiler` | `int | None` | `0` | Transpiler seed. |
| `pauli_twirling` | `bool` | `False` | Applies best-effort Pauli twirling before compilation. |
| `num_twirls` | `int` | `1` | Must be at least `1`. |
| `dynamical_decoupling` | `bool` | `False` | Adds a DD pass manager after compilation when supported. |
| `dd_sequence` | `str` | `"XY4"` | DD sequence name such as `"XY4"`, `"XX"`, or `"YY"`. |
| `measurement_twirling` | `bool` | `False` | Applies measurement bit flips and records a flip map for untwirling counts. |
| `seed_suppression` | `int | None` | `0` | Seed for suppression transforms. |
| `mthree` | `bool` | `False` | Enables M3 mitigation during execution workflows. |
| `zne` | `bool` | `False` | Enables zero-noise extrapolation during execution workflows. |
| `zne_factors` | `tuple[float, ...]` | `(1.0, 2.0, 3.0)` | Must be finite, sorted, include `1.0`, and be `>= 1.0` when `zne=True`. |
| `zne_degree` | `int` | `1` | Must be non-negative generally and `1 <= degree < len(zne_factors)` when `zne=True`. |
| `cutting` | `bool` | `False` | Requests best-effort circuit cutting before compilation. See the note below: such candidates are currently always skipped. |
| `max_subcircuit_qubits` | `int | None` | `None` | Required when `cutting=True`; if provided, must be `>= 1`. |
| `resilience_level` | `int | None` | `None` | Optional IBM Runtime-style level; valid values are `0`, `1`, or `2`. Carried through to artifacts as metadata; the local compile/execute path does not apply it. |

Boolean values are rejected for integer-like fields even though Python treats `bool` as a subclass of `int`.

Unknown fields are rejected as well, so a misspelled key fails loudly instead of silently producing a default strategy.

> **`qbalance_noise_aware` ignores connectivity.** The layout ranks physical qubits by calibration quality alone (readout error, T1, T2) and assigns the busiest logical qubits to the best-scoring ones, without consulting the coupling map. On a well-connected device that is roughly neutral (depth 63 vs 61 on an all-to-all 20-qubit backend), but on sparse hardware the chosen qubits are typically not adjacent — on a 127-qubit heavy-hex snapshot it selected eight qubits with no adjacent pair at all — so routing has to bridge them with swaps: depth 594 vs 72 and 442 vs 66 two-qubit gates against the default layout. Candidate search correctly rejects such candidates on the objective, but prefer a Qiskit layout method when targeting sparse devices.

> **Cutting is incomplete.** `find_cuts` returns a circuit containing QPD placeholder gates (`qpd_1q`/`qpd_2q`). Executing them requires the full cutting workflow — partition the problem, generate subexperiments, run each, then reconstruct expectation values — which qbalance does not implement; it hands the cut circuit straight to compilation, and no backend basis can express a QPD gate. `qiskit-addon-cutting` also rejects circuits that carry classical bits, which every measured dataset circuit does. A `cutting=True` candidate is therefore skipped, with the reason logged, and takes no part in selection.

## Default candidate generation

`qbalance.search.default_candidate_strategies(max_candidates=24, seed=0)` builds a deterministic candidate pool that includes:

- optimization-level sweeps,
- SABRE routing/layout combinations,
- qbalance noise-aware layout candidates,
- Pauli twirling, dynamical decoupling, and measurement-twirling variants,
- optional M3/ZNE mitigation candidates,
- an optional cutting candidate.

The first candidate is stable and the remaining candidates are shuffled deterministically by `seed`.

## Explicit strategy sets

Use explicit strategies when you need a reproducible curated search space:

```python
from qbalance import StrategySpec, Workload, load_data

strategies = [
    StrategySpec(optimization_level=1, routing_method="sabre"),
    StrategySpec(
        optimization_level=2,
        layout_method="qbalance_noise_aware",
        routing_method="sabre",
    ),
]

balanced = (
    Workload.from_dataset(load_data("tiny"))
    .set_target("fake:generic:5")
    .adjust(strategies=strategies)
)
```

`Workload.adjust(strategies=...)` accepts any iterable of `StrategySpec` objects or mapping objects. Inputs are validated and duplicate strategies are removed while preserving first-seen order. When explicit strategies intentionally exclude the baseline, `allow_regression=False` can still select the baseline as a guarded fallback if the best feasible candidate is worse than the finite-safe baseline objective; equal scores and incomparable baselines do not trigger fallback.

## Strategy JSON formats

`qbalance.load_strategy_specs(path)` accepts these JSON shapes:

### Single strategy object

```json
{"optimization_level": 1, "routing_method": "sabre"}
```

### List of strategy objects

```json
[
  {"optimization_level": 1, "routing_method": "sabre"},
  {"optimization_level": 2, "measurement_twirling": true}
]
```

### Wrapped list

```json
{
  "strategies": [
    {"optimization_level": 1, "routing_method": "sabre"},
    {"optimization_level": 2, "measurement_twirling": true}
  ]
}
```

### Saved qbalance outputs

The loader can also extract strategies from saved balanced workload result files containing `selections` and from matrix JSON files containing `results` with `strategy` entries. This makes it possible to reuse selected or benchmarked strategies in later runs.
