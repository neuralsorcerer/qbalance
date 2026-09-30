# Strategy configuration

A strategy is represented by `qbalance.StrategySpec`. It is immutable and combines compilation, suppression, mitigation, cutting, and runtime knobs.

## Fields

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `optimization_level` | `int` | `1` | Qiskit preset optimization level, `0..3`. |
| `layout_method` | `str` | `None` | `None` | Qiskit layout method or `"qbalance_noise_aware"`. See the note below for how it places qubits. |
| `routing_method` | `str` | `None` | `None` | Qiskit routing method, commonly `"sabre"`. |
| `translation_method` | `str` | `None` | `None` | Qiskit translation method. |
| `seed_transpiler` | `int` | `None` | `0` | Transpiler seed; must be non-negative. |
| `pauli_twirling` | `bool` | `False` | Twirls the two-qubit gates of the *compiled* circuit into `num_twirls` random, target-native instances (twirling before transpilation lets optimization re-synthesize the twirl away). Every instance is executed, on an equal share of the shots, because a twirl only suppresses noise averaged over its instances. |
| `num_twirls` | `int` | `1` | Number of randomized instances for Pauli and/or measurement twirling; must be at least `1`. |
| `dynamical_decoupling` | `bool` | `False` | Adds a DD pass manager after compilation when supported. |
| `dd_sequence` | `str` | `"XY4"` | One of `"XY4"`, `"XX"`, or `"YY"` (case-insensitive, stored upper-case); any other name is rejected. When the target cannot run `y`, an `XY4` request is padded with `XX` instead. |
| `measurement_twirling` | `bool` | `False` | Flips each terminal measurement at random and records the flips, per instance (`measurement_flip_maps`), so every instance's counts are untwirled with its own pattern. Readout error is symmetrized only on average over the instances: with `num_twirls=1` the single fixed pattern just relabels which outcomes suffer the larger error, so use several instances (the default candidates use 8). |
| `seed_suppression` | `int` | `None` | `0` | Seed for suppression transforms; must be non-negative. Instance `i` of a measurement-twirled ensemble uses `seed_suppression + i`. |
| `mthree` | `bool` | `False` | Enables M3 mitigation during execution workflows. |
| `zne` | `bool` | `False` | Enables zero-noise extrapolation during execution workflows. |
| `zne_factors` | `tuple[float, ...]` | `(1.0, 3.0, 5.0)` | Must be finite, sorted, include `1.0`, and be `>= 1.0` when `zne=True`. Global folding realizes odd integer factors, so each factor is rounded up to the next odd integer (`2.0` runs at `3`); extrapolation uses these realized factors, which are recorded as `zne_realized_factors`. The default asks for three distinct noise levels (the former default `(1, 2, 3)` ran at `(1, 3, 3)`). |
| `zne_degree` | `int` | `1` | Must be non-negative generally and, when `zne=True`, at least `1` and smaller than the number of *distinct realized* fold factors. |
| `cutting` | `bool` | `False` | Cuts circuits wider than `max_subcircuit_qubits` into subexperiments with `qiskit-addon-cutting`, compiles each, and (when executed) reconstructs the measured Z-parity. Cannot be combined with `pauli_twirling`, `measurement_twirling`, `mthree`, or `zne`. See the note below. |
| `max_subcircuit_qubits` | `int` | `None` | `None` | Required when `cutting=True`; if provided, must be `>= 1`. |
| `resilience_level` | `int` | `None` | `None` | Optional IBM Runtime-style level; valid values are `0`, `1`, or `2`. Carried through to artifacts as metadata; the local compile/execute path does not apply it. |

Boolean values are rejected for integer-like fields even though Python treats `bool` as a subclass of `int`.

Unknown fields are rejected as well, so a misspelled key fails loudly instead of silently producing a default strategy.

> **How `qbalance_noise_aware` places qubits.** Physical qubits are scored by readout error, with T1/T2 relative to the backend median as a tie-breaker. On a backend with a coupling map, couplers calibrated as broken (error 1.0) are ignored and logical qubits are placed one at a time, the busiest first: each goes next to the partners it interacts with whenever a free neighbour exists, and among the qubits that need equally few routing hops, to the cheapest in error units (its readout error, the calibrated two-qubit gate error of the coupled pairs it uses, and, for a pair that must be routed, a SWAP's three gate errors per extra hop plus the readout error of the qubit routing moves it onto). The placement is grown from the 24 most promising start qubits and the cheapest wins; the [API reference](api-references.md#noise_aware_initial_layout) gives the exact cost. It therefore keeps interacting qubits adjacent and routes around broken couplers: on the `fake:ibm:sherbrooke` snapshot an 8-qubit chain compiles at optimization level 2 to depth 29, 7 two-qubit gates, and estimated error 0.125, where the SABRE layout lands on error-1.0 couplers (estimated error 1.0). Without a coupling map the best-scoring qubits go to the busiest logical qubits.

> **How cutting is evaluated.** Terminal measurements are split off (they define the observable: `Z` on every measured qubit), the unitary part is cut with `find_cuts`, partitioned, and expanded into every quasi-probability term exactly (`num_samples=inf`, so results are reproducible). Each subexperiment is compiled on its own; the candidate's `depth`, `size`, `width`, `two_qubit_ops`, and `estimated_error` are the maxima over them (every subexperiment runs as a separate circuit), `compile_time_s` is the total, and `cut_count`, `cut_types`, `sampling_overhead`, `num_subcircuits`, `subcircuit_widths`, and `num_subexperiments` describe the cut. With execution, every subexperiment runs on `shots` shots through `BackendSamplerV2` and the reconstructed value is stored as `cut_parity_expval` (compare with `raw_parity_expval` of uncut candidates). A circuit no wider than `max_subcircuit_qubits` runs uncut (`cut_count: 0`, no subexperiment metrics). Qubits no gate acts on are dropped before cutting (a measured idle qubit reads `|0>`, contributing exactly `+1` to the parity), and a wider circuit whose connected parts each fit is split into those parts without any cut (`cut_count: 0`, `sampling_overhead: 1.0`), so no subcircuit ever exceeds the limit. A candidate is recorded as infeasible, with the reason logged and kept in its metrics (`cutting_error`, or `compile_error` when a subexperiment cannot be compiled for the backend), when the circuit has non-terminal measurements or classically controlled operations, when a subexperiment cannot be compiled, or when the cut would need more than 1024 subexperiments. The default objective charges `sampling_overhead` (9 per cut CNOT, multiplied over the cuts; 1 for an uncut circuit) so shallower subexperiments are weighed against their shot cost.

## Default candidate generation

`qbalance.search.default_candidate_strategies(max_candidates=24, seed=0)` builds a deterministic pool of 23 distinct candidates:

- optimization levels 0-3, each with default routing, SABRE routing, SABRE layout and routing, and the qbalance noise-aware layout with SABRE routing (16 candidates);
- at optimization level 2 with SABRE routing: Pauli twirling (8 instances), dynamical decoupling (`XY4`), measurement twirling (8 instances), and all three combined;
- M3 and ZNE, each with measurement twirling over 8 instances, at optimization level 2 with SABRE routing;
- a cutting candidate (optimization level 1, noise-aware layout, SABRE routing, 4-qubit subcircuits: at optimization level 1 Qiskit would otherwise keep each small subexperiment on qubits 0, 1, ... whatever their error rates).

`StrategySpec(optimization_level=0)` always comes first; the other 22 are shuffled deterministically by `seed`, and the first `max_candidates` of the result are returned, so a smaller `max_candidates` keeps a seed-dependent subset.

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

`Workload.adjust(strategies=...)` accepts any iterable of `StrategySpec` objects or mapping objects. Inputs are validated and duplicate strategies are removed while preserving first-seen order. When explicit strategies intentionally exclude the baseline, `allow_regression=False` can still select the baseline as a guarded fallback when every candidate in the selection pool is worse than the finite-safe baseline objective on the objective terms both report (a regressing top candidate is first replaced by the best one that does not regress); equal scores and incomparable baselines do not trigger fallback.

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
  {"optimization_level": 2, "measurement_twirling": true, "num_twirls": 8}
]
```

### Wrapped list

```json
{
  "strategies": [
    {"optimization_level": 1, "routing_method": "sabre"},
    {"optimization_level": 2, "measurement_twirling": true, "num_twirls": 8}
  ]
}
```

### Saved qbalance outputs

The loader can also extract strategies from saved balanced workload result files containing `selections` and from matrix JSON files containing `results` with `strategy` entries. This makes it possible to reuse selected or benchmarked strategies in later runs.
