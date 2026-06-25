# qbalance documentation

qbalance is a workflow toolkit for balancing quantum compilation, noise-suppression, and error-mitigation strategy choices over datasets of Qiskit circuits.

## Documentation map

- [Getting started](getting-started.md): installation, a minimal Python workflow, and the equivalent CLI flow.
- [Strategy configuration](strategies.md): `StrategySpec` fields, validation rules, default candidates, and JSON strategy files.
- [CLI guide](cli.md): command-by-command reference for `dataset`, `adjust`, `matrix`, `report`, `plugins`, and `compile`.
- [Artifacts and reports](artifacts.md): dataset layouts, balanced workload outputs, matrix JSON, and report generation.
- [API reference](api-references.md): public Python API, lower-level modules, extension points, and validation behavior.

## Core flow

```text
CircuitDataset
  -> Workload.from_dataset(...) or Workload.from_path(...)
  -> set_target("fake:generic:5" | "aer:..." | custom plugin)
  -> adjust(...)
  -> BalancedWorkload.save(...) or BalancedWorkload.to_download(...)
```

For fixed cross-product experiments, use `qbalance.benchmarking.run_matrix(...)` or the `qbalance matrix` CLI command.
