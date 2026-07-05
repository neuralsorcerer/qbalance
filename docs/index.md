# qbalance documentation

qbalance is a workflow toolkit for balancing quantum compilation, noise-suppression, and error-mitigation strategy choices over datasets of Qiskit circuits.

## Documentation map

- [Getting started](getting-started.md): installation, a minimal Python workflow, and the equivalent CLI flow.
- [Strategy configuration](strategies.md): `StrategySpec` fields, validation rules, default candidates, and JSON strategy files.
- [CLI guide](cli.md): command-by-command reference for `dataset`, `adjust`, `matrix`, `report`, `plugins`, and `compile`.
- [Artifacts and reports](artifacts.md): dataset layouts, balanced workload outputs, matrix JSON, and report generation.
- [API reference](api-references.md): public Python API, lower-level modules, extension points, and validation behavior.

```{toctree}
:maxdepth: 2
:caption: User guide

getting-started
strategies
cli
artifacts
api-references
```

## Citation

If you use qbalance in work, please cite the project metadata in [`CITATION.cff`](../CITATION.cff). The accompanying paper is available on arXiv: [QBalance: A Reproducible Multi-Objective Workflow for Quantum Compilation, Noise Suppression, and Error-Mitigation Strategy Selection](https://arxiv.org/abs/2605.02966) ([DOI: 10.48550/arXiv.2605.02966](https://doi.org/10.48550/arXiv.2605.02966)).

## Core flow

```text
CircuitDataset
  -> Workload.from_dataset(...) or Workload.from_path(...)
  -> set_target("fake:generic:5" | "aer:..." | custom plugin)
  -> adjust(...)
  -> BalancedWorkload.save(...) or BalancedWorkload.to_download(...)
  -> load_balanced_workload(...) for saved-directory reloads
```

For fixed cross-product experiments, use `qbalance.benchmarking.run_matrix(...)` or the `qbalance matrix` CLI command.
