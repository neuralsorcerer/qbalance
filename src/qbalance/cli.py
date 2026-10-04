# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from qbalance.benchmarking.matrix import run_matrix
from qbalance.builtin_data import _make_tiny
from qbalance.dataset import _build_unique_artifact, load_dataset, save_dataset
from qbalance.errors import QBalanceError
from qbalance.objectives import load_objective
from qbalance.plugins import list_plugins
from qbalance.reports.html import render_html
from qbalance.reports.markdown import render_markdown
from qbalance.strategies import StrategySpec, load_strategy_specs
from qbalance.utils import replacing_directory, strict_json_value
from qbalance.workflow.workload import Workload


def _strategy_file_or_none(value: Optional[Path]) -> Optional[Path]:
    """Return a real strategy path, ignoring Typer defaults in direct calls."""
    return value if isinstance(value, (str, Path)) else None


def _int_option_or_none(value: Any) -> Optional[int]:
    """Return an integer option value, ignoring Typer defaults in direct calls."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# Help text is Rich markup (hence the escaped brackets in install hints).
# Typer's default markup mode has changed between releases, so pin it.
app = typer.Typer(
    add_completion=False,
    help="qbalance: balance quantum compilation, suppression, and mitigation "
    "strategy choices over a dataset of circuits.",
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)

# Errors that describe bad input or an unusable environment: a missing or
# existing path, an unknown backend, an invalid option or strategy, a
# candidate set with nothing feasible.  ``main`` reports them in one line; any
# other exception is a bug and keeps its traceback.
_USER_ERRORS = (QBalanceError, ValueError, OSError)


def _check_output_dir(
    out: Path, overwrite: bool, source: Optional[Path] = None
) -> None:
    """Reject an unusable output directory before any work is done.

    ``adjust`` would otherwise run the whole search and only then find, while
    saving, that ``out`` already exists or would delete its own dataset.

    Raises:
        BadParameter: If ``out`` exists without ``--overwrite``, is not a
            directory, or contains ``source``.
    """
    out = Path(out)
    # is_symlink() catches a dangling link, which exists() reports as absent.
    if not (out.exists() or out.is_symlink()):
        return
    if not overwrite:
        raise typer.BadParameter(f"{out} exists (use --overwrite)")
    if not out.is_dir():
        raise typer.BadParameter(f"{out} exists and is not a directory")
    if source is not None:
        source_root = Path(source).resolve()
        out_resolved = out.resolve()
        if source_root == out_resolved or out_resolved in source_root.parents:
            raise typer.BadParameter(
                f"{out} contains the source dataset; choose a different --out"
            )


def main() -> None:
    """Console entry point: run the CLI, reporting user errors without a traceback.

    Raises:
        SystemExit: With status 1 after reporting a user error, or with the
            status the command itself exits with.
    """
    try:
        app()
    except _USER_ERRORS as e:
        # soft_wrap keeps paths in the message intact instead of breaking
        # them; highlight=False leaves the message itself uncoloured.
        err_console.print(
            f"[red]Error:[/red] {escape(str(e))}", soft_wrap=True, highlight=False
        )
        raise SystemExit(1) from None


@app.command("dataset")
def dataset_cmd(
    kind: str = typer.Argument(
        ...,
        help="Dataset action; only 'examples' (the built-in bell, ghz3 and qft4 "
        "circuits) is supported",
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Output dataset directory"),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing output directory"
    ),
):
    """Write the built-in example circuits as a dataset."""
    if kind != "examples":
        raise typer.BadParameter("Only 'examples' is supported right now")
    _check_output_dir(out, overwrite)
    circuits = _make_tiny()
    ds = save_dataset(out, circuits, overwrite=overwrite)
    console.print(
        f"[green]Wrote dataset[/green] {escape(str(ds.root))} "
        f"with {len(ds)} circuits"
    )


@app.command("adjust")
def adjust_cmd(
    dataset_dir: Path = typer.Argument(..., help="Dataset directory"),
    backend: str = typer.Option(
        ..., "--backend", "-b", help="Backend spec, e.g. fake:generic:5"
    ),
    out: Path = typer.Option(
        ..., "--out", "-o", help="Output directory for the balanced workload"
    ),
    search: str = typer.Option("grid", "--search", help="Search mode: grid or bandit"),
    pareto: bool = typer.Option(
        False,
        "--pareto",
        help="Select from the Pareto front before the objective tie-break",
    ),
    max_candidates: int = typer.Option(
        24,
        "--max-candidates",
        help="Number of generated candidates when --strategies is not given",
    ),
    max_evaluations: Optional[int] = typer.Option(
        None,
        "--max-evaluations",
        help="Evaluate at most this many candidates per circuit (default: all)",
    ),
    warmup: Optional[int] = typer.Option(
        None,
        "--warmup",
        help="Randomly ordered candidates bandit search evaluates before its "
        "surrogate proposes (default: 6)",
    ),
    strategies_json: Optional[Path] = typer.Option(
        None,
        "--strategies",
        help="Strategy JSON file, the complete candidate set: one strategy, a "
        "list, a strategies object, or a saved results or matrix file",
    ),
    objective_json: Optional[Path] = typer.Option(
        None,
        "--objective",
        help="Objective-weight JSON file: a metric-to-weight mapping, "
        '{"weights": ...}, or a saved results file',
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Execute circuits (needs runnable backend or qbalance\\[aer])",
    ),
    shots: int = typer.Option(1024, "--shots", help="Execution shots per circuit"),
    profile: bool = typer.Option(
        False, "--profile", help="Record pass-level transpiler profiles"
    ),
    cache_root: Optional[Path] = typer.Option(
        None,
        "--cache-root",
        help="Compiled-circuit cache directory (defaults to the platform cache)",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Seed for the candidate shuffle, bandit search, and execution",
    ),
    no_regression: bool = typer.Option(
        False,
        "--no-regression",
        help="Never select a candidate that scores worse than the baseline: fall "
        "back to the best one that does not, else to the baseline strategy",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing output directory"
    ),
):
    """Select one strategy per circuit of a dataset for a target backend.

    Every circuit is compiled with a baseline strategy and with each
    candidate, scored with the objective, and the best feasible candidate is
    kept.  The copied dataset, results.json, and summary.txt are written to
    --out.
    """
    _check_output_dir(out, overwrite, source=dataset_dir)
    strategy_path = _strategy_file_or_none(strategies_json)
    objective_path = _strategy_file_or_none(objective_json)
    # Forward --warmup only when given, so adjust() keeps owning the default.
    warmup_value = _int_option_or_none(warmup)
    search_options: Dict[str, Any] = (
        {} if warmup_value is None else {"warmup": warmup_value}
    )
    bw = (
        Workload.from_path(dataset_dir)
        .set_target(backend)
        .adjust(
            objective=load_objective(objective_path) if objective_path else None,
            search=search,
            pareto=pareto,
            max_candidates=max_candidates,
            strategies=load_strategy_specs(strategy_path) if strategy_path else None,
            execute=execute,
            shots=shots,
            profile=profile,
            cache_root=cache_root,
            seed=seed,
            allow_regression=not no_regression,
            max_evaluations=_int_option_or_none(max_evaluations),
            **search_options,
        )
    )
    bw.save(out, overwrite=overwrite)
    console.print("[green]Done[/green]")
    console.print(bw.summary(), markup=False)


@app.command("matrix")
def matrix_cmd(
    dataset_dir: Path = typer.Argument(..., help="Dataset directory"),
    backend: List[str] = typer.Option(
        ..., "--backend", "-b", help="Backend spec; repeat for several backends"
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Output matrix JSON path"),
    strategies_json: Optional[Path] = typer.Option(
        None,
        "--strategies",
        help="Strategy JSON file (default: five built-in strategies)",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Execute circuits and record counts and mitigation results "
        "(needs runnable backend or qbalance\\[aer])",
    ),
    shots: int = typer.Option(1024, "--shots", help="Execution shots per circuit"),
    profile: bool = typer.Option(
        False, "--profile", help="Record pass-level transpiler profiles"
    ),
    seed: int = typer.Option(
        0, "--seed", help="Simulator seed; twirl instance i runs with seed + i"
    ),
):
    """Evaluate every circuit under a fixed strategy list on each backend.

    Writes one result row per backend, circuit, and strategy to the matrix
    JSON, which 'qbalance report' renders.
    """
    strategy_path = _strategy_file_or_none(strategies_json)
    strategies = (
        load_strategy_specs(strategy_path)
        if strategy_path
        else [
            StrategySpec(optimization_level=1, routing_method="sabre"),
            StrategySpec(
                optimization_level=2,
                routing_method="sabre",
                layout_method="qbalance_noise_aware",
            ),
            StrategySpec(
                optimization_level=2,
                routing_method="sabre",
                pauli_twirling=True,
                num_twirls=8,
            ),
            StrategySpec(
                optimization_level=2,
                routing_method="sabre",
                dynamical_decoupling=True,
                dd_sequence="XY4",
            ),
            # Several flip patterns: one fixed pattern does not average readout
            # error, it only relabels which outcomes suffer the larger error.
            StrategySpec(
                optimization_level=2,
                routing_method="sabre",
                measurement_twirling=True,
                num_twirls=8,
            ),
        ]
    )
    p = run_matrix(
        dataset_dir,
        backend_specs=backend,
        strategies=strategies,
        out_json=out,
        execute=execute,
        shots=shots,
        seed=seed,
        profile=profile,
    )
    console.print(f"[green]Wrote[/green] {escape(str(p))}")


@app.command("report")
def report_cmd(
    matrix_json: Path = typer.Argument(
        ..., help="Matrix JSON file written by 'qbalance matrix'"
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Report output directory"),
    html: bool = typer.Option(
        False, "--html", help="Also emit HTML (requires qbalance\\[report])"
    ),
):
    """Render a matrix JSON file as report.md (and report.html with --html)."""
    md = render_markdown(matrix_json, out)
    console.print(f"[green]Wrote[/green] {escape(str(md))}")
    if html:
        h = render_html(matrix_json, out)
        console.print(f"[green]Wrote[/green] {escape(str(h))}")


@app.command("plugins")
def plugins_cmd(
    sub: str = typer.Argument(..., help="Plugins action; only 'list' is supported"),
):
    """List the registered qbalance entry-point plugins."""
    if sub != "list":
        raise typer.BadParameter("Only 'list' supported")
    plugins = list_plugins()
    table = Table(title="qbalance plugins")
    table.add_column("Group")
    table.add_column("Entries")
    for g, entries in plugins.items():
        table.add_row(g, ", ".join(entries) if entries else "-")
    console.print(table)


@app.command("compile")
def compile_cmd(
    dataset_dir: Path = typer.Argument(..., help="Dataset directory"),
    backend: str = typer.Option(..., "--backend", "-b", help="Backend spec"),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
    optimization_level: int = typer.Option(
        1, "--optimization-level", help="Qiskit optimization level, 0-3"
    ),
    routing_method: Optional[str] = typer.Option(
        "sabre", "--routing-method", help="Routing method"
    ),
    layout_method: Optional[str] = typer.Option(
        None,
        "--layout-method",
        help="Layout method, e.g. sabre or qbalance_noise_aware (default: "
        "Qiskit's choice)",
    ),
    pauli_twirling: bool = typer.Option(
        False, "--pauli-twirling", help="Twirl the compiled two-qubit gates"
    ),
    num_twirls: int = typer.Option(1, "--num-twirls", help="Number of twirl instances"),
    dynamical_decoupling: bool = typer.Option(
        False, "--dd", help="Pad idle windows with dynamical decoupling"
    ),
    measurement_twirling: bool = typer.Option(
        False, "--meas-twirl", help="Twirl the terminal measurements"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace a previous output directory"
    ),
):
    """Compile every circuit of a dataset with one strategy given as flags.

    Writes each circuit's compiled instances (every twirl instance) under
    --out/compiled and their compile metrics to --out/meta.json.
    """
    spec = StrategySpec(
        optimization_level=optimization_level,
        routing_method=routing_method,
        layout_method=layout_method,
        pauli_twirling=pauli_twirling,
        num_twirls=num_twirls,
        dynamical_decoupling=dynamical_decoupling,
        measurement_twirling=measurement_twirling,
    )
    from qiskit.exceptions import QiskitError

    from qbalance.backends import resolve_backend
    from qbalance.transpile.pipeline import compile_ensemble

    # --overwrite replaces a previous output directory only: never a file,
    # and never a directory holding the dataset being compiled.
    _check_output_dir(out, overwrite, source=dataset_dir)
    ds = load_dataset(dataset_dir)
    circuits = ds.load_circuits()
    b = resolve_backend(backend)
    out = Path(out)
    meta: Dict[str, Any] = {
        "backend": backend,
        "strategy": spec.model_dump(),
        "circuits": {},
        # Record name -> compiled file, which the stem de-duplication below
        # makes impossible to infer from the names alone.
        "artifacts": {},
    }
    try:
        from qiskit import qpy
    except Exception:
        qpy = None
    # Artifact names are unique, but their stems need not be ("bell.qpy" and
    # "bell.qasm"), and case-insensitive filesystems merge "Bell" and "bell".
    used_outputs: set[str] = set()
    # Written beside --out and swapped in at the end, so a circuit that fails
    # to compile leaves a previous output intact rather than deleted.
    with replacing_directory(out) as staging:
        (staging / "compiled").mkdir()
        for qc, rec in zip(circuits, ds.records):
            # A twirled strategy only suppresses noise averaged over its random
            # instances, so every instance is written (in one QPY file, in the
            # order of the per-instance measurement_flip_maps).
            try:
                instances, m = compile_ensemble(qc, backend=b, spec=spec, profile=False)
            except QiskitError as e:
                # Qiskit rejecting the input -- a circuit wider than the device,
                # an unknown layout or routing method -- is a user error, as
                # adjust and matrix treat it; name the circuit it concerns.
                reason = getattr(e, "message", None) or str(e)
                raise QBalanceError(
                    f"Circuit {rec.name!r} could not be compiled for {backend}: "
                    f"{reason}"
                ) from e
            meta["circuits"][rec.name] = m
            if qpy is not None:
                output_name = _build_unique_artifact(
                    Path(rec.artifact).stem, used_outputs
                )
                used_outputs.add(output_name.casefold())
                with (staging / "compiled" / output_name).open("wb") as f:
                    qpy.dump(instances, f)
                meta["artifacts"][rec.name] = f"compiled/{output_name}"
        (staging / "meta.json").write_text(
            json.dumps(strict_json_value(meta), indent=2, allow_nan=False),
            encoding="utf-8",
        )
    console.print(f"[green]Wrote[/green] {escape(str(out))}")


if __name__ == "__main__":
    main()
