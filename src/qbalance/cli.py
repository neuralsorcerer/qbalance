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
from rich.table import Table

from qbalance.benchmarking.matrix import run_matrix
from qbalance.builtin_data import _make_tiny
from qbalance.dataset import load_dataset, save_dataset
from qbalance.plugins import list_plugins
from qbalance.reports.html import render_html
from qbalance.reports.markdown import render_markdown
from qbalance.strategies import StrategySpec
from qbalance.workflow.workload import Workload

app = typer.Typer(
    add_completion=False, help="qbalance: balance-style quantum workflow toolkit"
)
console = Console()


@app.command("dataset")
def dataset_cmd(
    kind: str = typer.Argument(..., help="Dataset action: 'examples'"),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite output"),
):
    """Dataset cmd used by the qbalance workflow.

    Args:
        kind (default: typer.Argument(..., help="Dataset action: 'examples'")): Kind value consumed by this routine.
        out (default: typer.Option(..., '--out', '-o', help='Output directory')): Destination path for generated output files.
        overwrite (default: typer.Option(False, '--overwrite', help='Overwrite output')): Whether existing files/directories may be replaced.

    Returns:
        Computed value produced by this routine.

    Raises:
        BadParameter: Raised when input validation fails or a dependent operation cannot be completed.
    """
    if kind != "examples":
        raise typer.BadParameter("Only 'examples' is supported right now")
    circuits = _make_tiny()
    ds = save_dataset(out, circuits, overwrite=overwrite)
    console.print(f"[green]Wrote dataset[/green] {ds.root} with {len(ds)} circuits")


@app.command("adjust")
def adjust_cmd(
    dataset_dir: Path = typer.Argument(..., help="Dataset directory"),
    backend: str = typer.Option(
        ..., "--backend", "-b", help="Backend spec, e.g. fake:generic:5"
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
    search: str = typer.Option("grid", "--search", help="grid|bandit"),
    pareto: bool = typer.Option(False, "--pareto", help="Use Pareto-front selection"),
    max_candidates: int = typer.Option(24, "--max-candidates"),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Execute circuits (needs runnable backend or qbalance[aer])",
    ),
    shots: int = typer.Option(1024, "--shots"),
    profile: bool = typer.Option(False, "--profile", help="Enable per-pass profiling"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Adjust cmd used by the qbalance workflow.

    Args:
        dataset_dir (default: typer.Argument(..., help='Dataset directory')): Directory containing the dataset index and circuit artifacts.
        backend (default: typer.Option(..., '--backend', '-b', help='Backend spec, e.g. fake:generic:5')): Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        out (default: typer.Option(..., '--out', '-o', help='Output directory')): Destination path for generated output files.
        search (default: typer.Option('grid', '--search', help='grid|bandit')): Search value consumed by this routine.
        pareto (default: typer.Option(False, '--pareto', help='Use Pareto-front selection')): Pareto value consumed by this routine.
        max_candidates (default: typer.Option(24, '--max-candidates')): Max candidates value consumed by this routine.
        execute (default: typer.Option(False, '--execute', help='Execute circuits (needs runnable backend or qbalance[aer])')): Whether to run compiled circuits and collect counts.
        shots (default: typer.Option(1024, '--shots')): Number of shots used when executing circuits on a backend.
        profile (default: typer.Option(False, '--profile', help='Enable per-pass profiling')): Whether pass-level transpiler profiling is enabled.
        overwrite (default: typer.Option(False, '--overwrite')): Whether existing files/directories may be replaced.

    Returns:
        Computed value produced by this routine.

    Raises:
        None.
    """
    bw = (
        Workload.from_path(dataset_dir)
        .set_target(backend)
        .adjust(
            search=search,
            pareto=pareto,
            max_candidates=max_candidates,
            execute=execute,
            shots=shots,
            profile=profile,
        )
    )
    bw.save(out, overwrite=overwrite)
    console.print("[green]Done[/green]")
    console.print(bw.summary())


@app.command("matrix")
def matrix_cmd(
    dataset_dir: Path = typer.Argument(...),
    backend: List[str] = typer.Option(
        ..., "--backend", "-b", help="Repeatable backend spec"
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Output JSON"),
    execute: bool = typer.Option(False, "--execute"),
    shots: int = typer.Option(1024, "--shots"),
    profile: bool = typer.Option(False, "--profile"),
):
    # Default strategies

    """Matrix cmd used by the qbalance workflow.

    Args:
        dataset_dir (default: typer.Argument(...)): Directory containing the dataset index and circuit artifacts.
        backend (default: typer.Option(..., '--backend', '-b', help='Repeatable backend spec')): Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        out (default: typer.Option(..., '--out', '-o', help='Output JSON')): Destination path for generated output files.
        execute (default: typer.Option(False, '--execute')): Whether to run compiled circuits and collect counts.
        shots (default: typer.Option(1024, '--shots')): Number of shots used when executing circuits on a backend.
        profile (default: typer.Option(False, '--profile')): Whether pass-level transpiler profiling is enabled.

    Returns:
        Computed value produced by this routine.

    Raises:
        None.
    """
    strategies = [
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
        StrategySpec(
            optimization_level=2, routing_method="sabre", measurement_twirling=True
        ),
    ]
    p = run_matrix(
        dataset_dir,
        backend_specs=backend,
        strategies=strategies,
        out_json=out,
        execute=execute,
        shots=shots,
        profile=profile,
    )
    console.print(f"[green]Wrote[/green] {p}")


@app.command("report")
def report_cmd(
    matrix_json: Path = typer.Argument(...),
    out: Path = typer.Option(..., "--out", "-o"),
    html: bool = typer.Option(
        False, "--html", help="Also emit HTML (requires qbalance[report])"
    ),
):
    """Report cmd used by the qbalance workflow.

    Args:
        matrix_json (default: typer.Argument(...)): Matrix json value consumed by this routine.
        out (default: typer.Option(..., '--out', '-o')): Destination path for generated output files.
        html (default: typer.Option(False, '--html', help='Also emit HTML (requires qbalance[report])')): Html value consumed by this routine.

    Returns:
        Computed value produced by this routine.

    Raises:
        None.
    """
    md = render_markdown(matrix_json, out)
    console.print(f"[green]Wrote[/green] {md}")
    if html:
        h = render_html(matrix_json, out)
        console.print(f"[green]Wrote[/green] {h}")


@app.command("plugins")
def plugins_cmd(
    sub: str = typer.Argument(..., help="list"),
):
    """Plugins cmd used by the qbalance workflow.

    Args:
        sub (default: typer.Argument(..., help='list')): Sub value consumed by this routine.

    Returns:
        Computed value produced by this routine.

    Raises:
        BadParameter: Raised when input validation fails or a dependent operation cannot be completed.
    """
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
    dataset_dir: Path = typer.Argument(...),
    backend: str = typer.Option(..., "--backend", "-b"),
    out: Path = typer.Option(..., "--out", "-o"),
    optimization_level: int = typer.Option(1, "--optimization-level"),
    routing_method: Optional[str] = typer.Option("sabre", "--routing-method"),
    layout_method: Optional[str] = typer.Option(None, "--layout-method"),
    pauli_twirling: bool = typer.Option(False, "--pauli-twirling"),
    num_twirls: int = typer.Option(1, "--num-twirls"),
    dynamical_decoupling: bool = typer.Option(False, "--dd"),
    measurement_twirling: bool = typer.Option(False, "--meas-twirl"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Compile cmd used by the qbalance workflow.

    Args:
        dataset_dir (default: typer.Argument(...)): Directory containing the dataset index and circuit artifacts.
        backend (default: typer.Option(..., '--backend', '-b')): Backend object (or backend-like handle) used for compilation, property lookup, or execution.
        out (default: typer.Option(..., '--out', '-o')): Destination path for generated output files.
        optimization_level (default: typer.Option(1, '--optimization-level')): Optimization level value consumed by this routine.
        routing_method (default: typer.Option('sabre', '--routing-method')): Routing method value consumed by this routine.
        layout_method (default: typer.Option(None, '--layout-method')): Layout method value consumed by this routine.
        pauli_twirling (default: typer.Option(False, '--pauli-twirling')): Pauli twirling value consumed by this routine.
        num_twirls (default: typer.Option(1, '--num-twirls')): Num twirls value consumed by this routine.
        dynamical_decoupling (default: typer.Option(False, '--dd')): Dynamical decoupling value consumed by this routine.
        measurement_twirling (default: typer.Option(False, '--meas-twirl')): Measurement twirling value consumed by this routine.
        overwrite (default: typer.Option(False, '--overwrite')): Whether existing files/directories may be replaced.

    Returns:
        Computed value produced by this routine.

    Raises:
        BadParameter: Raised when input validation fails or a dependent operation cannot be completed.
    """
    Workload.from_path(dataset_dir).set_target(backend)
    spec = StrategySpec(
        optimization_level=optimization_level,
        routing_method=routing_method,
        layout_method=layout_method,
        pauli_twirling=pauli_twirling,
        num_twirls=num_twirls,
        dynamical_decoupling=dynamical_decoupling,
        measurement_twirling=measurement_twirling,
    )
    from qbalance.backends import resolve_backend
    from qbalance.transpile.pipeline import compile_one

    ds = load_dataset(dataset_dir)
    circuits = ds.load_circuits()
    b = resolve_backend(backend)
    out = Path(out)
    if out.exists():
        if not overwrite:
            raise typer.BadParameter(f"{out} exists (use --overwrite)")
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compiled").mkdir(exist_ok=True)
    meta: Dict[str, Any] = {
        "backend": backend,
        "strategy": spec.model_dump(),
        "circuits": {},
    }
    try:
        from qiskit import qpy
    except Exception:
        qpy = None
    for qc, rec in zip(circuits, ds.records):
        c, m = compile_one(qc, backend=b, spec=spec, profile=False)
        meta["circuits"][rec.name] = m
        if qpy is not None:
            with (out / "compiled" / f"{rec.name}.qpy").open("wb") as f:
                qpy.dump(c, f)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {out}")


if __name__ == "__main__":
    app()
