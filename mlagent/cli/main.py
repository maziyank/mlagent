"""ML Agent CLI — primary user interaction layer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from mlagent.config import Settings, get_settings
from mlagent.pipeline.datasets import build_file_dataset_config
from mlagent.pipeline.models import RunStatus
from mlagent.pipeline.runner import PipelineRunner
from mlagent.reporting.html_report import write_html_report
from mlagent.reporting.markdown_report import write_markdown_report
from mlagent.storage.workspace import WorkspaceManager

app = typer.Typer(
    name="mlagent",
    help="Multi-agent tabular ML pipeline automation (LangChain Deep Agents)",
    no_args_is_help=True,
)
console = Console()


@app.command("configs")
def list_configs() -> None:
    """List supported pipeline configurations and datasets."""
    mgr = WorkspaceManager()
    pipelines = mgr.list_pipelines()
    datasets = mgr.list_datasets()

    console.print("\n[bold]Pipelines[/bold]")
    table = Table("Name", "Description", "Target Metric", "Min Metric")
    for name, cfg in pipelines.items():
        table.add_row(
            name,
            cfg.get("description", ""),
            cfg.get("target_metric", ""),
            str(cfg.get("min_metric", "")),
        )
    console.print(table)

    console.print("\n[bold]Datasets[/bold]")
    dtable = Table("Name", "Task", "Default Pipeline", "Description")
    for name, cfg in datasets.items():
        dtable.add_row(
            name,
            cfg.get("task_type", ""),
            cfg.get("default_pipeline", ""),
            (cfg.get("description", "")[:60] + "...")
            if len(cfg.get("description", "")) > 60
            else cfg.get("description", ""),
        )
    console.print(dtable)


@app.command("run")
def run_pipeline(
    dataset: str = typer.Argument(
        ...,
        help="Dataset name from configs/datasets.yaml, or any label when using --csv",
    ),
    pipeline: Optional[str] = typer.Option(None, "--pipeline", "-p", help="Pipeline config name"),
    csv: Optional[Path] = typer.Option(
        None,
        "--csv",
        "-c",
        help="Path to your own CSV file (overrides built-in dataset loading)",
        exists=True,
        readable=True,
    ),
    target: Optional[str] = typer.Option(
        None,
        "--target",
        "-t",
        help="Name of the label column in your CSV (required with --csv)",
    ),
    task_type: Optional[str] = typer.Option(
        None,
        "--task-type",
        help="classification or regression (default: classification)",
    ),
    mode: Optional[str] = typer.Option(
        None, "--mode", "-m", help="Execution mode: agent | template"
    ),
    min_metric: Optional[float] = typer.Option(None, "--min-metric", help="Override minimum metric"),
    model: Optional[str] = typer.Option(None, "--model", help="LLM model (provider:model)"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Show live progress"),
) -> None:
    """Initiate end-to-end pipeline execution."""
    settings = get_settings()
    if mode:
        settings.execution_mode = mode  # type: ignore[assignment]
    if model:
        settings.mlagent_model = model
    params = {}
    if min_metric is not None:
        params["min_metric"] = min_metric

    dataset_config = None
    if csv is not None:
        if not target:
            console.print("[red]--target is required when using --csv[/red]")
            raise typer.Exit(1)
        tt = task_type or "classification"
        if tt not in ("classification", "regression"):
            console.print("[red]--task-type must be classification or regression[/red]")
            raise typer.Exit(1)
        dataset_config = build_file_dataset_config(
            csv,
            target_column=target,
            task_type=tt,
            name=dataset,
        )
        if pipeline is None:
            pipeline = dataset_config["default_pipeline"]
        console.print(
            f"[dim]Using custom CSV:[/dim] {csv}\n"
            f"[dim]Target column:[/dim] {target}  [dim]Task:[/dim] {tt}"
        )

    runner = PipelineRunner(settings)
    progress_state: dict = {"stage": "", "event": ""}

    def on_progress(run_id: str, stage: str, event: str) -> None:
        progress_state["run_id"] = run_id
        progress_state["stage"] = stage
        progress_state["event"] = event

    if watch:
        with Live(console=console, refresh_per_second=4) as live:
            def _run():
                return runner.run(
                    dataset, pipeline, params,
                    on_progress=on_progress,
                    dataset_config=dataset_config,
                )

            import threading

            result_holder: list = []

            def target():
                result_holder.append(_run())

            t = threading.Thread(target=target)
            t.start()
            while t.is_alive():
                live.update(
                    f"[cyan]Run {progress_state.get('run_id', '...')}[/cyan] "
                    f"— {progress_state.get('stage', 'starting')}: "
                    f"{progress_state.get('event', '')}"
                )
                time.sleep(0.25)
            t.join()
            run = result_holder[0]
    else:
        run = runner.run(
            dataset, pipeline, params,
            on_progress=on_progress,
            dataset_config=dataset_config,
        )

    mgr = WorkspaceManager(settings)
    ws = Path(run.workspace_path)
    write_markdown_report(run, ws)
    write_html_report(run, ws)

    if run.status == RunStatus.COMPLETED:
        console.print(f"\n[green]Success[/green] — metrics: {run.final_metrics}")
        console.print(f"Reports: {ws / 'reports'}")
    else:
        raise typer.Exit(code=1)


@app.command("status")
def status(
    run_id: Optional[str] = typer.Argument(None, help="Run ID (latest if omitted)"),
) -> None:
    """Monitor agent / pipeline progress for a run."""
    mgr = WorkspaceManager()
    if run_id:
        run = mgr.load_run(run_id)
    else:
        runs = mgr.list_runs()
        if not runs:
            console.print("[yellow]No runs found.[/yellow]")
            raise typer.Exit(1)
        run = runs[0]

    table = Table("Field", "Value")
    table.add_row("Run ID", run.run_id)
    table.add_row("Status", run.status.value)
    table.add_row("Dataset", run.dataset)
    table.add_row("Pipeline", run.pipeline)
    table.add_row("Metrics", json.dumps(run.final_metrics))
    console.print(table)

    console.print("\n[bold]Stages[/bold]")
    st = Table("Stage", "Status", "Iterations", "Best Metric", "Error")
    for name, stage in run.stages.items():
        best = ""
        if stage.optimization and stage.optimization.best_value is not None:
            best = (
                f"{stage.optimization.target_metric}="
                f"{stage.optimization.best_value:.4f}"
            )
        st.add_row(
            name,
            stage.status.value,
            str(len(stage.iterations)),
            best,
            stage.error or "",
        )
    console.print(st)


@app.command("artifacts")
def artifacts(
    run_id: str = typer.Argument(..., help="Run ID"),
    stage: Optional[str] = typer.Option(None, "--stage", "-s"),
) -> None:
    """Inspect intermediate code and data artifacts."""
    mgr = WorkspaceManager()
    run = mgr.load_run(run_id)
    ws = Path(run.workspace_path)

    if stage:
        code_dir = ws / "code" / stage
        console.print(f"[bold]Code — {stage}[/bold]")
        for f in sorted(code_dir.glob("*.py")):
            console.print(f"  {f.name} ({f.stat().st_size} bytes)")
        data_dir = ws / "data"
        console.print(f"[bold]Data[/bold]")
        for f in sorted(data_dir.glob("*")):
            console.print(f"  {f.name}")
    else:
        for sub in ("code", "data", "models", "logs", "artifacts"):
            d = ws / sub
            if not d.exists():
                continue
            console.print(f"\n[bold]{sub}/[/bold]")
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    console.print(f"  {f.relative_to(ws)}")


@app.command("history")
def history(
    run_id: str = typer.Argument(..., help="Run ID"),
    stage: Optional[str] = typer.Option(None, "--stage", "-s"),
) -> None:
    """View sandbox iteration history and execution logs."""
    mgr = WorkspaceManager()
    run = mgr.load_run(run_id)
    ws = Path(run.workspace_path)

    stages = [stage] if stage else [s.value for s in run.stages.keys()]
    for st_name in stages:
        state = run.stages.get(st_name)
        if state:
            console.print(f"\n[bold cyan]{st_name}[/bold cyan] — {len(state.iterations)} iterations")
            for it in state.iterations:
                console.print(
                    f"  #{it.iteration} exit={it.result.exit_code} "
                    f"metrics={it.result.metrics} duration={it.result.duration_seconds:.2f}s"
                )
        log_dir = ws / "logs" / st_name
        if log_dir.exists():
            for log in sorted(log_dir.glob("*.json")):
                data = json.loads(log.read_text())
                if not state:
                    console.print(f"  [dim]{log.name}[/dim] success={data.get('success')}")


@app.command("report")
def report(
    run_id: str = typer.Argument(..., help="Run ID"),
    format: str = typer.Option("both", "--format", "-f", help="md | html | both"),
) -> None:
    """Generate comprehensive HTML/Markdown pipeline reports."""
    mgr = WorkspaceManager()
    run = mgr.load_run(run_id)
    ws = Path(run.workspace_path)
    paths = []
    if format in ("md", "both"):
        paths.append(write_markdown_report(run, ws))
    if format in ("html", "both"):
        paths.append(write_html_report(run, ws))
    for p in paths:
        console.print(f"[green]Report written:[/green] {p}")


@app.command("export")
def export_code(
    run_id: str = typer.Argument(..., help="Run ID"),
    output: Path = typer.Option(Path("./export"), "--output", "-o"),
) -> None:
    """Export final production-ready pipeline code."""
    mgr = WorkspaceManager()
    dest = mgr.export_production_code(run_id, output)
    console.print(f"[green]Exported to[/green] {dest}")


@app.command("rerun")
def rerun(
    run_id: str = typer.Argument(..., help="Previous run ID"),
    min_metric: Optional[float] = typer.Option(None, "--min-metric"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m"),
) -> None:
    """Re-run pipeline with modified parameters."""
    settings = get_settings()
    if mode:
        settings.execution_mode = mode  # type: ignore[assignment]
    params = {}
    if min_metric is not None:
        params["min_metric"] = min_metric
    runner = PipelineRunner(settings)
    old = WorkspaceManager(settings).load_run(run_id)
    new_run = runner.run(old.dataset, old.pipeline, {**old.parameters, **params})
    console.print(f"[green]New run:[/green] {new_run.run_id}")


@app.command("runs")
def list_runs() -> None:
    """List all pipeline runs."""
    mgr = WorkspaceManager()
    runs = mgr.list_runs()
    if not runs:
        console.print("[yellow]No runs.[/yellow]")
        return
    table = Table("Run ID", "Dataset", "Status", "Created")
    for r in runs[:20]:
        table.add_row(r.run_id, r.dataset, r.status.value, r.created_at.strftime("%Y-%m-%d %H:%M"))
    console.print(table)


if __name__ == "__main__":
    app()
