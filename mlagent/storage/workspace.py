"""Pipeline run workspace management."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlagent.config import Settings, get_settings
from mlagent.pipeline.models import (
    PipelineRun,
    RunStatus,
    StageName,
    StageState,
    StageStatus,
)


class WorkspaceManager:
    """Manages per-run directories, manifests, and artifact paths."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.root = self.settings.workspace_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_dir = Path(__file__).resolve().parents[2] / "configs"

    def load_yaml_config(self, name: str) -> dict[str, Any]:
        path = self.config_dir / f"{name}.yaml"
        with path.open() as f:
            return yaml.safe_load(f)

    def list_pipelines(self) -> dict[str, Any]:
        return self.load_yaml_config("pipelines").get("pipelines", {})

    def list_datasets(self) -> dict[str, Any]:
        return self.load_yaml_config("datasets").get("datasets", {})

    def create_run(
        self,
        dataset: str,
        pipeline: str,
        parameters: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> PipelineRun:
        run_id = run_id or uuid.uuid4().hex[:12]
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        for sub in ("code", "data", "models", "reports", "logs", "artifacts"):
            (run_dir / sub).mkdir(exist_ok=True)

        stages = {
            s.value: StageState(name=s)
            for s in StageName
        }
        run = PipelineRun(
            run_id=run_id,
            dataset=dataset,
            pipeline=pipeline,
            parameters=parameters or {},
            stages={k: v for k, v in stages.items()},
            workspace_path=str(run_dir),
        )
        self.save_run(run)
        return run

    def run_path(self, run_id: str) -> Path:
        return self.root / run_id

    def manifest_path(self, run_id: str) -> Path:
        return self.run_path(run_id) / "manifest.json"

    def save_run(self, run: PipelineRun) -> None:
        run.updated_at = datetime.now(timezone.utc)
        path = self.manifest_path(run.run_id)
        path.write_text(run.model_dump_json(indent=2))

    def load_run(self, run_id: str) -> PipelineRun:
        path = self.manifest_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        return PipelineRun.model_validate_json(path.read_text())

    def list_runs(self) -> list[PipelineRun]:
        runs = []
        for d in sorted(self.root.iterdir(), reverse=True):
            manifest = d / "manifest.json"
            if manifest.exists():
                runs.append(PipelineRun.model_validate_json(manifest.read_text()))
        return runs

    def stage_dir(self, run_id: str, stage: StageName) -> Path:
        p = self.run_path(run_id) / "code" / stage.value
        p.mkdir(parents=True, exist_ok=True)
        return p

    def log_iteration(
        self,
        run_id: str,
        stage: StageName,
        iteration: int,
        stdout: str,
        stderr: str,
        metrics: dict[str, float],
    ) -> Path:
        log_dir = self.run_path(run_id) / "logs" / stage.value
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"iteration_{iteration:03d}.json"
        log_path.write_text(
            json.dumps(
                {
                    "iteration": iteration,
                    "stdout": stdout[-50000:],
                    "stderr": stderr[-50000:],
                    "metrics": metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )
        return log_path

    def export_production_code(self, run_id: str, dest: Path) -> Path:
        run = self.load_run(run_id)
        dest.mkdir(parents=True, exist_ok=True)
        export_dir = dest / f"pipeline_{run_id}"
        if export_dir.exists():
            shutil.rmtree(export_dir)
        shutil.copytree(self.run_path(run_id) / "code", export_dir / "code")
        shutil.copytree(self.run_path(run_id) / "data", export_dir / "data", ignore_errors=True)
        readme = export_dir / "README.md"
        readme.write_text(
            f"# Production ML Pipeline — Run {run_id}\n\n"
            f"- Dataset: {run.dataset}\n"
            f"- Pipeline: {run.pipeline}\n"
            f"- Metrics: {json.dumps(run.final_metrics, indent=2)}\n\n"
            "Execute stages in order: data_understanding → data_preparation → modeling → evaluation.\n"
        )
        return export_dir
