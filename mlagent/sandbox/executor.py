"""Isolated Python sandbox executor with logging."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from mlagent.pipeline.models import ExecutionResult
from mlagent.sandbox.isolation import build_restricted_env


METRIC_PATTERN = re.compile(
    r"MLAGENT_METRIC:(\w+)=([-\d.eE+]+)"
)


class SandboxExecutor:
    """Execute Python scripts in an isolated workspace with timeout and logging."""

    def __init__(self, workspace: Path, timeout_seconds: int = 300):
        self.workspace = workspace.resolve()
        self.timeout_seconds = timeout_seconds
        (self.workspace / "tmp").mkdir(parents=True, exist_ok=True)

    def execute_script(
        self,
        script_path: Path,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        script_path = script_path.resolve()
        if not script_path.exists():
            return ExecutionResult(
                success=False,
                exit_code=1,
                stderr=f"Script not found: {script_path}",
            )
        if not str(script_path).startswith(str(self.workspace)):
            return ExecutionResult(
                success=False,
                exit_code=1,
                stderr="Script path outside sandbox workspace",
            )

        env = build_restricted_env(self.workspace)
        if extra_env:
            env.update(extra_env)

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            duration = time.perf_counter() - start
            metrics = self._parse_metrics(proc.stdout)
            artifacts = self._detect_artifacts(proc.stdout)
            return ExecutionResult(
                success=proc.returncode == 0,
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_seconds=duration,
                metrics=metrics,
                artifacts_produced=artifacts,
            )
        except subprocess.TimeoutExpired as e:
            duration = time.perf_counter() - start
            return ExecutionResult(
                success=False,
                exit_code=124,
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + f"\nTimeout after {self.timeout_seconds}s",
                duration_seconds=duration,
            )

    def _parse_metrics(self, stdout: str) -> dict[str, float]:
        metrics = {}
        for match in METRIC_PATTERN.finditer(stdout):
            try:
                metrics[match.group(1)] = float(match.group(2))
            except ValueError:
                continue
        metrics_path = self.workspace / "metrics.json"
        if metrics_path.exists():
            try:
                metrics.update(json.loads(metrics_path.read_text()))
            except json.JSONDecodeError:
                pass
        return metrics

    def _detect_artifacts(self, stdout: str) -> list[str]:
        artifacts = []
        for line in stdout.splitlines():
            if line.startswith("MLAGENT_ARTIFACT:"):
                artifacts.append(line.split(":", 1)[1].strip())
        return artifacts
