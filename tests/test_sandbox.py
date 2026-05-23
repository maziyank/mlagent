"""Sandbox executor tests."""

from pathlib import Path

from mlagent.sandbox.executor import SandboxExecutor


def test_sandbox_executes_simple_script(tmp_path: Path) -> None:
    script = tmp_path / "hello.py"
    script.write_text('print("MLAGENT_METRIC:test=1.0")\n')
    ex = SandboxExecutor(tmp_path, timeout_seconds=30)
    result = ex.execute_script(script)
    assert result.success
    assert result.metrics.get("test") == 1.0
