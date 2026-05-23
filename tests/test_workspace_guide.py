"""Workspace guide and layout tests."""

from pathlib import Path

from mlagent.agents.workspace_guide import (
    WORKSPACE_LAYOUT_FILENAME,
    build_workspace_context,
    seed_stage_directories,
    write_workspace_layout,
)
from mlagent.storage.workspace import WorkspaceManager


def test_workspace_layout_written_on_create_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLAGENT_WORKSPACE_ROOT", str(tmp_path))
    from mlagent.config import Settings

    mgr = WorkspaceManager(Settings(workspace_root=tmp_path))
    run = mgr.create_run("iris", "multiclass_classification")
    ws = Path(run.workspace_path)
    assert (ws / WORKSPACE_LAYOUT_FILENAME).exists()
    assert (ws / "code" / "data_understanding" / "README.md").exists()


def test_build_workspace_context_includes_stage(tmp_path: Path) -> None:
    ws = tmp_path / "run1"
    (ws / "data").mkdir(parents=True)
    (ws / "data" / "raw.csv").write_text("a\n1")
    write_workspace_layout(ws)
    seed_stage_directories(ws)
    ctx = build_workspace_context(ws, stage="data_understanding")
    assert ctx["current_stage"]["name"] == "data_understanding"
    assert "data/eda_summary.json" in ctx["current_stage"]["required_outputs"]
    assert "data/raw.csv" in ctx["existing_files"]["data"]
