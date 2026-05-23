"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn import datasets as sk_datasets


def load_dataset_config(name: str, config_dir: Path) -> dict[str, Any]:
    path = config_dir / "datasets.yaml"
    with path.open() as f:
        all_ds = yaml.safe_load(f)["datasets"]
    if name not in all_ds:
        raise KeyError(f"Unknown dataset: {name}. Available: {list(all_ds)}")
    return all_ds[name]


def build_file_dataset_config(
    csv_path: Path,
    *,
    target_column: str,
    task_type: str = "classification",
    default_pipeline: str | None = None,
    name: str = "custom",
) -> dict[str, Any]:
    """Build an in-memory dataset config for a local CSV file."""
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if default_pipeline is None:
        default_pipeline = (
            "regression" if task_type == "regression" else "binary_classification"
        )
    return {
        "name": name,
        "source": "file",
        "path": str(csv_path),
        "target_column": target_column,
        "task_type": task_type,
        "default_pipeline": default_pipeline,
        "description": f"Custom CSV: {csv_path.name}",
    }


def materialize_dataset(
    name: str,
    dest_dir: Path,
    config_dir: Path,
    *,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Download or load dataset into workspace data/raw.csv."""
    cfg = cfg or load_dataset_config(name, config_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_path = dest_dir / "raw.csv"

    if cfg["source"] == "sklearn":
        loader_name = cfg["sklearn_dataset"]
        bundle = getattr(sk_datasets, f"load_{loader_name}")()
        df = pd.DataFrame(bundle.data, columns=bundle.feature_names)
        target_col = cfg.get("target_column", "target")
        df[target_col] = bundle.target
        df.to_csv(raw_path, index=False)
    elif cfg["source"] == "url":
        df = pd.read_csv(cfg["url"])
        df.to_csv(raw_path, index=False)
    elif cfg["source"] == "file":
        file_path = Path(cfg["path"]).expanduser().resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        df = pd.read_csv(file_path)
        target_col = cfg.get("target_column", "target")
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not in CSV. "
                f"Columns: {list(df.columns)}"
            )
        df.to_csv(raw_path, index=False)
    else:
        raise ValueError(f"Unsupported source: {cfg['source']}")

    meta = {
        "name": name,
        "target_column": cfg.get("target_column", "target"),
        "task_type": cfg.get("task_type", "classification"),
        "description": cfg.get("description", ""),
    }
    (dest_dir / "dataset_meta.json").write_text(
        __import__("json").dumps(meta, indent=2)
    )
    return raw_path
