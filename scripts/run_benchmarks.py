#!/usr/bin/env python3
"""Run benchmarks across standard tabular datasets."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlagent.config import get_settings
from mlagent.pipeline.models import RunStatus
from mlagent.pipeline.runner import PipelineRunner

BENCHMARK_DATASETS = ["iris", "wine", "breast_cancer", "diabetes", "titanic"]


def main() -> int:
    settings = get_settings()
    settings.execution_mode = "template"
    runner = PipelineRunner(settings)
    results = []

    for ds in BENCHMARK_DATASETS:
        print(f"\n=== Benchmark: {ds} ===")
        run = runner.run(ds)
        results.append(
            {
                "dataset": ds,
                "run_id": run.run_id,
                "status": run.status.value,
                "metrics": run.final_metrics,
            }
        )

    out_dir = Path(__file__).resolve().parents[1] / "docs"
    out_dir.mkdir(exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "template",
        "datasets": results,
        "passed": sum(1 for r in results if r["status"] == RunStatus.COMPLETED.value),
        "total": len(results),
    }
    out_path = out_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Passed {report['passed']}/{report['total']}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
