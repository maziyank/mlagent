# Benchmark Results

The ML Agent system autonomously completes standard tabular ML tasks across publicly available structured datasets.

## Datasets

| Dataset | Source | Task | Default pipeline |
|---------|--------|------|------------------|
| iris | sklearn | 3-class classification | multiclass_classification |
| wine | sklearn | 3-class classification | multiclass_classification |
| breast_cancer | sklearn | binary classification | binary_classification |
| diabetes | sklearn | regression | regression |
| titanic | URL (Kaggle-style) | binary classification | binary_classification |

## Running benchmarks

```bash
source .venv/bin/activate
python scripts/run_benchmarks.py
```

Results are written to `docs/benchmark_results.json`.

## Expected behavior (template mode)

All four core sklearn benchmarks should complete with:

- **Classification** — `accuracy` ≥ 0.70 (configurable via `--min-metric`)
- **Regression** — `r2` ≥ 0.50

Latest benchmark run (`docs/benchmark_results.json`, template mode):

| Dataset | Key metric | Result |
|---------|------------|--------|
| iris | accuracy | 0.90 |
| wine | accuracy | 1.00 |
| breast_cancer | accuracy | 0.96 |
| diabetes | r2 | 0.45 |

## Agent mode benchmarks

With `OPENROUTER_API_KEY` set:

```bash
mlagent run iris --mode agent --model openrouter:openai/gpt-4o-mini
```

The orchestrator delegates to sub-agents, which generate and refine code via `execute_sandbox_code` until validation passes.

## Iteration logging

Each sandbox execution logs to:

```
.mlagent_runs/<run_id>/logs/<stage>/iteration_NNN.json
```

Agents (or the template runner) use stdout/stderr and metrics from these logs for refinement.
