# Open Agent Eval Lab

Open-source benchmark harness for evaluating tool-using LLM agents on reproducible tasks.

## Vision

Open Agent Eval Lab helps you compare agent setups with consistent tasks and deterministic scoring.

## What is included in this repository (v0)

- Benchmark task format (`tasks/`)
- Validation and linting for task specs (`open_agent_eval.validators`)
- Baseline runner and sample agents (`open_agent_eval.runner`, `open_agent_eval.agents`)
- Local artifacts in JSONL/JSON for reproducible experiment history (`results/`)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Validate tasks:

```bash
PYTHONPATH=src python -m open_agent_eval.cli validate --tasks-dir tasks/examples
```

Run baseline evaluation:

```bash
PYTHONPATH=src python -m open_agent_eval.cli run --tasks-dir tasks/examples --output-dir results --agent echo-reference
```

## Output artifacts

Each run writes to `results/<run-id>/`:

- `results.jsonl`: one JSON record per task
- `summary.json`: aggregate metrics (pass rate, latency, totals)

## Roadmap

1. Expand task corpus (public + hidden split)
2. Add more baseline agents
3. Add grader plugins and richer metrics
4. Add leaderboard dashboard and CI smoke benchmarks
