# Open Agent Eval Lab

Open Agent Eval Lab is a reproducible benchmark harness for tool-using LLM agents.

It helps you compare agent configurations using a split-aware task dataset, deterministic grading, versioned run artifacts, and a generated leaderboard.

## Features

- Split-aware benchmark dataset (`public` and `hidden`)
- Task schema validation with duplicate ID detection and strict directory layout checks
- Multiple grading strategies:
  - `exact`
  - `contains`
  - `json` (strict or subset mode)
- Baseline agents for repeatable local benchmarks
- Optional API-backed agents (OpenAI and Anthropic)
- Rich run metrics:
  - pass rate
  - average score
  - tool precision/recall
  - latency
  - category breakdown
- Leaderboard generation in:
  - JSON
  - Markdown
  - HTML

## Repository layout

```text
src/open_agent_eval/
  agents.py
  cli.py
  graders.py
  leaderboard.py
  runner.py
  schemas.py
  validators.py

tasks/
  public/
  hidden/

tests/
.github/workflows/ci.yml
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

List available baseline agents:

```bash
PYTHONPATH=src python -m open_agent_eval.cli agents
```

Validate the full dataset:

```bash
PYTHONPATH=src python -m open_agent_eval.cli validate --tasks-dir tasks --split all
```

Run a benchmark on the public split:

```bash
PYTHONPATH=src python -m open_agent_eval.cli run \
  --tasks-dir tasks \
  --output-dir results \
  --agent echo-reference \
  --split public
```

Run a benchmark with a real model via OpenAI:

```bash
export OPENAI_API_KEY="your-key"
PYTHONPATH=src python -m open_agent_eval.cli run \
  --tasks-dir tasks \
  --output-dir results \
  --agent openai-chat \
  --model gpt-4o-mini \
  --split public \
  --max-tasks 5
```

Run a benchmark with a real model via Anthropic:

```bash
export ANTHROPIC_API_KEY="your-key"
PYTHONPATH=src python -m open_agent_eval.cli run \
  --tasks-dir tasks \
  --output-dir results \
  --agent anthropic-chat \
  --model claude-3-5-haiku-latest \
  --split public \
  --max-tasks 5
```

Generate leaderboard artifacts:

```bash
PYTHONPATH=src python -m open_agent_eval.cli leaderboard \
  --results-dir results \
  --output-dir leaderboard
```

## CLI reference

### `validate`

```bash
PYTHONPATH=src python -m open_agent_eval.cli validate --tasks-dir tasks --split [public|hidden|all]
```

### `run`

```bash
PYTHONPATH=src python -m open_agent_eval.cli run \
  --tasks-dir tasks \
  --output-dir results \
  --agent [echo-reference|instruction-snippet|tool-hint|openai-chat|anthropic-chat] \
  --split [public|hidden|all] \
  --model <optional-model-name> \
  --temperature 0.0 \
  --max-output-tokens 512 \
  --timeout-seconds 45 \
  --max-tasks 5
```

### `leaderboard`

```bash
PYTHONPATH=src python -m open_agent_eval.cli leaderboard --results-dir results --output-dir leaderboard
```

## Run artifact format

Each run writes to `results/<run-id>/`:

- `config.json`: run configuration and selected tasks
- `results.jsonl`: one line per task result
- `summary.json`: aggregate metrics for the run
- `categories.json`: per-category metrics

Leaderboard generation writes to `leaderboard/`:

- `leaderboard.json`
- `leaderboard.md`
- `leaderboard.html`

## Task format

Task files must be inside `tasks/public/**` or `tasks/hidden/**` and follow the schema in `docs/TASK_TEMPLATE.json`.

### Minimum required fields

- `task_id`
- `title`
- `category`
- `instructions`
- `expected_response`

## Development

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Run a local smoke pipeline:

```bash
PYTHONPATH=src python -m open_agent_eval.cli validate --tasks-dir tasks --split public
PYTHONPATH=src python -m open_agent_eval.cli run --tasks-dir tasks --output-dir results --agent instruction-snippet --split public --max-tasks 3 --run-id run-smoke-local
PYTHONPATH=src python -m open_agent_eval.cli leaderboard --results-dir results --output-dir leaderboard
```

## Contributing

See `CONTRIBUTING.md` for task and code contribution guidelines.

## Hosted leaderboard

The workflow in `/Users/dominik/Projects/Machine Learning/.github/workflows/deploy-leaderboard.yml` publishes a static leaderboard site to GitHub Pages whenever `main` is updated or the workflow is manually triggered.

To enable it in your GitHub repository:

1. Open repository settings.
2. Go to Pages.
3. Set source to GitHub Actions.

Detailed steps are also documented in `/Users/dominik/Projects/Machine Learning/docs/HOSTED_LEADERBOARD.md`.
