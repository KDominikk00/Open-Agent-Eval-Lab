# Real-Model Benchmark Quickstart

## Required environment variables

```bash
export OPENAI_API_KEY="<your-openai-key>"
export ANTHROPIC_API_KEY="<your-anthropic-key>"
```

## Run the prepared comparison profile

```bash
PYTHONPATH=src python -m open_agent_eval.cli run-profile \
  --profile profiles/real-models-public.json
```

This executes:

- `openai-chat` with `gpt-4o-mini`
- `anthropic-chat` with `claude-3-5-haiku-latest`

And refreshes `leaderboard/` artifacts.

## Use a fixed timestamp (recommended for reporting)

```bash
PYTHONPATH=src python -m open_agent_eval.cli run-profile \
  --profile profiles/real-models-public.json \
  --timestamp 20260221-180000
```

## Optional overrides

```bash
PYTHONPATH=src python -m open_agent_eval.cli run-profile \
  --profile profiles/real-models-public.json \
  --output-dir results-experiment \
  --leaderboard-output-dir leaderboard-experiment
```

## Build a report

Use `/Users/dominik/Projects/Machine Learning/docs/reports/BENCHMARK_REPORT_TEMPLATE.md` as the launch report template.
