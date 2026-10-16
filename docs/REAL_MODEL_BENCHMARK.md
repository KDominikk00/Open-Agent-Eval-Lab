# Real-Model Benchmark Quickstart

Two routes: hosted APIs (needs keys and budget) or local models through Ollama (free, needs a machine that can run them).

## Local models with Ollama (no API key)

```bash
ollama pull llama3.1 && ollama pull qwen2.5      # models that support tool calling
open-agent-eval run-profile --profile profiles/local-ollama.json
```

`ollama-chat` talks to Ollama's OpenAI-compatible endpoint (`$OLLAMA_HOST`, default `http://localhost:11434`). Pick a
model with `--model` or `$OLLAMA_MODEL`. The model must support tool calling; models that do not will answer without
calling tools and fail the tool-use tasks, which is a valid result but not a fair one. Different models appear as separate
leaderboard rows (`ollama-chat (llama3.1)`).

Not yet verified against a live Ollama server: the adapter is tested against mocked HTTP only. Check one run by hand
(`open-agent-eval run ... --agent ollama-chat --max-tasks 2`, then open `traces.html`) before trusting a full run.

## Hosted APIs

## Required environment variables

```bash
export OPENAI_API_KEY="<your-openai-key>"
export ANTHROPIC_API_KEY="<your-anthropic-key>"
```

## Run the prepared comparison profile

```bash
open-agent-eval run-profile \
  --profile profiles/real-models-public.json
```

This executes:

- `openai-chat` with `gpt-4o-mini`
- `anthropic-chat` with `claude-haiku-5-5`

And refreshes `leaderboard/` artifacts.

## Use a fixed timestamp (recommended for reporting)

```bash
open-agent-eval run-profile \
  --profile profiles/real-models-public.json \
  --timestamp 20260221-180000
```

## Optional overrides

```bash
open-agent-eval run-profile \
  --profile profiles/real-models-public.json \
  --output-dir results-experiment \
  --leaderboard-output-dir leaderboard-experiment
```

## Build a report

Use `docs/reports/BENCHMARK_REPORT_TEMPLATE.md` as the launch report template.
