# Contributing to Open Agent Eval Lab

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
make install        # pip install -e ".[dev]"
make check          # ruff + strict mypy + pytest with coverage (what CI runs)
```

## Adding a benchmark task

1. Copy `docs/TASK_TEMPLATE.json` into `tasks/public/<category>/` or `tasks/hidden/<category>/`.
2. Use a globally unique `task_id` of the form `<split>.<category>.<slug>`.
3. For tool-use tasks, define an `environment` (files, logs, tables, http fixtures), the enabled `tools`, and a `reference_trajectory` that solves it. The `oracle` agent replays this trajectory, so it doubles as proof that the task is solvable.
4. Grade behaviour, not just the answer: `required_calls`, `call_order`, `forbidden_calls`, `forbidden_needles`, `final_files`.
5. `expected_response` must satisfy its own grader (validation enforces this).
6. Validate, then refresh the published baselines if you changed public tasks:

```bash
open-agent-eval validate --tasks-dir tasks --split all
make baselines      # rewrites benchmarks/published; commit the result
```

A test fails when `benchmarks/published` was generated from a different public task set.

## Adding an agent

- Single-turn: implement `run(task) -> AgentOutput`.
- Tool-using: implement `decide(task, history) -> ToolCall | FinalAnswer` and call `run_episode`; see `src/open_agent_eval/policies.py`.
- Stochastic agents should expose `with_trial(index)` so `--trials` gives independent but reproducible runs.
- Register built-ins in `_builtin_agent_factories` (`agents.py`); external agents use `module.path:Symbol` (see `docs/CUSTOM_AGENTS.md`).

## Pull requests

- `make check` passes; new behaviour has tests.
- Dataset validation passes and the published snapshot is current.
- Interface changes update the README and `CHANGELOG.md`.

## Releasing

1. `make check && make baselines`, then commit any snapshot changes.
2. Keep `version` in `pyproject.toml` and `src/open_agent_eval/__init__.py` in sync.
3. Move `CHANGELOG.md` entries from Unreleased to the new version and tag it (`git tag vX.Y.Z`).

## Roadmap

- Retries and rate-limit backoff for API agents; optional result caching.
- Ollama adapter so real local-model results can be published without API keys.
- Per-tool failure analysis and richer leaderboard filtering.
- Benchmark packs for code review and triage tasks.

## Issue templates

Use the templates for bug reports, task proposals and feature requests.
