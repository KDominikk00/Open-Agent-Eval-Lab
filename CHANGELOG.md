# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Sandboxed tool simulators (`read_file`, `write_file`, `list_files`, `search_logs`, `db_query`, `http_get`, `calculator`) and a multi-step episode loop that records full tool traces.
- Behavioural graders: required, ordered and forbidden tool calls, forbidden response snippets, and final file-state checks.
- Native tool-calling loops for the OpenAI and Anthropic agents on episode tasks.
- Offline agents `oracle`, `keyword-retriever` and `random` (seeded).
- 31 episode tasks (log forensics, SQL + math, config edits, prompt injection, API chaining, secret hygiene).
- `--trials` for repeated runs, `--label` for tagging runs, Wilson and bootstrap confidence intervals, per-category/difficulty/grader breakdowns (`breakdowns.json`).
- Leaderboard: ranked agents with 95% CIs, category heatmap, trace viewer (`traces.html`); `smoke` runs hidden unless `--include-smoke`.
- Committed baseline snapshot in `benchmarks/published/` (`scripts/publish_baselines.sh`) used by the Pages deploy.
- Ruff, strict mypy, pytest with an 85% coverage gate, a Python 3.10-3.13 CI matrix, a snapshot-freshness CI job, `Makefile` and pre-commit config.
- `GullibleAgent` example showing the benchmark catching an agent that obeys instructions found in documents.
- `docs/ARCHITECTURE.md`, `docs/DATASET.md`, `docs/DESIGN_DECISIONS.md`; results chart rendered from the committed snapshot.

### Changed

- Validation rejects tasks whose `expected_response` fails their own grader.
- `RunSummary.total_tasks` counts unique tasks; `total_attempts` counts task x trial attempts and `passed_tasks` counts passed attempts.
- Default Anthropic model is now `claude-haiku-5-5` (was `claude-3-5-haiku-latest`).
- README rewritten; `ROADMAP.md` and `docs/RELEASE_CHECKLIST.md` folded into `CONTRIBUTING.md`.

### Fixed

- Re-using a `--run-id` now fails immediately with a clear message instead of crashing after the agent has run.
- Blank `OPENAI_MODEL` / `ANTHROPIC_MODEL` environment variables fall back to the default model.
- `public.security.parameterized-query` reference answer did not satisfy its grader.

## [0.4.0] - 2026-02-21

### Added

- Custom agent plugin system via import specs (`module.path:SymbolName`).
- Profile-driven benchmark execution command: `run-profile`.
- Built-in profiles for local smoke and real-model comparison.
- Framework extension docs and custom agent examples.
- Launch-ready benchmark report template.
- GitHub Pages deployment workflow for static leaderboard publishing.
- Issue templates, PR template, roadmap, and security policy.

### Changed

- CLI `run --agent` now accepts either built-in names or custom import specs.
- Runner now supports model/runtime parameters in reproducible run config artifacts.
- Packaging metadata updated for public release readiness and typed marker (`py.typed`).

## [0.3.0] - 2026-02-21

### Added

- Split-aware public/hidden benchmark dataset format.
- Multi-strategy graders (`exact`, `contains`, `json`).
- Baseline runner with task/category metrics.
- Leaderboard generation (`json`, `md`, `html`).

### Changed

- CLI expanded with `validate`, `run`, `leaderboard`, and `agents` commands.

## [0.1.0] - 2026-02-21

### Added

- Initial project scaffold and package structure.
