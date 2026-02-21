# Changelog

All notable changes to this project are documented in this file.

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
