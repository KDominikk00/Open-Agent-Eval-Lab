"""Stable Python API for framework users."""

from __future__ import annotations

from pathlib import Path

from open_agent_eval.leaderboard import generate_leaderboard
from open_agent_eval.profiles import (
    BenchmarkProfile,
    ProfileExecutionResult,
    execute_profile,
    load_profile,
)
from open_agent_eval.runner import RunSummary, run_from_directory
from open_agent_eval.validators import ValidationReport, validate_tasks_dir


def validate_dataset(tasks_dir: str | Path, *, split: str | None = None) -> ValidationReport:
    """Validate a benchmark dataset directory."""

    return validate_tasks_dir(Path(tasks_dir), only_split=split)


def run_benchmark(
    *,
    tasks_dir: str | Path,
    output_dir: str | Path,
    agent: str,
    split: str | None = "public",
    run_id: str | None = None,
    max_tasks: int | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
    timeout_seconds: float = 45.0,
    trials: int = 1,
    label: str | None = None,
) -> tuple[Path, RunSummary]:
    """Execute one benchmark run and return artifact path + run summary."""

    return run_from_directory(
        tasks_dir=Path(tasks_dir),
        output_dir=Path(output_dir),
        agent_name=agent,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        only_split=split,
        run_id=run_id,
        max_tasks=max_tasks,
        trials=trials,
        label=label,
    )


def build_leaderboard(
    *, results_dir: str | Path, output_dir: str | Path, include_smoke: bool = False
) -> Path:
    """Generate leaderboard artifacts from run summaries."""

    return generate_leaderboard(
        results_dir=Path(results_dir), output_dir=Path(output_dir), include_smoke=include_smoke
    )


def load_benchmark_profile(profile_path: str | Path) -> BenchmarkProfile:
    """Load and validate a benchmark profile file."""

    return load_profile(Path(profile_path))


def run_benchmark_profile(
    *,
    profile: BenchmarkProfile,
    profile_path: str | Path,
    timestamp: str | None = None,
    tasks_dir_override: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    leaderboard_output_dir_override: str | Path | None = None,
    skip_leaderboard: bool = False,
) -> ProfileExecutionResult:
    """Run all benchmark entries from a loaded profile."""

    return execute_profile(
        profile,
        profile_path=Path(profile_path),
        timestamp=timestamp,
        tasks_dir_override=Path(tasks_dir_override) if tasks_dir_override else None,
        output_dir_override=Path(output_dir_override) if output_dir_override else None,
        leaderboard_output_dir_override=Path(leaderboard_output_dir_override)
        if leaderboard_output_dir_override
        else None,
        skip_leaderboard=skip_leaderboard,
    )
