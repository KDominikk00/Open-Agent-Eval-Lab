"""Profile-driven benchmark execution utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from open_agent_eval.leaderboard import generate_leaderboard
from open_agent_eval.runner import RunSummary, run_from_directory


@dataclass(frozen=True)
class ProfileRunConfig:
    """One run entry from a benchmark profile."""

    agent: str
    model: str | None = None
    split: str | None = None
    max_tasks: int | None = None
    temperature: float = 0.0
    max_output_tokens: int = 512
    timeout_seconds: float = 45.0
    run_id_prefix: str | None = None
    trials: int = 1
    label: str | None = None


@dataclass(frozen=True)
class BenchmarkProfile:
    """Top-level benchmark profile definition."""

    name: str
    tasks_dir: str = "tasks"
    output_dir: str = "results"
    split: str | None = "public"
    max_tasks: int | None = None
    generate_leaderboard: bool = True
    leaderboard_output_dir: str = "leaderboard"
    runs: tuple[ProfileRunConfig, ...] = ()


@dataclass(frozen=True)
class ProfileRunOutcome:
    """Result metadata for one executed profile run."""

    run_id: str
    run_label: str
    artifacts_dir: str
    summary: RunSummary


@dataclass(frozen=True)
class ProfileExecutionResult:
    """Result metadata for a full profile execution."""

    profile_name: str
    outcomes: tuple[ProfileRunOutcome, ...]
    leaderboard_dir: str | None


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"profile field '{field}' must be numeric")


def _as_int_or_none(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError(f"profile field '{field}' must be a positive integer or null")


def _as_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError(f"profile field '{field}' must be a positive integer")


def _as_positive_float(value: object, *, field: str) -> float:
    parsed = _as_float(value, field=field)
    if parsed <= 0:
        raise ValueError(f"profile field '{field}' must be greater than zero")
    return parsed


def _as_bool(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"profile field '{field}' must be a boolean")


def _as_str(value: object, *, field: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"profile field '{field}' must be a non-empty string")


def _as_optional_split(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"public", "hidden", "all"}:
            return None if normalized == "all" else normalized
    raise ValueError(f"profile field '{field}' must be one of: public, hidden, all")


def _parse_run(raw: object, *, index: int) -> ProfileRunConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"profile runs[{index}] must be an object")

    agent = _as_str(raw.get("agent"), field=f"runs[{index}].agent")

    model = raw.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError(f"profile field 'runs[{index}].model' must be a non-empty string or null")

    run_id_prefix = raw.get("run_id_prefix")
    if run_id_prefix is not None and (
        not isinstance(run_id_prefix, str) or not run_id_prefix.strip()
    ):
        raise ValueError(
            f"profile field 'runs[{index}].run_id_prefix' must be a non-empty string or null"
        )

    label = raw.get("label")
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise ValueError(f"profile field 'runs[{index}].label' must be a non-empty string or null")

    return ProfileRunConfig(
        agent=agent,
        model=model.strip() if isinstance(model, str) else None,
        split=_as_optional_split(raw.get("split", None), field=f"runs[{index}].split")
        if "split" in raw
        else None,
        max_tasks=_as_int_or_none(raw.get("max_tasks", None), field=f"runs[{index}].max_tasks")
        if "max_tasks" in raw
        else None,
        temperature=_as_float(raw.get("temperature", 0.0), field=f"runs[{index}].temperature"),
        max_output_tokens=_as_positive_int(
            raw.get("max_output_tokens", 512), field=f"runs[{index}].max_output_tokens"
        ),
        timeout_seconds=_as_positive_float(
            raw.get("timeout_seconds", 45.0), field=f"runs[{index}].timeout_seconds"
        ),
        run_id_prefix=run_id_prefix.strip() if isinstance(run_id_prefix, str) else None,
        trials=_as_positive_int(raw.get("trials", 1), field=f"runs[{index}].trials"),
        label=label.strip() if isinstance(label, str) else None,
    )


def load_profile(profile_path: Path) -> BenchmarkProfile:
    """Load and validate a benchmark profile JSON file."""

    if not profile_path.exists():
        raise ValueError(f"profile file does not exist: {profile_path}")

    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"profile file contains invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("profile root must be a JSON object")

    name = _as_str(payload.get("name"), field="name")
    tasks_dir = _as_str(payload.get("tasks_dir", "tasks"), field="tasks_dir")
    output_dir = _as_str(payload.get("output_dir", "results"), field="output_dir")
    split = _as_optional_split(payload.get("split", "public"), field="split")
    max_tasks = _as_int_or_none(payload.get("max_tasks", None), field="max_tasks")
    generate_lb = _as_bool(payload.get("generate_leaderboard", True), field="generate_leaderboard")
    lb_output_dir = _as_str(
        payload.get("leaderboard_output_dir", "leaderboard"), field="leaderboard_output_dir"
    )

    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("profile field 'runs' must be a non-empty array")

    runs = tuple(_parse_run(raw, index=index) for index, raw in enumerate(raw_runs))

    return BenchmarkProfile(
        name=name,
        tasks_dir=tasks_dir,
        output_dir=output_dir,
        split=split,
        max_tasks=max_tasks,
        generate_leaderboard=generate_lb,
        leaderboard_output_dir=lb_output_dir,
        runs=runs,
    )


def _resolve_from_profile(profile_path: Path, maybe_relative: str) -> Path:
    candidate = Path(maybe_relative)
    if candidate.is_absolute():
        return candidate
    return (profile_path.parent / candidate).resolve()


def _default_run_id_prefix(run: ProfileRunConfig) -> str:
    model_token = run.model.replace("/", "-") if run.model else "default-model"
    model_token = model_token.replace(" ", "-")
    return f"{run.agent}-{model_token}"


def execute_profile(
    profile: BenchmarkProfile,
    *,
    profile_path: Path,
    timestamp: str | None = None,
    tasks_dir_override: Path | None = None,
    output_dir_override: Path | None = None,
    leaderboard_output_dir_override: Path | None = None,
    skip_leaderboard: bool = False,
) -> ProfileExecutionResult:
    """Execute all runs in a benchmark profile and optionally regenerate leaderboard."""

    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    tasks_dir = tasks_dir_override or _resolve_from_profile(profile_path, profile.tasks_dir)
    output_dir = output_dir_override or _resolve_from_profile(profile_path, profile.output_dir)
    leaderboard_output_dir = leaderboard_output_dir_override or _resolve_from_profile(
        profile_path, profile.leaderboard_output_dir
    )

    outcomes: list[ProfileRunOutcome] = []

    for run in profile.runs:
        split = run.split if run.split is not None else profile.split
        max_tasks = run.max_tasks if run.max_tasks is not None else profile.max_tasks
        run_prefix = run.run_id_prefix or _default_run_id_prefix(run)
        run_id = f"{run_prefix}-{ts}"

        run_dir, summary = run_from_directory(
            tasks_dir=tasks_dir,
            output_dir=output_dir,
            agent_name=run.agent,
            model=run.model,
            temperature=run.temperature,
            max_output_tokens=run.max_output_tokens,
            timeout_seconds=run.timeout_seconds,
            run_id=run_id,
            only_split=split,
            max_tasks=max_tasks,
            trials=run.trials,
            label=run.label,
        )

        label = f"{run.agent} ({run.model})" if run.model else run.agent
        outcomes.append(
            ProfileRunOutcome(
                run_id=run_id,
                run_label=label,
                artifacts_dir=str(run_dir),
                summary=summary,
            )
        )

    leaderboard_dir: str | None = None
    if profile.generate_leaderboard and not skip_leaderboard:
        generated_dir = generate_leaderboard(results_dir=output_dir, output_dir=leaderboard_output_dir)
        leaderboard_dir = str(generated_dir)

    return ProfileExecutionResult(
        profile_name=profile.name,
        outcomes=tuple(outcomes),
        leaderboard_dir=leaderboard_dir,
    )
