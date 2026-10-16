"""Benchmark runner implementation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from open_agent_eval.agents import Agent, AgentOutput, get_agent
from open_agent_eval.graders import compute_tool_match, grade_task
from open_agent_eval.schemas import TaskSpec
from open_agent_eval.stats import bootstrap_mean_interval, mean, wilson_interval
from open_agent_eval.validators import ValidationIssue, validate_tasks_dir


@dataclass(frozen=True)
class TaskRunResult:
    task_id: str
    split: str
    category: str
    difficulty: str
    success: bool
    score: float
    grade_reason: str
    response: str
    expected_response: str
    latency_ms: float
    steps: int
    max_steps: int
    step_ratio: float
    tool_calls: list[str]
    required_tools: list[str]
    tool_precision: float
    tool_recall: float
    error: str | None
    trace: list[dict[str, object]] = field(default_factory=list)
    trial: int = 0
    grader_type: str = "exact"


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    agent_name: str
    split: str
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    pass_rate: float
    average_score: float
    average_latency_ms: float
    average_tool_precision: float
    average_tool_recall: float
    average_step_ratio: float
    started_at_utc: str
    finished_at_utc: str
    trials: int = 1
    total_attempts: int = 0
    label: str | None = None
    pass_rate_ci_low: float = 0.0
    pass_rate_ci_high: float = 1.0
    score_ci_low: float = 0.0
    score_ci_high: float = 1.0
    average_steps: float = 0.0


@dataclass(frozen=True)
class CategorySummary:
    category: str
    total_tasks: int
    passed_tasks: int
    pass_rate: float
    average_score: float
    pass_rate_ci_low: float = 0.0
    pass_rate_ci_high: float = 1.0


def _per_task_means(results: list[TaskRunResult]) -> tuple[list[float], list[float]]:
    """Collapse repeated trials so that tasks, not attempts, are the independent units."""

    by_task: dict[str, list[TaskRunResult]] = {}
    for result in results:
        by_task.setdefault(result.task_id, []).append(result)
    passes = [mean([1.0 if r.success else 0.0 for r in group]) for group in by_task.values()]
    scores = [mean([r.score for r in group]) for group in by_task.values()]
    return passes, scores


def _build_summary(
    *,
    run_id: str,
    agent_name: str,
    split: str,
    results: list[TaskRunResult],
    started_at: datetime,
    finished_at: datetime,
    trials: int = 1,
    label: str | None = None,
) -> RunSummary:
    total = len(results)
    passed = sum(1 for result in results if result.success)
    task_passes, task_scores = _per_task_means(results)
    pass_low, pass_high = wilson_interval(mean(task_passes), len(task_passes))
    score_low, score_high = bootstrap_mean_interval(task_scores)

    def _avg(values: list[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    return RunSummary(
        run_id=run_id,
        agent_name=agent_name,
        split=split,
        total_tasks=len(task_passes),
        total_attempts=total,
        passed_tasks=passed,
        failed_tasks=total - passed,
        pass_rate=round((passed / total) if total else 0.0, 4),
        average_score=round(_avg([result.score for result in results]), 4),
        average_latency_ms=round(_avg([result.latency_ms for result in results]), 3),
        average_tool_precision=round(_avg([result.tool_precision for result in results]), 4),
        average_tool_recall=round(_avg([result.tool_recall for result in results]), 4),
        average_step_ratio=round(_avg([result.step_ratio for result in results]), 4),
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        trials=trials,
        label=label,
        pass_rate_ci_low=round(pass_low, 4),
        pass_rate_ci_high=round(pass_high, 4),
        score_ci_low=round(score_low, 4),
        score_ci_high=round(score_high, 4),
        average_steps=round(_avg([float(result.steps) for result in results]), 3),
    )


def build_breakdown(results: list[TaskRunResult], dimension: str) -> list[CategorySummary]:
    """Aggregate results by a ``TaskRunResult`` field (category, difficulty, grader_type)."""

    groups: dict[str, list[TaskRunResult]] = {}
    for result in results:
        groups.setdefault(str(getattr(result, dimension)), []).append(result)

    summaries: list[CategorySummary] = []
    for name in sorted(groups):
        bucket = groups[name]
        task_passes, task_scores = _per_task_means(bucket)
        low, high = wilson_interval(mean(task_passes), len(task_passes))
        summaries.append(
            CategorySummary(
                category=name,
                total_tasks=len(task_passes),
                passed_tasks=sum(1 for result in bucket if result.success),
                pass_rate=round(mean(task_passes), 4),
                average_score=round(mean(task_scores), 4),
                pass_rate_ci_low=round(low, 4),
                pass_rate_ci_high=round(high, 4),
            )
        )
    return summaries


def build_category_summaries(results: list[TaskRunResult]) -> list[CategorySummary]:
    """Compute per-category aggregate metrics from task results."""

    return build_breakdown(results, "category")


def run_benchmark(
    tasks: list[TaskSpec],
    agent: Agent,
    run_id: str,
    split_label: str,
    trials: int = 1,
    label: str | None = None,
) -> tuple[list[TaskRunResult], RunSummary]:
    """Run ``trials`` benchmark passes over tasks with a selected agent.

    Agents exposing ``with_trial(index)`` get a per-trial variant (e.g. a new random seed),
    so repeated trials of stochastic agents are independent but reproducible.
    """

    if trials <= 0:
        raise ValueError("trials must be a positive integer")

    results: list[TaskRunResult] = []
    started_at = datetime.now(timezone.utc)

    for trial in range(trials):
        reseed = getattr(agent, "with_trial", None)
        trial_agent = reseed(trial) if callable(reseed) else agent
        results.extend(_run_trial(tasks, trial_agent, trial))

    finished_at = datetime.now(timezone.utc)
    summary = _build_summary(
        run_id=run_id,
        agent_name=agent.name,
        split=split_label,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
        trials=trials,
        label=label,
    )

    return results, summary


def _run_trial(tasks: list[TaskSpec], agent: Agent, trial: int) -> list[TaskRunResult]:
    results: list[TaskRunResult] = []

    for task in tasks:
        start = perf_counter()

        try:
            output = agent.run(task)
            error = None
        except Exception as exc:  # pragma: no cover - defensive for third-party agents
            output = AgentOutput(response="", tool_calls=[], steps=0)
            error = str(exc)

        latency_ms = round((perf_counter() - start) * 1000, 3)
        grade = (
            grade_task(task, output.response, output.tool_calls, output.final_files)
            if error is None
            else None
        )

        if error is not None:
            score = 0.0
            success = False
            grade_reason = f"agent error: {error}"
        else:
            score = grade.score
            success = grade.success
            grade_reason = grade.reason

        tool_precision, tool_recall = compute_tool_match(task.required_tools, output.tool_calls)
        capped_steps = min(output.steps, task.max_steps)
        step_ratio = (capped_steps / task.max_steps) if task.max_steps else 0.0

        results.append(
            TaskRunResult(
                task_id=task.task_id,
                split=task.split,
                category=task.category,
                difficulty=task.difficulty,
                success=success,
                score=round(score, 4),
                grade_reason=grade_reason,
                response=output.response,
                expected_response=task.expected_response,
                latency_ms=latency_ms,
                steps=output.steps,
                max_steps=task.max_steps,
                step_ratio=round(step_ratio, 4),
                tool_calls=output.tool_calls,
                required_tools=task.required_tools,
                tool_precision=round(tool_precision, 4),
                tool_recall=round(tool_recall, 4),
                error=error,
                trace=list(output.trace),
                trial=trial,
                grader_type=task.grader.type,
            )
        )

    return results


def write_run_artifacts(
    output_dir: Path,
    results: list[TaskRunResult],
    summary: RunSummary,
    categories: list[CategorySummary],
    run_config: dict[str, object],
    breakdowns: dict[str, list[CategorySummary]] | None = None,
) -> Path:
    """Write run artifacts to disk and return the run directory path."""

    run_dir = output_dir / summary.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    categories_path = run_dir / "categories.json"
    config_path = run_dir / "config.json"

    with results_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=True) + "\n")

    summary_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    categories_path.write_text(
        json.dumps([asdict(category) for category in categories], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if breakdowns is not None:
        (run_dir / "breakdowns.json").write_text(
            json.dumps(
                {dim: [asdict(item) for item in items] for dim, items in breakdowns.items()},
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
    config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=True), encoding="utf-8")
    return run_dir


def _collect_validation_errors(issues: list[ValidationIssue]) -> str:
    return "\n".join(f"- {issue.path}: {issue.message}" for issue in issues)


def _select_tasks(tasks: list[TaskSpec], max_tasks: int | None) -> list[TaskSpec]:
    ordered = sorted(tasks, key=lambda task: task.task_id)
    if max_tasks is None:
        return ordered
    return ordered[:max_tasks]


def run_from_directory(
    tasks_dir: Path,
    output_dir: Path,
    agent_name: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
    timeout_seconds: float = 45.0,
    run_id: str | None = None,
    only_split: str | None = "public",
    max_tasks: int | None = None,
    trials: int = 1,
    label: str | None = None,
) -> tuple[Path, RunSummary]:
    """Validate tasks, run benchmark, and persist artifacts."""

    if trials <= 0:
        raise ValueError("trials must be a positive integer")

    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be a positive integer")

    report = validate_tasks_dir(tasks_dir, only_split=only_split)
    if not report.is_valid:
        raise ValueError(f"Task validation failed:\n{_collect_validation_errors(report.issues)}")

    split_label = only_split if only_split is not None else "all"

    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"run-{timestamp}-{split_label}"

    tasks = _select_tasks(report.tasks, max_tasks=max_tasks)
    if not tasks:
        raise ValueError("No tasks selected after filtering")

    agent = get_agent(
        agent_name,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
    results, summary = run_benchmark(
        tasks, agent, run_id=run_id, split_label=split_label, trials=trials, label=label
    )
    categories = build_category_summaries(results)

    run_config: dict[str, object] = {
        "tasks_dir": str(tasks_dir),
        "split": split_label,
        "agent_name": agent_name,
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "timeout_seconds": timeout_seconds,
        "max_tasks": max_tasks,
        "trials": trials,
        "label": label,
        "selected_task_ids": [task.task_id for task in tasks],
    }

    run_dir = write_run_artifacts(
        output_dir,
        results,
        summary,
        categories,
        run_config,
        breakdowns={
            "category": categories,
            "difficulty": build_breakdown(results, "difficulty"),
            "grader_type": build_breakdown(results, "grader_type"),
        },
    )

    return run_dir, summary
