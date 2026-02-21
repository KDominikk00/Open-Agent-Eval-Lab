"""Benchmark runner implementation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from open_agent_eval.agents import Agent, AgentOutput, get_agent
from open_agent_eval.graders import compute_tool_match, grade_response
from open_agent_eval.schemas import TaskSpec
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


@dataclass(frozen=True)
class CategorySummary:
    category: str
    total_tasks: int
    passed_tasks: int
    pass_rate: float
    average_score: float


def _build_summary(
    *,
    run_id: str,
    agent_name: str,
    split: str,
    results: list[TaskRunResult],
    started_at: datetime,
    finished_at: datetime,
) -> RunSummary:
    total = len(results)
    passed = sum(1 for result in results if result.success)

    def _avg(values: list[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    return RunSummary(
        run_id=run_id,
        agent_name=agent_name,
        split=split,
        total_tasks=total,
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
    )


def build_category_summaries(results: list[TaskRunResult]) -> list[CategorySummary]:
    """Compute per-category aggregate metrics from task results."""

    category_to_results: dict[str, list[TaskRunResult]] = {}
    for result in results:
        category_to_results.setdefault(result.category, []).append(result)

    summaries: list[CategorySummary] = []
    for category in sorted(category_to_results):
        bucket = category_to_results[category]
        total = len(bucket)
        passed = sum(1 for result in bucket if result.success)
        average_score = (sum(result.score for result in bucket) / total) if total else 0.0
        summaries.append(
            CategorySummary(
                category=category,
                total_tasks=total,
                passed_tasks=passed,
                pass_rate=round((passed / total) if total else 0.0, 4),
                average_score=round(average_score, 4),
            )
        )

    return summaries


def run_benchmark(
    tasks: list[TaskSpec],
    agent: Agent,
    run_id: str,
    split_label: str,
) -> tuple[list[TaskRunResult], RunSummary]:
    """Run one benchmark pass over tasks with a selected agent."""

    results: list[TaskRunResult] = []
    started_at = datetime.now(timezone.utc)

    for task in tasks:
        start = perf_counter()

        try:
            output = agent.run(task)
            error = None
        except Exception as exc:  # pragma: no cover - defensive for third-party agents
            output = AgentOutput(response="", tool_calls=[], steps=0)
            error = str(exc)

        latency_ms = round((perf_counter() - start) * 1000, 3)
        grade = grade_response(task, output.response) if error is None else None

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
            )
        )

    finished_at = datetime.now(timezone.utc)
    summary = _build_summary(
        run_id=run_id,
        agent_name=agent.name,
        split=split_label,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
    )

    return results, summary


def write_run_artifacts(
    output_dir: Path,
    results: list[TaskRunResult],
    summary: RunSummary,
    categories: list[CategorySummary],
    run_config: dict[str, object],
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
) -> tuple[Path, RunSummary]:
    """Validate tasks, run benchmark, and persist artifacts."""

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
    results, summary = run_benchmark(tasks, agent, run_id=run_id, split_label=split_label)
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
        "selected_task_ids": [task.task_id for task in tasks],
    }

    run_dir = write_run_artifacts(output_dir, results, summary, categories, run_config)

    return run_dir, summary
