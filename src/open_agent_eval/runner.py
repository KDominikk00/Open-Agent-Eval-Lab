"""Benchmark runner implementation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from open_agent_eval.agents import Agent, AgentOutput, get_agent
from open_agent_eval.schemas import TaskSpec
from open_agent_eval.validators import ValidationIssue, validate_tasks_dir


@dataclass(frozen=True)
class TaskRunResult:
    task_id: str
    success: bool
    response: str
    expected_response: str
    latency_ms: float
    steps: int
    tool_calls: list[str]
    error: str | None


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    agent_name: str
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    pass_rate: float
    average_latency_ms: float
    started_at_utc: str
    finished_at_utc: str


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip()


def _score_exact(actual: str, expected: str) -> bool:
    return _normalize(actual) == _normalize(expected)


def run_benchmark(tasks: list[TaskSpec], agent: Agent, run_id: str) -> tuple[list[TaskRunResult], RunSummary]:
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

        latency_ms = (perf_counter() - start) * 1000
        success = error is None and _score_exact(output.response, task.expected_response)

        results.append(
            TaskRunResult(
                task_id=task.task_id,
                success=success,
                response=output.response,
                expected_response=task.expected_response,
                latency_ms=round(latency_ms, 3),
                steps=output.steps,
                tool_calls=output.tool_calls,
                error=error,
            )
        )

    finished_at = datetime.now(timezone.utc)
    passed = sum(1 for result in results if result.success)
    total = len(results)
    average_latency = (sum(result.latency_ms for result in results) / total) if total else 0.0

    summary = RunSummary(
        run_id=run_id,
        agent_name=agent.name,
        total_tasks=total,
        passed_tasks=passed,
        failed_tasks=total - passed,
        pass_rate=round((passed / total) if total else 0.0, 4),
        average_latency_ms=round(average_latency, 3),
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
    )

    return results, summary


def write_run_artifacts(output_dir: Path, results: list[TaskRunResult], summary: RunSummary) -> Path:
    """Write run artifacts to disk and return the run directory path."""

    run_dir = output_dir / summary.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    with results_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=True) + "\n")

    summary_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    return run_dir


def _collect_validation_errors(issues: list[ValidationIssue]) -> str:
    return "\n".join(f"- {issue.path}: {issue.message}" for issue in issues)


def run_from_directory(tasks_dir: Path, output_dir: Path, agent_name: str, run_id: str | None = None) -> tuple[Path, RunSummary]:
    """Validate tasks, run benchmark, and persist artifacts."""

    report = validate_tasks_dir(tasks_dir)
    if not report.is_valid:
        raise ValueError(f"Task validation failed:\n{_collect_validation_errors(report.issues)}")

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")

    agent = get_agent(agent_name)
    results, summary = run_benchmark(report.tasks, agent, run_id=run_id)
    run_dir = write_run_artifacts(output_dir, results, summary)

    return run_dir, summary
