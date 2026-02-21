"""Task validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from open_agent_eval.schemas import TaskSchemaError, TaskSpec, task_from_dict


@dataclass(frozen=True)
class ValidationIssue:
    path: Path
    message: str


@dataclass
class ValidationReport:
    tasks: list[TaskSpec] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues


def _iter_task_files(tasks_dir: Path) -> list[Path]:
    return sorted(path for path in tasks_dir.glob("*.json") if path.is_file())


def validate_tasks_dir(tasks_dir: Path) -> ValidationReport:
    """Validate all task JSON files in ``tasks_dir``."""

    report = ValidationReport()
    seen_ids: dict[str, Path] = {}

    if not tasks_dir.exists() or not tasks_dir.is_dir():
        report.issues.append(ValidationIssue(tasks_dir, "tasks directory does not exist"))
        return report

    task_files = _iter_task_files(tasks_dir)
    if not task_files:
        report.issues.append(ValidationIssue(tasks_dir, "no task JSON files found"))
        return report

    for task_file in task_files:
        try:
            raw = json.loads(task_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.issues.append(ValidationIssue(task_file, f"invalid JSON: {exc.msg}"))
            continue

        try:
            task = task_from_dict(raw, task_file)
        except TaskSchemaError as exc:
            report.issues.append(ValidationIssue(task_file, str(exc)))
            continue

        if task.task_id in seen_ids:
            first_path = seen_ids[task.task_id]
            report.issues.append(
                ValidationIssue(
                    task_file,
                    f"duplicate task_id '{task.task_id}' (first seen in {first_path.name})",
                )
            )
            continue

        seen_ids[task.task_id] = task_file
        report.tasks.append(task)

    return report
