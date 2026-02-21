"""Task and run schema models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class TaskSchemaError(ValueError):
    """Raised when a task file does not satisfy the expected schema."""


@dataclass(frozen=True)
class TaskSpec:
    """Canonical task schema used by the evaluator."""

    task_id: str
    title: str
    category: str
    instructions: str
    expected_response: str
    required_tools: list[str] = field(default_factory=list)
    max_steps: int = 6
    tags: list[str] = field(default_factory=list)


def _require_string(record: dict[str, Any], key: str, source: Path) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskSchemaError(f"{source}: '{key}' must be a non-empty string")
    return value.strip()


def _optional_string_list(record: dict[str, Any], key: str, source: Path) -> list[str]:
    value = record.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise TaskSchemaError(f"{source}: '{key}' must be a list of non-empty strings")
    return [item.strip() for item in value]


def _optional_positive_int(record: dict[str, Any], key: str, source: Path, default: int) -> int:
    value = record.get(key, default)
    if not isinstance(value, int) or value <= 0:
        raise TaskSchemaError(f"{source}: '{key}' must be a positive integer")
    return value


def task_from_dict(record: dict[str, Any], source: Path) -> TaskSpec:
    """Build a ``TaskSpec`` from an untrusted dictionary."""

    if not isinstance(record, dict):
        raise TaskSchemaError(f"{source}: root JSON value must be an object")

    return TaskSpec(
        task_id=_require_string(record, "task_id", source),
        title=_require_string(record, "title", source),
        category=_require_string(record, "category", source),
        instructions=_require_string(record, "instructions", source),
        expected_response=_require_string(record, "expected_response", source),
        required_tools=_optional_string_list(record, "required_tools", source),
        max_steps=_optional_positive_int(record, "max_steps", source, default=6),
        tags=_optional_string_list(record, "tags", source),
    )
