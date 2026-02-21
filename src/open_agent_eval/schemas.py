"""Task and run schema models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_SPLITS = {"public", "hidden"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
ALLOWED_GRADER_TYPES = {"exact", "contains", "json"}


class TaskSchemaError(ValueError):
    """Raised when a task file does not satisfy the expected schema."""


@dataclass(frozen=True)
class GraderSpec:
    """Defines how a task response should be graded."""

    type: str = "exact"
    ignore_case: bool = True
    collapse_whitespace: bool = True
    needles: list[str] = field(default_factory=list)
    allow_extra_keys: bool = False


@dataclass(frozen=True)
class TaskSpec:
    """Canonical task schema used by the evaluator."""

    task_id: str
    title: str
    category: str
    instructions: str
    expected_response: str
    split: str = "public"
    difficulty: str = "medium"
    required_tools: list[str] = field(default_factory=list)
    max_steps: int = 6
    tags: list[str] = field(default_factory=list)
    grader: GraderSpec = field(default_factory=GraderSpec)


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


def _optional_bool(config: dict[str, Any], key: str, source: Path, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise TaskSchemaError(f"{source}: grader.{key} must be a boolean")
    return value


def _parse_split(record: dict[str, Any], source: Path, inferred_split: str | None) -> str:
    split = record.get("split", inferred_split or "public")
    if not isinstance(split, str):
        raise TaskSchemaError(f"{source}: 'split' must be a string")

    split = split.strip().lower()
    if split not in ALLOWED_SPLITS:
        options = ", ".join(sorted(ALLOWED_SPLITS))
        raise TaskSchemaError(f"{source}: 'split' must be one of: {options}")

    if inferred_split is not None and split != inferred_split:
        raise TaskSchemaError(
            f"{source}: split mismatch (declared '{split}', directory indicates '{inferred_split}')"
        )

    return split


def _parse_difficulty(record: dict[str, Any], source: Path) -> str:
    difficulty = record.get("difficulty", "medium")
    if not isinstance(difficulty, str):
        raise TaskSchemaError(f"{source}: 'difficulty' must be a string")

    difficulty = difficulty.strip().lower()
    if difficulty not in ALLOWED_DIFFICULTIES:
        options = ", ".join(sorted(ALLOWED_DIFFICULTIES))
        raise TaskSchemaError(f"{source}: 'difficulty' must be one of: {options}")

    return difficulty


def _parse_grader(record: dict[str, Any], source: Path) -> GraderSpec:
    raw = record.get("grader")
    if raw is None:
        return GraderSpec()

    if not isinstance(raw, dict):
        raise TaskSchemaError(f"{source}: 'grader' must be an object")

    grader_type = raw.get("type", "exact")
    if not isinstance(grader_type, str):
        raise TaskSchemaError(f"{source}: grader.type must be a string")

    grader_type = grader_type.strip().lower()
    if grader_type not in ALLOWED_GRADER_TYPES:
        options = ", ".join(sorted(ALLOWED_GRADER_TYPES))
        raise TaskSchemaError(f"{source}: grader.type must be one of: {options}")

    ignore_case = _optional_bool(raw, "ignore_case", source, default=True)
    collapse_whitespace = _optional_bool(raw, "collapse_whitespace", source, default=True)

    needles = _optional_string_list(raw, "needles", source)
    allow_extra_keys = _optional_bool(raw, "allow_extra_keys", source, default=False)

    if grader_type == "contains" and not needles:
        raise TaskSchemaError(f"{source}: contains grader requires non-empty grader.needles")

    if grader_type != "contains" and needles:
        raise TaskSchemaError(f"{source}: grader.needles is only valid for contains grader")

    if grader_type != "json" and allow_extra_keys:
        raise TaskSchemaError(f"{source}: grader.allow_extra_keys is only valid for json grader")

    return GraderSpec(
        type=grader_type,
        ignore_case=ignore_case,
        collapse_whitespace=collapse_whitespace,
        needles=needles,
        allow_extra_keys=allow_extra_keys,
    )


def task_from_dict(record: dict[str, Any], source: Path, inferred_split: str | None = None) -> TaskSpec:
    """Build a ``TaskSpec`` from an untrusted dictionary."""

    if not isinstance(record, dict):
        raise TaskSchemaError(f"{source}: root JSON value must be an object")

    return TaskSpec(
        task_id=_require_string(record, "task_id", source),
        title=_require_string(record, "title", source),
        category=_require_string(record, "category", source),
        instructions=_require_string(record, "instructions", source),
        expected_response=_require_string(record, "expected_response", source),
        split=_parse_split(record, source, inferred_split=inferred_split),
        difficulty=_parse_difficulty(record, source),
        required_tools=_optional_string_list(record, "required_tools", source),
        max_steps=_optional_positive_int(record, "max_steps", source, default=6),
        tags=_optional_string_list(record, "tags", source),
        grader=_parse_grader(record, source),
    )
