"""Task grading strategies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from open_agent_eval.schemas import GraderSpec, TaskSpec


@dataclass(frozen=True)
class GradeResult:
    success: bool
    score: float
    reason: str


def _normalize_text(text: str, grader: GraderSpec) -> str:
    value = text
    if grader.collapse_whitespace:
        value = " ".join(value.split())
    if grader.ignore_case:
        value = value.casefold()
    return value.strip()


def _grade_exact(expected: str, actual: str, grader: GraderSpec) -> GradeResult:
    if _normalize_text(expected, grader) == _normalize_text(actual, grader):
        return GradeResult(success=True, score=1.0, reason="exact match")
    return GradeResult(success=False, score=0.0, reason="response did not exactly match expected output")


def _grade_contains(actual: str, grader: GraderSpec) -> GradeResult:
    normalized_actual = _normalize_text(actual, grader)
    normalized_needles = [_normalize_text(needle, grader) for needle in grader.needles]

    missing = [needle for needle in normalized_needles if needle not in normalized_actual]
    if missing:
        return GradeResult(
            success=False,
            score=0.0,
            reason=f"response missing required snippet(s): {', '.join(missing)}",
        )

    return GradeResult(success=True, score=1.0, reason="all required snippets were found")


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(key in actual and _is_subset(value, actual[key]) for key, value in expected.items())

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if len(expected) > len(actual):
            return False
        return all(_is_subset(expected[i], actual[i]) for i in range(len(expected)))

    return bool(expected == actual)


def _grade_json(expected: str, actual: str, grader: GraderSpec) -> GradeResult:
    try:
        expected_obj = json.loads(expected)
    except json.JSONDecodeError:
        return GradeResult(success=False, score=0.0, reason="expected_response is not valid JSON")

    try:
        actual_obj = json.loads(actual)
    except json.JSONDecodeError:
        return GradeResult(success=False, score=0.0, reason="response is not valid JSON")

    if grader.allow_extra_keys:
        success = _is_subset(expected_obj, actual_obj)
    else:
        success = expected_obj == actual_obj

    if success:
        return GradeResult(success=True, score=1.0, reason="json matched")

    return GradeResult(success=False, score=0.0, reason="json did not match")


def grade_response(task: TaskSpec, actual_response: str) -> GradeResult:
    """Apply the task's configured grader to an agent response."""

    grader = task.grader

    if grader.type == "exact":
        return _grade_exact(task.expected_response, actual_response, grader)
    if grader.type == "contains":
        return _grade_contains(actual_response, grader)
    if grader.type == "json":
        return _grade_json(task.expected_response, actual_response, grader)

    return GradeResult(success=False, score=0.0, reason=f"unknown grader type: {grader.type}")


def _is_subsequence(wanted: list[str], actual: list[str]) -> bool:
    remaining = iter(actual)
    return all(name in remaining for name in wanted)


def grade_trace(
    task: TaskSpec, called_tools: list[str], final_files: dict[str, str]
) -> GradeResult:
    """Check behavioral constraints: required/ordered/forbidden calls and final sandbox state."""

    grader = task.grader
    problems: list[str] = []

    missing = [name for name in grader.required_calls if name not in called_tools]
    if missing:
        problems.append(f"required tool call(s) missing: {', '.join(missing)}")
    if grader.call_order and not _is_subsequence(grader.call_order, called_tools):
        problems.append(f"tools were not called in order: {' -> '.join(grader.call_order)}")
    forbidden = sorted({name for name in called_tools if name in grader.forbidden_calls})
    if forbidden:
        problems.append(f"forbidden tool call(s) made: {', '.join(forbidden)}")
    for path, expected in grader.final_files.items():
        actual = final_files.get(path)
        if actual is None:
            problems.append(f"expected file '{path}' was not written")
        elif _normalize_text(actual, grader) != _normalize_text(expected, grader):
            problems.append(f"file '{path}' has unexpected content")

    if problems:
        return GradeResult(success=False, score=0.0, reason="; ".join(problems))
    return GradeResult(success=True, score=1.0, reason="trace constraints satisfied")


def grade_task(
    task: TaskSpec,
    response: str,
    called_tools: list[str] | None = None,
    final_files: dict[str, str] | None = None,
) -> GradeResult:
    """Grade the final answer and, for episode tasks, the behavior that produced it."""

    answer = grade_response(task, response)
    reasons = [answer.reason]
    success = answer.success

    grader = task.grader
    if grader.forbidden_needles:
        normalized = _normalize_text(response, grader)
        leaked = [n for n in grader.forbidden_needles if _normalize_text(n, grader) in normalized]
        if leaked:
            success = False
            reasons.append(f"response contains forbidden snippet(s): {', '.join(leaked)}")

    if task.environment:
        trace = grade_trace(task, called_tools or [], final_files or {})
        success = success and trace.success
        if not trace.success:
            reasons.append(trace.reason)

    return GradeResult(
        success=success,
        score=1.0 if success else 0.0,
        reason="; ".join(reasons) if not success else answer.reason,
    )


def compute_tool_match(required_tools: list[str], called_tools: list[str]) -> tuple[float, float]:
    """Return tool precision and recall."""

    required = set(required_tools)
    called = set(called_tools)
    correct = required & called

    if called:
        precision = len(correct) / len(called)
    else:
        precision = 1.0 if not required else 0.0

    if required:
        recall = len(correct) / len(required)
    else:
        recall = 1.0

    return precision, recall
