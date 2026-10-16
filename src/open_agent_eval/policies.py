"""Offline scripted agents that exercise the episode loop without any API access.

They give the leaderboard a meaningful spread: ``oracle`` replays the reference
solution (ceiling), ``keyword-retriever`` is a heuristic search baseline, and
``random`` samples valid-looking tool calls (floor).
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, replace

from open_agent_eval.agents import (
    AgentOutput,
    EchoReferenceAgent,
    InstructionSnippetAgent,
    output_from_episode,
)
from open_agent_eval.episode import Action, FinalAnswer, ToolCall, TraceStep, run_episode
from open_agent_eval.schemas import TaskSpec
from open_agent_eval.tools import KNOWN_TOOLS, Environment

_STOPWORDS = frozenset(
    ["the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "from", "that", "this", "these", "those", "is", "are", "was", "were", "be", "been", "it", "its", "as", "at", "by", "your", "you", "we", "our", "which", "what", "when", "where", "how", "why", "use", "using", "then", "first", "second", "after", "before", "into", "out", "not", "any", "all", "each", "find", "tell", "give", "return", "report", "identify", "explain", "check"]
)


def _keywords(text: str, limit: int = 4) -> list[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_\-\.]{3,}", text)
    seen: dict[str, None] = {}
    for word in words:
        lowered = word.casefold()
        if lowered not in _STOPWORDS:
            seen.setdefault(lowered, None)
    return sorted(seen, key=len, reverse=True)[:limit]


def _enabled_tools(task: TaskSpec) -> list[str]:
    return sorted(task.tools) if task.tools else sorted(KNOWN_TOOLS)


@dataclass(frozen=True)
class OracleAgent:
    """Replays ``reference_trajectory`` then returns the expected answer (task-solvability ceiling)."""

    name: str = "oracle"

    def decide(self, task: TaskSpec, history: list[TraceStep]) -> Action:
        if len(history) < len(task.reference_trajectory):
            step = task.reference_trajectory[len(history)]
            return ToolCall(step["tool"], dict(step.get("args", {})))
        return FinalAnswer(task.expected_response)

    def run(self, task: TaskSpec) -> AgentOutput:
        if not task.environment:
            return EchoReferenceAgent().run(task)
        return output_from_episode(run_episode(task, self))


@dataclass(frozen=True)
class KeywordRetrieverAgent:
    """Heuristic baseline: searches logs/files/tables for instruction keywords and quotes what it finds."""

    name: str = "keyword-retriever"

    def decide(self, task: TaskSpec, history: list[TraceStep]) -> Action:
        tools = _enabled_tools(task)
        plan = self._plan(task, tools)
        if len(history) < len(plan):
            name, args = plan[len(history)]
            return ToolCall(name, args)
        observations = [step.observation for step in history if not step.is_error]
        return FinalAnswer("\n".join(observations)[:600])

    def _plan(self, task: TaskSpec, tools: list[str]) -> list[tuple[str, dict[str, object]]]:
        env = Environment(task.environment, tools=tools)
        words = _keywords(task.instructions)
        plan: list[tuple[str, dict[str, object]]] = []
        if "search_logs" in tools and env.logs:
            plan += [("search_logs", {"query": word}) for word in words[:2]]
        if "read_file" in tools:
            mentioned = [path for path in sorted(env.files) if path.casefold() in task.instructions.casefold()]
            plan += [("read_file", {"path": path}) for path in mentioned[:2]]
        if "db_query" in tools and env.tables:
            plan.append(("db_query", {"table": sorted(env.tables)[0]}))
        return plan[: max(1, task.max_steps - 1)]

    def run(self, task: TaskSpec) -> AgentOutput:
        if not task.environment:
            return InstructionSnippetAgent().run(task)
        return output_from_episode(run_episode(task, self))


@dataclass(frozen=True)
class RandomAgent:
    """Samples random valid tool calls and answers with a random observation line (chance floor)."""

    name: str = "random"
    seed: int = 0

    def with_trial(self, trial: int) -> RandomAgent:
        return replace(self, seed=self.seed + trial)

    def _rng(self, task: TaskSpec, step: int) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}:{task.task_id}:{step}".encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def decide(self, task: TaskSpec, history: list[TraceStep]) -> Action:
        rng = self._rng(task, len(history))
        if len(history) >= min(3, task.max_steps - 1) or rng.random() < 0.25:
            lines = [ln for step in history for ln in step.observation.splitlines() if ln]
            return FinalAnswer(rng.choice(lines) if lines else "I don't know")

        env = Environment(task.environment, tools=_enabled_tools(task))
        tool = rng.choice(_enabled_tools(task))
        words = _keywords(task.instructions, limit=8) or ["error"]
        args: dict[str, object]
        if tool in ("read_file", "write_file"):
            args = {"path": rng.choice(sorted(env.files) or ["notes.txt"])}
            if tool == "write_file":
                args["content"] = rng.choice(words)
        elif tool == "search_logs":
            args = {"query": rng.choice(words)}
        elif tool == "db_query":
            args = {"table": rng.choice(sorted(env.tables) or ["users"])}
        elif tool == "http_get":
            args = {"url": rng.choice(sorted(env.http) or ["https://example.com"])}
        elif tool == "calculator":
            args = {"expression": f"{rng.randint(1, 99)} + {rng.randint(1, 99)}"}
        else:
            args = {}
        return ToolCall(tool, args)

    def run(self, task: TaskSpec) -> AgentOutput:
        if not task.environment:
            return AgentOutput(response="I don't know", tool_calls=[], steps=1)
        return output_from_episode(run_episode(task, self))
